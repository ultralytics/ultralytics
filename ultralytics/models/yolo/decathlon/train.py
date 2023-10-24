# Ultralytics YOLO 🚀, AGPL-3.0 license

import torch
import torchvision
from torch import nn, optim

import ultralytics.nn.modules.head
from ultralytics.data import build_dataloader
from ultralytics.engine.trainer import BaseTrainer
from ultralytics.models import yolo
from ultralytics.nn.tasks import ClassificationModel, SegmentationModel, attempt_load_one_weight
from ultralytics.utils import DEFAULT_CFG, LOGGER, RANK, colorstr
from ultralytics.utils.plotting import plot_images, plot_results
from ultralytics.utils.torch_utils import is_parallel, strip_optimizer, torch_distributed_zero_first


class MultiHeadModel(ultralytics.nn.tasks.BaseModel):

    def __init__(self, model, head, src_model):
        super().__init__()
        self.__dict__.update(src_model.__dict__)  # copy all object fields
        modules = model + [head]
        self.model = torch.nn.Sequential(*modules)
        self.criterion = src_model.init_criterion()


class DecathlonTrainer(BaseTrainer):
    """
    A class extending the BaseTrainer class for training based on a classification model.

    Notes:
        - Torchvision classification models can also be passed to the 'model' argument, i.e. model='resnet18'.

    Example:
        ```python
        from ultralytics.models.yolo.classify import ClassificationTrainer

        args = dict(model='yolov8n-cls.pt', data='imagenet10', epochs=3)
        trainer = ClassificationTrainer(overrides=args)
        trainer.train()
        ```
    """

    def __init__(self, cfg=DEFAULT_CFG, overrides=None, _callbacks=None):
        """Initialize a DecathlonTrainer object with optional configuration overrides and callbacks."""
        self.overrides = overrides
        if overrides is None:
            overrides = {}
        overrides['task'] = 'decathlon'
        if overrides.get('imgsz') is None:
            overrides['imgsz'] = 224
        overrides['model'] = 'yolov8.yaml'
        super().__init__(cfg, overrides, _callbacks)

        self.trainers = None
        self.models = None
        self.predict_heads = None
        self.curr_trainer = None
        self.cfg = cfg
        self.fitness = 0

    def set_model_attributes(self):
        pass

    @staticmethod
    def get_training_program():
        return [
            #{'task': 'classify', 'data': 'cifar100'},
            #{'task': 'classify', 'data': 'imagenet'},
            {
                'task': 'classify',
                'data': 'caltech101'},
            #{'task': 'segment', 'data': 'coco-seg.yaml'},
            #{'task': 'pose', 'data': 'coco-pose.yaml'},
            {
                'task': 'detect',
                'data': 'SKU-110K.yaml'}, ]

    def setup_decathlon_trainers(self):
        if self.trainers is not None: return

        from ultralytics.models.yolo.model import YOLO
        task_map = YOLO().task_map

        self.trainers = []
        for decathlon_elem in self.get_training_program():
            task = decathlon_elem['task']

            sub_trainer_class = task_map[task]['trainer']
            overrides = {k: v for k, v in self.overrides.items()}
            overrides['task'] = task
            overrides['data'] = decathlon_elem['data']
            sub_trainer = sub_trainer_class(cfg=self.cfg, overrides=overrides)

            model = task_map[task]['model']()
            sub_trainer.model = model
            sub_trainer.model.args = self.args

            self.trainers.append(sub_trainer)

    def get_model(self, cfg=None, weights=None, verbose=True):
        """Returns a modified PyTorch model configured for training YOLO."""

        self.setup_decathlon_trainers()

        number_classes = 1
        model = SegmentationModel(cfg, ch=3, nc=number_classes, verbose=verbose
                                  and RANK == -1)  # arbitrarily take a model with 22 layers
        model.args = self.args
        del model.model[-1]  #remove prediction head

        self.models, self.predict_heads = [], []
        for trainer, task_data in zip(self.trainers, DecathlonTrainer.get_training_program()):
            task = task_data['task']
            head = trainer.model.model[-1]
            del trainer.model.model[-1]
            if task == 'classify':
                newmodel = MultiHeadModel(list(model.to(self.device).model)[0:8], head, trainer.model)  #
                ClassificationModel.reshape_outputs(newmodel, trainer.data['nc'])
            else:
                newmodel = MultiHeadModel(list(model.to(self.device).model), head, trainer.model)
            trainer.model = newmodel
            self.models.append(newmodel)
            self.predict_heads.append(head)
            trainer.model = newmodel

        return model

    def _setup_train(self, world_size):

        self.setup_decathlon_trainers()

        for st in self.trainers:
            st._setup_train(world_size)

        super()._setup_train(world_size)

        for m in self.model.modules():
            if not self.args.pretrained and hasattr(m, 'reset_parameters'):
                m.reset_parameters()
            if isinstance(m, torch.nn.Dropout) and self.args.dropout:
                m.p = self.args.dropout  # set dropout
        for p in self.model.parameters():
            p.requires_grad = True  # for training
        for h in self.predict_heads:
            for name, p in self.model.named_parameters():
                if name.endswith('.dfl.conv.weight'):
                    p.requires_grad = False

        self.models = [m.to(self.device) for m in self.models]
        self.predict_heads = [h.to(self.device) for h in self.predict_heads]

    def build_dataset(self, img_path, mode='train', batch=None):
        """No dataset, each sub task has its own dataset."""
        return None

    def get_dataloader(self, dataset_path, batch_size=16, rank=0, mode='train'):
        """Instantiate the dataloader for each sub task."""
        self.setup_decathlon_trainers()

        # switch between datasets
        # if a dataset is much shorter than others, this will
        # sample from the short dataset too much
        class FakeDataset():

            def __init__(self, len):
                self.len = len

            def __len__(self):
                return self.len

        class StripingIterator():

            def __init__(self, striping_loader):
                self.sub_loaders = striping_loader.sub_loaders
                self.iterators = [sl.__iter__() for sl in striping_loader.sub_loaders]
                self.current = -1
                self.num_loop = 0
                self.max_loop = max([len(sl.dataset) for sl in striping_loader.sub_loaders])

            def __next__(self):
                self.current += 1
                if self.current >= len(self.sub_loaders):
                    self.current = 0
                    self.num_loop += 1
                    if self.num_loop >= self.max_loop:
                        self.num_loop = 0
                        raise StopIteration
                try:
                    ret_value = next(self.iterators[self.current])
                except StopIteration:
                    self.iterators[self.current] = self.sub_loaders[self.current].__iter__()
                    ret_value = next(self.iterators[self.current])
                return ret_value

            def _reset(self):
                self.current = -1
                self.iterators = [sl.__iter__() for sl in self.sub_loaders]

        class StripingDataLoader():

            def __init__(self, sub_loaders, sub_trainers, multi_threaded):
                self.sub_loaders = sub_loaders
                self.sub_trainers = sub_trainers
                self.multi_threaded = multi_threaded
                self.len = max([len(sl.dataset) for sl in self.sub_loaders])
                self.num_workers = sum([sl.num_workers for sl in self.sub_loaders]) / len(self.sub_loaders)
                self.dataset = FakeDataset(self.len)
                self.multi_threaded = self.num_workers > 0
                self._iterator = None

            def __len__(self):
                return self.len * len(self.sub_loaders)

            def __iter__(self):
                if self.multi_threaded:
                    if self._iterator is None:
                        self._iterator = self._get_iterator()
                    else:
                        self._iterator._reset()
                    return self._iterator
                else:
                    self._iterator = self._get_iterator()  #we're peeking into the iterator and cannot iterate twice :/
                    return self._iterator

            def _get_iterator(self):
                return StripingIterator(self)

        sub_loaders = [st.train_loader for st in self.trainers]
        return StripingDataLoader(sub_loaders, self.trainers, multi_threaded=RANK != 1)

    def switch_trainer(self, trainer_num):
        self.curr_trainer = self.trainers[trainer_num]

        self.loss = self.curr_trainer.loss
        self.model = self.curr_trainer.model
        self.loss_names = self.curr_trainer.loss_names
        self.amp = self.curr_trainer.amp
        self.ema = self.curr_trainer.ema
        self.scaler = self.curr_trainer.scaler
        self.validator = self.curr_trainer.validator

    def preprocess_batch(self, batch):
        """Preprocesses a batch of images and classes."""
        if self.curr_trainer is not None:
            self.curr_trainer.loss = self.loss
            self.curr_trainer.loss_items = self.loss_items
            self.curr_trainer.epoch = self.epoch

        curr_sub_task = self.train_loader._iterator.current  # peeks into the iterator :/
        self.switch_trainer(curr_sub_task)

        # in case switching model gives memory fragmentation problems
        # torch.cuda.empty_cache()
        # import gc
        # gc.collect()

        return self.curr_trainer.preprocess_batch(batch)

    def progress_string(self):
        """Returns a formatted string showing training progress."""
        return ('\n' + '%11s' * (4 + len(self.loss_names))) % \
            ('Epoch', 'GPU_mem', *self.loss_names, 'Instances', 'Size')

    def get_validator(self):
        """DecathlonTrainer has no validator and uses sub-tasks for validation."""
        return yolo.decathlon.DecathlonValidator(self.test_loader, self.save_dir)

    def validate(self):
        """
        Runs validation on test set using self.validator.

        The returned dict is expected to contain "fitness" key.
        """
        all_metrics = {}
        for i, t in enumerate(self.trainers):
            self.switch_trainer(i)
            metrics, fitness = t.validate()
            metrics['fitness'] = fitness
            prefix = DecathlonTrainer.get_training_program()[i]['task'] + '_' + DecathlonTrainer.get_training_program(
            )[i]['data']
            for k, v in metrics.items():
                all_metrics[prefix + '_' + k] = v

        self.fitness -= 1  #no early stopping for multiple objective functions, a fitness function that returns a tuple does not have a total order

        return all_metrics, self.fitness

    def label_loss_items(self, loss_items=None, prefix='train'):
        """
        Returns a loss dict with labelled training loss items tensor.

        Not needed for classification but necessary for segmentation & detection
        """
        return ['val/Loss']

    def final_eval(self):
        """Evaluate trained model and save validation results."""
        for i, t in enumerate(self.trainers):
            self.switch_trainer(i)
            t.final_eval()  # TODO check saved names do not overlap

    def build_optimizer(self, model, name='auto', lr=0.001, momentum=0.9, decay=1e-5, iterations=1e5):
        """
        Constructs an optimizer for the given model, based on the specified optimizer name, learning rate, momentum,
        weight decay, and number of iterations.

        Args:
            model (torch.nn.Module): The model for which to build an optimizer.
            name (str, optional): The name of the optimizer to use. If 'auto', the optimizer is selected
                based on the number of iterations. Default: 'auto'.
            lr (float, optional): The learning rate for the optimizer. Default: 0.001.
            momentum (float, optional): The momentum factor for the optimizer. Default: 0.9.
            decay (float, optional): The weight decay for the optimizer. Default: 1e-5.
            iterations (float, optional): The number of iterations, which determines the optimizer if
                name is 'auto'. Default: 1e5.

        Returns:
            (torch.optim.Optimizer): The constructed optimizer.
        """

        # because there is a base model and different prediction heads, the optimiser
        # code is copy pasted to get all parameters from all sub-models

        g = [[], [], []]  # optimizer parameter groups
        bn = tuple(v for k, v in nn.__dict__.items() if 'Norm' in k)  # normalization layers, i.e. BatchNorm2d()
        if name == 'auto':
            LOGGER.info(f"{colorstr('optimizer:')} 'optimizer=auto' found, "
                        f"ignoring 'lr0={self.args.lr0}' and 'momentum={self.args.momentum}' and "
                        f"determining best 'optimizer', 'lr0' and 'momentum' automatically... ")
            nc = getattr(model, 'nc', 10)  # number of classes
            lr_fit = round(0.002 * 5 / (4 + nc), 6)  # lr0 fit equation to 6 decimal places
            name, lr, momentum = ('SGD', 0.01, 0.9) if iterations > 10000 else ('AdamW', lr_fit, 0.9)
            self.args.warmup_bias_lr = 0.0  # no higher than 0.01 for Adam

        for st in self.predict_heads + [self.model.model]:
            for module_name, module in st.named_modules():
                for param_name, param in module.named_parameters(recurse=False):
                    fullname = f'{module_name}.{param_name}' if module_name else param_name
                    if 'bias' in fullname:  # bias (no decay)
                        g[2].append(param)
                    elif isinstance(module, bn):  # weight (no decay)
                        g[1].append(param)
                    else:  # weight (with decay)
                        g[0].append(param)

        if name in ('Adam', 'Adamax', 'AdamW', 'NAdam', 'RAdam'):
            optimizer = getattr(optim, name, optim.Adam)(g[2], lr=lr, betas=(momentum, 0.999), weight_decay=1e-16)
        elif name == 'RMSProp':
            optimizer = optim.RMSprop(g[2], lr=lr, momentum=momentum)
        elif name == 'SGD':
            optimizer = optim.SGD(g[2], lr=lr, momentum=momentum, nesterov=True)
        else:
            raise NotImplementedError(
                f"Optimizer '{name}' not found in list of available optimizers "
                f'[Adam, AdamW, NAdam, RAdam, RMSProp, SGD, auto].'
                'To request support for addition optimizers please visit https://github.com/ultralytics/ultralytics.')

        optimizer.add_param_group({'params': g[0], 'weight_decay': decay})  # add g0 with weight_decay
        optimizer.add_param_group({'params': g[1], 'weight_decay': 1e-16})  # add g1 (BatchNorm2d weights)
        LOGGER.info(
            f"{colorstr('optimizer:')} {type(optimizer).__name__}(lr={lr}, momentum={momentum}) with parameter groups "
            f'{len(g[1])} weight(decay=0.0), {len(g[0])} weight(decay={decay}), {len(g[2])} bias(decay=0.0)')
        return optimizer
