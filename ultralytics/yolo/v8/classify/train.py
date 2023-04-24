# Ultralytics YOLO 🚀, AGPL-3.0 license

import torch
import torchvision

from ultralytics.nn.tasks import ClassificationModel, attempt_load_one_weight
from ultralytics.yolo import v8
from ultralytics.yolo.data import build_classification_dataloader
from ultralytics.yolo.engine.trainer import BaseTrainer
from ultralytics.yolo.utils import DEFAULT_CFG, LOGGER, RANK, colorstr
from ultralytics.yolo.utils.torch_utils import is_parallel, strip_optimizer


class ClassificationTrainer(BaseTrainer):

    def __init__(self, cfg=DEFAULT_CFG, overrides=None, _callbacks=None):
        """Initialize a ClassificationTrainer object with optional configuration overrides and callbacks."""
        if overrides is None:
            overrides = {}
        overrides['task'] = 'classify'
        super().__init__(cfg, overrides, _callbacks)

    def set_model_attributes(self):
        """Set the YOLO model's class names from the loaded dataset."""
        self.model.names = self.data['names']

    def get_model(self, cfg=None, weights=None, verbose=True):
        """Returns a modified PyTorch model configured for training YOLO."""
        model = ClassificationModel(cfg, nc=self.data['nc'], verbose=verbose and RANK == -1)
        if weights:
            model.load(weights)

        pretrained = self.args.pretrained
        for m in model.modules():
            if not pretrained and hasattr(m, 'reset_parameters'):
                m.reset_parameters()
            if isinstance(m, torch.nn.Dropout) and self.args.dropout:
                m.p = self.args.dropout  # set dropout
        for p in model.parameters():
            p.requires_grad = True  # for training

        # Update defaults
        if self.args.imgsz == 640:
            self.args.imgsz = 224

        return model

    def setup_model(self):
        """
        load/create/download model for any task
        """
        # Classification models require special handling

        if isinstance(self.model, torch.nn.Module):  # if model is loaded beforehand. No setup needed
            return

        model = str(self.model)
        # Load a YOLO model locally, from torchvision, or from Ultralytics assets
        if model.endswith('.pt'):
            self.model, _ = attempt_load_one_weight(model, device='cpu')
            for p in self.model.parameters():
                p.requires_grad = True  # for training
        elif model.endswith('.yaml'):
            self.model = self.get_model(cfg=model)
        elif model in torchvision.models.__dict__:
            pretrained = True
            self.model = torchvision.models.__dict__[model](weights='IMAGENET1K_V1' if pretrained else None)
        else:
            FileNotFoundError(f'ERROR: model={model} not found locally or online. Please check model name.')
        ClassificationModel.reshape_outputs(self.model, self.data['nc'])

        return  # dont return ckpt. Classification doesn't support resume

    def get_dataloader(self, dataset_path, batch_size=16, rank=0, mode='train'):
        """Returns PyTorch DataLoader with transforms to preprocess images for inference."""
        loader = build_classification_dataloader(path=dataset_path,
                                                 imgsz=self.args.imgsz,
                                                 batch_size=batch_size if mode == 'train' else (batch_size * 2),
                                                 augment=mode == 'train',
                                                 rank=rank,
                                                 workers=self.args.workers)
        # Attach inference transforms
        if mode != 'train':
            if is_parallel(self.model):
                self.model.module.transforms = loader.dataset.torch_transforms
            else:
                self.model.transforms = loader.dataset.torch_transforms
        return loader

    def preprocess_batch(self, batch):
        """Preprocesses a batch of images and classes."""
        batch['img'] = batch['img'].to(self.device)
        batch['cls'] = batch['cls'].to(self.device)
        return batch

    def progress_string(self):
        """Returns a formatted string showing training progress."""
        return ('\n' + '%11s' * (4 + len(self.loss_names))) % \
            ('Epoch', 'GPU_mem', *self.loss_names, 'Instances', 'Size')

    def get_validator(self):
        """Returns an instance of ClassificationValidator for validation."""
        self.loss_names = ['loss']
        return v8.classify.ClassificationValidator(self.test_loader, self.save_dir)

    def criterion(self, preds, batch):
        """Compute the classification loss between predictions and true labels."""
        loss = torch.nn.functional.cross_entropy(preds, batch['cls'], reduction='sum') / self.args.nbs
        loss_items = loss.detach()
        return loss, loss_items

    def label_loss_items(self, loss_items=None, prefix='train'):
        """
        Returns a loss dict with labelled training loss items tensor
        """
        # Not needed for classification but necessary for segmentation & detection
        keys = [f'{prefix}/{x}' for x in self.loss_names]
        if loss_items is None:
            return keys
        loss_items = [round(float(loss_items), 5)]
        return dict(zip(keys, loss_items))

    def resume_training(self, ckpt):
        """Resumes training from a given checkpoint."""
        pass

    def final_eval(self):
        """Evaluate trained model and save validation results."""
        for f in self.last, self.best:
            if f.exists():
                strip_optimizer(f)  # strip optimizers
                # TODO: validate best.pt after training completes
                # if f is self.best:
                #     LOGGER.info(f'\nValidating {f}...')
                #     self.validator.args.save_json = True
                #     self.metrics = self.validator(model=f)
                #     self.metrics.pop('fitness', None)
                #     self.run_callbacks('on_fit_epoch_end')
        LOGGER.info(f"Results saved to {colorstr('bold', self.save_dir)}")


def train(cfg=DEFAULT_CFG, use_python=False):
    """Train the YOLO classification model."""
    model = cfg.model or 'yolov8n-cls.pt'  # or "resnet18"
    data = cfg.data or 'mnist160'  # or yolo.ClassificationDataset("mnist")
    device = cfg.device if cfg.device is not None else ''

    args = dict(model=model, data=data, device=device)
    if use_python:
        from ultralytics import YOLO
        YOLO(model).train(**args)
    else:
        trainer = ClassificationTrainer(overrides=args)
        trainer.train()


if __name__ == '__main__':
    train()
