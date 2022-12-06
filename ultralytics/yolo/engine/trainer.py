"""
Simple training loop; Boilerplate that could apply to any arbitrary neural network,
"""

import os
import time
from collections import defaultdict
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from typing import Dict, Union

import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
from omegaconf import OmegaConf
from torch.cuda import amp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import lr_scheduler
from tqdm import tqdm

import ultralytics.yolo.utils as utils
import ultralytics.yolo.utils.callbacks as callbacks
from ultralytics.yolo.data.utils import check_dataset, check_dataset_yaml
from ultralytics.yolo.utils import LOGGER, ROOT, TQDM_BAR_FORMAT, colorstr
from ultralytics.yolo.utils.checks import check_file, print_args
from ultralytics.yolo.utils.configs import get_config
from ultralytics.yolo.utils.files import get_latest_run, increment_path, save_yaml
from ultralytics.yolo.utils.modeling import get_model
from ultralytics.yolo.utils.torch_utils import ModelEMA, de_parallel, init_seeds, one_cycle, strip_optimizer

DEFAULT_CONFIG = ROOT / "yolo/utils/configs/default.yaml"
RANK = int(os.getenv('RANK', -1))


class BaseTrainer:

    def __init__(self, config=DEFAULT_CONFIG, overrides={}):
        self.args = get_config(config, overrides)
        self.check_resume()
        init_seeds(self.args.seed + 1 + RANK, deterministic=self.args.deterministic)

        self.console = LOGGER
        self.validator = None
        self.model = None
        self.callbacks = defaultdict(list)
        self.save_dir = increment_path(Path(self.args.project) / self.args.name, exist_ok=self.args.exist_ok)
        self.wdir = self.save_dir / 'weights'  # weights dir
        self.wdir.mkdir(parents=True, exist_ok=True)  # make dir
        self.last, self.best = self.wdir / 'last.pt', self.wdir / 'best.pt'  # checkpoint paths
        self.batch_size = self.args.batch_size
        self.epochs = self.args.epochs
        self.start_epoch = 0
        print_args(dict(self.args))

        # Save run settings
        save_yaml(self.save_dir / 'args.yaml', OmegaConf.to_container(self.args, resolve=True))

        # device
        self.device = utils.torch_utils.select_device(self.args.device, self.batch_size)
        self.scaler = amp.GradScaler(enabled=self.device.type != 'cpu')

        # Model and Dataloaders.
        self.data = self.args.data
        if self.data.endswith(".yaml"):
            self.data = check_dataset_yaml(self.data)
        else:
            self.data = check_dataset(self.data)
        self.trainset, self.testset = self.get_dataset(self.data)
        self.ema = None

        # Optimization utils init
        self.lf = None
        self.scheduler = None

        # epoch level metrics
        self.best_fitness = None
        self.fitness = None
        self.loss = None
        self.tloss = None
        self.csv = self.save_dir / 'results.csv'

        for callback, func in callbacks.default_callbacks.items():
            self.add_callback(callback, func)
        callbacks.add_integration_callbacks(self)

    def add_callback(self, onevent: str, callback):
        """
        appends the given callback
        """
        self.callbacks[onevent].append(callback)

    def set_callback(self, onevent: str, callback):
        """
        overrides the existing callbacks with the given callback
        """
        self.callbacks[onevent] = [callback]

    def trigger_callbacks(self, onevent: str):
        for callback in self.callbacks.get(onevent, []):
            callback(self)

    def train(self):
        world_size = torch.cuda.device_count()
        if world_size > 1:
            mp.spawn(self._do_train, args=(world_size,), nprocs=world_size, join=True)
        else:
            # self._do_train(int(os.getenv("RANK", -1)), world_size)
            self._do_train()

    def _setup_ddp(self, rank, world_size):
        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = '9020'
        torch.cuda.set_device(rank)
        self.device = torch.device('cuda', rank)
        self.console.info(f"RANK - WORLD_SIZE - DEVICE: {rank} - {world_size} - {self.device} ")

        dist.init_process_group("nccl" if dist.is_nccl_available() else "gloo", rank=rank, world_size=world_size)

    def _setup_train(self, rank, world_size):
        """
        Builds dataloaders and optimizer on correct rank process
        """
        # model
        ckpt = self.setup_model()
        self.set_model_attributes()
        if world_size > 1:
            self.model = DDP(self.model, device_ids=[rank])
        # Optimizer
        self.accumulate = max(round(self.args.nbs / self.batch_size), 1)  # accumulate loss before optimizing
        self.args.weight_decay *= self.batch_size * self.accumulate / self.args.nbs  # scale weight_decay
        self.optimizer = build_optimizer(model=self.model,
                                         name=self.args.optimizer,
                                         lr=self.args.lr0,
                                         momentum=self.args.momentum,
                                         decay=self.args.weight_decay)
        # Scheduler
        if self.args.cos_lr:
            self.lf = one_cycle(1, self.args.lrf, self.epochs)  # cosine 1->hyp['lrf']
        else:
            self.lf = lambda x: (1 - x / self.epochs) * (1.0 - self.args.lrf) + self.args.lrf  # linear
        self.scheduler = lr_scheduler.LambdaLR(self.optimizer, lr_lambda=self.lf)
        self.resume_training(ckpt)
        self.scheduler.last_epoch = self.start_epoch - 1  # do not move

        # dataloaders
        batch_size = self.batch_size // world_size
        self.train_loader = self.get_dataloader(self.trainset, batch_size=batch_size, rank=rank, mode="train")
        if rank in {0, -1}:
            self.test_loader = self.get_dataloader(self.testset, batch_size=batch_size * 2, rank=-1, mode="val")
            validator = self.get_validator()
            # init metric, for plot_results
            metric_keys = validator.metric_keys + self.label_loss_items(prefix="val")
            self.metrics = dict(zip(metric_keys, [0] * len(metric_keys)))
            self.validator = validator
            self.ema = ModelEMA(self.model)

    def _do_train(self, rank=-1, world_size=1):
        if world_size > 1:
            self._setup_ddp(rank, world_size)

        self._setup_train(rank, world_size)
        self.trigger_callbacks("before_train")

        self.epoch_time = None
        self.epoch_time_start = time.time()
        self.train_time_start = time.time()
        nb = len(self.train_loader)  # number of batches
        nw = max(round(self.args.warmup_epochs * nb), 100)  # number of warmup iterations
        last_opt_step = -1
        for epoch in range(self.start_epoch, self.epochs):
            self.epoch = epoch
            self.trigger_callbacks("on_epoch_start")
            self.model.train()
            if rank != -1:
                self.train_loader.sampler.set_epoch(epoch)
            pbar = enumerate(self.train_loader)
            if rank in {-1, 0}:
                self.console.info(self.progress_string())
                pbar = tqdm(enumerate(self.train_loader), total=len(self.train_loader), bar_format=TQDM_BAR_FORMAT)
            self.tloss = None
            self.optimizer.zero_grad()
            for i, batch in pbar:
                self.trigger_callbacks("on_batch_start")
                # forward
                batch = self.preprocess_batch(batch)

                # warmup
                ni = i + nb * epoch
                if ni <= nw:
                    xi = [0, nw]  # x interp
                    self.accumulate = max(1, np.interp(ni, xi, [1, self.args.nbs / self.batch_size]).round())
                    for j, x in enumerate(self.optimizer.param_groups):
                        # bias lr falls from 0.1 to lr0, all other lrs rise from 0.0 to lr0
                        x['lr'] = np.interp(
                            ni, xi, [self.args.warmup_bias_lr if j == 0 else 0.0, x['initial_lr'] * self.lf(epoch)])
                        if 'momentum' in x:
                            x['momentum'] = np.interp(ni, xi, [self.args.warmup_momentum, self.args.momentum])

                preds = self.model(batch["img"])
                self.loss, self.loss_items = self.criterion(preds, batch)
                if rank != -1:
                    self.loss *= world_size
                self.tloss = (self.tloss * i + self.loss_items) / (i + 1) if self.tloss is not None \
                                else self.loss_items

                # backward
                self.scaler.scale(self.loss).backward()

                # optimize
                if ni - last_opt_step >= self.accumulate:
                    self.optimizer_step()
                    last_opt_step = ni

                # log
                mem = f'{torch.cuda.memory_reserved() / 1E9 if torch.cuda.is_available() else 0:.3g}G'  # (GB)
                loss_len = self.tloss.shape[0] if len(self.tloss.size()) else 1
                losses = self.tloss if loss_len > 1 else torch.unsqueeze(self.tloss, 0)
                if rank in {-1, 0}:
                    pbar.set_description(
                        ('%11s' * 2 + '%11.4g' * (2 + loss_len)) %
                        (f'{epoch + 1}/{self.epochs}', mem, *losses, batch["cls"].shape[0], batch["img"].shape[-1]))
                    self.trigger_callbacks('on_batch_end')
                    if self.args.plots and ni < 3:
                        self.plot_training_samples(batch, ni)

            lr = {f"lr{ir}": x['lr'] for ir, x in enumerate(self.optimizer.param_groups)}  # for loggers
            self.scheduler.step()

            if rank in [-1, 0]:
                # validation
                self.trigger_callbacks('on_val_start')
                self.ema.update_attr(self.model, include=['yaml', 'nc', 'args', 'names', 'stride', 'class_weights'])
                final_epoch = (epoch + 1 == self.epochs)
                if not self.args.noval or final_epoch:
                    self.metrics, self.fitness = self.validate()
                self.trigger_callbacks('on_val_end')
                log_vals = self.label_loss_items(self.tloss) | self.metrics | lr
                self.save_metrics(metrics=log_vals)

                # save model
                if (not self.args.nosave) or (epoch + 1 == self.epochs):
                    self.save_model()
                    self.trigger_callbacks('on_model_save')

            tnow = time.time()
            self.epoch_time = tnow - self.epoch_time_start
            self.epoch_time_start = tnow

            # TODO: termination condition

        if rank in [-1, 0]:
            # do the last evaluation with best.pt
            self.final_eval()
            if self.args.plots:
                self.plot_metrics()
            self.log(f"\nTraining complete ({(time.time() - self.train_time_start) / 3600:.3f} hours)")
            self.trigger_callbacks('on_train_end')
        dist.destroy_process_group() if world_size != 1 else None
        torch.cuda.empty_cache()

    def save_model(self):
        ckpt = {
            'epoch': self.epoch,
            'best_fitness': self.best_fitness,
            'model': deepcopy(de_parallel(self.model)).half(),
            'ema': deepcopy(self.ema.ema).half(),
            'updates': self.ema.updates,
            'optimizer': self.optimizer.state_dict(),
            'train_args': self.args,
            'date': datetime.now().isoformat()}

        # Save last, best and delete
        torch.save(ckpt, self.last)
        if self.best_fitness == self.fitness:
            torch.save(ckpt, self.best)
        del ckpt

    def get_dataset(self, data):
        """
        Get train, val path from data dict if it exists. Returns None if data format is not recognized
        """
        return data["train"], data.get("val") or data.get("test")

    def setup_model(self):
        """
        load/create/download model for any task
        """
        model = self.args.model
        pretrained = not (str(model).endswith(".yaml"))
        # config
        if not pretrained:
            model = check_file(model)
        ckpt = self.load_ckpt(model) if pretrained else None
        self.model = self.load_model(model_cfg=None if pretrained else model, weights=ckpt).to(self.device)  # model
        return ckpt

    def load_ckpt(self, ckpt):
        return torch.load(ckpt, map_location='cpu')

    def optimizer_step(self):
        self.scaler.unscale_(self.optimizer)  # unscale gradients
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=10.0)  # clip gradients
        self.scaler.step(self.optimizer)
        self.scaler.update()
        self.optimizer.zero_grad()
        if self.ema:
            self.ema.update(self.model)

    def preprocess_batch(self, batch):
        """
        Allows custom preprocessing model inputs and ground truths depending on task type
        """
        return batch

    def validate(self):
        """
        Runs validation on test set using self.validator.
        # TODO: discuss validator class. Enforce that a validator metrics dict should contain
        "fitness" metric.
        """
        metrics = self.validator(self)
        fitness = metrics.pop("fitness", -self.loss.detach().cpu().numpy())  # use loss as fitness measure if not found
        if not self.best_fitness or self.best_fitness < fitness:
            self.best_fitness = self.fitness
        return metrics, fitness

    def log(self, text, rank=-1):
        """
        Logs the given text to given ranks process if provided, otherwise logs to all ranks
        :param text: text to log
        :param rank: List[Int]

        """
        if rank in {-1, 0}:
            self.console.info(text)

    def load_model(self, model_cfg, weights):
        raise NotImplementedError("This task trainer doesn't support loading cfg files")

    def get_validator(self):
        raise NotImplementedError("get_validator function not implemented in trainer")

    def get_dataloader(self, dataset_path, batch_size=16, rank=0):
        """
        Returns dataloader derived from torch.data.Dataloader
        """
        raise NotImplementedError("get_dataloader function not implemented in trainer")

    def criterion(self, preds, batch):
        """
        Returns loss and individual loss items as Tensor
        """
        raise NotImplementedError("criterion function not implemented in trainer")

    def label_loss_items(self, loss_items=None, prefix="train"):
        """
        Returns a loss dict with labelled training loss items tensor
        """
        # Not needed for classification but necessary for segmentation & detection
        return {"loss": loss_items} if loss_items is not None else ["loss"]

    def set_model_attributes(self):
        """
        To set or update model parameters before training.
        """
        self.model.names = self.data["names"]

    def build_targets(self, preds, targets):
        pass

    def progress_string(self):
        return ""

    # TODO: may need to put these following functions into callback
    def plot_training_samples(self, batch, ni):
        pass

    def save_metrics(self, metrics):
        keys, vals = list(metrics.keys()), list(metrics.values())
        n = len(metrics) + 1  # number of cols
        s = '' if self.csv.exists() else (('%23s,' * n % tuple(['epoch'] + keys)).rstrip(',') + '\n')  # header
        with open(self.csv, 'a') as f:
            f.write(s + ('%23.5g,' * n % tuple([self.epoch] + vals)).rstrip(',') + '\n')

    def plot_metrics(self):
        pass

    def final_eval(self):
        # TODO: need standalone evaluator to do this
        for f in self.last, self.best:
            if f.exists():
                strip_optimizer(f)  # strip optimizers
                if f is self.best:
                    self.console.info(f'\nValidating {f}...')

    def check_resume(self):
        resume = self.args.resume
        if resume:
            last = Path(check_file(resume) if isinstance(resume, str) else get_latest_run())
            args_yaml = last.parent.parent / 'args.yaml'  # train options yaml
            if args_yaml.is_file():
                args = self._get_config(args_yaml)  # replace
            args.model, args.resume, args.exist_ok = str(last), True, True  # reinstate
            self.args = args

    def resume_training(self, ckpt):
        if ckpt is None:
            return
        best_fitness = 0.0
        start_epoch = ckpt['epoch'] + 1
        if ckpt['optimizer'] is not None:
            self.optimizer.load_state_dict(ckpt['optimizer'])  # optimizer
            best_fitness = ckpt['best_fitness']
        if self.ema and ckpt.get('ema'):
            self.ema.ema.load_state_dict(ckpt['ema'].float().state_dict())  # EMA
            self.ema.updates = ckpt['updates']
        if self.args.resume:
            assert start_epoch > 0, f'{self.args.model} training to {self.epochs} epochs is finished, nothing to resume.\n' \
                                    f"Start a new training without --resume, i.e. 'yolo task=... mode=train model={self.args.model}'"
            LOGGER.info(
                f'Resuming training from {self.args.model} from epoch {start_epoch} to {self.epochs} total epochs')
        if self.epochs < start_epoch:
            LOGGER.info(
                f"{self.args.model} has been trained for {ckpt['epoch']} epochs. Fine-tuning for {self.epochs} more epochs."
            )
            self.epochs += ckpt['epoch']  # finetune additional epochs
        self.best_fitness = best_fitness
        self.start_epoch = start_epoch


def build_optimizer(model, name='Adam', lr=0.001, momentum=0.9, decay=1e-5):
    # TODO: 1. docstring with example? 2. Move this inside Trainer? or utils?
    # YOLOv5 3-param group optimizer: 0) weights with decay, 1) weights no decay, 2) biases no decay
    g = [], [], []  # optimizer parameter groups
    bn = tuple(v for k, v in nn.__dict__.items() if 'Norm' in k)  # normalization layers, i.e. BatchNorm2d()
    for v in model.modules():
        if hasattr(v, 'bias') and isinstance(v.bias, nn.Parameter):  # bias (no decay)
            g[2].append(v.bias)
        if isinstance(v, bn):  # weight (no decay)
            g[1].append(v.weight)
        elif hasattr(v, 'weight') and isinstance(v.weight, nn.Parameter):  # weight (with decay)
            g[0].append(v.weight)

    if name == 'Adam':
        optimizer = torch.optim.Adam(g[2], lr=lr, betas=(momentum, 0.999))  # adjust beta1 to momentum
    elif name == 'AdamW':
        optimizer = torch.optim.AdamW(g[2], lr=lr, betas=(momentum, 0.999), weight_decay=0.0)
    elif name == 'RMSProp':
        optimizer = torch.optim.RMSprop(g[2], lr=lr, momentum=momentum)
    elif name == 'SGD':
        optimizer = torch.optim.SGD(g[2], lr=lr, momentum=momentum, nesterov=True)
    else:
        raise NotImplementedError(f'Optimizer {name} not implemented.')

    optimizer.add_param_group({'params': g[0], 'weight_decay': decay})  # add g0 with weight_decay
    optimizer.add_param_group({'params': g[1], 'weight_decay': 0.0})  # add g1 (BatchNorm2d weights)
    LOGGER.info(f"{colorstr('optimizer:')} {type(optimizer).__name__}(lr={lr}) with parameter groups "
                f"{len(g[1])} weight(decay=0.0), {len(g[0])} weight(decay={decay}), {len(g[2])} bias")
    return optimizer


# Dummy validator
def val(trainer: BaseTrainer):
    trainer.console.info("validating")
    return {"metric_1": 0.1, "metric_2": 0.2, "fitness": 1}
