"""
Simple training loop; Boilerplate that could apply to any arbitrary neural network,
so nothing in this file really has anything to do with GPT specifically.
"""

from collections import defaultdict
import logging
import os
from pydoc import resolve
from typing import Union
import time
from pathlib import Path

import torch
from torch.cuda import amp
import torch.nn as nn
from torch.utils.data.dataloader import DataLoader
from omegaconf import DictConfig, OmegaConf
import hydra
import ultralytics.yolo.utils as utils

LOGGER = logging.getLogger()
CONFIG_PATH_REL = "../utils/configs"
CONFIG_PATH_ABS = Path(__file__).parents[1] / "utils/configs"
DEFAULT_CONFIG = "defaults.yaml"


class BaseTrainer:
        
    def __init__(self, model, dataset, criterion, config=CONFIG_PATH_ABS/DEFAULT_CONFIG):
        self.console = LOGGER
        self.model = model
        self.dataset = dataset
        self.callbacks = defaultdict(list)
        self.train, self.hyps = self._get_config(config)
        self.console.info(f"Training config: \n Train: \n {self.train} \n hyps: \n {self.hyps}") # delete this
        # Directories
        self.save_dir = utils.increment_path(Path(self.train.project) / self.train.name, exist_ok=self.train.exist_ok)
        self.wdir = self.save_dir / 'weights'
        self.wdir.mkdir(parents=True, exist_ok=True)  # make dir
        self.last, self.best = self.wdir / 'last.pt', self.wdir / 'best.pt'

        # Save run settings
        utils.save_yaml(self.save_dir / 'train.yaml', OmegaConf.to_container(self.train, resolve=True))

        self.optimizer = build_optimizer(
                                        model=self.model,
                                        name=self.train.optimizer,
                                        lr=self.hyps.lr0,
                                        momentum=self.hyps.momentum,
                                        decay=self.hyps.weight_decay
                                        )
        self.criterion = criterion # ComputeLoss object TODO: create yolo.Loss classes
        if self.train.device == '':
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = self.train.device
        self.model = self.model.to(self.device)
        self.console.info(f"running on device {self.device}")

        self.scaler = amp.GradScaler(enabled=self.device!="cpu")

    def _get_config(self, config: Union[str, Path, DictConfig]=None):
        """
        Accepts yaml file name or DictConfig containing experiment configuration.
        Returns train and hyps namespace
        :param conf: Optional file name or DictConfig object
        """
        try:
            if isinstance(config, str) or isinstance(config, Path):
                config = OmegaConf.load(config)
            return config.train, config.hyps
        except:
            raise Exception("Missing key(s) in config")

    def add_callback(self, onevent: str, callback):
        self.callbacks[onevent].append(callback)

    def set_callback(self, onevent: str, callback):
        self.callbacks[onevent] = [callback]

    def trigger_callbacks(self, onevent: str):
        for callback in self.callbacks.get(onevent, []):
            callback(self)

    def train(self):
        model = self.get_model(self.model)

        # setup the dataloader
        trainset, testset = self.get_dataset()
        train_loader = self.get_dataloader(trainset)
        test_loader = self.get_dataloader(testset)
        # TODO: callback hook. before_train

        self.epochs = 1
        self.epoch_time = time.time()
        data_iter = iter(train_loader)
        for epoch in range(self.train.epochs): 
            # TODO: callback hook. on_epoch_start
            model.train()
            pbar = enumerate(train_loader)

            for i, (images, labels) in pbar:  # progress bar
                images, labels = images.to(self.device, non_blocking=True), labels.to(self.device)
                self.loss = self.criterion(model(images), labels)

            # backprop and update the parameters
            model.zero_grad(set_to_none=True)
            self.scaler.scale(self.loss).backward()

            # optimize
            self.scaler.unscale_(self.optimizer)  # unscale gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)  # clip gradients
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.optimizer.zero_grad()

            self.trigger_callbacks('on_batch_end')

            self.epochs += 1
            tnow = time.time()
            self.iter_dt = tnow - self.iter_time
            self.iter_time = tnow

            # termination conditions

    
    def get_dataloader(self, path):
        """
        Returns dataloader derived from torch.data.Dataloader
        """
        pass

    def get_dataset(self):
        """
        Uses self.dataset to download the dataset if needed and verify it.
        Returns train and val split datasets
        """
        pass
    
    def get_model(self):
        """
        Uses self.model to load/create/download dataset for any task
        """
        pass
    
    def set_criterion(self, criterion):
        """
        :param criterion: yolo.Loss object.
        """
        self.criterion = criterion

def build_optimizer(model, name='Adam', lr=0.001, momentum=0.9, decay=1e-5):
    # TODO: docstring with example?
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
    LOGGER.info(f"{'optimizer:'} {type(optimizer).__name__}(lr={lr}) with parameter groups "
                f"{len(g[1])} weight(decay=0.0), {len(g[0])} weight(decay={decay}), {len(g[2])} bias")
    return optimizer



if __name__ == "__main__":
    model = torch.nn.Sequential(nn.Linear(10,100))
    Trainer(model, "dataset")
