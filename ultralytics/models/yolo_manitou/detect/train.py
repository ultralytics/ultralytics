"""
@Description: Trainer to support training Manitou dataset.
@Author: Sijie Hu
@Date: 2025-05-06
"""
import os
import math
import numpy as np
from copy import copy

import torch
import torch.nn as nn
from torch import distributed as dist

from pathlib import Path
from ultralytics.nn.tasks import DetectionModel
from ultralytics.data import build_dataloader, build_manitou_dataset, get_manitou_dataset
from ultralytics.engine.trainer import BaseTrainer
from ultralytics.cfg import cfg2dict, check_cfg, get_save_dir
from ultralytics.models import yolo, yolo_manitou
from ultralytics.models.yolo.detect import DetectionTrainer
from ultralytics.utils.torch_utils import de_parallel, torch_distributed_zero_first
from ultralytics.utils.checks import check_amp, check_file, check_imgsz, check_model_file_from_stem, print_args
from ultralytics.utils import (
    DEFAULT_CFG,
    DEFAULT_CFG_DICT,
    LOCAL_RANK,
    LOGGER,
    RANK,
    IterableSimpleNamespace,
    TQDM,
    callbacks,
    yaml_load,
    yaml_save,
)
from ultralytics.utils.torch_utils import (
    TORCH_2_4,
    EarlyStopping,
    ModelEMA,
)


class ManitouTrainer(DetectionTrainer):
    """
    A class extending the DetectionTrainer class for training on the Manitou dataset.

    Attributes:
        model: The model to be trained.
        data: Dictionary containing dataset information including class names and number of classes.
        loss_names (Tuple[str]): Names of the loss components used in training (box_loss, cls_loss, dfl_loss).
        
    Modified Methods:
        __init__: Initialize the trainer with configuration, device, and dataset.
        get_dataset: Get the annotation path for training or validation.
        build_dataset: Build Manitou dataset for training or validation based on the annotation path.
    """
    def __init__(self, cfg=DEFAULT_CFG, overrides=None, _callbacks=None):
        """
        Initialize the BaseTrainer class.

        Args:
            cfg (str): Path to a configuration file.
            overrides (dict, optional): Configuration overrides. Defaults to None.
            _callbacks (list, optional): List of callback functions. Defaults to None.
        """
        super().__init__(cfg=cfg, overrides=overrides, _callbacks=_callbacks)
        
    def _setup_train(self, world_size):
        """Build dataloaders and optimizer on correct rank process."""
        # Model
        self.run_callbacks("on_pretrain_routine_start")
        ckpt = self.setup_model()
        self.model = self.model.to(self.device)
        self.set_model_attributes()

        # Freeze layers
        freeze_list = (
            self.args.freeze
            if isinstance(self.args.freeze, list)
            else range(self.args.freeze)
            if isinstance(self.args.freeze, int)
            else []
        )
        always_freeze_names = [".dfl"]  # always freeze these layers  
        freeze_layer_names = [f"model.{x}." for x in freeze_list] + always_freeze_names
        self.freeze_layer_names = freeze_layer_names
        for k, v in self.model.named_parameters():
            # v.register_hook(lambda x: torch.nan_to_num(x))  # NaN to 0 (commented for erratic training results)
            if any(x in k for x in freeze_layer_names):
                LOGGER.info(f"Freezing layer '{k}'")
                v.requires_grad = False
            elif not v.requires_grad and v.dtype.is_floating_point:  # only floating point Tensor can require gradients
                LOGGER.warning(
                    f"setting 'requires_grad=True' for frozen layer '{k}'. "
                    "See ultralytics.engine.trainer for customization of frozen layers."
                )
                v.requires_grad = True

        # Check AMP
        self.amp = torch.tensor(self.args.amp).to(self.device)  # True or False
        if self.amp and RANK in {-1, 0}:  # Single-GPU and DDP
            callbacks_backup = callbacks.default_callbacks.copy()  # backup callbacks as check_amp() resets them
            self.amp = torch.tensor(check_amp(self.model), device=self.device)
            callbacks.default_callbacks = callbacks_backup  # restore callbacks
        if RANK > -1 and world_size > 1:  # DDP
            dist.broadcast(self.amp.int(), src=0)  # broadcast from rank 0 to all other ranks; gloo errors with boolean
        self.amp = bool(self.amp)  # as boolean
        self.scaler = (
            torch.amp.GradScaler("cuda", enabled=self.amp) if TORCH_2_4 else torch.amp.GradScaler(enabled=self.amp)
        )
        if world_size > 1:
            self.model = nn.parallel.DistributedDataParallel(self.model, device_ids=[RANK], find_unused_parameters=True)

        # Check imgsz
        gs = max(int(self.model.stride.max() if hasattr(self.model, "stride") else 32), 32)  # grid size (max stride)
        # target_w = math.ceil(self.args.imgsz[1] / gs) * gs
        # target_h = math.ceil
        # self.args.imgsz = (self.args.imgsz[0] // gs * gs, self.args.imgsz[1] // gs * gs)  # grid size (multiple of gs)
        # LOGGER.info(f"Image will be cropped to {self.args.imgsz} for training. i.e. crop (0: self.args.imgsz[0], 0: self.args.imgsz[1])")
        self.stride = gs  # for multiscale training

        # Batch size
        if self.batch_size < 1 and RANK == -1:  # single-GPU only, estimate best batch size
            self.args.batch = self.batch_size = self.auto_batch()

        # Dataloaders
        batch_size = self.batch_size // max(world_size, 1)
        self.train_loader = self.get_dataloader(self.trainset, batch_size=batch_size, rank=LOCAL_RANK, mode="train")
        if RANK in {-1, 0}:
            # Note: When training DOTA dataset, double batch size could get OOM on images with >2000 objects.
            self.test_loader = self.get_dataloader(
                self.testset, batch_size=batch_size if self.args.task == "obb" else batch_size * 2, rank=-1, mode="val"
            )
            self.validator = self.get_validator()
            metric_keys = self.validator.metrics.keys + self.label_loss_items(prefix="val")
            self.metrics = dict(zip(metric_keys, [0] * len(metric_keys)))
            self.ema = ModelEMA(self.model)
            if self.args.plots:
                self.plot_training_labels()

        # Optimizer
        self.accumulate = max(round(self.args.nbs / self.batch_size), 1)  # accumulate loss before optimizing
        weight_decay = self.args.weight_decay * self.batch_size * self.accumulate / self.args.nbs  # scale weight_decay
        iterations = math.ceil(len(self.train_loader.dataset) / max(self.batch_size, self.args.nbs)) * self.epochs
        self.optimizer = self.build_optimizer(
            model=self.model,
            name=self.args.optimizer,
            lr=self.args.lr0,
            momentum=self.args.momentum,
            decay=weight_decay,
            iterations=iterations,
        )
        # Scheduler
        self._setup_scheduler()
        self.stopper, self.stop = EarlyStopping(patience=self.args.patience), False
        self.resume_training(ckpt)
        self.scheduler.last_epoch = self.start_epoch - 1  # do not move
        self.run_callbacks("on_pretrain_routine_end")
            
    def get_cfg(self, cfg, overrides):
        """
        Load and merge configuration data from a YAML file, with optional overrides.
        
        Modified from: ultralytics.cfg.get_cfg() to get rid of the checking of the official custom cfg.
        """
        cfg = cfg2dict(cfg)

        # Merge overrides
        if overrides:
            overrides = cfg2dict(overrides)
            if "save_dir" not in cfg:
                overrides.pop("save_dir", None)  # special override keys to ignore
            cfg = {**cfg, **overrides}  # merge c
        
        # Special handling for numeric project/name
        for k in "project", "name":
            if k in cfg and isinstance(cfg[k], (int, float)):
                cfg[k] = str(cfg[k])
        if cfg.get("name") == "model":  # assign model to 'name' arg
            cfg["name"] = str(cfg.get("model", "")).split(".")[0]
            LOGGER.warning(f"'name=model' automatically updated to 'name={cfg['name']}'.")

        # Type and Value checks
        check_cfg(cfg)

        # Return instance
        return IterableSimpleNamespace(**cfg)
        
    def get_validator(self):
        """Return a DetectionValidator for YOLO model validation."""
        self.loss_names = "box_loss", "cls_loss", "dfl_loss"
        return yolo_manitou.detect.ManitouValidator(
            self.test_loader, save_dir=self.save_dir, args=copy(self.args), _callbacks=self.callbacks
        )
        
    def get_dataset(self):
        """
        Get the annotation path for training or validation.
        
        Returns:
            (tuple): Tuple containing training and validation datasets.
        """        
        self.data = get_manitou_dataset(self.args.data)
        
        return self.data["train"], self.data["val"]
    
    def build_dataset(self, ann_path, mode="train", batch=None):
        """
        Build Manitou dataset for training or validation based on the annotation path.

        Args:
            ann_path (str): Path to the json annotation.
            mode (str): `train` mode or `val` mode, users are able to customize different augmentations for each mode.
            batch (int, optional): Size of batches, this is for `rect`.

        Returns:
            Dataset: A dataset object for the specified mode.
        """
        gs = max(int(de_parallel(self.model).stride.max() if self.model else 0), 32)
        return build_manitou_dataset(self.args, ann_path, batch, self.data, mode=mode, rect=mode == "val", stride=gs)
        
    def preprocess_batch(self, batch):
        """
        Preprocess a batch of images by scaling and converting to float.

        Args:
            batch (dict): Dictionary containing batch data with 'img' tensor.

        Returns:
            (dict): Preprocessed batch with normalized images.
        """
        batch = super().preprocess_batch(batch)
        return batch
    
    def get_model(self, cfg=None, weights=None, verbose=True):
        """
        Return a YOLO detection model.

        Args:
            cfg (str, optional): Path to model configuration file.
            weights (str, optional): Path to model weights.
            verbose (bool): Whether to display model information.

        Returns:
            (DetectionModel): YOLO detection model.
        """
        model = DetectionModel(cfg, nc=self.data["nc"], ch=self.data["channels"], verbose=verbose and RANK == -1)
        if weights:
            model.load(weights)
        return model

        