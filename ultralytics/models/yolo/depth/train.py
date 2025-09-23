# Ultralytics YOLO 🚀, AGPL-3.0 license

"""
Training script for MDE (Monocular Depth Estimation) model.

This module provides training functionality for YOLO models with depth estimation capabilities.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast

from ultralytics.engine.trainer import BaseTrainer
from ultralytics.utils import DEFAULT_CFG, LOGGER, RANK
from ultralytics.utils.loss import v8DetectionLoss


class MDELoss(nn.Module):
    """
    MDE Loss combining detection loss and depth estimation loss.

    This loss function combines the standard YOLO detection loss with a depth
    estimation loss to train the MDE model end-to-end.

    Attributes:
        det_loss (v8DetectionLoss): Detection loss component.
        depth_loss_weight (float): Weight for depth loss.
        depth_loss_type (str): Type of depth loss ('l1', 'l2', 'smooth_l1').

    Methods:
        forward: Compute combined detection and depth loss.
        depth_loss: Compute depth estimation loss.
    """

    def __init__(self, model, depth_loss_weight: float = 1.0, depth_loss_type: str = "l1"):
        """
        Initialize MDE loss.

        Args:
            model: The MDE model.
            depth_loss_weight (float): Weight for depth loss component.
            depth_loss_type (str): Type of depth loss ('l1', 'l2', 'smooth_l1').
        """
        super().__init__()
        self.det_loss = v8DetectionLoss(model)
        self.depth_loss_weight = depth_loss_weight
        self.depth_loss_type = depth_loss_type

        # Depth loss functions
        if depth_loss_type == "l1":
            self.depth_criterion = nn.L1Loss()
        elif depth_loss_type == "l2":
            self.depth_criterion = nn.MSELoss()
        elif depth_loss_type == "smooth_l1":
            self.depth_criterion = nn.SmoothL1Loss()
        else:
            raise ValueError(f"Unsupported depth loss type: {depth_loss_type}")

    @property
    def loss_names(self):
        """Return loss component names for MDE model."""
        return ["box", "cls", "dfl", "depth"]

    def forward(self, preds, targets):
        """
        Forward pass to compute combined loss.

        Args:
            preds: Model predictions.
            targets: Ground truth targets.

        Returns:
            tuple: (total_loss, loss_items_tensor)
        """
        # Compute detection loss
        det_loss, det_loss_items = self.det_loss(preds, targets)

        # Compute depth loss
        depth_loss = self.depth_loss(preds, targets)

        # Combine losses
        total_loss = det_loss + self.depth_loss_weight * depth_loss

        # Create loss items tensor in the format expected by BaseValidator
        # Standard format: [box_loss, cls_loss, dfl_loss, depth_loss]
        loss_items = torch.cat(
            [det_loss_items, depth_loss.unsqueeze(0)]  # [box_loss, cls_loss, dfl_loss]  # [depth_loss]
        )

        # Return in the same format as standard YOLO losses: (total_loss, loss_items.detach())
        return total_loss, loss_items.detach()

    def depth_loss(self, preds, targets):
        """
        Compute depth estimation loss.

        Args:
            preds: Model predictions with depth.
            targets: Ground truth targets with depth.

        Returns:
            torch.Tensor: Depth loss.
        """
        depth_loss = 0.0
        num_scales = len(preds)

        for i, pred in enumerate(preds):
            # Extract depth predictions (last channel)
            pred_depth = pred[..., -1:]  # [B, H, W, 1]

            # Get corresponding target depth
            target_depth = targets["depth"][i] if "depth" in targets else None

            if target_depth is not None:
                # Resize target to match prediction size
                if target_depth.shape[-2:] != pred_depth.shape[-2:]:
                    target_depth = F.interpolate(
                        target_depth, size=pred_depth.shape[-2:], mode="bilinear", align_corners=False
                    )

                # Compute depth loss
                depth_loss += self.depth_criterion(pred_depth, target_depth)

        return depth_loss / num_scales if num_scales > 0 else torch.tensor(0.0, device=preds[0].device)


class MDETrainer(BaseTrainer):
    """
    MDE Trainer for training YOLO models with depth estimation.

    This trainer extends the base YOLO trainer to handle depth estimation
    training with appropriate loss functions and metrics.

    Attributes:
        model: The MDE model.
        criterion: The MDE loss function.
        metrics: Detection and depth metrics.

    Methods:
        criterion: Get the MDE loss function.
        get_model: Get the MDE model.
        train_one_epoch: Train for one epoch.
        validate: Validate the model.
    """

    def __init__(self, cfg=DEFAULT_CFG, overrides=None, _callbacks=None):
        """Initialize MDE trainer."""
        super().__init__(cfg, overrides, _callbacks)

    def get_model(self, cfg=None, weights=None, verbose=True):
        """
        Get MDE model.

        Args:
            cfg: Model configuration.
            weights: Model weights.
            verbose: Whether to print model information.

        Returns:
            DetectionModel with MDE heads.
        """
        from ultralytics.nn.tasks import DetectionModel

        model = DetectionModel(cfg, nc=self.data["nc"], verbose=verbose and RANK == -1)
        if weights:
            model.load(weights)

        return model

    def get_validator(self):
        """Return a MDEValidator for YOLO MDE model validation."""
        self.loss_names = "box_loss", "cls_loss", "dfl_loss", "depth_loss"
        from copy import copy

        from .val import MDEValidator

        return MDEValidator(self.test_loader, save_dir=self.save_dir, args=copy(self.args), _callbacks=self.callbacks)

    def label_loss_items(self, loss_items: list[float] | None = None, prefix: str = "train"):
        """
        Return a loss dict with labeled training loss items tensor.

        Args:
            loss_items (list[float], optional): List of loss values.
            prefix (str): Prefix for keys in the returned dictionary.

        Returns:
            (dict | list): Dictionary of labeled loss items if loss_items is provided, otherwise list of keys.
        """
        keys = [f"{prefix}/{x}" for x in self.loss_names]
        if loss_items is not None:
            loss_items = [round(float(x), 5) for x in loss_items]  # convert tensors to 5 decimal place floats
            return dict(zip(keys, loss_items))
        else:
            return keys

    def build_dataset(self, img_path: str, mode: str = "train", batch: int | None = None):
        """
        Build YOLO Dataset for training or validation.

        Args:
            img_path (str): Path to the folder containing images.
            mode (str): 'train' mode or 'val' mode.
            batch (int, optional): Batch size for dataset.

        Returns:
            YOLODataset: Dataset object.
        """
        from ultralytics.data import build_yolo_dataset

        gs = max(int(self.model.stride.max() if hasattr(self.model, "stride") else 32), 32)
        return build_yolo_dataset(self.args, img_path, batch, self.data, mode=mode, rect=mode == "val", stride=gs)

    def get_dataloader(self, dataset_path: str, batch_size: int = 16, rank: int = 0, mode: str = "train"):
        """
        Construct and return dataloader for the specified mode.

        Args:
            dataset_path (str): Path to the dataset.
            batch_size (int): Number of images per batch.
            rank (int): Process rank for distributed training.
            mode (str): 'train' for training dataloader, 'val' for validation dataloader.

        Returns:
            (DataLoader): PyTorch dataloader object.
        """
        from ultralytics.data import build_dataloader
        from ultralytics.utils.torch_utils import torch_distributed_zero_first

        assert mode in {"train", "val"}, f"Mode must be 'train' or 'val', not {mode}."
        with torch_distributed_zero_first(rank):  # init dataset *.cache only once if DDP
            dataset = self.build_dataset(dataset_path, mode, batch_size)
        shuffle = mode == "train"
        if getattr(dataset, "rect", False) and shuffle:
            LOGGER.warning("'rect=True' is incompatible with DataLoader shuffle, setting shuffle=False")
            shuffle = False
        return build_dataloader(
            dataset,
            batch=batch_size,
            workers=self.args.workers if mode == "train" else self.args.workers * 2,
            shuffle=shuffle,
            rank=rank,
            drop_last=self.args.compile and mode == "train",
        )

    def preprocess_batch(self, batch: dict) -> dict:
        """
        Preprocess a batch of images by scaling and converting to float.

        This method handles device placement and normalization for MDE training,
        ensuring all tensors are moved to the correct device before training.

        Args:
            batch (dict): Dictionary containing batch data with 'img' tensor and other components.

        Returns:
            (dict): Preprocessed batch with normalized images and tensors moved to device.
        """
        # Move all tensors to device (critical for MDE training)
        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                batch[k] = v.to(self.device, non_blocking=True)

        # Normalize images to [0, 1] range
        batch["img"] = batch["img"].float() / 255

        # Handle multi-scale training if enabled
        if self.args.multi_scale:
            import math
            import random

            import torch.nn as nn

            imgs = batch["img"]
            sz = (
                random.randrange(int(self.args.imgsz * 0.5), int(self.args.imgsz * 1.5 + self.stride))
                // self.stride
                * self.stride
            )  # size
            sf = sz / max(imgs.shape[2:])  # scale factor
            if sf != 1:
                ns = [
                    math.ceil(x * sf / self.stride) * self.stride for x in imgs.shape[2:]
                ]  # new shape (stretched to gs-multiple)
                imgs = nn.functional.interpolate(imgs, size=ns, mode="bilinear", align_corners=False)
            batch["img"] = imgs

        return batch

    def set_model_attributes(self):
        """Set model attributes based on dataset information."""
        self.model.nc = self.data["nc"]  # attach number of classes to model
        self.model.names = self.data["names"]  # attach class names to model
        self.model.args = self.args  # attach hyperparameters to model

    def criterion(self, preds, batch):
        """
        Get MDE loss function.

        Args:
            preds: Model predictions.
            batch: Training batch.

        Returns:
            tuple: (loss, loss_dict)
        """
        if not hasattr(self, "_criterion"):
            self._criterion = MDELoss(self.model)

        return self._criterion(preds, batch)

    def progress_string(self):
        """Return a formatted string of training progress with epoch, GPU memory, loss, instances and size."""
        return ("\n" + "%11s" * (4 + len(self.loss_names))) % (
            "Epoch",
            "GPU_mem",
            *self.loss_names,
            "Instances",
            "Size",
        )

    def train_one_epoch(self):
        """Train for one epoch."""
        self.model.train()
        self.loss = torch.zeros(3, device=self.device)  # box, cls, dfl
        self.loss_items = []

        pbar = enumerate(self.train_loader)
        if RANK in (-1, 0):
            pbar = self.pbar(pbar, total=len(self.train_loader))

        self.optimizer.zero_grad()

        for i, batch in pbar:
            self.run_callbacks("on_train_batch_start")

            # Preprocess batch (move to device and normalize)
            batch = self.preprocess_batch(batch)

            # Forward pass
            with autocast(self.amp):
                preds = self.model(batch["img"])
                loss, loss_dict = self.criterion(preds, batch)

            # Backward pass
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.optimizer.zero_grad()

            # Update loss
            self.loss_items = [
                loss_dict.get("box_loss", 0.0),
                loss_dict.get("cls_loss", 0.0),
                loss_dict.get("dfl_loss", 0.0),
                loss_dict.get("depth_loss", 0.0),
            ]

            # Update progress bar
            if RANK in (-1, 0):
                self.pbar.set_description(
                    f"Epoch {self.epoch}/{self.epochs - 1} "
                    f"Box: {self.loss_items[0]:.4f} "
                    f"Cls: {self.loss_items[1]:.4f} "
                    f"Dfl: {self.loss_items[2]:.4f} "
                    f"Depth: {self.loss_items[3]:.4f}"
                )

            self.run_callbacks("on_train_batch_end")

        self.scheduler.step()
        self.run_callbacks("on_train_epoch_end")


def train_mde(cfg=None, overrides=None):
    """
    Train MDE model.

    Args:
        cfg: Training configuration.
        overrides: Configuration overrides.

    Returns:
        Training results.
    """
    trainer = MDETrainer(cfg, overrides)
    trainer.train()
    return trainer


if __name__ == "__main__":
    # Example usage
    train_mde("yolov8n.yaml", {"data": "kitti.yaml", "epochs": 100, "imgsz": 640})
