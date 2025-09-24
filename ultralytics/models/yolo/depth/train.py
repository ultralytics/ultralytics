# Ultralytics YOLO 🚀, AGPL-3.0 license

"""
Training script for MDE (Monocular Depth Estimation) model.

This module provides training functionality for YOLO models with depth estimation capabilities.
"""

from __future__ import annotations

from pathlib import Path

import torch
from torch.cuda.amp import autocast

from ultralytics.models.yolo.detect import DetectionTrainer
from ultralytics.nn.tasks import MDEModel
from ultralytics.utils import LOGGER, RANK


class MDETrainer(DetectionTrainer):
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

    def get_model(self, cfg: str | None = None, weights: str | None = None, verbose: bool = True):
        """
        Get MDE model.

        Args:
            cfg: Model configuration.
            weights: Model weights.
            verbose: Whether to print model information.

        Returns:
            DetectionModel with MDE heads.
        """
        model = MDEModel(
            cfg,
            nc=self.data["nc"],
            ch=self.data["channels"],
            verbose=verbose and RANK == -1,
        )
        if weights:
            model.load(weights)

        return model

    def get_validator(self):
        """Return a MDEValidator for YOLO MDE model validation."""
        self.loss_names = "box_loss", "cls_loss", "dfl_loss", "depth_loss"
        from copy import copy

        from .val import MDEValidator

        return MDEValidator(
            self.test_loader,
            save_dir=self.save_dir,
            args=copy(self.args),
            _callbacks=self.callbacks,
        )

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
        Build MDE Dataset for training or validation with depth information.

        Args:
            img_path (str): Path to the folder containing images.
            mode (str): 'train' mode or 'val' mode.
            batch (int, optional): Batch size for dataset.

        Returns:
            MDEDataset: Dataset object with depth information.
        """
        import sys

        from ultralytics.data import build_yolo_dataset

        sys.path.append("/root/ultralytics")
        from build_mde_dataset import build_mde_dataset

        # For MDE training, use the custom MDE dataset
        try:
            # Create data config from current args
            data_config = {
                "path": str(
                    Path(self.data["path"])
                    if isinstance(self.data.get("path"), str)
                    else self.data.get("path", img_path)
                ),
                "train": self.data.get("train", "images"),
                "val": self.data.get("val", "images"),
                "nc": self.data["nc"],
                "names": self.data["names"],
                "depth_loss_weight": getattr(self.args, "depth_loss_weight", 1.0),
                "depth_loss_type": getattr(self.args, "depth_loss_type", "l1"),
            }

            # Use MDE dataset
            mde_dataset = build_mde_dataset(data_config, mode=mode, batch=batch)
            return mde_dataset

        except Exception as e:
            LOGGER.warning(f"Failed to create MDE dataset, falling back to standard dataset: {e}")
            # Fallback to standard YOLO dataset
            gs = max(
                int(self.model.stride.max() if hasattr(self.model, "stride") else 32),
                32,
            )
            return build_yolo_dataset(
                self.args,
                img_path,
                batch,
                self.data,
                mode=mode,
                rect=mode == "val",
                stride=gs,
            )

    def get_dataloader(
        self,
        dataset_path: str,
        batch_size: int = 16,
        rank: int = 0,
        mode: str = "train",
    ):
        """
        Construct and return MDE dataloader for the specified mode.

        Args:
            dataset_path (str): Path to the dataset.
            batch_size (int): Number of images per batch.
            rank (int): Process rank for distributed training.
            mode (str): 'train' for training dataloader, 'val' for validation dataloader.

        Returns:
            (DataLoader): PyTorch dataloader object with MDE batches.
        """
        import sys

        from ultralytics.data import build_dataloader
        from ultralytics.utils.torch_utils import torch_distributed_zero_first

        sys.path.append("/root/ultralytics")
        from build_mde_dataset import create_mde_dataloader

        assert mode in {"train", "val"}, f"Mode must be 'train' or 'val', not {mode}."

        # Try to use MDE dataloader first
        try:
            # Create data config
            data_config = {
                "path": str(
                    Path(self.data["path"])
                    if isinstance(self.data.get("path"), str)
                    else self.data.get("path", dataset_path)
                ),
                "train": self.data.get("train", "images"),
                "val": self.data.get("val", "images"),
                "nc": self.data["nc"],
                "names": self.data["names"],
                "depth_loss_weight": getattr(self.args, "depth_loss_weight", 1.0),
                "depth_loss_type": getattr(self.args, "depth_loss_type", "l1"),
            }

            # Use MDE dataloader
            mde_dataloader = create_mde_dataloader(
                data_config,
                batch_size=batch_size,
                mode=mode,
                workers=self.args.workers if mode == "train" else self.args.workers * 2,
            )
            LOGGER.info(f"✅ Created MDE dataloader for {mode} mode")
            return mde_dataloader

        except Exception as e:
            LOGGER.warning(f"Failed to create MDE dataloader, falling back to standard dataloader: {e}")

            # Fallback to standard dataloader
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

        # Debug logging
        if hasattr(self, "_debug_count"):
            self._debug_count += 1
        else:
            self._debug_count = 1

        if self._debug_count <= 3:  # Only log first 3 batches
            LOGGER.info(f"DEBUG Batch {self._debug_count} keys: {list(batch.keys())}")
            for k, v in batch.items():
                if isinstance(v, torch.Tensor):
                    LOGGER.info(f"  {k}: {v.shape}")

        # Normalize images to [0, 1] range (if not already normalized)
        if batch["img"].max() > 1.0:
            batch["img"] = batch["img"].float() / 255

        # Convert MDE batch format to standard YOLO format
        if "labels" in batch and "depth" in batch:
            # MDE batch format - convert to standard format
            labels = batch["labels"]
            if len(labels) > 0:
                # Split labels into components
                batch_idx = labels[:, 0].long()
                cls = labels[:, 1].long()
                bboxes = labels[:, 2:6]

                # Create standard batch format
                batch["batch_idx"] = batch_idx
                batch["cls"] = cls
                batch["bboxes"] = bboxes

                # Keep depth information
                batch["depth"] = batch["depth"]
            else:
                # No labels - create empty tensors
                batch["batch_idx"] = torch.zeros(0, dtype=torch.long, device=self.device)
                batch["cls"] = torch.zeros(0, dtype=torch.long, device=self.device)
                batch["bboxes"] = torch.zeros(0, 4, dtype=torch.float32, device=self.device)

        # Ensure all required keys exist for YOLO format
        if "batch_idx" not in batch:
            batch["batch_idx"] = torch.zeros(0, dtype=torch.long, device=self.device)
        if "cls" not in batch:
            batch["cls"] = torch.zeros(0, dtype=torch.long, device=self.device)
        if "bboxes" not in batch:
            batch["bboxes"] = torch.zeros(0, 4, dtype=torch.float32, device=self.device)

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

                # Also resize depth maps if present
                if "depth" in batch:
                    depth_maps = batch["depth"]
                    depth_maps = nn.functional.interpolate(depth_maps, size=ns, mode="bilinear", align_corners=False)
                    batch["depth"] = depth_maps

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
            # Use the new v11DetectionLoss_MDE instead of the old MDELoss
            from ultralytics.utils.loss import v11DetectionLoss_MDE

            self._criterion = v11DetectionLoss_MDE(self.model)

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
                loss_dict[0],
                loss_dict[1],
                loss_dict[2],
                loss_dict[3],
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
