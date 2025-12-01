# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

"""Stereo 3D Object Detection trainer extending Ultralytics BaseTrainer."""

from __future__ import annotations

from copy import copy
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from ultralytics.data.stereo.dataset import KITTIStereo3DDataset
from ultralytics.data.stereo.target import TargetGenerator
from ultralytics.engine.trainer import BaseTrainer
from ultralytics.nn.modules.stereo.head import StereoCenterNetHead
from ultralytics.nn.modules.stereo.loss import StereoLoss
from ultralytics.nn.tasks import DetectionModel, parse_model
from ultralytics.utils import DEFAULT_CFG, LOGGER, RANK
from ultralytics.utils.torch_utils import torch_distributed_zero_first


class StereoTrainer(BaseTrainer):
    """Trainer for Stereo 3D Object Detection models.

    Extends BaseTrainer to support stereo image pairs, 10-branch outputs, and
    uncertainty-weighted multi-task loss.
    """

    def __init__(self, cfg=DEFAULT_CFG, overrides: dict[str, Any] | None = None, _callbacks=None):
        """Initialize Stereo Trainer.

        Args:
            cfg: Default configuration dictionary.
            overrides: Dictionary of parameter overrides.
            _callbacks: List of callback functions.
        """
        super().__init__(cfg, overrides, _callbacks)

        # Stereo-specific attributes
        self.loss_names = (
            "heatmap_loss",
            "offset_loss",
            "bbox_size_loss",
            "lr_distance_loss",
            "right_width_loss",
            "dimensions_loss",
            "orientation_loss",
            "vertices_loss",
            "vertex_offset_loss",
            "vertex_dist_loss",
        )

        # Target generator
        self.target_generator = TargetGenerator(
            output_size=(96, 320),  # H/4, W/4 for 384×1280 input
            num_classes=self.data.get("nc", 3),
        )

    def build_dataset(self, img_path: str, mode: str = "train", batch: int | None = None):
        """Build KITTI stereo dataset.

        Args:
            img_path: Path to dataset root directory.
            mode: 'train' or 'val'.
            batch: Batch size (unused, kept for compatibility).

        Returns:
            KITTIStereo3DDataset: Dataset instance.
        """
        return KITTIStereo3DDataset(
            root=img_path,
            split=mode,
            imgsz=self.args.imgsz,
            augment=(mode == "train"),
        )

    def get_dataloader(
        self, dataset_path: str, batch_size: int = 16, rank: int = 0, mode: str = "train"
    ):
        """Construct and return dataloader for stereo dataset.

        Args:
            dataset_path: Path to dataset.
            batch_size: Number of images per batch.
            rank: Process rank for distributed training.
            mode: 'train' or 'val'.

        Returns:
            DataLoader: PyTorch dataloader.
        """
        assert mode in {"train", "val"}, f"Mode must be 'train' or 'val', not {mode}."

        with torch_distributed_zero_first(rank):
            dataset = self.build_dataset(dataset_path, mode, batch_size)

        shuffle = mode == "train"
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=self.args.workers if mode == "train" else self.args.workers * 2,
            pin_memory=True,
            collate_fn=self.collate_fn,
        )

    def collate_fn(self, batch: list[dict[str, Any]]) -> dict[str, Any]:
        """Collate function for stereo dataset batches.

        Args:
            batch: List of samples from dataset.

        Returns:
            Dictionary with batched data.
        """
        # Stack images
        imgs = torch.stack([sample["img"] for sample in batch])  # [B, 6, H, W]

        # Collect labels and calibration
        labels = [sample["labels"] for sample in batch]
        calibs = [sample["calib"] for sample in batch]
        image_ids = [sample["image_id"] for sample in batch]

        return {
            "img": imgs,
            "labels": labels,
            "calib": calibs,
            "image_id": image_ids,
        }

    def preprocess_batch(self, batch: dict[str, Any]) -> dict[str, Any]:
        """Preprocess batch for training.

        Args:
            batch: Batch dictionary.

        Returns:
            Preprocessed batch.
        """
        # Move to device
        batch["img"] = batch["img"].to(self.device, non_blocking=True)

        # Generate targets for all samples
        targets = []
        for labels, calib in zip(batch["labels"], batch["calib"]):
            target = self.target_generator.generate_targets(labels, input_size=(384, 1280))
            # Move targets to device
            target = {k: v.to(self.device) for k, v in target.items()}
            targets.append(target)

        batch["targets"] = targets
        return batch

    def get_model(self, cfg: str | None = None, weights: str | None = None, verbose: bool = True):
        """Get stereo detection model.

        Args:
            cfg: Path to model configuration file.
            weights: Path to model weights.
            verbose: Whether to display model information.

        Returns:
            Stereo detection model.
        """
        # Load configuration
        if cfg is None:
            cfg = self.args.model

        # Parse model configuration
        # For now, create a simple model structure
        # In full implementation, would parse YAML and build model
        model = self._build_stereo_model(cfg)

        if weights:
            model.load(weights)

        return model

    def _build_stereo_model(self, cfg: str | Path) -> nn.Module:
        """Build stereo detection model from configuration.

        Args:
            cfg: Model configuration path or dict.

        Returns:
            Stereo detection model.
        """
        # Simplified model structure
        # In full implementation, would parse YAML and use parse_model
        from ultralytics.nn.modules.stereo.backbone import StereoBackbone

        # For now, return a placeholder model
        # Full implementation would:
        # 1. Parse YAML config
        # 2. Build backbone with 6-channel input
        # 3. Build neck/FPN
        # 4. Build 10-branch head
        # 5. Combine into full model

        class StereoModel(nn.Module):
            def __init__(self):
                super().__init__()
                # Placeholder - would be built from config
                self.head = StereoCenterNetHead(in_channels=256, num_classes=self.data.get("nc", 3))

            def forward(self, x, *args, **kwargs):
                # Placeholder forward
                return self.head(x)

        return StereoModel()

    def get_validator(self):
        """Return validator for stereo model (placeholder).

        Returns:
            Validator instance (to be implemented).
        """
        # TODO: Implement stereo validator
        raise NotImplementedError("Stereo validator not yet implemented")

    def set_model_attributes(self):
        """Set model attributes based on dataset information."""
        if hasattr(self.model, "nc"):
            self.model.nc = self.data["nc"]
        if hasattr(self.model, "names"):
            self.model.names = self.data["names"]
        if hasattr(self.model, "args"):
            self.model.args = self.args

    def label_loss_items(self, loss_items: list[float] | None = None, prefix: str = "train"):
        """Return labeled loss items.

        Args:
            loss_items: List of loss values.
            prefix: Prefix for keys.

        Returns:
            Dictionary of labeled loss items.
        """
        keys = [f"{prefix}/{x}" for x in self.loss_names]
        if loss_items is not None:
            loss_items = [round(float(x), 5) for x in loss_items]
            return dict(zip(keys, loss_items))
        else:
            return keys

    def progress_string(self):
        """Return formatted training progress string."""
        return ("\n" + "%11s" * (4 + len(self.loss_names))) % (
            "Epoch",
            "GPU_mem",
            *self.loss_names,
            "Instances",
            "Size",
        )

    def save_model(self):
        """Save model training checkpoints with stereo-specific metadata.

        Inherits from BaseTrainer.save_model() and adds stereo-specific checkpoint information.
        The checkpoint includes:
        - Model state (EMA weights)
        - Optimizer state
        - Training metrics (including all 10 loss branches)
        - Epoch and best fitness
        - Training arguments
        """
        # BaseTrainer.save_model() handles the standard checkpoint saving
        # This method can be overridden to add stereo-specific metadata if needed
        super().save_model()

    def resume_training(self, ckpt):
        """Resume stereo training from a saved checkpoint.

        Inherits from BaseTrainer.resume_training() and ensures compatibility with stereo checkpoints.
        The checkpoint must contain:
        - Model state (EMA weights)
        - Optimizer state
        - Training metrics
        - Epoch number
        - Best fitness value

        Args:
            ckpt: Checkpoint dictionary loaded from .pt file.
        """
        # BaseTrainer.resume_training() handles the standard checkpoint loading
        # This method ensures stereo-specific components are properly restored
        if ckpt is None or not self.resume:
            return

        # Verify checkpoint contains required stereo training state
        if "train_args" in ckpt:
            # Ensure task is set correctly
            if ckpt["train_args"].get("task") != "stereo3ddet":
                LOGGER.warning(
                    "Checkpoint task mismatch. Expected 'stereo3ddet', "
                    f"got '{ckpt['train_args'].get('task')}'. Proceeding anyway."
                )

        # Call parent resume_training to handle standard checkpoint loading
        super().resume_training(ckpt)

        # Verify loss_names match (for compatibility)
        if "train_metrics" in ckpt:
            # Check if checkpoint has stereo loss names
            metrics = ckpt["train_metrics"]
            stereo_losses = [k for k in metrics.keys() if any(loss in k for loss in self.loss_names)]
            if stereo_losses:
                LOGGER.info(f"Resuming stereo training with {len(stereo_losses)} loss branches found in checkpoint")

