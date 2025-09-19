# Ultralytics YOLO 🚀, AGPL-3.0 license

"""
Training script for MDE (Monocular Depth Estimation) model.

This module provides training functionality for YOLO models with depth estimation capabilities.
"""

from __future__ import annotations

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast

from ultralytics.engine.trainer import BaseTrainer
from ultralytics.utils import LOGGER, colorstr, RANK, DEFAULT_CFG
from ultralytics.utils.loss import v8DetectionLoss
from ultralytics.utils.metrics import DetMetrics


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
    
    def __init__(self, model, depth_loss_weight: float = 1.0, depth_loss_type: str = 'l1'):
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
        if depth_loss_type == 'l1':
            self.depth_criterion = nn.L1Loss()
        elif depth_loss_type == 'l2':
            self.depth_criterion = nn.MSELoss()
        elif depth_loss_type == 'smooth_l1':
            self.depth_criterion = nn.SmoothL1Loss()
        else:
            raise ValueError(f"Unsupported depth loss type: {depth_loss_type}")
    
    def forward(self, preds, targets):
        """
        Forward pass to compute combined loss.
        
        Args:
            preds: Model predictions.
            targets: Ground truth targets.
            
        Returns:
            tuple: (total_loss, loss_dict)
        """
        # Compute detection loss
        det_loss, det_loss_dict = self.det_loss(preds, targets)
        
        # Compute depth loss
        depth_loss = self.depth_loss(preds, targets)
        
        # Combine losses
        total_loss = det_loss + self.depth_loss_weight * depth_loss
        
        # Create loss dictionary
        loss_dict = det_loss_dict.copy()
        loss_dict['depth_loss'] = depth_loss.item()
        loss_dict['total_loss'] = total_loss.item()
        
        return total_loss, loss_dict
    
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
            target_depth = targets['depth'][i] if 'depth' in targets else None
            
            if target_depth is not None:
                # Resize target to match prediction size
                if target_depth.shape[-2:] != pred_depth.shape[-2:]:
                    target_depth = F.interpolate(
                        target_depth, 
                        size=pred_depth.shape[-2:], 
                        mode='bilinear', 
                        align_corners=False
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
        
        model = DetectionModel(cfg, nc=self.data['nc'], verbose=verbose and RANK == -1)
        if weights:
            model.load(weights)
        
        return model
    
    def get_validator(self):
        """Return a MDEValidator for YOLO MDE model validation."""
        self.loss_names = "box_loss", "cls_loss", "dfl_loss", "depth_loss"
        from .val import MDEValidator
        from copy import copy
        return MDEValidator(
            self.test_loader, save_dir=self.save_dir, args=copy(self.args), _callbacks=self.callbacks
        )
    
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
        from ultralytics.utils.torch_utils import torch_distributed_zero_first
        
        gs = max(int(self.model.stride.max() if hasattr(self.model, 'stride') else 32), 32)
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
    
    def criterion(self, preds, batch):
        """
        Get MDE loss function.
        
        Args:
            preds: Model predictions.
            batch: Training batch.
            
        Returns:
            tuple: (loss, loss_dict)
        """
        if not hasattr(self, '_criterion'):
            self._criterion = MDELoss(self.model)
        
        return self._criterion(preds, batch)
    
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
            
            # Forward pass
            with autocast(self.amp):
                preds = self.model(batch['img'])
                loss, loss_dict = self.criterion(preds, batch)
            
            # Backward pass
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.optimizer.zero_grad()
            
            # Update loss
            self.loss_items = [
                loss_dict.get('box_loss', 0.0),
                loss_dict.get('cls_loss', 0.0), 
                loss_dict.get('dfl_loss', 0.0),
                loss_dict.get('depth_loss', 0.0)
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
    
    def validate(self):
        """Validate the model."""
        self.model.eval()
        self.metrics.reset()
        
        pbar = enumerate(self.val_loader)
        if RANK in (-1, 0):
            pbar = self.pbar(pbar, total=len(self.val_loader))
        
        with torch.no_grad():
            for i, batch in pbar:
                self.run_callbacks("on_val_batch_start")
                
                # Forward pass
                preds = self.model(batch['img'])
                
                # Update metrics
                self.metrics.update(preds, batch)
                
                self.run_callbacks("on_val_batch_end")
        
        self.run_callbacks("on_val_epoch_end")
        return self.metrics.results_dict


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


if __name__ == '__main__':
    # Example usage
    train_mde('yolov8n.yaml', {'data': 'kitti.yaml', 'epochs': 100, 'imgsz': 640})
