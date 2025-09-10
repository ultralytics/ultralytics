#!/usr/bin/env python3
# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

"""
Tennis Ball Pose Trainer

Custom trainer for tennis ball pose estimation using 4-channel input (RGB + motion mask).
Integrates TennisBallDataset and TennisBallPoseModel for enhanced tennis ball tracking.
"""

from __future__ import annotations

from copy import copy
from pathlib import Path
from typing import Any, Dict, Optional, Union

import torch
from ultralytics.models.yolo.pose.train import PoseTrainer
from ultralytics.data.ball_dataset import TennisBallDataset, MotionDataLoader, create_ball_dataloader
from ultralytics.data import build_dataloader
from ultralytics.nn.tasks import TennisBallPoseModel
from ultralytics.nn.modules.motion_utils import MotionConfig
from ultralytics.utils import LOGGER, DEFAULT_CFG, colorstr
from ultralytics.utils.torch_utils import torch_distributed_zero_first, de_parallel


class TennisBallTrainer(PoseTrainer):
    """
    Tennis Ball Pose Trainer with 4-channel input support.
    
    Extends PoseTrainer to use TennisBallDataset and TennisBallPoseModel
    for enhanced tennis ball tracking with motion mask integration.
    
    Attributes:
        motion_config (MotionConfig): Configuration for motion detection
        use_motion_masks (bool): Whether to use motion masks
        motion_cache_dir (Optional[Path]): Directory for caching motion masks
        
    Examples:
        >>> trainer = TennisBallTrainer(
        ...     model="ultralytics/cfg/models/11/yolo11-tennis-pose.yaml",
        ...     data="Dataset_YOLO/data.yaml"
        ... )
        >>> trainer.train()
    """
    
    def __init__(self, cfg=DEFAULT_CFG, overrides: Dict[str, Any] | None = None, _callbacks=None):
        """
        Initialize TennisBallTrainer.
        
        Args:
            cfg: Default configuration
            overrides: Configuration overrides
            _callbacks: Callback functions
        """
        if overrides is None:
            overrides = {}
        
        # Extract custom parameters before parent initialization
        self.use_motion_masks = overrides.pop("use_motion_masks", True)
        self.motion_cache_dir = overrides.pop("motion_cache_dir", None)
        self.frame_window_size = overrides.pop("frame_window_size", 5)
        self.precompute_motion = overrides.pop("precompute_motion", False)
        pixel_threshold = overrides.pop("pixel_threshold", 15)
        
        # Ensure task is set to pose
        # TODO: Should we using a new task type like ball pose detection?
        overrides["task"] = "pose"
        
        # Set default motion configuration
        self.motion_config = MotionConfig(
            pixel_threshold=pixel_threshold,
            delta=1,
            window_size=self.frame_window_size,
            adaptive_threshold=True,
            min_motion_pixels=100,
            max_motion_pixels=50000
        )
        
        # Initialize parent trainer (without custom parameters)
        super().__init__(cfg, overrides, _callbacks)
        
        LOGGER.info(f"{colorstr('TennisBallTrainer')}: Initialized with motion_masks={self.use_motion_masks}")
    
    def build_dataset(self, img_path: str, mode: str = "train", batch: int | None = None):
        """
        Build TennisBallDataset for training or validation.
        
        Args:
            img_path: Path to the folder containing images (Dataset_YOLO/images/train or val)
            mode: 'train' or 'val' mode
            batch: Batch size for rectangular training
            
        Returns:
            TennisBallDataset instance
        """
        LOGGER.info(f"{colorstr('TennisBallTrainer')}: Building {mode} dataset from {img_path}")
        
        gs = max(int(de_parallel(self.model).stride.max() if self.model else 0), 32)
        
        # Create TennisBallDataset with motion support and multi-clip structure
        dataset = TennisBallDataset(
            img_path=img_path,
            imgsz=self.args.imgsz,
            batch_size=batch,
            augment=mode == "train",
            hyp=self.args,
            rect=self.args.rect or (mode == "val"),
            cache=self.args.cache or None,
            single_cls=self.args.single_cls or False,
            stride=gs,
            pad=0.0 if mode == "train" else 0.5,
            prefix=colorstr(f"{mode}: "),
            task=self.args.task,
            classes=self.args.classes,
            data=self.data,
            fraction=self.args.fraction if mode == "train" else 1.0,
            # Tennis ball specific parameters
            motion_config=self.motion_config,
            use_motion_masks=self.use_motion_masks,
            motion_cache_dir=self.motion_cache_dir,
            frame_window_size=self.frame_window_size,
            precompute_motion=self.precompute_motion and mode == "train"
        )
        
        LOGGER.info(f"{colorstr('TennisBallTrainer')}: Built {mode} dataset with {len(dataset)} samples")
        LOGGER.info(f"  Motion masks: {dataset.use_motion_masks}")
        LOGGER.info(f"  Clips discovered: {len(dataset.game_clips)}")
        LOGGER.info(f"  Frame window size: {dataset.frame_window_size}")
        
        return dataset
    
    def get_dataloader(self, dataset_path: str, batch_size: int = 16, rank: int = 0, mode: str = "train"):
        """
        Construct and return dataloader for tennis ball training.
        
        Args:
            dataset_path: Path to dataset
            batch_size: Batch size
            rank: Process rank for distributed training
            mode: 'train' or 'val' mode
            
        Returns:
            DataLoader instance compatible with parent class
        """
        assert mode in {"train", "val"}, f"Mode must be 'train' or 'val', not {mode}."
        
        with torch_distributed_zero_first(rank):
            dataset = self.build_dataset(dataset_path, mode, batch_size)
        
        shuffle = mode == "train"
        if getattr(dataset, "rect", False) and shuffle:
            LOGGER.warning("'rect=True' is incompatible with DataLoader shuffle, setting shuffle=False")
            shuffle = False
        
        workers = self.args.workers if mode == "train" else self.args.workers * 2
        
        # Use standard YOLO dataloader (compatible with parent class)
        dataloader = build_dataloader(dataset, batch_size, workers, shuffle, rank)
        
        LOGGER.info(f"{colorstr('TennisBallTrainer')}: Created {mode} dataloader with batch_size={batch_size}")
        return dataloader
    
    def get_model(
        self, 
        cfg: str | Path | Dict[str, Any] | None = None, 
        weights: str | Path | None = None, 
        verbose: bool = True
    ) -> TennisBallPoseModel:
        """
        Get TennisBallPoseModel with 4-channel input support.
        
        Args:
            cfg: Model configuration
            weights: Path to weights file
            verbose: Whether to display model info
            
        Returns:
            TennisBallPoseModel instance
        """
        # Use tennis ball pose model configuration if not specified
        if cfg is None:
            cfg = "ultralytics/cfg/models/11/yolo11-tennis-pose.yaml"
        
        # Set channels based on motion mask usage
        ch = 4 if self.use_motion_masks else 3
        
        LOGGER.info(f"{colorstr('TennisBallTrainer')}: Creating TennisBallPoseModel with {ch} channels")
        
        model = TennisBallPoseModel(
            cfg=str(cfg) if cfg is not None else "ultralytics/cfg/models/11/yolo11-tennis-pose.yaml",
            ch=ch,
            nc=self.data["nc"],
            data_kpt_shape=self.data["kpt_shape"],
            verbose=verbose
        )
        
        # Handle weights loading for 4-channel model
        if weights:
            if str(weights).endswith('.pt') and ch == 4:
                # Check if this is a pretrained 3-channel pose model
                if any(pose_indicator in str(weights).lower() for pose_indicator in ['pose', 'yolo11n-pose', 'yolo11s-pose', 'yolo11m-pose', 'yolo11l-pose', 'yolo11x-pose']):
                    LOGGER.info(f"{colorstr('TennisBallTrainer')}: Loading pretrained pose weights and adapting for 4-channel input")
                    success = model.load_pretrained_pose_weights(str(weights), adapt_first_layer=True)
                    if success:
                        LOGGER.info(f"{colorstr('TennisBallTrainer')}: Successfully adapted pretrained pose weights from {weights}")
                    else:
                        LOGGER.warning(f"{colorstr('TennisBallTrainer')}: Failed to load pretrained weights, using random initialization")
                else:
                    # Standard weight loading for tennis ball specific weights
                    try:
                        model.load(weights)
                        LOGGER.info(f"{colorstr('TennisBallTrainer')}: Loaded tennis ball specific weights from {weights}")
                    except Exception as e:
                        LOGGER.warning(f"{colorstr('TennisBallTrainer')}: Failed to load weights {weights}: {e}")
            else:
                # Standard weight loading for 3-channel model or non-.pt files
                model.load(weights)
                LOGGER.info(f"{colorstr('TennisBallTrainer')}: Loaded weights from {weights}")
        
        return model
    
    def set_model_attributes(self):
        """Set model attributes for tennis ball tracking."""
        super().set_model_attributes()
        
        # Set tennis ball specific attributes if model supports them
        model = getattr(self, 'model', None)
        if model is not None and hasattr(model, 'model'):
            actual_model = model.model
            if hasattr(actual_model, 'use_motion_masks'):
                actual_model.use_motion_masks = self.use_motion_masks
            if hasattr(actual_model, 'motion_config'):
                setattr(actual_model, 'motion_config', self.motion_config)
        
        LOGGER.info(f"{colorstr('TennisBallTrainer')}: Set model attributes for tennis ball tracking")
    
    def preprocess_batch(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        """
        Preprocess batch with 4-channel support.
        
        Args:
            batch: Input batch dictionary
            
        Returns:
            Preprocessed batch
        """
        # Handle 4-channel input
        batch["img"] = batch["img"].to(self.device, non_blocking=True).float() / 255
        
        # Apply multi-scale training if enabled
        if self.args.multi_scale:
            import random
            import math
            import torch.nn.functional as F
            
            imgs = batch["img"]
            sz = (
                random.randrange(int(self.args.imgsz * 0.5), int(self.args.imgsz * 1.5 + self.stride))
                // self.stride * self.stride
            )
            sf = sz / max(imgs.shape[2:])
            if sf != 1:
                ns = [
                    math.ceil(x * sf / self.stride) * self.stride for x in imgs.shape[2:]
                ]
                imgs = F.interpolate(imgs, size=ns, mode="bilinear", align_corners=False)
            batch["img"] = imgs
        
        return batch
    
    def get_validator(self):
        """Return a tennis ball pose validator."""
        from ultralytics.models.yolo.pose.val import PoseValidator
        
        self.loss_names = "box_loss", "pose_loss", "kobj_loss", "cls_loss", "dfl_loss"
        
        # Create validator with tennis ball specific configuration
        validator = PoseValidator(
            self.test_loader,
            save_dir=self.save_dir,
            args=copy(self.args),
            _callbacks=self.callbacks
        )
        
        # Set tennis ball specific validation attributes if supported
        if hasattr(validator, 'use_motion_masks'):
            setattr(validator, 'use_motion_masks', self.use_motion_masks)
        if hasattr(validator, 'motion_config'):
            setattr(validator, 'motion_config', self.motion_config)
        
        return validator


def train_tennis_ball_model(
    model: str = "ultralytics/cfg/models/11/yolo11-tennis-pose.yaml",
    data: str = "Dataset_YOLO/data.yaml",
    epochs: int = 100,
    imgsz: int = 640,
    batch: int = 16,
    use_motion_masks: bool = True,
    motion_cache_dir: Optional[str] = None,
    **kwargs
):
    """
    Train tennis ball pose model with motion mask support.
    
    Args:
        model: Path to model configuration YAML
        data: Path to dataset configuration YAML
        epochs: Number of training epochs
        imgsz: Image size for training
        batch: Batch size
        use_motion_masks: Whether to use motion masks
        motion_cache_dir: Directory for caching motion masks
        **kwargs: Additional training arguments
        
    Returns:
        Trained model results
        
    Examples:
        >>> results = train_tennis_ball_model(
        ...     model="ultralytics/cfg/models/11/yolo11-tennis-pose.yaml",
        ...     data="Dataset_YOLO/data.yaml",
        ...     epochs=100,
        ...     batch=16,
        ...     use_motion_masks=True
        ... )
    """
    # Training configuration
    overrides = {
        "model": model,
        "data": data,
        "epochs": epochs,
        "imgsz": imgsz,
        "batch": batch,
        "task": "pose",
        "use_motion_masks": use_motion_masks,
        "motion_cache_dir": motion_cache_dir,
        **kwargs
    }
    
    LOGGER.info(f"{colorstr('Tennis Ball Training')}: Starting training with configuration:")
    for key, value in overrides.items():
        LOGGER.info(f"  {key}: {value}")
    
    # Create and run trainer
    trainer = TennisBallTrainer(overrides=overrides)
    trainer.train()
    
    return trainer.best


if __name__ == "__main__":
    # Example training configuration
    results = train_tennis_ball_model(
        model="ultralytics/cfg/models/11/yolo11-tennis-pose.yaml",
        data="Dataset_YOLO/data.yaml",
        epochs=100,
        imgsz=640,
        batch=16,
        lr0=0.01,
        device="cuda" if torch.cuda.is_available() else "cpu",
        use_motion_masks=True,
        workers=8,
        patience=50,
        save_period=10,
        project="tennis_ball_training",
        name="yolo11_tennis_pose_4channel"
    )
    
    print(f"Training completed! Best results: {results}")
