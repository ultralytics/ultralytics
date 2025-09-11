#!/usr/bin/env python3
# Tennis Ball Training Configuration Script

"""
Complete training setup for tennis ball pose estimation using:
1. Custom yolo11-tennis-pose.yaml model (4-channel input)
2. TennisBallDataset (with motion mask support)
3. TennisBallTrainer (custom trainer)

Usage:
    python train_tennis_ball.py
    
Or with custom parameters:
    python train_tennis_ball.py --epochs 200 --batch 32 --imgsz 640
"""

import argparse
import sys
import torch
from pathlib import Path

# Add ultralytics to path
sys.path.insert(0, str(Path(__file__).parent / "ultralytics"))

from tennis_ball_trainer import TennisBallTrainer, train_tennis_ball_model
from ultralytics.nn.modules.motion_utils import MotionConfig
from ultralytics.utils import LOGGER, colorstr


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Tennis Ball Pose Training")
    
    # Model and data configuration
    parser.add_argument("--model", type=str, 
                       default="ultralytics/cfg/models/11/yolo11-tennis-pose.yaml",
                       help="Path to model configuration YAML")
    parser.add_argument("--data", type=str, 
                       default="Dataset_YOLO/data.yaml",
                       help="Path to dataset configuration YAML")
    parser.add_argument("--pretrained", type=str, default="yolo11n-pose.pt",
                       help="Path to pretrained pose weights (e.g., yolo11n-pose.pt)")
    parser.add_argument("--auto-download", action="store_true",
                       help="Automatically download pretrained weights if not found")
    
    # Training parameters
    parser.add_argument("--epochs", type=int, default=100, 
                       help="Number of training epochs")
    parser.add_argument("--batch", type=int, default=16, 
                       help="Batch size")
    parser.add_argument("--imgsz", type=int, default=640, 
                       help="Image size for training")
    parser.add_argument("--lr0", type=float, default=0.01, 
                       help="Initial learning rate")
    parser.add_argument("--workers", type=int, default=4, 
                       help="Number of dataloader workers")
    
    # Device and optimization
    parser.add_argument("--device", type=str, 
                       default="cuda" if torch.cuda.is_available() else "cpu",
                       help="Training device")
    parser.add_argument("--patience", type=int, default=50, 
                       help="Early stopping patience")
    parser.add_argument("--save-period", type=int, default=10, 
                       help="Save checkpoint every N epochs")
    
    # Motion detection parameters
    parser.add_argument("--use-motion-masks", action="store_true", default=True,
                       help="Use motion masks for enhanced tracking")
    parser.add_argument("--no-motion-masks", dest="use_motion_masks", 
                       action="store_false",
                       help="Disable motion masks (3-channel input)")
    parser.add_argument("--motion-cache-dir", type=str, 
                       default="motion_cache",
                       help="Directory for caching motion masks")
    parser.add_argument("--pixel-threshold", type=int, default=15,
                       help="Motion detection pixel threshold")
    parser.add_argument("--precompute-motion", action="store_true", 
                       help="Pre-compute motion masks before training")
    
    # Output configuration
    parser.add_argument("--project", type=str, 
                       default="tennis_ball_training",
                       help="Project name for saving results")
    parser.add_argument("--name", type=str, 
                       default="yolo11_tennis_pose",
                       help="Experiment name")
    parser.add_argument("--resume", type=str, default="",
                       help="Resume training from checkpoint")
    
    return parser.parse_args()


def setup_environment():
    """Set up training environment and check dependencies."""
    print(f"{colorstr('Setup')}: Checking environment...")
    
    # Check required files
    required_files = [
        "ultralytics/cfg/models/11/yolo11-tennis-pose.yaml",
        "ultralytics/data/ball_dataset.py",
        "ultralytics/nn/tasks.py",  # Should contain TennisBallPoseModel
        "tennis_ball_trainer.py"
    ]
    
    missing_files = []
    for file_path in required_files:
        if not Path(file_path).exists():
            missing_files.append(file_path)
    
    if missing_files:
        print(f"{colorstr('Error')}: Missing required files:")
        for file_path in missing_files:
            print(f"  - {file_path}")
        return False
    
    # Check dataset
    dataset_paths = ["Dataset_YOLO/data.yaml", "Dataset_YOLO/images", "Dataset_YOLO/labels"]
    missing_dataset = []
    for path in dataset_paths:
        if not Path(path).exists():
            missing_dataset.append(path)
    
    if missing_dataset:
        print(f"{colorstr('Warning')}: Dataset files not found:")
        for path in missing_dataset:
            print(f"  - {path}")
        print("Make sure your dataset is properly structured.")
    
    print(f"{colorstr('Setup')}: Environment check completed ✓")
    return True


def download_pretrained_weights(weights_name, auto_download=False):
    """Download pretrained weights if not available locally."""
    if not weights_name or Path(weights_name).exists():
        return weights_name
    
    print(f"{colorstr('Pretrained')}: Checking for {weights_name}...")
    
    # List of available pretrained pose models
    available_models = {
        'yolo11n-pose.pt': 'https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11n-pose.pt',
        'yolo11s-pose.pt': 'https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11s-pose.pt',
        'yolo11m-pose.pt': 'https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11m-pose.pt',
        'yolo11l-pose.pt': 'https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11l-pose.pt',
        'yolo11x-pose.pt': 'https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11x-pose.pt'
    }
    
    if weights_name in available_models:
        if auto_download:
            print(f"{colorstr('Download')}: Downloading {weights_name}...")
            try:
                import urllib.request
                urllib.request.urlretrieve(available_models[weights_name], weights_name)
                print(f"{colorstr('Download')}: Successfully downloaded {weights_name}")
                return weights_name
            except Exception as e:
                print(f"{colorstr('Error')}: Failed to download {weights_name}: {e}")
                return None
        else:
            print(f"{colorstr('Info')}: {weights_name} not found locally.")
            print(f"  Available for download from: {available_models[weights_name]}")
            print(f"  Use --auto-download to automatically download pretrained weights")
            return None
    else:
        print(f"{colorstr('Warning')}: Unknown pretrained model: {weights_name}")
        print(f"  Available models: {', '.join(available_models.keys())}")
        return None


def create_motion_config(args):
    """Create motion detection configuration."""
    return MotionConfig(
        pixel_threshold=args.pixel_threshold,
        delta=1,
        window_size=5,
        adaptive_threshold=True,
        min_motion_pixels=100,
        max_motion_pixels=50000
    )


def main():
    """Main training function."""
    args = parse_args()
    
    print("🎾 Tennis Ball Pose Training")
    print("=" * 50)
    
    # Set up environment
    if not setup_environment():
        print(f"{colorstr('Error')}: Environment setup failed!")
        return
    
    # Display configuration
    print(f"\n{colorstr('Configuration')}:")
    print(f"  Model: {args.model}")
    print(f"  Data: {args.data}")
    print(f"  Pretrained: {args.pretrained if args.pretrained else 'None (random initialization)'}")
    print(f"  Epochs: {args.epochs}")
    print(f"  Batch Size: {args.batch}")
    print(f"  Image Size: {args.imgsz}")
    print(f"  Device: {args.device}")
    print(f"  Motion Masks: {args.use_motion_masks}")
    if args.use_motion_masks:
        print(f"  Motion Cache: {args.motion_cache_dir}")
        print(f"  Pixel Threshold: {args.pixel_threshold}")
    
    # Handle pretrained weights
    pretrained_path = None
    if args.pretrained:
        pretrained_path = download_pretrained_weights(args.pretrained, args.auto_download)
        if pretrained_path:
            print(f"{colorstr('Pretrained')}: Will use {pretrained_path}")
        else:
            print(f"{colorstr('Warning')}: Proceeding without pretrained weights")
    
    # Create motion configuration
    motion_config = create_motion_config(args)
    
    # Training configuration
    training_config = {
        "model": args.model,
        "data": args.data,
        "pretrained": pretrained_path,
        "epochs": args.epochs,
        "imgsz": args.imgsz,
        "batch": args.batch,
        "lr0": args.lr0,
        "device": args.device,
        "workers": args.workers,
        "patience": args.patience,
        "save_period": args.save_period,
        "project": args.project,
        "name": args.name,
        "use_motion_masks": args.use_motion_masks,
        "motion_cache_dir": args.motion_cache_dir,
        "precompute_motion": args.precompute_motion,
        "resume": args.resume if args.resume else False,
        # Additional training parameters
        "warmup_epochs": 3,
        "warmup_momentum": 0.8,
        "warmup_bias_lr": 0.1,
        "box": 7.5,
        "cls": 0.5,
        "dfl": 1.5,
        "pose": 12.0,
        "kobj": 2.0,
        "label_smoothing": 0.0,
        "nbs": 64,
        "hsv_h": 0.015,
        "hsv_s": 0.7,
        "hsv_v": 0.4,
        "degrees": 0.0,
        "translate": 0.1,
        "scale": 0.5,
        "shear": 0.0,
        "perspective": 0.0,
        "flipud": 0.0,
        "fliplr": 0.5,
        "bgr": 0.0,
        "mosaic": 1.0,
        "mixup": 0.0,
        "copy_paste": 0.0
    }
    
    print(f"\n{colorstr('Training')}: Starting tennis ball pose training...")
    
    # Create trainer with custom configuration
    trainer = TennisBallTrainer(overrides=training_config)
    
    # Add motion configuration to trainer
    trainer.motion_config = motion_config
    
    print(f"{colorstr('Training')}: Trainer initialized successfully")
    print(f"{colorstr('Training')}: Starting training for {args.epochs} epochs...")
    
    # Start training
    results = trainer.train()
    
    print(f"\n{colorstr('Success')}: Training completed!")
    print(f"Results saved to: {trainer.save_dir}")
    print(f"Best weights: {trainer.best}")
    
    # Display training metrics
    if hasattr(trainer, 'metrics') and trainer.metrics:
        print(f"\n{colorstr('Metrics')}:")
        for key, value in trainer.metrics.items():
            print(f"  {key}: {value:.4f}")
    
    return results
    
if __name__ == "__main__":
    results = main()
