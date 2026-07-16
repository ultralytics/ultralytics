# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

"""Stereo 3D Object Detection module for YOLO.

This module implements stereo-based 3D object detection with CenterNet-style outputs,
including geometric construction, dense alignment, and occlusion handling.

Key Components:
    - Stereo3DDetModel: Main model class for stereo 3D detection
    - Stereo3DDetTrainer: Training logic with stereo-specific augmentation
    - Stereo3DDetValidator: Validation with KITTI AP3D metrics
    - Stereo3DDetPredictor: Inference pipeline for stereo images
"""

from .model import Stereo3DDetModel
from .train import Stereo3DDetTrainer
from .val import Stereo3DDetValidator
from .predict import Stereo3DDetPredictor

__all__ = [
    "Stereo3DDetModel",
    "Stereo3DDetTrainer",
    "Stereo3DDetValidator",
    "Stereo3DDetPredictor",
]
