# Ultralytics YOLO 🚀, AGPL-3.0 license

from .predict import OBB_PosePredictor
from .train import OBB_PoseTrainer
from .val import OBB_PoseValidator

__all__ = "OBB_PosePredictor", "OBB_PoseTrainer", "OBB_PoseValidator"
