# Ultralytics YOLO 🚀, AGPL-3.0 license

from .predict import OBB_SegmentationPredictor
from .train import OBB_SegmentationTrainer
from .val import OBB_SegmentationValidator

__all__ = "OBB_SegmentationPredictor", "OBB_SegmentationTrainer", "OBB_SegmentationValidator"
