# Ultralytics YOLO 🚀, GPL-3.0 license

from .predict import DetectionPredictor, predict
from .train import DetectionTrainer, train
from .val import DetectionValidator, val

__all__ = ['DetectionPredictor', 'predict', 'DetectionTrainer', 'train', 'DetectionValidator', 'val']
