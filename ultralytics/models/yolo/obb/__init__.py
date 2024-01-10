# Ultralytics YOLO 🚀, AGPL-3.0 license

from .predict import OBBPredictor
from .train import OBBTrainer
from .val import OBBValidator

__all__ = "OBBPredictor", "OBBTrainer", "OBBValidator"
