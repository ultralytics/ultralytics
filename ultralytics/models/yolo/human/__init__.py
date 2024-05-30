# Ultralytics YOLO 🚀, AGPL-3.0 license

from .predict import HumanPredictor
from .train import HumanTrainer
from .val import HumanValidator

__all__ = "HumanPredictor", "HumanTrainer", "HumanValidator"
