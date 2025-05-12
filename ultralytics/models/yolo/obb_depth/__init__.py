# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from .predict import OBBDPredictor
from .train import OBBDTrainer
from .val import OBBDValidator

__all__ = "OBBDPredictor", "OBBDTrainer", "OBBDValidator"
