# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from .predict import OBBPredictor
from .train import OBBTrainer
from .val import OBBValidator

__all__ = "OBBPredictor", "OBBTrainer", "OBBValidator"
