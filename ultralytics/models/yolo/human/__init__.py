# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from .predict import HumanPredictor
from .train import HumanTrainer
from .val import HumanValidator

__all__ = "HumanPredictor", "HumanTrainer", "HumanValidator"
