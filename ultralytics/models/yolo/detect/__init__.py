# Ultralytics YOLO 🚀, AGPL-3.0 license

from .neuron_predict import NeuronDetectionPredictor
from .predict import DetectionPredictor
from .train import DetectionTrainer
from .val import DetectionValidator

__all__ = (
    "DetectionPredictor",
    "DetectionTrainer",
    "DetectionValidator",
    "NeuronDetectionPredictor",
)
