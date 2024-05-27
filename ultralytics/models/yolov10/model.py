# Ultralytics YOLO 🚀, AGPL-3.0 license

from ultralytics.engine.model import Model
from ultralytics.nn.tasks import v10DetectionModel, DetectionModel

from ultralytics.models import yolo
from .detect.train import YOLOv10DetectionTrainer


class YOLOv10(Model):
    @property
    def task_map(self):
        """Map head to model, trainer, validator, and predictor classes."""
        return {
            "detect": {
                "model": DetectionModel,
                "trainer": YOLOv10DetectionTrainer,
                "validator": yolo.detect.DetectionValidator,
                "predictor": yolo.detect.DetectionPredictor,
            },
        }
