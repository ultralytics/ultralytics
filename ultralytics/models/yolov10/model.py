# Ultralytics YOLO 🚀, AGPL-3.0 license

from ultralytics.engine.model import Model
from ultralytics.nn.tasks import YOLOv10DetectionModel
from .detect.predict import YOLOv10DetectionPredictor
from .detect.val import YOLOv10DetectionValidator
from .detect.train import YOLOv10DetectionTrainer


class YOLOv10(Model):
    @property
    def task_map(self):
        """Map head to model, trainer, validator, and predictor classes."""
        return {
            "detect": {
                "model": YOLOv10DetectionModel,
                "trainer": YOLOv10DetectionTrainer,
                "validator": YOLOv10DetectionValidator,
                "predictor": YOLOv10DetectionPredictor,
            },
        }
