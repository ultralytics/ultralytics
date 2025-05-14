# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from pathlib import Path

from ultralytics.data.build import load_inference_source
from ultralytics.engine.model import Model
from ultralytics.models import yolo, yolo_manitou
from ultralytics.nn.tasks import (
    DetectionModel,
)
from ultralytics.utils import ROOT, yaml_load


class YOLOManitou(Model):
    """YOLO (You Only Look Once) object detection model."""

    @property
    def task_map(self):
        """Map head to model, trainer, validator, and predictor classes."""
        return {
            "detect": {
                "model": DetectionModel,
                "trainer": yolo_manitou.detect.ManitouTrainer,
                "validator": yolo_manitou.detect.ManitouValidator,
                "predictor": yolo_manitou.detect.ManitouPredictor,
            },
        }