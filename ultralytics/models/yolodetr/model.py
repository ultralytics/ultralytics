# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""YOLO-DETR: DEIMv2-style DINOv3+STA backbone with DeimDecoder head, packaged for inference."""

from ultralytics.engine.model import Model
from ultralytics.models.rtdetr.predict import RTDETRPredictor
from ultralytics.models.rtdetr.train import RTDETRTrainer
from ultralytics.models.rtdetr.val import RTDETRValidator
from ultralytics.nn.tasks import RTDETRDetectionModel


class YOLODETR(Model):
    """Interface for YOLO-DETR, a DEIMv2-style DINOv3+STA detector with a DeimDecoder head.

    Reuses the RT-DETR predict/val/train pipeline because the decoder output contract is identical (bs, num_queries, [x,
    y, w, h, conf, cls]) and the BaseModel wiring is shared.

    Examples:
        Run inference from a YAML
        >>> from ultralytics import YOLODETR
        >>> model = YOLODETR("yolo27-detr.yaml")
        >>> results = model("image.jpg")
    """

    def __init__(self, model: str = "yolo27-detr.yaml") -> None:
        """Initialize YOLO-DETR from a YAML config or .pt weights.

        Args:
            model (str): Path to a .yaml or .pt file.
        """
        super().__init__(model=model, task="detect")

    @property
    def task_map(self) -> dict:
        """Map the detect task to RT-DETR's predictor, validator, trainer, and the shared model class."""
        return {
            "detect": {
                "predictor": RTDETRPredictor,
                "validator": RTDETRValidator,
                "trainer": RTDETRTrainer,
                "model": RTDETRDetectionModel,
            }
        }
