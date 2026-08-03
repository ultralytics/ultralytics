# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""YOLO-DETR models with YOLO26 CSP or UltraViT backbones and DeimDecoder heads."""

from ultralytics.engine.model import Model
from ultralytics.models.rtdetr.predict import RTDETRPredictor
from ultralytics.nn.tasks import YOLODETRDetectionModel

from .train import _YOLODETR_DEFAULTS, YOLODETRTrainer, YOLODETRValidator


class YOLODETR(Model):
    """Interface for YOLO-DETR models with YOLO26 CSP or UltraViT backbones and DeimDecoder heads.

    YOLO27l uses a YOLO26-style CSP backbone and FPN/PAN neck, while YOLO27x uses an UltraViT backbone and
    HybridEncoder neck. Both variants reuse the RT-DETR prediction and validation pipeline because their decoder
    output contract is identical (bs, num_queries, [x, y, w, h, conf, cls]). Training is routed through
    YOLODETRTrainer for augmentation decay, flat-cosine learning rates, and separate head/backbone learning rates.
    YOLODETRDetectionModel dispatches DfineLoss for each DeimDecoder head with the full FGL/DDF terms.

    Examples:
        Run inference from a YAML
        >>> from ultralytics import YOLODETR
        >>> model = YOLODETR("yolo27x-detr.yaml")
        >>> results = model("image.jpg")
    """

    _DEIM_KWARGS = tuple(_YOLODETR_DEFAULTS)

    def __init__(self, model: str = "yolo27x-detr.yaml") -> None:
        """Initialize YOLO-DETR from a YAML config or .pt weights.

        Args:
            model (str): Path to a .yaml or .pt file.
        """
        super().__init__(model=model, task="detect")

    def train(self, trainer=None, **kwargs):
        """Forward DEIM-specific kwargs through self.overrides so they survive get_cfg's alignment check."""
        deim = {k: kwargs.pop(k) for k in list(kwargs) if k in self._DEIM_KWARGS}
        if deim:
            self.overrides = {**self.overrides, **deim}
        return super().train(trainer=trainer, **kwargs)

    @property
    def task_map(self) -> dict:
        """Map the detect task to YOLODETRTrainer + YOLODETRDetectionModel + RT-DETR predict/val."""
        return {
            "detect": {
                "predictor": RTDETRPredictor,
                "validator": YOLODETRValidator,
                "trainer": YOLODETRTrainer,
                "model": YOLODETRDetectionModel,
            }
        }
