# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""YOLO-DETR: DEIMv2-style DINOv3-ViT + STA backbone with DEIM/DFine/RT-DETR decoder variants."""

from ultralytics.engine.model import Model
from ultralytics.models.rtdetr.predict import RTDETRPredictor
from ultralytics.nn.tasks import YOLODETRDetectionModel

from .train import _YOLODETR_DEFAULTS, YOLODETRTrainer, YOLODETRValidator


class YOLODETR(Model):
    """Interface for YOLO-DETR, a DEIMv2-style DINOv3-ViT + STA detector with DEIM/DFine/RT-DETR decoder heads.

    Scale-to-backbone mapping: n/s/m/l scales use a CNN CSP trunk with RTDETRDecoderV2; x scale uses a
    DINOv3-ViT-S/16-plus + STA backbone with DeimDecoder; xxl scale uses a DINOv3-ViT-B/16 + STA backbone with
    DeimDecoder. Reuses the RT-DETR predict/val pipeline because the decoder output contract is identical
    (bs, num_queries, [x, y, w, h, conf, cls]). Training is routed through YOLODETRTrainer for augment decay,
    flat-cosine LR, and the head/backbone LR split. The underlying detection model is YOLODETRDetectionModel which
    dispatches DfineLoss for D-Fine/DEIM heads with the full FGL/DDF terms and for RTDETRDecoderV2 with FGL/DDF gains
    zeroed and union-set matching off (RTDETRDecoderV2 emits no pred_corners / pre-stage tensors). The parent
    RTDETRDecoder head still uses RTDETRDetectionLoss.

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
