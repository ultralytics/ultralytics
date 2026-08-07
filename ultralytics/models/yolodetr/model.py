# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""YOLO-DETR models with YOLO26 CSP or UltraViT backbones and DeimDecoder heads."""

from ultralytics.engine.model import Model
from ultralytics.models.rtdetr.predict import RTDETRPredictor
from ultralytics.nn.tasks import YOLODETRDetectionModel

from .train import _YOLODETR_DEFAULTS, YOLODETRTrainer, YOLODETRValidator


class YOLODETR(Model):
    """Interface for YOLO-DETR models with YOLO26 CSP or UltraViT backbones and DeimDecoder heads.

    YOLO27l uses a YOLO26-style CSP backbone and FPN/PAN neck, while YOLO27x uses an UltraViT backbone and HybridEncoder
    neck. Both variants reuse the RT-DETR prediction and validation pipeline because their decoder output contract is
    identical (bs, num_queries, [x, y, w, h, conf, cls]). Training is routed through YOLODETRTrainer for augmentation
    decay, flat-cosine learning rates, and separate head/backbone learning rates. YOLODETRDetectionModel dispatches
    DfineLoss for each DeimDecoder head with the full FGL/DDF terms.

    Examples:
        Run inference from a YAML
        >>> from ultralytics import YOLODETR
        >>> model = YOLODETR("yolo27x.yaml")
        >>> results = model("image.jpg")
    """

    _DEIM_KWARGS = tuple(_YOLODETR_DEFAULTS)

    def __init__(self, model: str = "yolo27x.yaml") -> None:
        """Initialize YOLO-DETR from a YAML config or .pt weights.

        Args:
            model (str): Path to a .yaml or .pt file.
        """
        super().__init__(model=model, task="detect")

    def predict(self, source=None, stream: bool = False, predictor=None, **kwargs):
        """Run prediction, defaulting conf to 0.5 when unset since the decoder applies no NMS.

        Args:
            source (str | Path | int | Image.Image | list | tuple | np.ndarray | torch.Tensor, optional): Source of the
                images or videos to predict on.
            stream (bool): Treat the source as a continuous stream.
            predictor (BasePredictor, optional): Predictor instance overriding the default one.
            **kwargs (Any): Prediction arguments forwarded to the predictor.

        Returns:
            (list[Results]): Prediction results, one entry per image.

        Notes:
            The 0.5 default matches DETR visualization defaults. An explicitly passed conf always wins, including
            conf=0.0, because the check is against None rather than falsiness.
        """
        if kwargs.get("conf") is None:  # unset, or None from default.yaml; an explicit value always wins
            kwargs["conf"] = 0.5
        return super().predict(source, stream, predictor, **kwargs)

    def train(self, trainer=None, **kwargs):
        """Train the model, routing DEIM-specific kwargs so they survive get_cfg's alignment check.

        Args:
            trainer (BaseTrainer, optional): Trainer instance overriding the default one.
            **kwargs (Any): Training arguments; DEIM-specific keys are moved into self.overrides first because get_cfg
                rejects any key that default.yaml does not define.

        Returns:
            (dict): Training metrics.
        """
        deim = {k: kwargs.pop(k) for k in list(kwargs) if k in self._DEIM_KWARGS}
        if deim:
            self.overrides = {**self.overrides, **deim}
        return super().train(trainer=trainer, **kwargs)

    @property
    def task_map(self) -> dict:
        """Map the detect task to YOLODETRTrainer + YOLODETRDetectionModel + RT-DETR predict/val.

        Returns:
            (dict): Mapping of the detect task to its predictor, validator, trainer, and model classes.
        """
        return {
            "detect": {
                "predictor": RTDETRPredictor,
                "validator": YOLODETRValidator,
                "trainer": YOLODETRTrainer,
                "model": YOLODETRDetectionModel,
            }
        }
