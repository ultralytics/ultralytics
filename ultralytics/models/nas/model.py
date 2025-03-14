# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""
YOLO-NAS model interface.

Examples:
    >>> from ultralytics import NAS
    >>> model = NAS("yolo_nas_s")
    >>> results = model.predict("ultralytics/assets/bus.jpg")
"""

from pathlib import Path

import torch

from ultralytics.engine.model import Model
from ultralytics.utils import DEFAULT_CFG_DICT
from ultralytics.utils.downloads import attempt_download_asset
from ultralytics.utils.torch_utils import model_info

from .predict import NASPredictor
from .val import NASValidator


class NAS(Model):
    """
    YOLO NAS model for object detection.

    This class provides an interface for the YOLO-NAS models and extends the `Model` class from Ultralytics engine.
    It is designed to facilitate the task of object detection using pre-trained or custom-trained YOLO-NAS models.

    Attributes:
        model (torch.nn.Module): The loaded YOLO-NAS model.
        task (str): The task type for the model, defaults to 'detect'.
        predictor (NASPredictor): The predictor instance for making predictions.
        validator (NASValidator): The validator instance for model validation.

    Examples:
        >>> from ultralytics import NAS
        >>> model = NAS("yolo_nas_s")
        >>> results = model.predict("ultralytics/assets/bus.jpg")

    Notes:
        YOLO-NAS models only support pre-trained models. Do not provide YAML configuration files.
    """

    def __init__(self, model: str = "yolo_nas_s.pt") -> None:
        """Initialize the NAS model with the provided or default model."""
        assert Path(model).suffix not in {".yaml", ".yml"}, "YOLO-NAS models only support pre-trained models."
        super().__init__(model, task="detect")

    def _load(self, weights: str, task=None) -> None:
        """
        Load an existing NAS model weights or create a new NAS model with pretrained weights.

        Args:
            weights (str): Path to the model weights file or model name.
            task (str, optional): Task type for the model.
        """
        import super_gradients

        suffix = Path(weights).suffix
        if suffix == ".pt":
            self.model = torch.load(attempt_download_asset(weights))
        elif suffix == "":
            self.model = super_gradients.training.models.get(weights, pretrained_weights="coco")

        # Override the forward method to ignore additional arguments
        def new_forward(x, *args, **kwargs):
            """Ignore additional __call__ arguments."""
            return self.model._original_forward(x)

        self.model._original_forward = self.model.forward
        self.model.forward = new_forward

        # Standardize model
        self.model.fuse = lambda verbose=True: self.model
        self.model.stride = torch.tensor([32])
        self.model.names = dict(enumerate(self.model._class_names))
        self.model.is_fused = lambda: False  # for info()
        self.model.yaml = {}  # for info()
        self.model.pt_path = weights  # for export()
        self.model.task = "detect"  # for export()
        self.model.args = {**DEFAULT_CFG_DICT, **self.overrides}  # for export()

    def info(self, detailed: bool = False, verbose: bool = True):
        """
        Log model information.

        Args:
            detailed (bool): Show detailed information about model.
            verbose (bool): Controls verbosity.

        Returns:
            (dict): Model information dictionary.
        """
        return model_info(self.model, detailed=detailed, verbose=verbose, imgsz=640)

    @property
    def task_map(self):
        """Return a dictionary mapping tasks to respective predictor and validator classes."""
        return {"detect": {"predictor": NASPredictor, "validator": NASValidator}}
