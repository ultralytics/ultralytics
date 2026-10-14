# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from __future__ import annotations

from pathlib import Path
from types import MethodType
from typing import Any

import torch

from ultralytics.engine.model import Model
from ultralytics.utils import DEFAULT_CFG_DICT
from ultralytics.utils.checks import check_model_file_from_stem
from ultralytics.utils.downloads import attempt_download_asset
from ultralytics.utils.patches import torch_load
from ultralytics.utils.torch_utils import model_info

from .predict import NASPredictor
from .val import NASValidator


class NAS(Model):
    """YOLO-NAS model for object detection.

    This class provides an interface for the YOLO-NAS models and extends the `Model` class from Ultralytics engine. It
    is designed to facilitate the task of object detection using pre-trained or custom-trained YOLO-NAS models.

    Attributes:
        model (torch.nn.Module): The loaded YOLO-NAS model.
        task (str): The task type for the model, defaults to 'detect'.
        predictor (NASPredictor): The predictor instance for making predictions, created on the first prediction.

    Methods:
        info: Log model information and return model details.

    Examples:
        >>> from ultralytics import NAS
        >>> model = NAS("yolo_nas_s.pt")
        >>> results = model.predict("ultralytics/assets/bus.jpg")

    Notes:
        YOLO-NAS models only support pre-trained models. Do not provide YAML configuration files.
    """

    def __init__(self, model: str = "yolo_nas_s.pt") -> None:
        """Initialize the NAS model with the provided or default model.

        Args:
            model (str): Path to a pre-trained NAS model file (.pt) or a YOLO-NAS model name (e.g. "yolo_nas_s").
        """
        assert Path(model).suffix not in {".yaml", ".yml"}, "YOLO-NAS models only support pre-trained models."
        super().__init__(model, task="detect")

    def _load(self, weights: str, task=None) -> None:
        """Load NAS model weights from a file or a model name.

        Args:
            weights (str): Path to the model weights file or model name.
            task (str, optional): Task type for the model.
        """
        self.model = torch_load(attempt_download_asset(check_model_file_from_stem(weights)))

        def forward(m, x):
            """Return predictions in the Ultralytics detect layout (B, 4 + nc, N) with xywh boxes."""
            y = type(m).forward(m, x)
            b, scores = y if torch.jit.is_tracing() else y[0]  # super-gradients drops raw outputs when traced
            return torch.cat(((b[..., :2] + b[..., 2:]) / 2, b[..., 2:] - b[..., :2], scores), -1).permute(0, 2, 1)

        # Standardize model attributes for compatibility, binding methods to the module so its copies use their own weights
        self.model.forward = MethodType(forward, self.model)
        self.model.fuse = MethodType(lambda m, verbose=True, imgsz=640: m, self.model)
        self.model.stride = torch.tensor([32])
        self.model.names = dict(enumerate(self.model._class_names))
        self.model.is_fused = lambda: False  # for info()
        self.model.yaml = {}  # for info()
        self.model.pt_path = str(weights)  # for export()
        self.model.task = "detect"  # for export()
        self.model.args = {**DEFAULT_CFG_DICT, **self.overrides}  # for export()
        self.model.eval()
        self.model_name = weights

    def info(self, detailed: bool = False, verbose: bool = True, imgsz: int | list[int] = 640) -> tuple:
        """Log model information.

        Args:
            detailed (bool): Show detailed information about model.
            verbose (bool): Controls verbosity.
            imgsz (int | list[int]): Input image size used for FLOPs calculation.

        Returns:
            (tuple): Model information as a tuple of (layers, parameters, gradients, GFLOPs).
        """
        return model_info(self.model, detailed=detailed, verbose=verbose, imgsz=imgsz)

    @property
    def task_map(self) -> dict[str, dict[str, Any]]:
        """Return a dictionary mapping tasks to respective predictor and validator classes."""
        return {"detect": {"predictor": NASPredictor, "validator": NASValidator}}
