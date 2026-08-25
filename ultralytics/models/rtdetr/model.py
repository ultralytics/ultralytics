# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""
Interface for Baidu's RT-DETR, a Vision Transformer-based real-time object detector.

RT-DETR offers real-time performance and high accuracy, excelling in accelerated backends like CUDA with TensorRT.
It features an efficient hybrid encoder and IoU-aware query selection for enhanced detection accuracy.

References:
    https://arxiv.org/pdf/2304.08069.pdf
"""

from typing import Any

import torch

from ultralytics.engine.model import Model
from ultralytics.nn.tasks import RTDETRDetectionModel
from ultralytics.utils.torch_utils import TORCH_1_11

from .deim import RTDETRDEIMTrainer, RTDETRDEIMTrainerV2, RTDETRDEIMValidator
from .predict import RTDETRPredictor
from .segment import (
    RTDETRDEIMSegmentationPredictor,
    RTDETRDEIMSegmentationTrainer,
    RTDETRDEIMSegmentationValidator,
)
from .train import RTDETRTrainer
from .val import RTDETRValidator


class RTDETR(Model):
    """Interface for Baidu's RT-DETR model, a Vision Transformer-based real-time object detector.

    This model provides real-time performance with high accuracy. It supports efficient hybrid encoding, IoU-aware query
    selection, and adaptable inference speed.

    Attributes:
        model (str): Path to the pre-trained model.

    Methods:
        task_map: Return a task map for RT-DETR, associating tasks with corresponding Ultralytics classes.

    Examples:
        Initialize RT-DETR with a pre-trained model
        >>> from ultralytics import RTDETR
        >>> model = RTDETR("rtdetr-l.pt")
        >>> results = model("image.jpg")
    """

    def __init__(self, model: str = "rtdetr-l.pt") -> None:
        """Initialize the RT-DETR model with the given pre-trained model file.

        Args:
            model (str): Path to the pre-trained model. Supports .pt, .yaml, and .yml formats.
        """
        assert TORCH_1_11, "RTDETR requires torch>=1.11"
        super().__init__(model=model, task="detect")

    @property
    def task_map(self) -> dict:
        """Return a task map for RT-DETR, associating tasks with corresponding Ultralytics classes.

        Returns:
            (dict): A dictionary mapping task names to Ultralytics task classes for the RT-DETR model.
        """
        return {
            "detect": {
                "predictor": RTDETRPredictor,
                "validator": RTDETRValidator,
                "trainer": RTDETRTrainer,
                "model": RTDETRDetectionModel,
            }
        }


class RTDETRDEIMPredictor(RTDETRPredictor):
    """Predictor that applies ImageNet normalization when ``rtdetr_input_normalize`` is set."""

    @staticmethod
    def _normalize_input(img: torch.Tensor) -> torch.Tensor:
        mean = img.new_tensor((0.485, 0.456, 0.406)).view(1, 3, 1, 1)
        std = img.new_tensor((0.229, 0.224, 0.225)).view(1, 3, 1, 1)
        return (img - mean) / std

    def preprocess(self, im):
        im = super().preprocess(im)
        if getattr(self.args, "rtdetr_input_normalize", False):
            im = self._normalize_input(im)
        return im


class RTDETRDEIM(RTDETR):
    """RT-DETR interface that routes training/validation through isolated DEIM classes."""

    _EXTRA_CKPT_ARGS = {"rtdetr_input_normalize"}

    def __init__(self, model: str = "rtdetr-l.pt") -> None:
        """Initialize DEIM and infer detect versus segment from its decoder head."""
        assert TORCH_1_11, "RTDETR requires torch>=1.11"
        Model.__init__(self, model=model, task=None)

    @staticmethod
    def _reset_ckpt_args(args: dict[str, Any]) -> dict[str, Any]:
        include = {"imgsz", "data", "task", "single_cls"} | RTDETRDEIM._EXTRA_CKPT_ARGS
        return {k: v for k, v in args.items() if k in include}

    @property
    def task_map(self) -> dict:
        """Return a task map that uses DEIM-specific trainer/validator implementations."""
        return {
            "detect": {
                "predictor": RTDETRDEIMPredictor,
                "validator": RTDETRDEIMValidator,
                "trainer": RTDETRDEIMTrainer,
                "model": RTDETRDetectionModel,
            },
            "segment": {
                "predictor": RTDETRDEIMSegmentationPredictor,
                "validator": RTDETRDEIMSegmentationValidator,
                "trainer": RTDETRDEIMSegmentationTrainer,
                "model": RTDETRDetectionModel,
            },
        }


class RTDETRDEIMv2(RTDETR):
    """RT-DETR interface that routes training through DEIM v2 trainer with stage-switch EMA refresh."""

    @staticmethod
    def _reset_ckpt_args(args: dict[str, Any]) -> dict[str, Any]:
        include = {"imgsz", "data", "task", "single_cls"} | RTDETRDEIM._EXTRA_CKPT_ARGS
        return {k: v for k, v in args.items() if k in include}

    @property
    def task_map(self) -> dict:
        """Return a task map that uses DEIM v2 trainer implementation."""
        return {
            "detect": {
                "predictor": RTDETRDEIMPredictor,
                "validator": RTDETRDEIMValidator,
                "trainer": RTDETRDEIMTrainerV2,
                "model": RTDETRDetectionModel,
            }
        }
