# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from copy import copy

from ultralytics.models.yolo.segment import SegmentationTrainer
from ultralytics.nn.tasks import YOLOESegModel
from ultralytics.utils import RANK

from .train import YOLOEPETrainer, YOLOETrainer, YOLOETrainerFromScratch, YOLOEVPTrainer
from .val import YOLOESegValidator


class YOLOESegTrainer(YOLOETrainer, SegmentationTrainer):
    """Trainer class for YOLOE segmentation models.

    This class combines YOLOETrainer and SegmentationTrainer to provide training functionality specifically for YOLOE
    segmentation models, enabling both object detection and instance segmentation capabilities.

    Attributes:
        cfg (dict): Configuration dictionary with training parameters.
        overrides (dict): Dictionary with parameter overrides.
        _callbacks (dict): Dictionary of callback functions for training events.
    """

    def get_model(self, cfg=None, weights=None, verbose=True):
        """Return YOLOESegModel initialized with specified config and weights.

        Args:
            cfg (dict | str, optional): Model configuration dictionary or YAML file path.
            weights (str, optional): Path to pretrained weights file.
            verbose (bool): Whether to display model information.

        Returns:
            (YOLOESegModel): Initialized YOLOE segmentation model.
        """
        # NOTE: This `nc` here is the max number of different text samples in one image, rather than the actual `nc`.
        # NOTE: Following the official config, nc hard-coded to 80 for now.
        model = YOLOESegModel(
            cfg["yaml_file"] if isinstance(cfg, dict) else cfg,
            ch=self.data["channels"],
            nc=min(self.data["nc"], 80),
            verbose=verbose and RANK == -1,
        )
        if weights:
            model.load(weights)

        return model

    def get_validator(self):
        """Create and return a validator for YOLOE segmentation model evaluation.

        Returns:
            (YOLOESegValidator): Validator for YOLOE segmentation models.
        """
        return YOLOESegValidator(
            self.test_loader, save_dir=self.save_dir, args=copy(self.args), _callbacks=self.callbacks
        )


class YOLOEPESegTrainer(SegmentationTrainer):
    """Fine-tune YOLOE segmentation models in linear probing way."""

    get_model = YOLOEPETrainer.get_model  # shared linear-probing builder; SegmentationTrainer stays the sole base


class YOLOESegTrainerFromScratch(YOLOETrainerFromScratch, YOLOESegTrainer):
    """Trainer for YOLOE segmentation models trained from scratch without pretrained weights."""


class YOLOESegVPTrainer(YOLOEVPTrainer, YOLOESegTrainerFromScratch):
    """Trainer for YOLOE segmentation models with Vision Prompt (VP) capabilities."""
