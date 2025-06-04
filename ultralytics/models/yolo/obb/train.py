# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from copy import copy
from pathlib import Path
from typing import Any, List, Optional, Union

from ultralytics.models import yolo
from ultralytics.nn.tasks import OBBModel
from ultralytics.utils import DEFAULT_CFG, RANK


class OBBTrainer(yolo.detect.DetectionTrainer):
    """
    A class extending the DetectionTrainer class for training based on an Oriented Bounding Box (OBB) model.

    This trainer specializes in training YOLO models that detect oriented bounding boxes, which are useful for
    detecting objects at arbitrary angles rather than just axis-aligned rectangles.

    Attributes:
        loss_names (tuple): Names of the loss components used during training including box_loss, cls_loss,
            and dfl_loss.

    Methods:
        get_model: Return OBBModel initialized with specified config and weights.
        get_validator: Return an instance of OBBValidator for validation of YOLO model.

    Examples:
        >>> from ultralytics.models.yolo.obb import OBBTrainer
        >>> args = dict(model="yolo11n-obb.pt", data="dota8.yaml", epochs=3)
        >>> trainer = OBBTrainer(overrides=args)
        >>> trainer.train()
    """

    def __init__(self, cfg=DEFAULT_CFG, overrides: Optional[dict] = None, _callbacks: Optional[List[Any]] = None):
        """
        Initialize an OBBTrainer object for training Oriented Bounding Box (OBB) models.

        This trainer extends the DetectionTrainer class to specialize in training models that detect oriented
        bounding boxes. It automatically sets the task to 'obb' in the configuration.

        Args:
            cfg (dict, optional): Configuration dictionary for the trainer. Contains training parameters and
                model configuration.
            overrides (dict, optional): Dictionary of parameter overrides for the configuration. Any values here
                will take precedence over those in cfg.
            _callbacks (List[Any], optional): List of callback functions to be invoked during training.

        Examples:
            >>> from ultralytics.models.yolo.obb import OBBTrainer
            >>> args = dict(model="yolo11n-obb.pt", data="dota8.yaml", epochs=3)
            >>> trainer = OBBTrainer(overrides=args)
            >>> trainer.train()
        """
        if overrides is None:
            overrides = {}
        overrides["task"] = "obb"
        super().__init__(cfg, overrides, _callbacks)

    def get_model(
        self, cfg: Optional[Union[str, dict]] = None, weights: Optional[Union[str, Path]] = None, verbose: bool = True
    ) -> OBBModel:
        """
        Return OBBModel initialized with specified config and weights.

        Args:
            cfg (str | dict, optional): Model configuration. Can be a path to a YAML config file, a dictionary
                containing configuration parameters, or None to use default configuration.
            weights (str | Path, optional): Path to pretrained weights file. If None, random initialization is used.
            verbose (bool): Whether to display model information during initialization.

        Returns:
            (OBBModel): Initialized OBBModel with the specified configuration and weights.

        Examples:
            >>> trainer = OBBTrainer()
            >>> model = trainer.get_model(cfg="yolo11n-obb.yaml", weights="yolo11n-obb.pt")
        """
        model = OBBModel(cfg, nc=self.data["nc"], ch=self.data["channels"], verbose=verbose and RANK == -1)
        if weights:
            model.load(weights)

        return model

    def get_validator(self):
        """Return an instance of OBBValidator for validation of YOLO model."""
        self.loss_names = "box_loss", "cls_loss", "dfl_loss"
        return yolo.obb.OBBValidator(
            self.test_loader, save_dir=self.save_dir, args=copy(self.args), _callbacks=self.callbacks
        )
