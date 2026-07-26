# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from __future__ import annotations

from copy import copy
from pathlib import Path
from typing import Any

import numpy as np

from ultralytics.data.dataset import HumanDataset
from ultralytics.models import yolo
from ultralytics.nn.tasks import HumanModel
from ultralytics.utils import DEFAULT_CFG, RANK, colorstr
from ultralytics.utils.plotting import plot_attributes
from ultralytics.utils.torch_utils import unwrap_model


class HumanTrainer(yolo.detect.DetectionTrainer):
    """A class extending the DetectionTrainer class for training YOLO-Human attribute estimation models.

    Attributes:
        model (HumanModel): The human attribute estimation model being trained.
        data (dict): Dictionary containing dataset information including class names and number of classes.
        loss_names (tuple): Names of the loss components, derived from the loss dict returned by the criterion.

    Methods:
        get_model: Retrieve a human attribute estimation model with specified configuration.
        get_validator: Create a validator instance for model evaluation.
        build_dataset: Build the HumanDataset for training or validation.
        plot_training_labels: Plot box and human attribute statistics of the training labels.

    Examples:
        >>> from ultralytics.models.yolo.human import HumanTrainer
        >>> args = dict(model="yolov8n-human.pt", data="coco8-human.yaml", epochs=3)
        >>> trainer = HumanTrainer(overrides=args)
        >>> trainer.train()
    """

    def __init__(self, cfg=DEFAULT_CFG, overrides: dict[str, Any] | None = None, _callbacks: dict | None = None):
        """Initialize a HumanTrainer object for training YOLO-Human models.

        Args:
            cfg (dict, optional): Default configuration dictionary containing training parameters.
            overrides (dict, optional): Dictionary of parameter overrides for the default configuration.
            _callbacks (dict, optional): Dictionary of callback functions to be executed during training.
        """
        if overrides is None:
            overrides = {}
        overrides["task"] = "human"
        super().__init__(cfg, overrides, _callbacks)

    def get_model(
        self,
        cfg: str | Path | dict[str, Any] | None = None,
        weights: str | Path | None = None,
        verbose: bool = True,
    ) -> HumanModel:
        """Return a HumanModel initialized with the specified config and weights.

        Args:
            cfg (str | Path | dict, optional): Model configuration file path or dictionary.
            weights (str | Path, optional): Path to the model weights file.
            verbose (bool): Whether to display model information.

        Returns:
            (HumanModel): Initialized human attribute estimation model.
        """
        model = self.set_model_names_for_load(
            HumanModel(cfg, nc=self.data["nc"], ch=self.data["channels"], verbose=verbose and RANK == -1)
        )
        if weights:
            model.load(weights)
        return model

    def get_validator(self):
        """Return an instance of HumanValidator for validation of the YOLO-Human model."""
        return yolo.human.HumanValidator(
            self.test_loader, save_dir=self.save_dir, args=copy(self.args), _callbacks=self.callbacks
        )

    def build_dataset(self, img_path: str, mode: str = "train", batch: int | None = None) -> HumanDataset:
        """Build the HumanDataset for training or validation.

        Args:
            img_path (str): Path to the folder containing images.
            mode (str): 'train' mode or 'val' mode, users are able to customize different augmentations for each mode.
            batch (int, optional): Size of batches, this is for 'rect' mode.

        Returns:
            (HumanDataset): Dataset object configured for the specified mode.
        """
        cfg = self.args
        return HumanDataset(
            img_path=img_path,
            imgsz=cfg.imgsz,
            batch_size=batch,
            augment=mode == "train",  # augmentation
            hyp=cfg,
            rect=cfg.rect or mode == "val",  # rectangular batches
            cache=cfg.cache or None,
            single_cls=cfg.single_cls or False,
            stride=max(int(unwrap_model(self.model).stride.max()), 32),
            pad=0.0 if mode == "train" else 0.5,
            prefix=colorstr(f"{mode}: "),
            classes=cfg.classes,
            fraction=cfg.fraction if mode == "train" else 1.0,
        )

    def plot_training_labels(self):
        """Create labeled training plots for both boxes and human attributes."""
        attributes = np.concatenate([lb["attributes"] for lb in self.train_loader.dataset.labels], 0)
        plot_attributes(attributes, save_dir=self.save_dir, on_plot=self.on_plot)
        super().plot_training_labels()
