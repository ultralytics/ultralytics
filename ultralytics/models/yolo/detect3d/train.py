# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from __future__ import annotations

from copy import copy
from pathlib import Path
from typing import Any

import numpy as np

from ultralytics.models import yolo
from ultralytics.nn.modules.head import Detect3D
from ultralytics.nn.tasks import Detection3DModel
from ultralytics.utils import DEFAULT_CFG, RANK
from ultralytics.utils.geometry3d import image_transform_matrix, transform_projection
from ultralytics.utils.plotting import plot_detect3d_results, plot_images, plot_labels


class Detection3DTrainer(yolo.detect.DetectionTrainer):
    """A class extending the DetectionTrainer class for training based on a 3D detection model.

    This trainer specializes in training YOLO models that detect objects with 3D attributes (depth, position,
    dimensions, rotation).

    Attributes:
        loss_names (tuple): Names of the loss components, derived from the loss dict returned by the criterion.

    Methods:
        get_model: Return Detection3DModel initialized with specified config and weights.
        get_validator: Return an instance of Detection3DValidator for validation.

    Examples:
        >>> from ultralytics.models.yolo.detect3d import Detection3DTrainer
        >>> args = dict(model="yolo11n-3d.pt", data="kitti3d.yaml", epochs=100)
        >>> trainer = Detection3DTrainer(overrides=args)
        >>> trainer.train()
    """

    def __init__(
        self,
        cfg=DEFAULT_CFG,
        overrides: dict | None = None,
        _callbacks: dict | None = None,
    ):
        """Initialize a Detection3DTrainer object for training 3D detection models.

        Args:
            cfg (dict, optional): Configuration dictionary for the trainer.
            overrides (dict, optional): Dictionary of parameter overrides for the configuration.
            _callbacks (dict, optional): Dictionary of callback functions to be invoked during training.
        """
        if overrides is None:
            overrides = {}
        overrides["task"] = "detect3d"
        super().__init__(cfg, overrides, _callbacks)

    def get_model(
        self,
        cfg: str | dict | None = None,
        weights: str | Path | None = None,
        verbose: bool = True,
    ) -> Detection3DModel:
        """Return Detection3DModel initialized with specified config and weights.

        Args:
            cfg (str | dict, optional): Model configuration. Can be a path to a YAML config file, a dictionary
                containing configuration parameters, or None to use default configuration.
            weights (str | Path, optional): Path to pretrained weights file.
            verbose (bool): Whether to display model information during initialization.

        Returns:
            (Detection3DModel): Initialized Detection3DModel with the specified configuration and weights.
        """
        model = self.set_model_names_for_load(
            Detection3DModel(
                cfg,
                nc=self.data["nc"],
                ch=self.data["channels"],
                verbose=verbose and RANK == -1,
            )
        )
        if weights:
            model.load(weights)

        return model

    def get_validator(self):
        """Return an instance of Detection3DValidator for validation of 3D detection model."""
        return yolo.detect3d.Detection3DValidator(
            self.test_loader,
            save_dir=self.save_dir,
            args=copy(self.args),
            _callbacks=self.callbacks,
        )

    def set_model_attributes(self) -> None:
        """Attach dataset metadata and configure checkpointed Mono3D priors and score calibration."""
        super().set_model_attributes()
        head = self.model.model[-1]
        if not isinstance(head, Detect3D):
            raise TypeError(f"Detection3DTrainer requires Detect3D head, got {type(head).__name__}")
        head.set_dim_priors(self.data["names"], self.data.get("dimension_priors"))
        head.set_quality3d_power(self.args.quality3d_power)

    def preprocess_batch(self, batch: dict) -> dict:
        """Preprocess images and keep augmented camera projections synchronized with multi-scale interpolation."""
        old_h, old_w = batch["img"].shape[-2:]
        batch = super().preprocess_batch(batch)
        new_h, new_w = batch["img"].shape[-2:]
        if (new_h, new_w) == (old_h, old_w):
            return batch

        scale_x, scale_y = new_w / old_w, new_h / old_h
        scale_h = image_transform_matrix(scale_x=scale_x, scale_y=scale_y)
        p2s_aug = batch.get("p2s_aug")
        if p2s_aug is None or any(p2 is None for p2 in p2s_aug):
            raise RuntimeError("detect3d multi-scale training requires one valid augmented P2 matrix per image")
        batch["p2s_aug"] = [transform_projection(p2, scale_h) for p2 in p2s_aug]

        # Keep generic geometry metadata correct for callbacks and future consumers too.
        ratio_pad = batch.get("ratio_pad")
        if ratio_pad is not None:
            batch["ratio_pad"] = tuple(
                (
                    (float(rp[0][0]) * scale_y, float(rp[0][1]) * scale_x),
                    (float(rp[1][0]) * scale_x, float(rp[1][1]) * scale_y),
                )
                for rp in ratio_pad
            )
        return batch

    def plot_training_samples(self, batch: dict[str, Any], ni: int) -> None:
        """Plot training samples with 3D boxes."""
        p2s = batch.get("p2s_aug", None)
        ori_shapes = batch.get("ori_shape", None)
        plot_images(
            labels=batch,
            paths=batch["im_file"],
            fname=self.save_dir / f"train_batch{ni}.jpg",
            on_plot=self.on_plot,
            p2s=p2s,
            ori_shapes=ori_shapes,
            p2s_augmented=True,
        )

    def plot_training_labels(self) -> None:
        """Plot the 2D box columns from Detect3D's extended training labels."""
        boxes = np.concatenate([label["bboxes"] for label in self.train_loader.dataset.labels], axis=0)
        classes = np.concatenate([label["cls"] for label in self.train_loader.dataset.labels], axis=0)
        plot_labels(
            boxes[:, :4],
            classes.squeeze(),
            names=self.data["names"],
            save_dir=self.save_dir,
            on_plot=self.on_plot,
        )

    def plot_metrics(self) -> None:
        """Plot 3D-focused curves separately from the standard 2D detection metrics."""
        plot_detect3d_results(file=self.csv, on_plot=self.on_plot)
