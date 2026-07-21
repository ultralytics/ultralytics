# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from __future__ import annotations

from copy import copy
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from ultralytics.data.utils import add_polygon_background
from ultralytics.models import yolo
from ultralytics.models.yolo.detect import DetectionTrainer
from ultralytics.nn.tasks import SemanticSegmentationModel
from ultralytics.utils import DEFAULT_CFG, LOGGER, RANK
from ultralytics.utils.plotting import colors, plt_settings


class SemanticSegmentationTrainer(DetectionTrainer):
    """Trainer for YOLO semantic segmentation models.

    This trainer handles semantic segmentation training, including dataset building, model initialization, and
    validation setup.

    Examples:
        >>> from ultralytics.models.yolo.semantic import SemanticSegmentationTrainer
        >>> args = dict(model="yolo26n-sem.yaml", data="cityscapes8.yaml", epochs=3)
        >>> trainer = SemanticSegmentationTrainer(overrides=args)
        >>> trainer.train()
    """

    def __init__(self, cfg=DEFAULT_CFG, overrides: dict[str, Any] | None = None, _callbacks: dict | None = None):
        """Initialize SemanticSegmentationTrainer.

        Args:
            cfg (dict): Configuration dictionary with default training settings.
            overrides (dict, optional): Dictionary of parameter overrides.
            _callbacks (dict, optional): Callback functions.
        """
        if overrides is None:
            overrides = {}
        overrides["task"] = "semantic"
        super().__init__(cfg, overrides, _callbacks)

    def get_dataset(self):
        """Parse the dataset YAML and add background metadata for polygon labels when required."""
        return add_polygon_background(super().get_dataset())

    def get_model(self, cfg: str | None = None, weights: str | None = None, verbose: bool = True):
        """Return a SemanticSegmentationModel with optional pretrained backbone.

        Args:
            cfg (str, optional): Path to model configuration file.
            weights (str | Path, optional): Path to model weights.
            verbose (bool): Whether to display model information.

        Returns:
            (SemanticSegmentationModel): Semantic segmentation model.
        """
        model = SemanticSegmentationModel(
            cfg, nc=self.data["nc"], ch=self.data["channels"], verbose=verbose and RANK == -1
        )
        if weights:
            model.load(weights)

        return model

    def get_validator(self):
        """Return a SemanticSegmentationValidator for model evaluation."""
        return yolo.semantic.SemanticSegmentationValidator(
            self.test_loader, save_dir=self.save_dir, args=copy(self.args), _callbacks=self.callbacks
        )

    def set_class_weights(self):
        """Compute pixel-frequency class weights; skipped for binary (nc==1, unweighted BCE in the loss)."""
        if self.data["nc"] > 1:
            super().set_class_weights()

    def get_class_counts(self, max_masks=None):
        """Return per-class pixel counts from training masks, optionally sampled to max_masks."""
        nc = self.data["nc"]
        pixel_counts = np.zeros(nc, dtype=np.float32)
        dataset = self.train_loader.dataset
        labels = getattr(dataset, "labels", [])
        if not labels:
            return pixel_counts
        indices = np.arange(len(labels))
        if max_masks and len(indices) > max_masks:
            indices = np.linspace(0, len(labels) - 1, max_masks).astype(int)
        include_class = getattr(dataset, "include_class", None)
        for idx in indices:
            shape = labels[idx].get("shape")
            try:
                mask = dataset.load_mask(idx, image_shape=tuple(shape) if shape is not None else None)
            except Exception:
                continue
            if include_class is not None:
                mask[~np.isin(mask, include_class)] = 255
            valid = (mask >= 0) & (mask < nc) & (mask != 255)
            if valid.any():
                classes, counts = np.unique(mask[valid], return_counts=True)
                pixel_counts[classes.astype(int)] += counts
        return pixel_counts

    def compute_class_weights(self, class_counts):
        """Compute ENet inverse-log `(1/ln(1.02 + p))**cls_pw` weights (Paszke et al., 2016, arXiv:1606.02147)."""
        p = class_counts / max(class_counts.sum(), 1.0)  # pixel frequency, bounded for rare classes unlike detection
        return (1.0 / np.log(1.02 + p)) ** self.args.cls_pw

    @plt_settings()
    def plot_training_labels(self):
        """Plot training labels class distribution for semantic segmentation.

        Samples up to 1000 mask files from the training dataset, accumulates per-class pixel
        counts, and plots a bar chart of class distribution saved to 'labels.jpg'.
        """
        LOGGER.info(f"Plotting labels to {self.save_dir / 'labels.jpg'}...")
        nc = self.data["nc"]
        names = self.data["names"]
        pixel_counts = self.get_class_counts(max_masks=1000)
        if not pixel_counts.any():
            LOGGER.warning("No semantic mask files found, skipping label plot.")
            return

        _, ax = plt.subplots(1, 1, figsize=(8, 6), tight_layout=True)
        bars = ax.bar(range(nc), pixel_counts, color=[list(c / 255.0 for c in colors(i, False)) for i in range(nc)])
        ax.set_xlabel("Class")
        ax.set_ylabel("Pixels")
        ax.set_title("Training Labels Class Distribution")
        if 0 < len(names) < 30:
            ax.set_xticks(range(len(names)))
            ax.set_xticklabels(list(names.values()), rotation=90, fontsize=10)
        for bar in bars:
            height = bar.get_height()
            if height > 0:
                ax.text(
                    bar.get_x() + bar.get_width() / 2.0,
                    height,
                    f"{int(height):,}",
                    ha="center",
                    va="bottom",
                    fontsize=8,
                )
        for spine in ax.spines.values():
            spine.set_visible(False)

        fname = self.save_dir / "labels.jpg"
        plt.savefig(fname, dpi=200)
        plt.close()
        if self.on_plot:
            self.on_plot(fname)
