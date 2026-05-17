# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from __future__ import annotations

from copy import copy
from typing import Any

import cv2
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from ultralytics.data.dataset import PolygonSemsegDataset, SemsegDataset, add_polygon_background
from ultralytics.models import yolo
from ultralytics.models.yolo.detect import DetectionTrainer
from ultralytics.nn.tasks import SemanticSegmentationModel
from ultralytics.utils import DEFAULT_CFG, LOGGER, RANK
from ultralytics.utils.plotting import Annotator, colors, plt_settings


class SemanticSegmentationTrainer(DetectionTrainer):
    """Trainer for YOLO semantic segmentation models.

    This trainer handles semantic segmentation specific training including dataset building, model initialization, and
    validation setup.

    Examples:
        >>> from ultralytics.models.yolo.semseg import SemanticSegmentationTrainer
        >>> args = dict(model="yolo26n-semseg.yaml", data="cityscapes8.yaml", epochs=3)
        >>> trainer = SemanticSegmentationTrainer(overrides=args)
        >>> trainer.train()
    """

    def __init__(self, cfg=DEFAULT_CFG, overrides: dict[str, Any] | None = None, _callbacks: dict | None = None):
        """Initialize SemanticSegmentationTrainer.

        Args:
            cfg (dict): Configuration dictionary with default training settings.
            overrides (dict, optional): Dictionary of parameter overrides.
            _callbacks (list, optional): List of callback functions.
        """
        if overrides is None:
            overrides = {}
        overrides["task"] = "semseg"
        super().__init__(cfg, overrides, _callbacks)

    def get_dataset(self):
        """Parse the dataset YAML and bump nc/names with a background class for the polygon path."""
        return add_polygon_background(super().get_dataset())

    def build_dataset(self, img_path: str, mode: str = "train", batch: int | None = None):
        """Build semantic segmentation dataset.

        Routes to `PolygonSemsegDataset` when the dataset YAML lacks 'masks_dir' (polygon labels
        rasterized on the fly) and to `SemsegDataset` otherwise (PNG mask labels).

        Args:
            img_path (str): Path to the folder containing images.
            mode (str): 'train' or 'val' mode.
            batch (int, optional): Batch size for rect mode.

        Returns:
            (SemsegDataset): Semantic segmentation dataset.
        """
        use_rect = mode == "val"
        dataset_cls = SemsegDataset if self.data.get("masks_dir") else PolygonSemsegDataset
        return dataset_cls(
            img_path=img_path,
            imgsz=self.args.imgsz,
            augment=mode == "train",
            hyp=self.args,
            cache=self.args.cache or None,
            data=self.data,
            rect=use_rect,
            batch_size=batch,
            stride=max(int(self.model.stride.max() if hasattr(self.model, "stride") else 32), 32),
            prefix=f"{mode}: ",
            pad=0,
        )

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
        self.loss_names = "ce_loss", "dice_loss", "aux_loss"
        return yolo.semseg.SemanticSegmentationValidator(
            self.test_loader, save_dir=self.save_dir, args=copy(self.args), _callbacks=self.callbacks
        )

    def set_class_weights(self):
        """Skip bbox-based class weight computation for semantic segmentation.

        Semantic segmentation requires pixel-level class frequency counting from masks,
        which is not performed here. The loss function uses dataset-level Cityscapes
        weights when applicable instead.
        """
        pass

    def plot_training_samples(self, batch, ni):
        """Plot training samples with semantic mask overlay.

        Args:
            batch (dict): Batch data containing 'img' and 'semantic_mask'.
            ni (int): Batch index for naming output file.
        """
        images = batch["img"]  # [B, 3, H, W] float 0-1
        masks = batch["semantic_mask"]  # [B, H, W] long
        max_subplots = min(16, len(images))
        images = images[:max_subplots]
        masks = masks[:max_subplots]

        bs, _, h, w = images.shape
        # Create grid
        ns = int(np.ceil(bs**0.5))  # grid size
        mosaic = np.zeros((ns * h, ns * w, 3), dtype=np.uint8)

        for i in range(bs):
            # Image
            img = (images[i].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

            # Mask overlay
            mask = masks[i].cpu().numpy()
            annotator = Annotator(img, line_width=1)
            annotator.semantic_mask(mask, alpha=0.4)
            img = annotator.result()

            # Place in grid
            row, col = i // ns, i % ns
            mosaic[row * h : (row + 1) * h, col * w : (col + 1) * w] = img

        fname = self.save_dir / f"train_batch{ni}.jpg"
        cv2.imwrite(str(fname), mosaic)
        if self.on_plot:
            self.on_plot(fname)

    @plt_settings()
    def plot_training_labels(self):
        """Plot training labels class distribution for semantic segmentation.

        Samples up to 1000 mask files from the training dataset, accumulates per-class pixel
        counts, and plots a bar chart of class distribution saved to 'labels.jpg'.
        """
        LOGGER.info(f"Plotting labels to {self.save_dir / 'labels.jpg'}... ")
        nc = self.data["nc"]
        names = self.data["names"]
        pixel_counts = np.zeros(nc, dtype=np.int32)

        dataset = self.train_loader.dataset
        mask_files = getattr(dataset, "mask_files", [])
        if not mask_files:
            LOGGER.warning("No mask files found, skipping plot_training_labels")
            return

        sample_size = min(1000, len(mask_files))
        indices = np.linspace(0, len(mask_files) - 1, sample_size).astype(int)

        for idx in indices:
            try:
                mask = np.array(Image.open(mask_files[idx]))
            except Exception:
                continue
            if hasattr(dataset, "label_mapping") and dataset.label_mapping:
                for old, new in dataset.label_mapping.items():
                    mask[mask == old] = new
            valid = (mask >= 0) & (mask < nc) & (mask != 255)
            if valid.any():
                classes, counts = np.unique(mask[valid], return_counts=True)
                for c, count in zip(classes, counts):
                    pixel_counts[int(c)] += int(count)

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
