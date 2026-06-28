# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from PIL import Image

from ultralytics.data.dataset import SemanticDataset
from ultralytics.data.utils import add_polygon_background
from ultralytics.models.yolo.detect import DetectionValidator
from ultralytics.utils import LOGGER, RANK
from ultralytics.utils.metrics import ConfusionMatrix, SemanticMetrics
from ultralytics.utils.plotting import plot_images


class SemanticSegmentationValidator(DetectionValidator):
    """Validator for semantic segmentation models.

    This validator evaluates semantic segmentation models using mIoU and pixel accuracy metrics.

    Attributes:
        metrics (SemanticMetrics): Metrics calculator for semantic segmentation.

    Examples:
        >>> from ultralytics.models.yolo.semantic import SemanticSegmentationValidator
        >>> args = dict(model="yolo26n-sem.pt", data="cityscapes8.yaml")
        >>> validator = SemanticSegmentationValidator(args=args)
        >>> validator()
    """

    def __init__(self, dataloader=None, save_dir=None, args=None, _callbacks=None):
        """Initialize SemanticSegmentationValidator.

        Args:
            dataloader (DataLoader, optional): DataLoader for validation.
            save_dir (Path, optional): Directory to save results.
            args (dict, optional): Arguments for the validator.
            _callbacks (dict, optional): Callback functions.
        """
        super().__init__(dataloader, save_dir, args, _callbacks)
        self.args.task = "semantic"
        self.dataset = None
        self.results_dir = None
        self.metrics = SemanticMetrics()
        self.image_shapes = {}
        self._semantic_target_shape = None

    def init_metrics(self, model):
        """Initialize metrics with model class names.

        Args:
            model (nn.Module): Model to validate.
        """
        self.names = model.names
        self.nc = len(self.names)
        self.metrics = SemanticMetrics(names=self.names)
        self.seen = 0
        self.dataset = getattr(self.dataloader, "dataset", None)
        labels = getattr(self.dataset, "labels", []) if self.dataset is not None else []
        self.image_shapes = {lb["im_file"]: tuple(lb["shape"]) for lb in labels if "im_file" in lb and "shape" in lb}
        self.results_dir = None
        if self.args.save_json:
            self.results_dir = self.save_dir / "results"
            self.results_dir.mkdir(parents=True, exist_ok=True)
        cm_nc = self.metrics.cm_nc
        if cm_nc == 2 and len(self.names) == 1:  # binary segmentation, expand to include background
            cm_names = {0: "background", 1: next(iter(self.names.values()))}
        else:
            base = list(self.names.values()) + [str(i) for i in range(len(self.names), cm_nc)]
            cm_names = {i: base[i] for i in range(cm_nc)}
        self.confusion_matrix = ConfusionMatrix(names=cm_names, task="semantic")

    def preprocess(self, batch):
        """Preprocess a batch of images and masks.

        Args:
            batch (dict): Batch data containing images and masks.

        Returns:
            (dict): Preprocessed batch.
        """
        batch = super().preprocess(batch)
        batch["semantic_mask"] = batch["semantic_mask"].to(self.device, dtype=torch.int32)
        self._semantic_target_shape = tuple(batch["semantic_mask"].shape[-2:])
        return batch

    def postprocess(self, preds):
        """Convert logits or baked class maps to class predictions.

        Args:
            preds (torch.Tensor): Raw model output logits [B, nc, H, W] or baked class map [B, H, W].

        Returns:
            (torch.Tensor): Predicted class IDs [B, H, W].
        """
        if isinstance(preds, (tuple, list)):
            preds = preds[0]
        if preds.ndim == 3:
            # [B, H, W] class map with argmax already baked into the graph. Nearest-resize only.
            if tuple(preds.shape[-2:]) != self._semantic_target_shape:
                preds = F.interpolate(preds[:, None].float(), size=self._semantic_target_shape, mode="nearest")[:, 0]
            return preds.to(torch.int32)
        pred_hw = preds.shape[2:]
        if pred_hw[0] != self._semantic_target_shape[0] or pred_hw[1] != self._semantic_target_shape[1]:
            preds = F.interpolate(preds, size=self._semantic_target_shape, mode="bilinear", align_corners=False)
        return preds.argmax(dim=1).to(torch.int32) if self.nc > 1 else preds.gt(0).squeeze(1).to(torch.int32)

    def update_metrics(self, preds, batch):
        """Update metrics with predictions and ground truth.

        Args:
            preds (torch.Tensor): Predicted class IDs [B, H, W].
            batch (dict): Batch containing 'semantic_mask'.
        """
        if self.args.save_json:
            self.save_pred_masks(preds, batch)
        self.metrics.update_stats(preds, batch["semantic_mask"])
        self.seen += preds.shape[0]

    def gather_stats(self):
        """Reduce semantic confusion matrix to rank 0 during DDP validation."""
        if RANK == -1 or not dist.is_available() or not dist.is_initialized():
            return
        if self.metrics.matrix is None:
            cm_nc = self.metrics.cm_nc
            self.metrics.matrix = torch.zeros((cm_nc, cm_nc), device=self.device, dtype=torch.float32)
        dist.reduce(self.metrics.matrix, dst=0, op=dist.ReduceOp.SUM)
        # Gather nt_per_image across ranks
        if RANK == 0:
            gathered_nt = [None] * dist.get_world_size()
            dist.gather_object(self.metrics.nt_per_image, gathered_nt, dst=0)
            self.metrics.nt_per_image = np.sum(gathered_nt, axis=0)
        elif RANK > 0:
            dist.gather_object(self.metrics.nt_per_image, None, dst=0)

    def save_pred_masks(self, preds: torch.Tensor, batch: dict[str, Any]) -> None:
        """Save semantic predictions as single-channel PNG masks."""
        if self.results_dir is None:
            return
        im_files = batch.get("im_file", [])
        if not im_files:
            return
        preds = preds.cpu().numpy()
        if isinstance(self.dataset, SemanticDataset) and self.dataset.label_mapping:
            preds = self.dataset.convert_label(preds, inverse=True)
        preds = preds.astype(np.uint8, copy=False)
        for pred, im_file in zip(preds, im_files):
            orig_shape = self.image_shapes.get(im_file)
            if orig_shape and pred.shape != orig_shape:
                pred = cv2.resize(pred, (orig_shape[1], orig_shape[0]), interpolation=cv2.INTER_NEAREST)
            save_path = self.results_dir / Path(im_file).with_suffix(".png").name
            Image.fromarray(pred).save(save_path)

    def get_stats(self):
        """Return validation statistics.

        Returns:
            (dict): Dictionary of validation metrics.
        """
        self.metrics.process(save_dir=self.save_dir, plot=self.args.plots, on_plot=self.on_plot)
        if self.metrics.matrix is not None:
            # Internal layout is [gt, pred]; transpose to [pred, gt] for ConfusionMatrix export format.
            self.confusion_matrix.matrix = self.metrics.matrix.detach().cpu().numpy().T.astype(float)
        return self.metrics.results_dict

    def get_desc(self):
        """Return a formatted description of evaluation metrics.

        Returns:
            (str): Formatted string with metric names.
        """
        return ("%22s" + "%11s" * 4) % ("Class", "Images", "Pixels", "mIoU", "PixAcc")

    def print_results(self) -> None:
        """Print training/validation set metrics per class."""
        super().print_results()
        if self.args.save_json and self.results_dir is not None:
            LOGGER.info(f"Semantic prediction masks saved to {self.results_dir}")

    def get_dataset(self):
        """Parse the dataset YAML and add background metadata for polygon labels when required."""
        return add_polygon_background(super().get_dataset())

    def plot_predictions(self, batch, preds, ni):
        """Plot predicted semantic masks on input images."""
        plot_images(
            images=batch["img"],
            labels={"semantic_mask": preds},
            paths=batch["im_file"],
            fname=self.save_dir / f"val_batch{ni}_pred.jpg",
            names=self.names,
            on_plot=self.on_plot,
        )
