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

from ultralytics.data.build import build_dataloader
from ultralytics.data.dataset import PolygonSemsegDataset, SemsegDataset, add_polygon_background
from ultralytics.models.yolo.detect import DetectionValidator
from ultralytics.utils import LOGGER, RANK
from ultralytics.utils.metrics import SemsegMetrics
from ultralytics.utils.plotting import Annotator, plot_images, plt_settings


class SemanticSegmentationValidator(DetectionValidator):
    """Validator for semantic segmentation models.

    This validator evaluates semantic segmentation models using mIoU and pixel accuracy metrics.

    Attributes:
        metrics (SemsegMetrics): Metrics calculator for semantic segmentation.

    Examples:
        >>> from ultralytics.models.yolo.semseg import SemanticSegmentationValidator
        >>> args = dict(model="yolo26n-semseg.pt", data="cityscapes8.yaml")
        >>> validator = SemanticSegmentationValidator(args=args)
        >>> validator()
    """

    def __init__(self, dataloader=None, save_dir=None, args=None, _callbacks=None):
        """Initialize SemanticSegmentationValidator.

        Args:
            dataloader (DataLoader, optional): DataLoader for validation.
            save_dir (Path, optional): Directory to save results.
            args (dict, optional): Arguments for the validator.
            _callbacks (list, optional): Callback functions.
        """
        super().__init__(dataloader, save_dir, args, _callbacks)
        self.args.task = "semseg"
        self.dataset = None
        self.results_dir = None
        self.metrics = SemsegMetrics()
        self.image_shapes = {}
        self._semantic_target_shape = None

    def init_metrics(self, model):
        """Initialize metrics with model class names.

        Args:
            model (nn.Module): Model to validate.
        """
        self.names = model.names
        self.nc = len(self.names)
        self.metrics = SemsegMetrics(names=self.names)
        self.seen = 0
        self.dataset = getattr(self.dataloader, "dataset", None)
        labels = getattr(self.dataset, "labels", []) if self.dataset is not None else []
        self.image_shapes = {lb["im_file"]: tuple(lb["shape"]) for lb in labels if "im_file" in lb and "shape" in lb}
        self.results_dir = None
        if self.args.save_json:
            self.results_dir = self.save_dir / "results"
            self.results_dir.mkdir(parents=True, exist_ok=True)

    def preprocess(self, batch):
        """Preprocess a batch of images and masks.

        Args:
            batch (dict): Batch data containing images and masks.

        Returns:
            (dict): Preprocessed batch.
        """
        batch = super().preprocess(batch)
        batch["semantic_mask"] = batch["semantic_mask"].to(torch.int32)
        self._semantic_target_shape = tuple(batch["semantic_mask"].shape[-2:])
        return batch

    def postprocess(self, preds):
        """Convert logits to class predictions.

        Args:
            preds (torch.Tensor): Raw model output logits [B, nc, H, W].

        Returns:
            (torch.Tensor): Predicted class IDs [B, H, W].
        """
        if isinstance(preds, (tuple, list)):
            preds = preds[0]
        preds = (
            F.interpolate(preds, size=self._semantic_target_shape, mode="bilinear", align_corners=False)
            if self._semantic_target_shape is not None
            else F.interpolate(preds, scale_factor=8, mode="bilinear", align_corners=False)
        )
        if self.nc > 1:
            pred_mask = preds.argmax(dim=1).to(torch.int32)
        else:
            pred_mask = preds.gt(0).squeeze(1).to(torch.int32)
        return pred_mask

    def update_metrics(self, preds, batch):
        """Update metrics with predictions and ground truth.

        Args:
            preds (torch.Tensor): Predicted class IDs [B, H, W].
            batch (dict): Batch containing 'semantic_mask'.
        """
        targets = batch["semantic_mask"]
        if preds.shape[1:] != targets.shape[1:]:
            preds = (
                F.interpolate(preds.float().unsqueeze(1), targets.shape[1:], mode="nearest").squeeze(1).to(torch.int32)
            )
        if self.args.save_json:
            self.save_pred_masks(preds, batch)
        self.metrics.update_stats(preds, targets)
        self.seen += preds.shape[0]

    def finalize_metrics(self):
        """Set final values on semantic metrics."""
        self.metrics.speed = self.speed
        self.metrics.save_dir = self.save_dir

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
        if isinstance(self.dataset, SemsegDataset) and self.dataset.label_mapping:
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
        self.metrics.clear_stats()
        return self.metrics.results_dict

    def get_desc(self):
        """Return a formatted description of evaluation metrics.

        Returns:
            (str): Formatted string with metric names.
        """
        return ("%22s" + "%11s" * 4) % ("Class", "Images", "Pixels", "mIoU", "PixAcc")

    def print_results(self):
        """Print validation results including per-class IoU."""
        pf = "%22s" + "%11i" * 2 + "%11.3g" * len(self.metrics.keys)
        LOGGER.info(pf % ("all", self.seen, self.metrics.nt_per_class.sum(), *self.metrics.mean_results()))
        if self.metrics.nt_per_class.sum() == 0:
            LOGGER.warning(f"no labels found in {self.args.task} set, cannot compute metrics without labels")
        if self.args.verbose and not self.training and self.nc > 1:
            for i, c in enumerate(self.metrics.ap_class_index):
                LOGGER.info(
                    pf
                    % (
                        self.names[c],
                        self.metrics.nt_per_image[c],
                        self.metrics.nt_per_class[c],
                        *self.metrics.class_result(i),
                    )
                )
        if self.args.save_json and self.results_dir is not None:
            LOGGER.info(f"Semantic prediction masks saved to {self.results_dir}")

    def build_dataset(self, img_path, mode="val", batch=None):
        """Build semantic segmentation dataset.

        Routes to `PolygonSemsegDataset` when the dataset YAML lacks 'masks_dir' and to
        `SemsegDataset` otherwise. The helper call is idempotent: in trainer-driven validation
        the data dict has already been bumped with the background class.

        Args:
            img_path (str): Path to images.
            mode (str): Dataset mode.
            batch (int, optional): Batch size.

        Returns:
            (SemsegDataset): Dataset object.
        """
        self.data = add_polygon_background(self.data)
        use_rect = mode == "val"
        dataset_cls = SemsegDataset if self.data.get("masks_dir") else PolygonSemsegDataset
        return dataset_cls(
            img_path=img_path,
            imgsz=self.args.imgsz,
            augment=False,
            hyp=self.args,
            cache=self.args.cache or None,
            data=self.data,
            rect=use_rect,
            batch_size=batch,
            stride=self.stride,
            pad=0,
            prefix=f"{mode}: ",
        )

    @plt_settings()
    def plot_val_samples(self, batch, ni):
        """Plot validation image samples with semantic mask overlays.

        Args:
            batch (dict): Batch containing images and semantic masks.
            ni (int): Batch index.
        """
        images = batch["img"]
        masks = batch["semantic_mask"]
        bs = min(len(images), 16)
        images = images[:bs]
        masks = masks[:bs]
        images_np = images.cpu().float().numpy()
        if images_np.max() <= 1:
            images_np *= 255
        masks_np = masks.cpu().numpy()
        overlaid = []
        for i in range(bs):
            img = images_np[i].transpose(1, 2, 0).astype(np.uint8)
            if img.shape[2] == 3:
                img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            mask = masks_np[i]
            annotator = Annotator(img, line_width=1)
            annotator.semantic_mask(mask, alpha=0.4)
            img = annotator.result()
            if img.shape[2] == 3:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            overlaid.append(img)
        overlaid = np.stack(overlaid).transpose(0, 3, 1, 2)
        plot_images(
            labels={"img": overlaid, "cls": np.zeros(0)},
            paths=batch.get("im_file", []),
            fname=self.save_dir / f"val_batch{ni}_labels.jpg",
            names=self.names,
            on_plot=self.on_plot,
        )

    @plt_settings()
    def plot_predictions(self, batch, preds, ni):
        """Plot predicted semantic masks on input images.

        Args:
            batch (dict): Batch containing images.
            preds (torch.Tensor): Predicted class IDs [B, H, W].
            ni (int): Batch index.
        """
        images = batch["img"]
        bs = min(len(images), 16)
        images = images[:bs]
        preds = preds[:bs]
        images_np = images.cpu().float().numpy()
        if images_np.max() <= 1:
            images_np *= 255
        preds_np = preds.cpu().numpy() if isinstance(preds, torch.Tensor) else preds
        overlaid = []
        for i in range(bs):
            img = images_np[i].transpose(1, 2, 0).astype(np.uint8)
            if img.shape[2] == 3:
                img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            mask = preds_np[i]
            annotator = Annotator(img, line_width=1)
            annotator.semantic_mask(mask, alpha=0.4)
            img = annotator.result()
            if img.shape[2] == 3:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            overlaid.append(img)
        overlaid = np.stack(overlaid).transpose(0, 3, 1, 2)
        plot_images(
            labels={"img": overlaid, "cls": np.zeros(0)},
            paths=batch.get("im_file", []),
            fname=self.save_dir / f"val_batch{ni}_pred.jpg",
            names=self.names,
            on_plot=self.on_plot,
        )
