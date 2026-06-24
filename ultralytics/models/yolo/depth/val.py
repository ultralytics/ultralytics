# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""Depth estimation validator for YOLO models."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from ultralytics.models.yolo.detect import DetectionValidator
from ultralytics.utils import LOGGER, RANK
from ultralytics.utils.metrics import DepthMetrics


class DepthValidator(DetectionValidator):
    """Validator for YOLO depth estimation models.

    Computes standard depth metrics: delta1, abs_rel, rmse, silog.
    Uses validation loss as the primary training signal.
    """

    def __init__(self, dataloader=None, save_dir=None, args=None, _callbacks=None):
        """Initialize DepthValidator."""
        super().__init__(dataloader, save_dir, args, _callbacks)
        self.args.task = "depth"

    def init_metrics(self, model):
        """Initialize the DepthMetrics accumulator."""
        self.metrics = DepthMetrics()
        self.metrics.clear_stats()

    def preprocess(self, batch):
        """Preprocess batch — move to device, normalize images, handle precision."""
        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                batch[k] = v.to(self.device, non_blocking=self.device.type == "cuda")
        # Normalize images to [0,1] (DepthFormat outputs uint8, same as detection pipeline)
        batch["img"] = (batch["img"].half() if self.args.half else batch["img"].float()) / 255
        if "depth" in batch:
            batch["depth"] = batch["depth"].float()  # depth always float32
        return batch

    def postprocess(self, preds):
        """No NMS needed for depth — return predictions as-is."""
        return preds

    def _extract_pred(self, preds):
        """Return the (B,1,H,W) predicted depth tensor from any model output container."""
        if isinstance(preds, dict):
            return preds.get("depth", preds.get("proto"))
        if isinstance(preds, (tuple, list)):
            return preds[0] if isinstance(preds[0], torch.Tensor) else preds
        return preds

    def update_metrics(self, preds, batch):
        """Accumulate depth metrics for a batch."""
        if "depth" not in batch:
            return

        pred_depth = self._extract_pred(preds)
        gt_depth = batch["depth"]
        if gt_depth.ndim == 3:
            gt_depth = gt_depth.unsqueeze(1)
        if pred_depth.ndim == 3:
            pred_depth = pred_depth.unsqueeze(1)
        if pred_depth.shape[-2:] != gt_depth.shape[-2:]:
            pred_depth = F.interpolate(pred_depth.float(), size=gt_depth.shape[-2:], mode="bilinear", align_corners=True)
        self.metrics.update_stats(pred_depth, gt_depth)

    def get_stats(self):
        """Reduce across ranks, finalize, and return the metrics dict."""
        self.metrics.reduce_ddp()
        self.metrics.process()
        return self.metrics.results_dict

    def gather_stats(self) -> None:
        """No-op DDP gather for depth validation.

        Depth metrics are reduced across ranks in DepthMetrics.reduce_ddp(); no per-stat gather needed.
        DetectionValidator.gather_stats() accesses self.metrics.stats and self.metrics.box which do not
        exist on DepthMetrics, so we override here to prevent AttributeError on multi-GPU val runs.
        """
        pass

    def print_results(self):
        """Log the headline depth metrics in the detection-style aligned table format.

        Columns line up with get_desc(): Class, Images, delta1, abs_rel, rmse, silog.
        Uses "depth_val" as the row label (depth has no classes, where detection prints "all").
        """
        r = self.metrics.results_dict
        n_images = len(self.dataloader.dataset) if self.dataloader is not None else (self.seen or 0)
        pf = "%22s" + "%11i" + "%11.4g" * 4  # label, Images, delta1, abs_rel, rmse, silog
        LOGGER.info(
            pf
            % (
                "depth_val",
                n_images,
                r.get("metrics/delta1", 0.0),
                r.get("metrics/abs_rel", 0.0),
                r.get("metrics/rmse", 0.0),
                r.get("metrics/silog", 0.0),
            )
        )

    def finalize_metrics(self, *args, **kwargs):
        """No-op; metrics finalized in get_stats()."""
        pass

    def get_desc(self):
        """Return description for progress bar."""
        return f"{'Class':>22}{'Images':>11}{'delta1':>11}{'abs_rel':>11}{'rmse':>11}{'silog':>11}"

    def plot_predictions(self, batch, preds, ni):
        """Skip detection-style prediction plotting for depth."""
        pass

    def plot_val_samples(self, batch, ni):
        """Skip detection-style sample plotting for depth."""
        pass
