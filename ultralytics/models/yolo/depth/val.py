# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""Depth estimation validator for YOLO models."""

from __future__ import annotations

import cv2
import numpy as np
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
        # Scale-only calibration: when enabled (by Model.calibrate), collect (log pred, log gt)
        # pairs during the val pass and fit the global log-affine in get_stats().
        self.calibrating = False
        self.calib = None
        self._cal_logp, self._cal_logg, self._cal_pts = [], [], 0

    def init_metrics(self, model):
        """Initialize the DepthMetrics accumulator and reset per-pass calibration state."""
        self.metrics = DepthMetrics()
        self.metrics.clear_stats()
        # Reset calibration accumulators for this val pass. These also live here (not only in
        # __init__) so the validator works when constructed via __new__ (e.g. unit tests). The
        # calibrating flag is set externally by Model.calibrate() before the pass, so preserve it.
        self.calibrating = getattr(self, "calibrating", False)
        self.calib = None
        self._cal_logp, self._cal_logg, self._cal_pts = [], [], 0

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

        if self.calibrating and self._cal_pts < 500_000:
            valid = (gt_depth > 1e-3) & (pred_depth > 1e-3) & torch.isfinite(pred_depth)
            if valid.any():
                lp = torch.log(pred_depth[valid]).flatten().cpu().numpy()
                lg = torch.log(gt_depth[valid]).flatten().cpu().numpy()
                if lp.size > 50_000:  # subsample per batch — calibration is only a 2-parameter fit
                    idx = np.random.default_rng(self._cal_pts).choice(lp.size, 50_000, replace=False)
                    lp, lg = lp[idx], lg[idx]
                self._cal_logp.append(lp)
                self._cal_logg.append(lg)
                self._cal_pts += lp.size

    def get_stats(self):
        """Reduce across ranks, finalize, and return the metrics dict.

        If calibration was enabled, fit the global log-affine ``(a, b)`` from the collected pairs.
        """
        self.metrics.reduce_ddp()
        self.metrics.process()
        if self.calibrating and self._cal_logp:
            from .calibrate import lstsq_affine

            import os

            self.calib = lstsq_affine(
                np.concatenate(self._cal_logp), np.concatenate(self._cal_logg),
                dist_power=float(os.environ.get("DEPTH_CAL_DIST_POWER", 0.0)),
            )
            LOGGER.info(f"Depth calibration fit on {self._cal_pts} pixels: a={self.calib[0]:.4f} b={self.calib[1]:.4f}")
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

    @staticmethod
    def _colorize_depth(depth, vmin: float, vmax: float):
        """Map a (H,W) metric-depth map to a BGR uint8 image (INFERNO), invalid pixels black."""
        d = depth.detach().float().cpu().numpy() if isinstance(depth, torch.Tensor) else np.asarray(depth, np.float32)
        valid = d > 0
        if vmax <= vmin:
            vmax = vmin + 1e-6
        dn = np.clip((d - vmin) / (vmax - vmin), 0.0, 1.0)
        color = cv2.applyColorMap((dn * 255).astype(np.uint8), cv2.COLORMAP_INFERNO)  # BGR
        color[~valid] = 0
        return color

    def plot_predictions(self, batch, preds, ni, max_images: int = 4):
        """Save a RGB | GT depth | predicted depth panel for the batch to val_batch{ni}.jpg.

        Depth has no boxes/classes, so the detection-style plotters are replaced with a
        side-by-side depth visualization. GT and prediction share GT's valid depth range
        so the colors are directly comparable. Called by BaseValidator for the first few
        batches when args.plots is set.
        """
        if "depth" not in batch:
            return
        try:
            imgs = batch["img"]
            gt = batch["depth"]
            pred = self._extract_pred(preds)
            if gt.ndim == 3:
                gt = gt.unsqueeze(1)
            if pred.ndim == 3:
                pred = pred.unsqueeze(1)
            h, w = imgs.shape[-2:]
            rows = []
            for i in range(min(imgs.shape[0], max_images)):
                rgb = (imgs[i].detach().float().cpu().clamp(0, 1).numpy() * 255).astype(np.uint8).transpose(1, 2, 0)
                rgb = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
                g, p = gt[i, 0], pred[i, 0]
                gv = g[g > 0]
                vmin = float(gv.min()) if gv.numel() else 0.0
                vmax = float(gv.max()) if gv.numel() else 1.0
                gc = cv2.resize(self._colorize_depth(g, vmin, vmax), (w, h), interpolation=cv2.INTER_NEAREST)
                pc = cv2.resize(self._colorize_depth(p, vmin, vmax), (w, h), interpolation=cv2.INTER_NEAREST)
                rows.append(np.hstack([rgb, gc, pc]))
            cv2.imwrite(str(self.save_dir / f"val_batch{ni}.jpg"), np.vstack(rows))
        except Exception as e:
            LOGGER.warning(f"DepthValidator: failed to plot val_batch{ni}: {e}")

    def plot_val_samples(self, batch, ni):
        """No-op: GT depth is shown alongside predictions in plot_predictions()."""
        pass
