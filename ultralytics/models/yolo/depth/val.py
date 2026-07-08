# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""Depth estimation validator for YOLO models."""

from __future__ import annotations

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from ultralytics.models.yolo.detect import DetectionValidator
from ultralytics.utils import LOGGER
from ultralytics.utils.metrics import DepthMetrics
from ultralytics.utils.plotting import colorize_depth


class DepthValidator(DetectionValidator):
    """Validator for YOLO depth estimation models.

    Computes standard depth metrics: delta1, abs_rel, rmse, silog. Uses validation loss as the primary training signal.
    """

    def __init__(self, dataloader=None, save_dir=None, args=None, _callbacks=None):
        """Initialize DepthValidator."""
        super().__init__(dataloader, save_dir, args, _callbacks)
        self.args.task = "depth"
        # Scale-only calibration: when enabled (by Model.calibrate), collect (log pred, log gt)
        # pairs during the val pass and fit the global log-affine in get_stats().
        self.calibrating = False

    def init_metrics(self, model):
        """Initialize the DepthMetrics accumulator and reset per-pass calibration state.

        The calibrating flag is not reset here: Model.calibrate() sets it after construction,
        before the val pass runs.
        """
        self.metrics = DepthMetrics()
        self.metrics.clear_stats()
        self.calib = None
        self._cal_logp, self._cal_logg, self._cal_pts = [], [], 0
        # Baked calibration of the model under validation, for the standalone-val comparison plot
        # (the head applies cal_a/cal_b in its forward, so predictions arrive already calibrated).
        from .calibrate import _depth_head

        head = _depth_head(model)
        self._cal_ab = (float(head.cal_a), float(head.cal_b)) if head is not None else None

    def preprocess(self, batch):
        """Preprocess batch — move to device, normalize images, handle precision."""
        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                batch[k] = v.to(self.device, non_blocking=self.device.type == "cuda")
        # Normalize images to [0,1] (DepthFormat outputs uint8, same as detection pipeline)
        batch["img"] = (batch["img"].half() if self.args.quantize == 16 else batch["img"].float()) / 255
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
            pred_depth = F.interpolate(
                pred_depth.float(), size=gt_depth.shape[-2:], mode="bilinear", align_corners=True
            )
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
        """Finalize and return the metrics dict.

        No cross-rank reduction happens here: DDP training validates on rank 0 alone, so a
        collective would deadlock waiting for ranks that never enter validation.

        If calibration was enabled, fit the global log-affine ``(a, b)`` from the collected pairs.
        """
        self.metrics.process()
        if self.calibrating and self._cal_logp:
            from .calibrate import lstsq_affine

            self.calib = lstsq_affine(
                np.concatenate(self._cal_logp),
                np.concatenate(self._cal_logg),
                dist_power=self.args.cal_dist_pw,
            )
            LOGGER.info(f"Depth calibration fit on {self._cal_pts} pixels: a={self.calib[0]:.4f} b={self.calib[1]:.4f}")
        return self.metrics.results_dict

    def gather_stats(self) -> None:
        """No-op DDP gather for depth validation.

        Validation runs on rank 0 only over the full val set, so the accumulators are already
        complete. DetectionValidator.gather_stats() accesses self.metrics.stats and self.metrics.box
        which do not exist on DepthMetrics, so we override here to prevent AttributeError.
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

    def plot_predictions(self, batch, preds, ni, max_images: int = 4):
        """Save a RGB | GT depth | predicted depth panel for the batch to val_batch{ni}.jpg.

        Depth has no boxes/classes, so the detection-style plotters are replaced with a
        side-by-side depth visualization (see plot_depth_panels). Called by BaseValidator
        for the first few batches when args.plots is set.

        Standalone val (``yolo val``) additionally writes ``val_batch{ni}_calibrated.jpg``
        comparing raw vs the checkpoint's baked calibration. The head already applies
        ``cal_a``/``cal_b`` in its forward, so the prediction here IS the calibrated output;
        raw is recovered by inverting the log-affine. Training-epoch validation skips this
        (buffers are identity until final_eval fits them, so the comparison says nothing).
        """
        if "depth" not in batch:
            return
        try:
            pred = self._extract_pred(preds)
            plot_depth_panels(
                batch["img"],
                batch["depth"],
                [pred],
                self.save_dir / f"val_batch{ni}.jpg",
                max_images=max_images,
            )
            cal = getattr(self, "_cal_ab", None)
            if cal is not None and not getattr(self, "training", True):
                a, b = cal
                raw = torch.exp((torch.log(pred.float().clamp(min=1e-3)) - b) / a)
                name = "identity" if (a, b) == (1.0, 0.0) else "baked"
                plot_depth_panels(
                    batch["img"],
                    batch["depth"],
                    [raw, pred],
                    self.save_dir / f"val_batch{ni}_calibrated.jpg",
                    titles=["RGB", "GT", "raw", f"calibrated ({name} x{np.exp(b):.2f})"],
                    max_images=max_images,
                )
        except Exception as e:
            LOGGER.warning(f"DepthValidator: failed to plot val_batch{ni}: {e}")

    def plot_val_samples(self, batch, ni):
        """No-op: GT depth is shown alongside predictions in plot_predictions()."""
        pass


def plot_depth_panels(imgs, gt, preds, fname, titles=None, max_images: int = 4):
    """Write a depth panel grid: one row per image, columns RGB | GT | one per entry of ``preds``.

    All depth columns share the GT valid-pixel range per row, so a scale error between GT and any prediction shows up
    directly as a color mismatch. Panels are resized to the RGB image size, so predictions at head stride need no prior
    interpolation.

    Args:
        imgs (torch.Tensor): (B,3,H,W) float image tensor in [0,1].
        gt (torch.Tensor): (B,1,H,W) or (B,H,W) ground-truth depth in meters (pixels <= 0 invalid, drawn black).
        preds (list): List of (B,1,H,W) or (B,H,W) predicted depth tensors; each adds one column.
        fname (str | Path): Output image path.
        titles (list, optional): List of ``2 + len(preds)`` column labels, drawn in a 24 px header strip. None (the
            val_batch{ni}.jpg default) keeps the historical strip-free layout.
        max_images: Maximum number of rows.
    """
    if gt.ndim == 3:
        gt = gt.unsqueeze(1)
    preds = [p.unsqueeze(1) if p.ndim == 3 else p for p in preds]
    h, w = imgs.shape[-2:]
    rows = []
    for i in range(min(imgs.shape[0], max_images)):
        rgb = (imgs[i].detach().float().cpu().clamp(0, 1).numpy() * 255).astype(np.uint8).transpose(1, 2, 0)
        panels = [cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)]
        g = gt[i, 0]
        gv = g[g > 0]
        vmin = float(gv.min()) if gv.numel() else 0.0
        vmax = float(gv.max()) if gv.numel() else 1.0
        for d in [g] + [p[i, 0] for p in preds]:
            d = d.detach().float().cpu().numpy() if isinstance(d, torch.Tensor) else np.asarray(d, np.float32)
            panels.append(cv2.resize(colorize_depth(d, vmin, vmax), (w, h), interpolation=cv2.INTER_NEAREST))
        rows.append(np.hstack(panels))
    grid = np.vstack(rows)
    if titles:
        strip = np.full((24, grid.shape[1], 3), 255, dtype=np.uint8)
        for j, t in enumerate(titles):
            cv2.putText(strip, str(t), (j * w + 4, 17), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
        grid = np.vstack([strip, grid])
    cv2.imwrite(str(fname), grid)
