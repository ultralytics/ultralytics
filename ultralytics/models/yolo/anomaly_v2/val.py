# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

"""YOLO Anomaly v2 validator — 2-pass (mask-on + mask-off) with AUROC.

Pass 1 (mask-on):  forward with GT bbox rendered mask.
Pass 2 (mask-off): forward with mask disabled (passthrough ≈ vanilla YOLO).

Metrics from pass 2 are prefixed ``mask_off/``. The canonical loss reported to
the trainer comes from pass 1.

In addition to standard detection metrics, accumulates image-AUROC and
pixel-AUROC from the model's stashed heatmap (``model._last_heatmap``) during
pass 1 only.

When ``prior_mode`` is set (standalone val), runs single-pass with that mode
and still computes AUROC.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from ultralytics.models.yolo.detect import DetectionValidator
from ultralytics.utils import LOGGER

from ._util import resolve_v2_model


class AnomalyV2Validator(DetectionValidator):
    """Detection validator: 2-pass during training (mask-on / mask-off), single-pass for prior modes."""

    def __init__(self, dataloader=None, save_dir=None, args=None, _callbacks=None,
                 prior_mode: str | None = None) -> None:
        if isinstance(args, dict):
            prior_mode = args.pop("prior_mode", prior_mode)
        super().__init__(dataloader, save_dir, args, _callbacks)
        self.args.task = "anomaly_v2"
        self.prior_mode = prior_mode
        self._mask_mode = "on"
        self._model_ref = None

        # AUROC accumulators
        self._auroc_image_scores: list[float] = []
        self._auroc_image_labels: list[int] = []
        self._auroc_pixel_scores: list[float] = []
        self._auroc_pixel_labels: list[int] = []

        from ultralytics.nn.modules.anomaly_v2 import BboxMaskRenderer
        self._eval_mask_renderer = BboxMaskRenderer(mask_size=256, mode="rect")

    # ------------------------------------------------------------------
    # Hooks
    # ------------------------------------------------------------------
    def init_metrics(self, model) -> None:
        super().init_metrics(model)
        self._model_ref = model
        m = resolve_v2_model(model)
        if m is not None and hasattr(m, "set_prior_mode") and self.prior_mode is not None:
            m.set_prior_mode(self.prior_mode)

    def preprocess(self, batch):
        batch = super().preprocess(batch)
        model = resolve_v2_model(self._model_ref)
        if model is None or not hasattr(model, "set_mask_input"):
            return batch
        if self.prior_mode is not None:
            # Single-pass prior_mode: "mask" mode injects GT bboxes
            if self.prior_mode == "mask":
                bb = batch.get("bboxes")
                bi = batch.get("batch_idx")
                if bb is not None and bi is not None:
                    model.set_mask_input(bb, bi)
        elif self._mask_mode == "on":
            bb = batch.get("bboxes")
            bi = batch.get("batch_idx")
            if bb is not None and bi is not None:
                model.set_mask_input(bb, bi)
            else:
                model.disable_mask_once()
        else:
            model.disable_mask_once()
        return batch

    # ------------------------------------------------------------------
    # 2-pass __call__ (training val) or single-pass (prior_mode)
    # ------------------------------------------------------------------
    def __call__(self, trainer=None, model=None):
        if self.prior_mode is not None:
            return self._single_pass(trainer, model)
        return self._two_pass(trainer, model)

    def _single_pass(self, trainer, model):
        self._reset_auroc()
        if self.prior_mode == "heatmap":
            self.args.rect = False
        return super().__call__(trainer=trainer, model=model)

    def _two_pass(self, trainer, model):
        # Pass 1: mask-on
        self._mask_mode = "on"
        self._reset_auroc()
        stats_on = super().__call__(trainer=trainer, model=model)

        # Snapshot mask-on metrics before pass 2 overwrites them
        metrics_on = self.metrics
        image_auroc = self._compute_auroc(self._auroc_image_scores, self._auroc_image_labels)
        pixel_auroc = self._compute_auroc(self._auroc_pixel_scores, self._auroc_pixel_labels)

        # Pass 2: mask-off
        self._mask_mode = "off"
        try:
            stats_off = super().__call__(trainer=trainer, model=model)
        except Exception as e:
            LOGGER.warning(f"AnomalyV2Validator: mask-off pass failed: {e}")
            stats_off = {}

        self._mask_mode = "on"

        # Restore mask-on metrics so results_dict shows mask-on by default
        self.metrics = metrics_on

        if not isinstance(stats_on, dict):
            return stats_on
        merged = dict(stats_on)
        merged["image_auroc"] = image_auroc
        merged["pixel_auroc"] = pixel_auroc
        if isinstance(stats_off, dict):
            for k, v in stats_off.items():
                merged[f"mask_off/{k}"] = v
            # Stash mask-off stats for external access
            self._mask_off_stats = dict(stats_off)
        return merged

    def plot_val_samples(self, batch, ni):
        """Slice the cached-prior channel off before plotting (plot_images expects 3-channel)."""
        if batch["img"].shape[1] == 4:
            batch = {**batch, "img": batch["img"][:, :3]}
        super().plot_val_samples(batch, ni)

    def plot_predictions(self, batch, preds, ni):
        """Slice the cached-prior channel off before plotting (plot_images expects 3-channel)."""
        if batch["img"].shape[1] == 4:
            batch = {**batch, "img": batch["img"][:, :3]}
        super().plot_predictions(batch, preds, ni)

    # ------------------------------------------------------------------
    # AUROC
    # ------------------------------------------------------------------
    def _reset_auroc(self):
        self._auroc_image_scores = []
        self._auroc_image_labels = []
        self._auroc_pixel_scores = []
        self._auroc_pixel_labels = []

    def update_metrics(self, preds, batch):
        super().update_metrics(preds, batch)
        m = resolve_v2_model(self._model_ref)
        if m is None:
            return
        heatmap = getattr(m, "_last_heatmap", None)
        if heatmap is None or heatmap.numel() == 0:
            return

        bboxes = batch.get("bboxes")
        batch_idx = batch.get("batch_idx")
        bs = heatmap.shape[0]

        for b in range(bs):
            has_anom = 0
            if bboxes is not None and batch_idx is not None and batch_idx.numel() > 0:
                has_anom = int((batch_idx == b).any().item())
            img_score = heatmap[b].max().item()
            self._auroc_image_scores.append(img_score)
            self._auroc_image_labels.append(has_anom)

            if has_anom and bboxes is not None and batch_idx is not None:
                bb_per_img = bboxes[batch_idx == b]
                if bb_per_img.numel() > 0:
                    gt_mask = self._eval_mask_renderer(
                        bb_per_img,
                        torch.zeros(bb_per_img.shape[0], dtype=torch.long, device=bb_per_img.device),
                        1,
                    )
                    hmap_b = heatmap[b:b + 1]
                    if hmap_b.shape[2] != 256 or hmap_b.shape[3] != 256:
                        hmap_b = F.interpolate(hmap_b, size=(256, 256),
                                               mode="bilinear", align_corners=False)
                    self._auroc_pixel_scores.extend(hmap_b.flatten().cpu().tolist())
                    self._auroc_pixel_labels.extend(gt_mask.flatten().cpu().tolist())

    @staticmethod
    def _compute_auroc(scores: list[float], labels: list[int]) -> float:
        try:
            from sklearn.metrics import roc_auc_score
        except ImportError:
            return float("nan")
        if len(scores) == 0 or len(set(labels)) < 2:
            return float("nan")
        return float(roc_auc_score(labels, scores))

    def finalize_metrics(self):
        image_auroc = self._compute_auroc(self._auroc_image_scores, self._auroc_image_labels)
        pixel_auroc = self._compute_auroc(self._auroc_pixel_scores, self._auroc_pixel_labels)
        self.metrics.image_auroc = image_auroc
        self.metrics.pixel_auroc = pixel_auroc
        super().finalize_metrics()

    def get_stats(self):
        stats = super().get_stats()
        stats["image_auroc"] = self._compute_auroc(self._auroc_image_scores, self._auroc_image_labels)
        if any(self._auroc_pixel_labels):
            stats["pixel_auroc"] = self._compute_auroc(self._auroc_pixel_scores, self._auroc_pixel_labels)
        else:
            stats["pixel_auroc"] = float("nan")
        return stats
