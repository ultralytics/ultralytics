# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

"""YOLO Anomaly v2 validator — single-pass with configurable prior mode + AUROC.

Supports the same four prior modes as the predictor:

  - ``"none"``    — passthrough (vanilla YOLO).
  - ``"segment"`` — SegBranch sigmoid output as prior.
  - ``"heatmap"`` — BackboneMemoryBank output as prior.
  - ``"mask"``    — GT bboxes rendered as mask (upper-bound).

In addition to standard detection metrics, accumulates image-AUROC and
pixel-AUROC from the model's stashed heatmap (``model._last_heatmap``).
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from ultralytics.models.yolo.detect import DetectionValidator
from ultralytics.utils import LOGGER

from ._util import resolve_v2_model


class AnomalyV2Validator(DetectionValidator):
    """Detection validator that evaluates the v2 model with a configurable prior mode.

    Single-pass (replaces the old 2-pass mask-on/mask-off design).
    """

    def __init__(self, dataloader=None, save_dir=None, args=None, _callbacks=None,
                 prior_mode: str | None = None) -> None:
        # Pop prior_mode from args before super().__init__ validates all keys
        if isinstance(args, dict):
            prior_mode = args.pop("prior_mode", prior_mode)
        super().__init__(dataloader, save_dir, args, _callbacks)
        self.args.task = "anomaly_v2"
        self.prior_mode = prior_mode
        self._model_ref = None

        # AUROC accumulators (computed from model._last_heatmap)
        self._auroc_image_scores: list[float] = []
        self._auroc_image_labels: list[int] = []
        self._auroc_pixel_scores: list[float] = []
        self._auroc_pixel_labels: list[int] = []

        # Pixel-level GT mask renderer at a fixed eval resolution
        from ultralytics.nn.modules.anomaly_v2 import BboxMaskRenderer
        self._eval_mask_renderer = BboxMaskRenderer(mask_size=256, mode="rect")

    # ------------------------------------------------------------------
    # Hooks
    # ------------------------------------------------------------------
    def init_metrics(self, model) -> None:
        """Capture model reference and set prior mode for the validation pass."""
        super().init_metrics(model)
        self._model_ref = model
        m = resolve_v2_model(model)
        if m is not None and hasattr(m, "set_prior_mode"):
            m.set_prior_mode(self.prior_mode)

    def preprocess(self, batch):
        """Inject GT bboxes for 'mask' mode; other modes need no per-batch injection."""
        batch = super().preprocess(batch)
        if self.prior_mode == "mask":
            model = resolve_v2_model(self._model_ref)
            if model is not None and hasattr(model, "set_mask_input"):
                bb = batch.get("bboxes")
                bi = batch.get("batch_idx")
                if bb is not None and bi is not None:
                    model.set_mask_input(bb, bi)
        return batch

    # ------------------------------------------------------------------
    # Single-pass __call__
    # ------------------------------------------------------------------
    def __call__(self, trainer=None, model=None):
        self._auroc_image_scores = []
        self._auroc_image_labels = []
        self._auroc_pixel_scores = []
        self._auroc_pixel_labels = []
        # rect padding features look anomalous to the memory bank, drowning
        # out real content differences. Force rect=False for heatmap mode.
        if self.prior_mode == "heatmap":
            self.args.rect = False
        return super().__call__(trainer=trainer, model=model)

    # ------------------------------------------------------------------
    # AUROC accumulation
    # ------------------------------------------------------------------
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

    # ------------------------------------------------------------------
    # AUROC helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _compute_auroc(scores: list[float], labels: list[int]) -> float:
        """Compute ROC-AUC; returns NaN when inputs are degenerate."""
        try:
            from sklearn.metrics import roc_auc_score
        except ImportError:
            return float("nan")
        if len(scores) == 0 or len(set(labels)) < 2:
            return float("nan")
        return float(roc_auc_score(labels, scores))

    def finalize_metrics(self):
        """Attach AUROC to metrics object (called AFTER get_stats by BaseValidator)."""
        image_auroc = self._compute_auroc(self._auroc_image_scores, self._auroc_image_labels)
        # pixel-AUROC needs at least one positive pixel label
        pixel_auroc = self._compute_auroc(self._auroc_pixel_scores, self._auroc_pixel_labels)
        self.metrics.image_auroc = image_auroc
        self.metrics.pixel_auroc = pixel_auroc
        super().finalize_metrics()

    def get_stats(self):
        """Compute stats including AUROC (called BEFORE finalize_metrics, so compute eagerly)."""
        stats = super().get_stats()
        stats["image_auroc"] = self._compute_auroc(self._auroc_image_scores, self._auroc_image_labels)
        # pixel-AUROC: must have at least one positive pixel
        if any(self._auroc_pixel_labels):
            stats["pixel_auroc"] = self._compute_auroc(self._auroc_pixel_scores, self._auroc_pixel_labels)
        else:
            stats["pixel_auroc"] = float("nan")
        return stats
