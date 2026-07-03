# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

"""YOLO Anomaly v2 validator — 2-pass (mask-on + mask-off) with AUROC.

Pass 1 (mask-on):  forward with GT bbox rendered mask.
Pass 2 (mask-off): forward with mask disabled (passthrough ≈ vanilla YOLO).

wandb / results.csv key scheme:
  - mask-off pass -> BARE standard keys (``metrics/...(B/M)``, ``val/..._loss``, ``fitness``) so
    the run reads like a vanilla YOLO run; fitness/best.pt tracks mask-off mAP for non-OOD runs.
  - mask-on pass  -> ``metrics(mask_prior)/`` (metrics + auroc + fitness) and ``val(mask_prior)/`` (losses).
  - MVTec OOD modes -> ``test_metrics(none_prior)/`` · ``test_metrics(heatmap_prior)/`` ·
    ``test_metrics(mask_prior)/`` (train.py).

The mask-off (vanilla floor) pass always runs; the mask-on prior pass is optional via
``anomaly_v2.val_mask_prior`` (default true) — set false to ~halve in-dist val time.

In addition to standard detection metrics, accumulates image-AUROC and
pixel-AUROC from the model's stashed heatmap (``model._last_heatmap``) during
pass 1 (logged under ``metrics(mask_prior)/``).

When ``prior_mode`` is set (standalone val), runs single-pass with that mode
and still computes AUROC.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F

from ultralytics.models.yolo.detect import DetectionValidator
from ultralytics.utils import LOGGER
from ultralytics.nn.modules.anomaly_v2 import BboxMaskRenderer
from ultralytics.utils.anomaly_v2 import PriorContext, PriorMode, prior_context_from_overrides

from ._util import resolve_v2_model

_SENTINEL = object()  # "prior mode not snapshotted" marker (distinct from a real None)


class YOLOAnomalyValidatorBase:
    """Shared anomaly-v2 validation logic, mixed into ``DetectionValidator``.

    Holds all anomaly behavior — prior-mode routing, 2-pass (mask-on / mask-off) training val,
    image/pixel AUROC, the memory-bank lifecycle, and the OOD / standalone single-pass path.
    Composed with ``DetectionValidator`` as :class:`YOLOAnomalyValidator`.
    """

    def __init__(
        self, dataloader=None, save_dir=None, args=None, _callbacks=None, prior_mode: str | None = None
    ) -> None:
        if isinstance(args, dict):
            prior_mode = args.pop("prior_mode", prior_mode)
        super().__init__(dataloader, save_dir, args, _callbacks)
        self.args.task = "anomaly_v2"
        self._mask_mode = "on"
        self._model_ref = None

        # Heatmap-prior memory-bank lifecycle (single-pass OOD eval). When prior_mode
        # is "heatmap", build the BackboneMemoryBank from the dataset's train (normal)
        # split inside this validator, then restore prior_mode + drop the bank after —
        # so a shared/EMA model is left clean (no ckpt bloat, no stale prior).
        self._ood_bank_size = 10000
        self._saved_prior_context = _SENTINEL
        self._built_bank = False

        # AUROC accumulators
        self._auroc_image_scores: list[float] = []
        self._auroc_image_labels: list[int] = []
        self._auroc_pixel_scores: list[float] = []
        self._auroc_pixel_labels: list[int] = []

        # GT mask renderer for pixel-AUROC (matches the 256x256 heatmap resize in update_metrics)
        self._eval_mask_renderer = BboxMaskRenderer(mask_size=256, mode="rect")

        # Set last so the property setter can configure the IoU vector.
        self.prior_mode = prior_mode

    @property
    def prior_mode(self) -> str | None:
        """Active prior mode."""
        return getattr(self, "_prior_mode", None)

    @prior_mode.setter
    def prior_mode(self, value: str | None) -> None:
        """Switch the IoU vector when entering/leaving the OOD/standalone single-pass path.

        OOD/standalone single-pass eval adds coarse low-IoU thresholds (0.10, 0.25) on top of the
        standard 0.5:0.95 grid — defect localization is coarse, so mAP@0.10/0.25 are the informative
        operating points. Training 2-pass val (prior_mode=None) keeps the default iouv so the
        trainer-facing box.map50/box.map stay standard.
        """
        self._prior_mode = value
        if value is not None:
            # Standard 0.5:0.95 grid FIRST (so box.map50/map75 + P/R stay at their standard IoU
            # operating points), low-IoU thresholds appended. ood_map_metrics indexes by IoU value,
            # so column order doesn't affect the reported mAP10/25/50/50-95.
            self.iouv = torch.cat([torch.linspace(0.5, 0.95, 10), torch.tensor([0.10, 0.25])])
            self.niou = self.iouv.numel()
        else:
            self.iouv = torch.linspace(0.5, 0.95, 10)
            self.niou = self.iouv.numel()

    # ------------------------------------------------------------------
    # Hooks
    # ------------------------------------------------------------------
    def init_metrics(self, model) -> None:
        super().init_metrics(model)
        self._model_ref = model
        m = resolve_v2_model(model)
        if m is not None and self.prior_mode is not None and hasattr(m, "set_prior_context"):
            self._saved_prior_context = m.prior_context  # snapshot for restore
            base = m.prior_context if isinstance(m.prior_context, PriorContext) else PriorContext()
            ctx = PriorContext(**{**base.__dict__, "mode": PriorMode(self.prior_mode)})
            if self.prior_mode == "heatmap":
                if not self._ensure_memory_bank(m):
                    # No usable bank (no bb_layers / no normal images / build failed): never run the
                    # heatmap forward against a missing-or-empty bank — fall back to bare detection so
                    # the run is honest (AUROC nan) instead of crashing or scoring against 0 vectors.
                    LOGGER.warning("YOLOAnomalyValidator: heatmap prior unavailable -> running prior_mode='none'.")
                    ctx = PriorContext(**{**base.__dict__, "mode": PriorMode.NONE})
                else:
                    ctx = prior_context_from_overrides(dict(vars(self.args)), ctx)
            m.set_prior_context(ctx)

    def preprocess(self, batch):
        batch = super().preprocess(batch)
        model = resolve_v2_model(self._model_ref)
        if model is None:
            return batch

        # Stash the per-batch prior mask on the model so the forward pass can consume it.
        # Training and mask-on validation now receive ``batch["prior_mask"]`` from the dataset
        # pipeline (built by LoadAnomalyPriorMask); heatmap/none modes leave it None/False.
        if self.prior_mode is not None:
            if self.prior_mode in ("mask", "box"):
                model._pending_prior_mask = batch.get("prior_mask")
            else:
                model._pending_prior_mask = None
        elif self._mask_mode == "on":
            model._pending_prior_mask = batch.get("prior_mask")
        else:
            model._pending_prior_mask = None
        return batch

    def postprocess(self, preds):
        """Post-process predictions."""
        return super().postprocess(preds)

    # ------------------------------------------------------------------
    # 2-pass __call__ (training val) or single-pass (prior_mode)
    # ------------------------------------------------------------------
    def __call__(self, trainer=None, model=None):
        self._built_bank = False
        self._saved_prior_context = _SENTINEL
        try:
            if self.prior_mode is not None:
                return self._single_pass(trainer, model)
            return self._two_pass(trainer, model)
        finally:
            self._restore_prior_state()  # undo prior_mode + any bank we built (YOLOE-style)

    def _single_pass(self, trainer, model):
        self._reset_auroc()
        if self.prior_mode == "heatmap":
            self.args.rect = False
        return super().__call__(trainer=trainer, model=model)

    def _two_pass(self, trainer, model):
        # Mask-off (vanilla floor) always runs; the mask-on (prior) pass is optional. Skipping it
        # ~halves in-dist val time; OOD runs take fitness from the OOD heatmap eval, so the prior
        # pass is pure monitoring there.
        run_on = self._resolve_mask_on_pass(trainer, model)

        # wandb / results.csv key scheme:
        #   mask-OFF pass -> BARE standard keys (metrics/...(B/M), val/..._loss, fitness) so the run
        #     reads like a vanilla YOLO run — mask-off IS passthrough ≈ vanilla YOLO. This also makes
        #     fitness/best.pt track the honest mask-off mAP for non-OOD runs (OOD runs override it).
        #   mask-ON pass  -> metrics(mask_prior)/ for metrics+auroc+fitness, val(mask_prior)/ for losses
        #     (in-distribution val WITH the GT bbox prior; prior-tagged sections mirror the bare det ones).
        def _regroup_on(k):
            if k.startswith("metrics/"):
                return f"metrics(mask_prior)/{k[len('metrics/') :]}"
            if k.startswith("val/"):
                return f"val(mask_prior)/{k[len('val/') :]}"
            return f"metrics(mask_prior)/{k}"  # fitness + any other top-level scalar

        merged: dict = {}
        metrics_on = None

        # Pass 1: mask-on (prior) -> metrics(mask_prior)/ + val(mask_prior)/ + image/pixel AUROC.
        if run_on:
            self._mask_mode = "on"
            self._reset_auroc()
            stats_on = super().__call__(trainer=trainer, model=model)
            metrics_on = self.metrics  # snapshot before pass 2 overwrites
            image_auroc = self._compute_auroc(self._auroc_image_scores, self._auroc_image_labels)
            pixel_auroc = self._compute_auroc(self._auroc_pixel_scores, self._auroc_pixel_labels)
            if not isinstance(stats_on, dict):
                return stats_on
            merged = {_regroup_on(k): v for k, v in stats_on.items()}
            merged["metrics(mask_prior)/image_auroc"] = image_auroc
            merged["metrics(mask_prior)/pixel_auroc"] = pixel_auroc

        # Pass 2: mask-off (bare vanilla floor + fitness) — always runs; this is the standard val.
        self._mask_mode = "off"
        try:
            stats_off = super().__call__(trainer=trainer, model=model)
        except Exception as e:
            LOGGER.warning(f"YOLOAnomalyValidator: mask-off pass failed: {e}")
            stats_off = {}
        if isinstance(stats_off, dict):
            self._mask_off_stats = dict(stats_off)
            # AUROC is meaningless without a prior heatmap; drop it from the bare namespace.
            merged.update({k: v for k, v in stats_off.items() if k not in ("image_auroc", "pixel_auroc")})

        # Restore mask-on metrics so results_dict / curves reflect the prior-on pass (when it ran).
        self._mask_mode = "on"
        if metrics_on is not None:
            self.metrics = metrics_on
        return merged

    def _resolve_mask_on_pass(self, trainer, model) -> bool:
        """Whether to run the in-dist mask-ON (prior) pass. ``anomaly_v2.val_mask_prior``, default True.

        The mask-OFF (bare vanilla floor) pass always runs — it carries the standard metrics and
        fitness. Skipping mask-on (no ``metrics(mask_prior)/`` or AUROC) ~halves in-dist val time.
        """
        mdl = model if model is not None else getattr(trainer, "model", None)
        m = resolve_v2_model(mdl if mdl is not None else self._model_ref)
        v2 = (getattr(m, "yaml", {}) or {}).get("anomaly_v2", {}) if m is not None else {}
        return bool(v2.get("val_mask_prior", True))

    # ------------------------------------------------------------------
    # Heatmap-prior memory-bank lifecycle (single-pass OOD eval)
    # ------------------------------------------------------------------
    def _collect_support_paths(self) -> list[str]:
        """Resolve normal (good) image paths from the dataset's train split for the bank."""
        from pathlib import Path

        data = self.data if isinstance(self.data, dict) else {}
        train = data.get("train")
        if train is None:
            return []
        if isinstance(train, (list, tuple)):
            return [str(p) for p in train]
        p = Path(train)
        if p.suffix.lower() == ".txt":  # image-list file (e.g. MVTec category train.txt)
            root = Path(data.get("path", p.parent))
            out = []
            for ln in p.read_text().splitlines():
                ln = ln.strip()
                if ln:
                    q = Path(ln)
                    out.append(str(q if q.is_absolute() else root / q))
            return out
        return [str(p)]  # a directory (build_memory_bank iterates it) or single image

    def _ensure_memory_bank(self, m) -> bool:
        """Build the bank from the train (normal) split if empty. Returns True iff a usable bank is ready."""
        mb = getattr(m, "memory_bank", None)
        if mb is None or getattr(m, "_bb_layers", None) is None:
            LOGGER.warning(
                "YOLOAnomalyValidator: prior_mode='heatmap' but no BackboneMemoryBank is configured "
                "(model YAML needs a bb_layers block)."
            )
            return False
        if mb.memory_bank is not None and mb.memory_bank.shape[0] > 0:
            return True  # caller already supplied a bank — use it, don't rebuild or clear it
        support = self._collect_support_paths()
        if not support:
            LOGGER.warning("YOLOAnomalyValidator: no train (normal) images found; cannot build memory bank.")
            return False
        imgsz = self.args.imgsz if isinstance(self.args.imgsz, int) else 640
        try:
            n = m.build_memory_bank(
                support, imgsz=imgsz, device=self.device, max_bank_size=self._ood_bank_size, verbose=False
            )
        except Exception as e:  # missing/corrupt images, OOM, etc. — never crash the val run
            LOGGER.warning(f"YOLOAnomalyValidator: memory bank build failed ({type(e).__name__}: {e}).")
            return False
        if not n:
            LOGGER.warning("YOLOAnomalyValidator: memory bank is empty after build.")
            return False
        LOGGER.info(f"YOLOAnomalyValidator: built memory bank ({n} features) from {len(support)} normal images.")
        self._built_bank = True
        return True

    def _restore_prior_state(self) -> None:
        """Undo prior_mode + any bank we built so a shared/EMA model is left clean."""
        if self._saved_prior_context is _SENTINEL and not self._built_bank:
            return
        m = resolve_v2_model(self._model_ref)
        if m is None:
            return
        if self._saved_prior_context is not _SENTINEL and hasattr(m, "set_prior_context"):
            m.set_prior_context(self._saved_prior_context)
            self._saved_prior_context = _SENTINEL
        if self._built_bank:
            mb = getattr(m, "memory_bank", None)
            if mb is not None and hasattr(mb, "reset_memory_bank"):
                mb.reset_memory_bank()  # free the bank; prevents EMA/ckpt bloat mid-training
            self._built_bank = False

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
                    renderer = self._eval_mask_renderer.to(bb_per_img.device)
                    gt_mask = renderer(
                        bb_per_img,
                        torch.zeros(bb_per_img.shape[0], dtype=torch.long, device=bb_per_img.device),
                        1,
                    )
                    hmap_b = heatmap[b : b + 1]
                    if hmap_b.shape[2] != 256 or hmap_b.shape[3] != 256:
                        hmap_b = F.interpolate(hmap_b, size=(256, 256), mode="bilinear", align_corners=False)
                    self._auroc_pixel_scores.extend(hmap_b.flatten().cpu().tolist())
                    self._auroc_pixel_labels.extend(gt_mask.flatten().cpu().tolist())

    @staticmethod
    def _compute_auroc(scores: list[float], labels: list[int]) -> float:
        """Binary ROC-AUC via the tie-aware rank statistic (Mann-Whitney U) — no sklearn dependency.

        Matches ``sklearn.metrics.roc_auc_score`` for the binary case. Returns NaN when there are no
        finite scores or only one label class is present. Self-contained on purpose: the ``ultra``
        training env has no sklearn, so the old ``import sklearn`` made every OOD AUROC silently NaN.
        """
        import numpy as np

        if not scores or len(set(labels)) < 2:
            return float("nan")
        s = np.asarray(scores, dtype=np.float64)
        y = np.asarray(labels, dtype=np.float64)
        finite = np.isfinite(s)
        if not finite.all():
            s, y = s[finite], y[finite]
        n_pos, n_neg = float((y == 1).sum()), float((y == 0).sum())
        if n_pos == 0 or n_neg == 0:
            return float("nan")
        order = np.argsort(s, kind="mergesort")
        s_sorted = s[order]
        ranks_sorted = np.empty(s.shape[0], dtype=np.float64)
        i, n = 0, s.shape[0]
        while i < n:  # average rank within each tie group (1-based)
            j = i + 1
            while j < n and s_sorted[j] == s_sorted[i]:
                j += 1
            ranks_sorted[i:j] = 0.5 * (i + j - 1) + 1.0
            i = j
        ranks = np.empty(n, dtype=np.float64)
        ranks[order] = ranks_sorted
        return float((ranks[y == 1].sum() - n_pos * (n_pos + 1.0) / 2.0) / (n_pos * n_neg))

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

    def ood_map_metrics(self) -> dict[str, float]:
        """mAP at IoU {0.10, 0.25, 0.50} and the {0.50:0.95} mean, from the extended-iouv ``all_ap``.

        Only meaningful on the OOD/standalone path (``prior_mode`` set), where ``iouv`` carries the
        low-IoU thresholds. Falls back to the standard box accessors when ``all_ap`` is unavailable.
        """
        import numpy as np

        box = self.metrics.box
        all_ap = getattr(box, "all_ap", [])
        out = {
            "mAP10": 0.0,
            "mAP25": 0.0,
            "mAP50": float(box.map50),
            "mAP50_95": float(box.map),
            "P": float(box.mp),
            "R": float(box.mr),
        }
        if not len(all_ap):
            return out
        iouv = self.iouv.cpu().numpy()
        _at = lambda thr: float(all_ap[:, int(np.argmin(np.abs(iouv - thr)))].mean())
        std = all_ap[:, iouv >= 0.5 - 1e-6]
        out["mAP10"], out["mAP25"], out["mAP50"] = _at(0.10), _at(0.25), _at(0.50)
        out["mAP50_95"] = float(std.mean()) if std.size else float(box.map)
        return out

    def get_desc(self) -> str:
        """OOD path: header with low-IoU mAP10/25 + im/px AUROC columns; standard header otherwise."""
        if self.prior_mode is None:
            return super().get_desc()
        return ("%22s" + "%11s" * 10) % (
            "Class",
            "Images",
            "Instances",
            "Box(P",
            "R",
            "mAP10",
            "mAP25",
            "mAP50",
            "mAP50-95)",
            "im_auroc",
            "px_auroc",
        )

    def print_results(self) -> None:
        """OOD/standalone path: print the "all" row in DetectionValidator's aligned column format,
        with correctly-labeled mAP10/25/50/50-95 from ``ood_map_metrics``.

        The extended iouv shifts the columns ``box.map50``/``box.map`` read, so the standard summary
        would mislabel mAP@0.10 as "mAP50". Print the right values in the same style; training 2-pass
        val (``prior_mode=None``, standard iouv) is unchanged.
        """
        if self.prior_mode is None:
            return super().print_results()
        mm = self.ood_map_metrics()
        nt = int(self.metrics.nt_per_class.sum()) if len(self.metrics.nt_per_class) else 0
        ia = float(getattr(self.metrics, "image_auroc", math.nan))
        pa = float(getattr(self.metrics, "pixel_auroc", math.nan))
        pf = "%22s" + "%11i" * 2 + "%11.3g" * 8  # DetectionValidator widths + im/px AUROC columns
        LOGGER.info(
            pf % ("all", self.seen, nt, mm["P"], mm["R"], mm["mAP10"], mm["mAP25"], mm["mAP50"], mm["mAP50_95"], ia, pa)
        )


class YOLOAnomalyValidator(YOLOAnomalyValidatorBase, DetectionValidator):
    """Anomaly-v2 validator with box metrics (training val + OOD / standalone path)."""



