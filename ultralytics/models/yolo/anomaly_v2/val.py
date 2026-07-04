# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

"""YOLO Anomaly v2 validator — single-pass with optional heatmap prior."""

from __future__ import annotations

import math

import torch

from ultralytics.models.yolo.detect import DetectionValidator
from ultralytics.utils import LOGGER
from ultralytics.utils.anomaly_v2 import PriorContext, PriorMode, prior_context_from_overrides

from ._util import resolve_v2_model

_SENTINEL = object()


def _ctx_with_mode(base: PriorContext, mode: PriorMode) -> PriorContext:
    """Return a fresh PriorContext with ``mode`` replaced, copying only known fields."""
    return PriorContext(
        mode=mode,
        heatmap_norm=base.heatmap_norm,
        heatmap_smooth_kernel=base.heatmap_smooth_kernel,
        heatmap_edge_weight=base.heatmap_edge_weight,
        heatmap_edge_p=base.heatmap_edge_p,
        heatmap_edge_m=base.heatmap_edge_m,
        heatmap_edge_sigma=base.heatmap_edge_sigma,
    )


class YOLOAnomalyValidator(DetectionValidator):
    """Anomaly-v2 validator.

    Runs a single validation pass. If a BackboneMemoryBank is configured and can be
    built from the dataset's train (normal) split, the pass uses the heatmap prior
    (extended IoU grid for coarse localization). Otherwise it falls back to a regular
    detection validation with a warning.
    """

    def __init__(self, dataloader=None, save_dir=None, args=None, _callbacks=None) -> None:
        super().__init__(dataloader, save_dir, args, _callbacks)
        self.args.task = "anomaly_v2"
        self._model_ref = None
        self._heatmap_active = False

        # Memory-bank lifecycle: built from the train split when empty, restored after.
        self._ood_bank_size = 10000
        self._saved_prior_context = _SENTINEL
        self._built_bank = False

        # Standard IoU grid by default; extended if the heatmap prior is active.
        self.iouv = torch.linspace(0.5, 0.95, 10)
        self.niou = self.iouv.numel()

    def init_metrics(self, model) -> None:
        """Set up prior context and try to build the memory bank for heatmap mode."""
        super().init_metrics(model)
        self._model_ref = model
        self._heatmap_active = False
        m = resolve_v2_model(model)
        if m is None or not hasattr(m, "set_prior_context"):
            return

        self._saved_prior_context = m.prior_context
        base = m.prior_context if isinstance(m.prior_context, PriorContext) else PriorContext()

        ctx = _ctx_with_mode(base, PriorMode.HEATMAP)
        if self._ensure_memory_bank(m):
            ctx = prior_context_from_overrides(dict(vars(self.args)), ctx)
            m.set_prior_context(ctx)
            self._heatmap_active = True
            # Extended IoU grid for coarse defect localization with the heatmap prior.
            self.iouv = torch.cat([torch.linspace(0.5, 0.95, 10), torch.tensor([0.10, 0.25])])
            self.niou = self.iouv.numel()
        else:
            LOGGER.warning("YOLOAnomalyValidator: heatmap prior unavailable -> running regular validation.")
            m.set_prior_context(_ctx_with_mode(base, PriorMode.NONE))

    def __call__(self, trainer=None, model=None):
        """Single validation pass; restore model state afterwards."""
        self._built_bank = False
        self._saved_prior_context = _SENTINEL
        self._heatmap_active = False
        try:
            return super().__call__(trainer=trainer, model=model)
        finally:
            self._restore_prior_state()

    # ------------------------------------------------------------------
    # Memory-bank lifecycle
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
        if p.suffix.lower() == ".txt":
            root = Path(data.get("path", p.parent))
            out = []
            for ln in p.read_text().splitlines():
                ln = ln.strip()
                if ln:
                    q = Path(ln)
                    out.append(str(q if q.is_absolute() else root / q))
            return out
        return [str(p)]

    def _ensure_memory_bank(self, m) -> bool:
        """Build the bank from the train (normal) split if empty. Returns True iff a usable bank is ready."""
        mb = getattr(m, "memory_bank", None)
        if mb is None or getattr(m, "_bb_layers", None) is None:
            return False
        if mb.memory_bank is not None and mb.memory_bank.shape[0] > 0:
            return True
        support = self._collect_support_paths()
        if not support:
            return False
        imgsz = self.args.imgsz if isinstance(self.args.imgsz, int) else 640
        try:
            n = m.build_memory_bank(
                support, imgsz=imgsz, device=self.device, max_bank_size=self._ood_bank_size, verbose=False
            )
        except Exception as e:
            LOGGER.warning(f"YOLOAnomalyValidator: memory bank build failed ({type(e).__name__}: {e}).")
            return False
        if not n:
            LOGGER.warning("YOLOAnomalyValidator: memory bank is empty after build.")
            return False
        LOGGER.info(f"YOLOAnomalyValidator: built memory bank ({n} features) from {len(support)} normal images.")
        self._built_bank = True
        return True

    def _restore_prior_state(self) -> None:
        """Undo prior context changes and free any bank built during validation."""
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
                mb.reset_memory_bank()
            self._built_bank = False

    # ------------------------------------------------------------------
    # Metrics formatting (extended IoU grid only when heatmap prior is active)
    # ------------------------------------------------------------------
    def _ood_map_metrics(self) -> dict[str, float]:
        """mAP at IoU {0.10, 0.25, 0.50} and the {0.50:0.95} mean from the extended iouv."""
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
        """Return the column header."""
        if not self._heatmap_active:
            return super().get_desc()
        return ("%22s" + "%11s" * 8) % (
            "Class",
            "Images",
            "Instances",
            "Box(P",
            "R",
            "mAP10",
            "mAP25",
            "mAP50",
            "mAP50-95)",
        )

    def print_results(self) -> None:
        """Print the 'all' row; use the extended-iou labels when the heatmap prior is active."""
        if not self._heatmap_active:
            return super().print_results()
        mm = self._ood_map_metrics()
        nt = int(self.metrics.nt_per_class.sum()) if len(self.metrics.nt_per_class) else 0
        pf = "%22s" + "%11i" * 2 + "%11.3g" * 6
        LOGGER.info(pf % ("all", self.seen, nt, mm["P"], mm["R"], mm["mAP10"], mm["mAP25"], mm["mAP50"], mm["mAP50_95"]))
