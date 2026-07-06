# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

"""YOLO Anomaly v2 validator — single-pass with optional heatmap prior."""

from __future__ import annotations

import numpy as np
import torch

from ultralytics.models.yolo.detect import DetectionValidator
from ultralytics.utils import LOGGER


class YOLOAnomalyValidator(DetectionValidator):
    """Anomaly-v2 validator.

    Runs a single validation pass. If an AnomalyMemoryBank is configured and can be
    built from the dataset's train (normal) split, the pass uses the heatmap prior
    (extended IoU grid for coarse localization). Otherwise it falls back to a regular
    detection validation with a warning.
    """

    def __init__(self, dataloader=None, save_dir=None, args=None, _callbacks=None) -> None:
        super().__init__(dataloader, save_dir, args, _callbacks)
        self.args.task = "anomaly_v2"
        self.args.rect = False

    def init_metrics(self, model) -> None:
        """Set up heatmap flag and try to build the memory bank for heatmap mode."""
        super().init_metrics(model)
        self.v2_model = model.model if hasattr(model, "backend") else model  # unwrap AutoBackend if present
        self._bank_built = False

        # Build the bank only when the prior is enabled; a bank-free pass leaves it untouched.
        if self.use_prior:
            self._ensure_memory_bank(self.v2_model)
        # Coarse IoU grid .10:.50 (step .05) — anomaly localization cares about coarse overlap,
        # not tight boxes. Same grid with or without the prior, so both are directly comparable.
        # Columns: .10=0, .25=3, .50=8. mAP10-50 is the mean over the whole grid.
        self.iouv = torch.linspace(0.1, 0.5, 9)
        self.niou = self.iouv.numel()

    @property
    def use_prior(self) -> bool:
        """Whether the heatmap prior is active for this pass (mirrors model._prior_enabled)."""
        return bool(getattr(getattr(self, "v2_model", None), "_prior_enabled", True))

    def __call__(self, trainer=None, model=None):
        """Single validation pass; restore model state afterwards."""
        metrics = super().__call__(trainer=trainer, model=model)
        # Only clear a bank this validator built itself; leave a pre-fitted bank intact
        # so the same model can be validated again (e.g. with vs. without the prior).
        if getattr(self, "_bank_built", False):
            self.v2_model.memory_bank.reset()
        return metrics

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

    def _ensure_memory_bank(self, model) -> bool:
        """Build the bank from the train (normal) split if empty. Returns True iff a usable bank is ready."""
        if model.memory_bank.is_ready:
            return True
        support = self._collect_support_paths()
        if not support:
            return False
        imgsz = self.args.imgsz if isinstance(self.args.imgsz, int) else 640
        try:
            n = model.build_memory_bank(support, imgsz=imgsz, device=self.device, verbose=False)
            if not n:
                LOGGER.warning("YOLOAnomalyValidator: memory bank is empty after build.")
                return False
        except Exception as e:
            # LOGGER.warning(f"YOLOAnomalyValidator: failed to build memory bank: {e}")
            return False
        self._bank_built = True
        LOGGER.info(f"YOLOAnomalyValidator: built memory bank ({n} features) from {len(support)} normal images.")
        return True

    def _ood_map_metrics(self) -> dict[str, float]:
        """mAP at IoU {0.10, 0.25, 0.50} and mAP10-50 (mean over the whole .10:.50 grid)."""
        box = self.metrics.box
        all_ap = getattr(box, "all_ap", [])
        out = {
            "mAP10": 0.0,
            "mAP25": 0.0,
            "mAP50": 0.0,
            "mAP10_50": 0.0,
            "P": float(box.mp),
            "R": float(box.mr),
        }
        if not len(all_ap):
            return out
        iouv = self.iouv.cpu().numpy()
        idx = {thr: int(np.where(np.isclose(iouv, thr))[0][0]) for thr in (0.10, 0.25, 0.50)}
        out["mAP10"] = float(all_ap[:, idx[0.10]].mean())
        out["mAP25"] = float(all_ap[:, idx[0.25]].mean())
        out["mAP50"] = float(all_ap[:, idx[0.50]].mean())
        out["mAP10_50"] = float(all_ap.mean())  # mean over the full .10:.50 grid
        return out

    def get_desc(self) -> str:
        """Return the column header with mAP10/mAP25 columns (always, for alignment)."""
        return ("%22s" + "%11s" * 8) % (
            "Class",
            "Images",
            "Instances",
            "Box(P",
            "R",
            "mAP10",
            "mAP25",
            "mAP50",
            "mAP10-50)",
        )

    def print_results(self) -> None:
        """Print the 'all' row with mAP10/mAP25/mAP50 and the mAP10:50 aggregate."""
        mm = self._ood_map_metrics()
        nt = int(self.metrics.nt_per_class.sum()) if len(self.metrics.nt_per_class) else 0
        pf = "%22s" + "%11i" * 2 + "%11.3g" * 6
        LOGGER.info(
            pf % ("all", self.seen, nt, mm["P"], mm["R"], mm["mAP10"], mm["mAP25"], mm["mAP50"], mm["mAP10_50"])
        )
