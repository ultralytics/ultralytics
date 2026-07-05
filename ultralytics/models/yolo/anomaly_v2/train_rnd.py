# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

"""Internal R&D trainer for YOLO Anomaly v2.

Extends ``AnomalyV2Trainer`` with periodic cross-dataset OOD validation. This is
not intended for normal users; it exists for internal experiments where best.pt
selection should be driven by a macro-average OOD metric (e.g. MVTec mAP50).

The OOD loop is intentionally simple:
  * resolve a list of data yamls,
  * for each yaml, build a memory bank from its normal train split,
  * run ``YOLOAnomalyValidator`` on its val split,
  * macro-average the metrics and use the result as training fitness.
"""

from __future__ import annotations

import math
from copy import deepcopy
from pathlib import Path

from ultralytics.models.yolo.anomaly_v2.train import AnomalyV2Trainer
from ultralytics.models.yolo.anomaly_v2.val import YOLOAnomalyValidator
from ultralytics.utils import LOGGER, RANK, YAML
from ultralytics.utils.torch_utils import unwrap_model


MVTEC_CATEGORIES = [
    "bottle",
    "cable",
    "capsule",
    "carpet",
    "grid",
    "hazelnut",
    "leather",
    "metal_nut",
    "pill",
    "screw",
    "tile",
    "toothbrush",
    "transistor",
    "wood",
    "zipper",
]


def _normal_dir_from_yaml(yaml_path: str | Path) -> Path:
    """Resolve the directory of normal images referenced by a data yaml.

    Follows the MVTec convention: if ``<train>/good`` exists, use it; otherwise
    use ``<train>``.
    """
    data = YAML.load(yaml_path)
    root = Path(data.get("path", Path(yaml_path).parent))
    train = Path(data["train"])
    if not train.is_absolute():
        train = root / train
    good = train / "good"
    return good if good.is_dir() else train


def _average_ood_rows(rows: list[dict]) -> dict[str, float]:
    """Macro-average over categories, ignoring NaNs."""
    out: dict[str, float] = {}
    for key in ("mAP10", "mAP25", "mAP50", "mAP50_95", "P", "R"):
        vals = [r[key] for r in rows if not math.isnan(r[key])]
        out[key] = sum(vals) / len(vals) if vals else math.nan
    return out


class AnomalyV2RNDTrainer(AnomalyV2Trainer):
    """Trainer for YOLOAnomalyV2 with periodic cross-dataset OOD validation."""

    def validate(self):
        """Run normal validation, then periodic OOD validation; fitness = OOD mAP50."""
        metrics, fitness = super().validate()

        if RANK not in (-1, 0) or self.ema is None:
            return metrics, fitness

        v2_cfg = getattr(unwrap_model(self.model), "yaml", {}).get("anomaly_v2", {})
        freq = int(v2_cfg.get("test_val_freq", 0))
        if freq <= 0 or (self.epoch + 1) % freq != 0:
            return metrics, fitness

        yamls = self._resolve_test_yamls(v2_cfg)
        if not yamls:
            return metrics, fitness

        ema_eval = deepcopy(self.ema.ema).eval()
        try:
            rows = self._run_ood_eval(ema_eval, yamls, v2_cfg)
            if rows:
                avg = _average_ood_rows(rows)
                fitness = float(avg["mAP50"])
                metrics["fitness"] = fitness
                self.best_fitness = max(self.best_fitness or -math.inf, fitness)
                LOGGER.info(
                    f"OOD eval @ep{self.epoch + 1}: mAP50={avg['mAP50']:.4f} "
                    f"mAP10={avg['mAP10']:.4f} (n={len(rows)} categories)"
                )
        finally:
            del ema_eval

        return metrics, fitness

    def _resolve_test_yamls(self, v2_cfg: dict) -> list[Path]:
        """Resolve explicit ``test_data_yamls`` or expand ``test_root`` + ``test_categories``."""
        if explicit := v2_cfg.get("test_data_yamls"):
            return [Path(p) for p in explicit]

        root = v2_cfg.get("test_root")
        if not root:
            LOGGER.warning("AnomalyV2RNDTrainer: no test_data_yamls or test_root configured; skipping OOD eval.")
            return []

        cats = v2_cfg.get("test_categories") or MVTEC_CATEGORIES
        yamls = []
        for cat in cats:
            for name in (f"{cat}_binary.yaml", f"{cat}.yaml"):
                p = Path(root) / cat / name
                if p.exists():
                    yamls.append(p)
                    break
            else:
                LOGGER.warning(f"AnomalyV2RNDTrainer: no yaml found for category '{cat}' under {root}")
        return yamls

    def _run_ood_eval(self, model, yamls: list[Path], v2_cfg: dict) -> list[dict]:
        """Fit bank per yaml and validate; return per-category metric rows."""
        rows = []
        batch = int(v2_cfg.get("test_batch", 8))
        device = self.device
        workers = self.args.workers

        for yaml in yamls:
            source = _normal_dir_from_yaml(yaml)
            try:
                model.memory_bank.reset()
                n = model.build_memory_bank(
                    str(source),
                    imgsz=640,
                    device=device,
                    batch=batch,
                    verbose=False,
                )
                if not n:
                    LOGGER.warning(f"OOD eval: empty bank for {yaml.name}; skipping.")
                    continue

                overrides = {
                    "task": "anomaly_v2",
                    "mode": "val",
                    "data": str(yaml),
                    "split": "val",
                    "imgsz": 640,
                    "batch": batch,
                    "workers": workers,
                    "device": str(device) if device is not None else None,
                    "rect": False,
                    "plots": False,
                    "verbose": False,
                    "save_json": False,
                    "single_cls": True,
                    "iou": 0.1,
                }
                validator = YOLOAnomalyValidator(args=overrides)
                validator(trainer=None, model=model)

                rows.append({"category": yaml.parent.name, **validator._ood_map_metrics()})
            except Exception as e:
                LOGGER.warning(f"OOD eval failed for {yaml}: {type(e).__name__}: {e}")

        return rows
