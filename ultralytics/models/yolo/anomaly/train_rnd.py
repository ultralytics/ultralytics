# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

"""Internal R&D trainer for YOLO Anomaly.

Extends ``AnomalyTrainer`` with periodic cross-dataset OOD validation. This is
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
from torch import distributed as dist

from ultralytics.models.yolo.anomaly.train import AnomalyTrainer
from ultralytics.models.yolo.anomaly.val import YOLOAnomalyValidator
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

_MVTEC_ROOT_CANDIDATES = (
    "/data/shared-datasets/louis_data/MVTec-YOLO",
    "/Users/louis/workspace/ultra_louis_work/buffer/AnomalyData/MVTEC/MVTec-YOLO",
    "/home/laughing/codes/datasets/MVTec-YOLO",
)


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
    """Macro-average over categories, ignoring NaNs and non-numeric fields (e.g. ``category``).

    Averages every numeric key present, so both the heatmap-prior metrics (``mAP50`` …) and the
    none-prior metrics (``none_mAP50`` …) are aggregated in one pass.
    """
    keys = [k for k in rows[0] if isinstance(rows[0].get(k), (int, float))]
    out: dict[str, float] = {}
    for key in keys:
        vals = [r[key] for r in rows if isinstance(r.get(key), (int, float)) and not math.isnan(r[key])]
        out[key] = sum(vals) / len(vals) if vals else math.nan
    return out


class AnomalyRNDTrainer(AnomalyTrainer):
    """Trainer for YOLOAnomaly with periodic cross-dataset OOD validation."""

    def validate(self):
        """Run normal validation, then periodic OOD validation; fitness = OOD mAP50."""
        if self.ema and self.world_size > 1:
            # Sync EMA buffers from rank 0 to all ranks
            for buffer in self.ema.ema.buffers():
                dist.broadcast(buffer, src=0)
        metrics = self.validator(self)
        if metrics is None:  # non-rank-0 DDP workers get no metrics — mirror BaseTrainer.validate()
            return None, None
        fitness = metrics.pop("fitness", -self.loss.detach().cpu().numpy())
        if RANK not in (-1, 0) or self.ema is None:
            return metrics, fitness

        v2_cfg = getattr(unwrap_model(self.model), "yaml", {}).get("anomaly", {})
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
                avg_metrics = {f"ood/{k}": v for k, v in avg.items()}
                fitness = float(avg["mAP50"])
                metrics["fitness"] = fitness
                metrics.update(avg_metrics)
                self.best_fitness = max(self.best_fitness or -math.inf, fitness)
                LOGGER.info(
                    f"OOD eval @ep{self.epoch + 1}: [heatmap] mAP50={avg['mAP50']:.4f} "
                    f"mAP10={avg['mAP10']:.4f} | [none] mAP50={avg.get('none_mAP50', float('nan')):.4f} "
                    f"mAP10={avg.get('none_mAP10', float('nan')):.4f} "
                    f"(fitness=heatmap mAP50; n={len(rows)} categories)"
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
            for candidate in _MVTEC_ROOT_CANDIDATES:
                if Path(candidate).is_dir():
                    root = candidate
                    break
        if not root:
            LOGGER.warning("AnomalyRNDTrainer: no test_data_yamls or test_root configured; skipping OOD eval.")
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
                LOGGER.warning(f"AnomalyRNDTrainer: no yaml found for category '{cat}' under {root}")
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
                n = model.build_memory_bank(str(source), imgsz=640, device=device, batch=batch)
                if not n:
                    LOGGER.warning(f"OOD eval: empty bank for {yaml.name}; skipping.")
                    continue

                overrides = {
                    "task": "detect",
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
                    "iou": 0.2,
                    "conf": 0.25,
                    "end2end": False,
                }
                # Pass 1: heatmap prior (memory bank active) — the yoloa_clean fitness signal.
                validator = YOLOAnomalyValidator(args=overrides)
                validator(trainer=None, model=model)
                row = {"category": yaml.parent.name, **validator._ood_map_metrics()}

                # Pass 2: none prior (bank disabled via the ``building`` flag, same toggle the viz
                # path uses) — bare-detector baseline, logged as ``none_*`` so the per-category
                # fusion lift (heatmap - none) is visible. Does not change fitness.
                mb = model.memory_bank
                saved_building = mb.building
                mb.building = True
                try:
                    validator_none = YOLOAnomalyValidator(args=overrides)
                    validator_none(trainer=None, model=model)
                    row.update({f"none_{k}": v for k, v in validator_none._ood_map_metrics().items()})
                finally:
                    mb.building = saved_building

                rows.append(row)
            except Exception as e:
                LOGGER.warning(f"OOD eval failed for {yaml}: {type(e).__name__}: {e}")

        return rows
