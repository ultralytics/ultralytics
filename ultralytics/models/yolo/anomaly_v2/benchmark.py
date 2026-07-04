# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

"""MVTec cross-dataset OOD benchmark for YOLO Anomaly v2.

This module is **not** imported during normal training/validation; it is loaded on
-demand by the trainer's periodic OOD callback. Keeping it separate keeps the core
validator / predictor paths small and makes it obvious that the MVTec sweep is a
post-hoc evaluation, not a required code path.
"""

from __future__ import annotations

import csv
import math
import os
from collections import defaultdict
from pathlib import Path

import torch

from ultralytics.data.build import build_dataloader
from ultralytics.utils import LOGGER
from ultralytics.utils.anomaly_v2 import PriorContext, PriorMode


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
from ultralytics.utils.torch_utils import unwrap_model

from ._util import resolve_v2_model
from .val import YOLOAnomalyValidator


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

_OOD_CSV_FIELDS = [
    "epoch",
    "category",
    "mode",
    "mAP10",
    "mAP25",
    "mAP50",
    "mAP50_95",
    "P",
    "R",
]

# Candidate dataset roots in priority order (env var wins, then ultra6, then laptop).
_MVTEC_ROOT_CANDIDATES = (
    "/data/shared-datasets/louis_data/MVTec-YOLO",
    "/Users/louis/workspace/ultra_louis_work/buffer/AnomalyData/MVTEC/MVTec-YOLO",
    "/home/laughing/codes/datasets/MVTec-YOLO",
)


def resolve_mvtec_root(explicit: str | None = None) -> Path | None:
    """First existing directory among: explicit arg, ``$MVTEC_ROOT``, ultra6 path, laptop path."""
    for c in (explicit, os.environ.get("MVTEC_ROOT"), *_MVTEC_ROOT_CANDIDATES):
        if c and Path(c).is_dir():
            return Path(c)
    return None


def _category_yaml(root: Path, cat: str) -> Path | None:
    """Prefer the single-class ``<cat>_binary.yaml`` over the multi-class ``<cat>.yaml``."""
    for name in (f"{cat}_binary.yaml", f"{cat}.yaml"):
        p = root / cat / name
        if p.exists():
            return p
    return None


def _inject_cat_bank(m, root: Path, cat: str, cache_dir: Path, imgsz, device, bank_size: int) -> bool:
    """Build-or-load a category's memory bank and inject it into ``m`` for reuse across modes.

    Mirrors the predict script's disk cache: the bank is saved to
    ``<cache_dir>/<cat>_sz<imgsz>_n<bank_size>.pt`` and reloaded on re-runs, skipping the slow
    feature extraction. Once injected, :meth:`YOLOAnomalyValidator._ensure_memory_bank` reuses it
    (its ``_built_bank`` stays False, so the bank is not dropped between modes). Returns True iff a
    usable bank is now in place.
    """
    mb = getattr(m, "memory_bank", None)
    if mb is None or getattr(m, "_bb_layers", None) is None:
        return False
    isz = imgsz if isinstance(imgsz, int) else 640
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = cache_dir / f"{cat}_sz{isz}_n{bank_size}.pt"
    if path.exists():
        d = torch.load(path, map_location="cpu")
        if not d.get("_calibrated"):
            LOGGER.warning(f"bank cache is old format (no calibration state); delete {path} to rebuild")
        mb.load_bank(d["memory_bank"])
        mb.temperature = d["temperature"]
        mb.update = False
        if d.get("_calibrated"):
            mb._threshold = d["_threshold"]
            mb._compactness = d["_compactness"]
            mb._calibrated = True
        LOGGER.info(f"MVTec OOD: {cat}: loaded cached bank ({mb.memory_bank.shape[0]} vecs) <- {path}")
        return True
    train_dir = root / cat / "train" / "good"
    if not train_dir.is_dir():
        train_dir = root / cat / "train"
    if not train_dir.is_dir():
        LOGGER.warning(f"MVTec OOD: {cat}: no train split for bank build; skipping cache.")
        return False
    mb.reset_memory_bank()
    try:
        n = m.build_memory_bank(str(train_dir), imgsz=isz, device=device, max_bank_size=bank_size, verbose=False)
    except Exception as e:
        LOGGER.warning(f"MVTec OOD: {cat}: bank build failed ({type(e).__name__}: {e}); skipping cache.")
        return False
    if not n or mb.memory_bank is None or mb.memory_bank.shape[0] == 0:
        return False
    entry = {
        "memory_bank": mb.memory_bank.detach().cpu(),
        "feature_dim": mb.feature_dim,
        "temperature": float(mb.temperature),
    }
    if getattr(mb, "_calibrated", False):
        entry["_threshold"] = mb._threshold
        entry["_compactness"] = mb._compactness
        entry["_calibrated"] = True
    torch.save(entry, path)
    mb.update = False
    LOGGER.info(f"MVTec OOD: {cat}: built+cached bank ({mb.memory_bank.shape[0]} vecs) -> {path}")
    return True


def run_mvtec_ood_eval(
    model,
    mvtec_root: str | Path,
    categories: list[str] | None = None,
    modes: tuple[str, ...] = ("none", "heatmap"),
    imgsz: int = 320,
    batch: int = 8,
    workers: int = 0,
    device=None,
    bank_size: int = 10000,
    save_dir: str | Path | None = None,
    epoch: int | None = None,
    e2e: bool | None = None,
    iou: float | None = None,
    heatmap_norm: str | None = None,
    heatmap_smooth_kernel: int | None = None,
    heatmap_edge_weight: bool | None = None,
    heatmap_edge_p: float | None = None,
    heatmap_edge_m: float | None = None,
    heatmap_edge_sigma: float | None = None,
    bank_cache_dir: str | Path | None = None,
    validator_cls: type | None = None,
) -> list[dict]:
    """Run the MVTec OOD mode sweep over ``categories``; ``model`` is a YOLOAnomalyV2Model.

    Each (category, mode) is isolated in try/except so one failure never aborts the sweep.
    Returns per-(category, mode) rows plus per-mode ``AVERAGE`` rows; appends ``mvtec_ood.csv``
    under ``save_dir`` when given. Reuses :class:`YOLOAnomalyValidator` via the ``model.val`` args
    path.

    ``bank_cache_dir`` (opt-in): persist each category's memory bank to
    ``<dir>/<cat>_sz<imgsz>_n<bank_size>.pt`` and inject it before the mode loop, so the validator
    reuses it across modes instead of rebuilding and dropping it per pass.
    """
    root = Path(mvtec_root)
    validator_cls = validator_cls or YOLOAnomalyValidator
    m = unwrap_model(model)
    if e2e is not None:
        m.model[-1].end2end = e2e
        m.end2end = e2e

    base_ctx = m.prior_context if isinstance(m.prior_context, PriorContext) else PriorContext()
    if heatmap_norm is not None:
        base_ctx.heatmap_norm = str(heatmap_norm).lower()
        if heatmap_smooth_kernel is not None:
            base_ctx.heatmap_smooth_kernel = int(heatmap_smooth_kernel)
    if heatmap_edge_weight is not None:
        base_ctx.heatmap_edge_weight = bool(heatmap_edge_weight)
        if heatmap_edge_p is not None:
            base_ctx.heatmap_edge_p = float(heatmap_edge_p)
        if heatmap_edge_m is not None:
            base_ctx.heatmap_edge_m = float(heatmap_edge_m)
        if heatmap_edge_sigma is not None:
            base_ctx.heatmap_edge_sigma = float(heatmap_edge_sigma)
    m.prior_context = base_ctx

    rows: list[dict] = []

    for cat in categories or MVTEC_CATEGORIES:
        yaml = _category_yaml(root, cat)
        if yaml is None:
            LOGGER.warning(f"MVTec OOD: no data yaml for category '{cat}' under {root}; skipping.")
            continue

        cat_dataset = None
        for mi, mode in enumerate(modes):
            if mode not in ("none", "heatmap"):
                LOGGER.warning(f"MVTec OOD: unsupported mode '{mode}'; skipping.")
                continue

            try:
                if mode == "heatmap":
                    if bank_cache_dir is not None:
                        _inject_cat_bank(m, root, cat, Path(bank_cache_dir), imgsz, device, bank_size)
                    else:
                        # Let the validator build the bank from the train split.
                        if m.memory_bank is not None:
                            m.memory_bank.reset_memory_bank()
                    m.prior_context = _ctx_with_mode(base_ctx, PriorMode.HEATMAP)
                else:
                    if m.memory_bank is not None:
                        m.memory_bank.reset_memory_bank()
                    m.prior_context = _ctx_with_mode(base_ctx, PriorMode.NONE)

                overrides = {
                    "task": "anomaly_v2",
                    "mode": "val",
                    "data": str(yaml),
                    "split": "val",
                    "imgsz": imgsz,
                    "batch": batch,
                    "workers": workers,
                    "device": str(device) if device is not None else None,
                    "rect": False,
                    "plots": False,
                    "verbose": False,
                    "save_json": False,
                    "single_cls": True,
                    "cache": "ram" if len(modes) > 1 else False,
                }
                if e2e is not None:
                    overrides["end2end"] = e2e
                if iou is not None:
                    overrides["iou"] = iou
                sd = Path(save_dir) / "mvtec_ood_runs" if save_dir is not None else None
                if mi == 0:
                    validator = validator_cls(args=overrides, save_dir=sd)
                    validator._ood_bank_size = bank_size
                    validator(trainer=None, model=m)
                    cat_dataset = validator.dataloader.dataset
                else:
                    cat_dl = build_dataloader(cat_dataset, batch, workers, shuffle=False, rank=-1)
                    validator = validator_cls(dataloader=cat_dl, args=overrides, save_dir=sd)
                    validator._ood_bank_size = bank_size
                    validator(trainer=None, model=m)
                mm = validator._ood_map_metrics()
                rows.append(
                    {
                        "epoch": epoch,
                        "category": cat,
                        "mode": mode,
                        "mAP10": mm["mAP10"],
                        "mAP25": mm["mAP25"],
                        "mAP50": mm["mAP50"],
                        "mAP50_95": mm["mAP50_95"],
                        "P": mm["P"],
                        "R": mm["R"],
                    }
                )
            except Exception as e:
                LOGGER.warning(f"MVTec OOD: {cat}/{mode} failed ({type(e).__name__}: {e}); recording NaN.")
                rows.append(
                    {
                        "epoch": epoch,
                        "category": cat,
                        "mode": mode,
                        "mAP10": math.nan,
                        "mAP25": math.nan,
                        "mAP50": math.nan,
                        "mAP50_95": math.nan,
                        "P": math.nan,
                        "R": math.nan,
                    }
                )
        if m.memory_bank is not None:
            m.memory_bank.reset_memory_bank()

    rows.extend(_ood_macro_average(rows, epoch))
    if save_dir is not None:
        _append_ood_csv(rows, Path(save_dir) / "mvtec_ood.csv")
    return rows


def _ood_macro_average(rows: list[dict], epoch: int | None) -> list[dict]:
    """One ``AVERAGE`` row per mode, averaging each metric over categories (ignoring NaN)."""
    by_mode: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        if r["category"] != "AVERAGE":
            by_mode[r["mode"]].append(r)
    out = []
    for mode, rs in by_mode.items():

        def _avg(key: str) -> float:
            vals = [r[key] for r in rs if not math.isnan(r[key])]
            return sum(vals) / len(vals) if vals else math.nan

        avg = {
            "epoch": epoch,
            "category": "AVERAGE",
            "mode": mode,
            "mAP10": _avg("mAP10"),
            "mAP25": _avg("mAP25"),
            "mAP50": _avg("mAP50"),
            "mAP50_95": _avg("mAP50_95"),
            "P": _avg("P"),
            "R": _avg("R"),
        }
        out.append(avg)
        LOGGER.info(
            f"MVTec OOD @ep{epoch} [{mode:8s}] mAP10={avg['mAP10']:.4f} mAP25={avg['mAP25']:.4f} "
            f"mAP50={avg['mAP50']:.4f} mAP50-95={avg['mAP50_95']:.4f} (n={len(rs)})"
        )
    return out


def _append_ood_csv(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.exists()
    with open(path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=_OOD_CSV_FIELDS)
        if write_header:
            w.writeheader()
        for r in rows:
            w.writerow({k: ("" if r.get(k) is None else r[k]) for k in _OOD_CSV_FIELDS})
