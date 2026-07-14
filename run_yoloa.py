#!/usr/bin/env python
"""Simplified YOLOA driver using the new high-level API.

- val      : per-category ``model.set_memory`` + ``model.val`` (matches ``runs/anomaly/test_anomaly.py``).
- predict  : ``model.set_memory`` + heatmap-prior prediction; falls back to no prior if no bank is ready.
- visualize: 1x3 grid:
    - none-prior prediction + GT boxes
    - original image with heatmap overlay
    - heatmap-prior prediction + GT boxes
  (no masks / seg heatmaps).

Usage:
  python run_yoloa.py --mode val --cat all
  python run_yoloa.py --mode predict --cat bottle --n-per-cat 5
  python run_yoloa.py --mode visualize --cat texture --n-per-cat 3
"""

import argparse
import csv
import os
from pathlib import Path

import cv2
import numpy as np
import torch

from ultralytics.models.yolo.anomaly.predict import AnomalyPredictorHM
from ultralytics.models.yolo.anomaly.val import YOLOAnomalyValidatorHM
from ultralytics.models.yolo.anomaly.train_rnd import MVTEC_CATEGORIES
from ultralytics.utils import LOGGER
from ultralytics import YOLOA
from yoloa_utils import (
    CAT_GROUPS,
    collect_test_images,
    good_dir,
    model_id_from_ckpt,
    save_simple_grid,
)

DEFAULT_CKPT = "/home/laughing/Downloads/best_converted.pt"
DEFAULT_MVTEC_ROOT = "/home/laughing/codes/datasets/MVTec-YOLO"


def resolve_categories(cat_arg: str):
    """Return category list from 'all', group names, or a single category name."""
    cat_arg = cat_arg.lower()
    if cat_arg == "all":
        return MVTEC_CATEGORIES
    if cat_arg in CAT_GROUPS:
        return CAT_GROUPS[cat_arg]
    return [cat_arg]


def _data_yaml(root: Path, cat: str) -> Path | None:
    """Pick the category data yaml, preferring the *_binary variant."""
    for name in (f"{cat}_binary.yaml", f"{cat}.yaml"):
        p = root / cat / name
        if p.exists():
            return p
    return None


def _bank_is_ready(model) -> bool:
    mb = getattr(model.model, "memory_bank", None)
    return mb is not None and mb.is_ready


def _average(rows: list[dict], key: str) -> float:
    vals = [r[key] for r in rows if not (isinstance(r[key], float) and np.isnan(r[key]))]
    return float(np.mean(vals)) if vals else float("nan")


def _predict_plot(model, img: str, prior: str, pkw: dict):
    """Run ``model.predict`` with the requested prior and return (plot_bgr, n_dets, heatmap_np).

    ``prior`` is one of ``none`` or ``heatmap``.  The memory bank is temporarily
    disabled for ``none`` so no heatmap prior is used.
    """
    mb = getattr(model.model, "memory_bank", None)
    saved_building = getattr(mb, "building", None)
    try:
        if prior == "none" and mb is not None:
            mb.building = True
        res = model.predict(source=img, stream=False, verbose=False, **pkw)[0]
    finally:
        if saved_building is not None:
            mb.building = saved_building
    n = 0 if res.boxes is None else len(res.boxes)
    hm = getattr(res, "heatmap", None)
    hm_np = hm.detach().cpu().numpy() if hm is not None else None
    return res.plot(), n, hm_np


def main():
    ap = argparse.ArgumentParser(description="Simplified YOLOA driver (new API).")
    ap.add_argument("--mode", choices=["predict", "val", "visualize"], default="predict")
    ap.add_argument("--ckpt", default=DEFAULT_CKPT, help="path to converted YOLOA checkpoint")
    ap.add_argument("--cat", default="bottle", help="category, 'all', 'object', or 'texture'")
    ap.add_argument("--device", default=None, help="cpu / cuda:0 / 0 (default: cuda:0 if available)")
    ap.add_argument("--batch", type=int, default=8, help="memory-bank feature-extraction batch size")
    ap.add_argument("--conf", type=float, default=0.1, help="predict conf threshold")
    ap.add_argument("--iou", type=float, default=0.1, help="NMS IoU")
    ap.add_argument("--e2e", "--end2end", action="store_true", help="use end-to-end NMS-free head")
    ap.add_argument("--hm-gate-blend", type=float, default=0.0, help="heatmap conf gate (0=on, 1=off)")
    ap.add_argument(
        "--no-memory", action="store_true", help="skip building/using the memory bank (bank-free, no heatmap prior)"
    )
    ap.add_argument("--n-per-cat", type=int, default=0, help="predict/visualize: images per cat (0=all)")
    ap.add_argument(
        "--bb-layers",
        type=int,
        nargs="+",
        default=None,
        help="override memory-bank backbone tap layers (applied before set_memory), e.g. --bb-layers 8",
    )
    ap.add_argument("--imgsz", type=int, default=640, help="inference image size")
    ap.add_argument("--mvtec-root", default=None, help="MVTec-YOLO root (default: MVTEC_ROOT env or built-in)")
    ap.add_argument("--bank-cache", default=None, help="bank cache dir (default: <out>/banks)")
    ap.add_argument("--out", default=None, help="output root (default: runs/temp/yoloa_new/<model_id>)")

    ap.add_argument("--rebuild", action="store_true", help="rebuild memory bank even if cache exists")

    ap.add_argument(
        "--hm-boxes",
        action="store_true",
        help="derive boxes from the heatmap via connected components (AnomalyPredictorHM) instead of the detection head",
    )

    args = ap.parse_args()

    # -- Resolve paths / device ------------------------------------------------
    root = Path(args.mvtec_root or os.environ.get("MVTEC_ROOT", DEFAULT_MVTEC_ROOT))
    assert root.is_dir(), f"MVTec root not found: {root}"

    device = args.device or ("cuda:0" if torch.cuda.is_available() else "cpu")
    cats = resolve_categories(args.cat)

    model = YOLOA(args.ckpt)
    if args.bb_layers:
        model.model.bb_layers = list(args.bb_layers)
        print(f"  bb_layers override -> {model.model.bb_layers}", flush=True)

    mid = model_id_from_ckpt(args.ckpt)
    out_root = Path(args.out) if args.out else Path("runs/temp/yoloa_new") / mid
    bank_cache = args.bank_cache or str(out_root / "banks")

    print(
        f"YOLOA {args.mode} | root: {root} | device: {device} | imgsz: {args.imgsz} | "
        f"cats({len(cats)}): {', '.join(cats)}",
        flush=True,
    )
    print(f"  model: {type(model.model).__name__}", flush=True)
    print(f"  out: {out_root}  |  bank-cache: {bank_cache}", flush=True)

    # -- VAL -------------------------------------------------------------------
    if args.mode == "val":
        rows = []
        for ci, cat in enumerate(cats, 1):
            yaml = _data_yaml(root, cat)
            if yaml is None:
                LOGGER.warning(f"[{ci}/{len(cats)}] {cat}: no data yaml; skipping")
                continue

            stage = "val (no memory)" if args.no_memory else "set_memory + val"
            print(f"\n{'=' * 60}\n[{ci}/{len(cats)}] {cat}: {stage}\n{'=' * 60}", flush=True)
            if not args.no_memory:
                model.reset_memory()
                model.to(device)
                model.set_memory(source=str(good_dir(root, cat)), batch=args.batch, imgsz=args.imgsz)

            metrics = model.val(
                validator=YOLOAnomalyValidatorHM if args.hm_boxes else None,
                data=str(yaml),
                imgsz=args.imgsz,
                iou=args.iou,
                end2end=args.e2e,
                single_cls=True,
                device=device,
                batch=args.batch,
                conf=args.conf,
                project=Path(args.ckpt).stem,
                name=cat,
            )

            # all_ap cols: iouv = linspace(.10, .50, 9) → .10=col0, .25=col3, .50=col8.
            ap_mat = metrics.box.all_ap
            ok = ap_mat.ndim == 2 and ap_mat.shape[1] >= 9
            map10 = float(ap_mat[:, 0].mean()) if ok else float("nan")
            map25 = float(ap_mat[:, 3].mean()) if ok else float("nan")
            map50 = float(ap_mat[:, 8].mean()) if ok else float("nan")
            map10_50 = float(ap_mat.mean()) if ok else float("nan")  # mean over IoU .10:.50

            rows.append({"category": cat, "mAP10": map10, "mAP25": map25, "mAP50": map50, "mAP10-50": map10_50})
            print(
                f"[{cat}] mAP10={map10:.4f} mAP25={map25:.4f} mAP50={map50:.4f} mAP10-50={map10_50:.4f}",
                flush=True,
            )

        if rows:
            rows.append(
                {
                    "category": "AVERAGE",
                    "mAP10": _average(rows, "mAP10"),
                    "mAP25": _average(rows, "mAP25"),
                    "mAP50": _average(rows, "mAP50"),
                    "mAP10-50": _average(rows, "mAP10-50"),
                }
            )

            print("\n" + "=" * 80)
            print(f"{'category':<15s} {'mAP10':>10s} {'mAP25':>10s} {'mAP50':>10s} {'mAP10-50':>10s}")
            print("-" * 80)
            for r in rows:
                print(
                    f"{r['category']:<15s} "
                    f"{r['mAP10']:>10.4f} {r['mAP25']:>10.4f} "
                    f"{r['mAP50']:>10.4f} {r['mAP10-50']:>10.4f}"
                )
            print("=" * 80)

            out_csv = out_root / f"val_{args.cat}.csv"
            out_csv.parent.mkdir(parents=True, exist_ok=True)
            with open(out_csv, "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=["category", "mAP10", "mAP25", "mAP50", "mAP10-50"])
                w.writeheader()
                w.writerows(rows)
            print(f"\nCSV -> {out_csv}", flush=True)

    # -- PREDICT / VISUALIZE ---------------------------------------------------
    else:
        pkw = dict(
            imgsz=args.imgsz,
            conf=args.conf,
            iou=args.iou,
            device=device,
            end2end=args.e2e,
        )
        if args.hm_boxes:
            pkw["predictor"] = AnomalyPredictorHM
            print("  predictor -> AnomalyPredictorHM (heatmap connected-component boxes)", flush=True)
        for ci, cat in enumerate(cats, 1):
            if not (root / cat).is_dir():
                LOGGER.warning(f"[{ci}/{len(cats)}] {cat}: missing category dir; skipping")
                continue

            if not args.no_memory:
                model.to(device)
                model.set_memory(source=str(good_dir(root, cat)), batch=args.batch)

            samples = collect_test_images(root / cat / "test", args.n_per_cat)
            if not samples:
                LOGGER.warning(f"[{ci}/{len(cats)}] {cat}: no test images; skipping")
                continue

            out = out_root / ("predict" if args.mode == "predict" else "visualize") / cat
            out.mkdir(parents=True, exist_ok=True)

            for img, label in samples:
                if args.mode == "predict":
                    prior = "heatmap" if (not args.no_memory and _bank_is_ready(model)) else "none"
                    pred_bgr, _, _ = _predict_plot(model, img, prior, pkw)
                    cv2.imwrite(str(out / f"{label}__{Path(img).stem}.jpg"), pred_bgr)
                else:  # visualize
                    original = cv2.imread(img)
                    none_pred, n_none, _ = _predict_plot(model, img, "none", pkw)
                    heat_pred, n_heat, heat_hmap = _predict_plot(model, img, "heatmap", pkw)
                    gt_txt = str(Path(img).with_suffix(".txt"))
                    save_simple_grid(
                        original,
                        none_pred,
                        heat_pred,
                        heat_hmap,
                        gt_txt_path=gt_txt,
                        out_path=out / f"{label}__{Path(img).stem}.jpg",
                        n_none=n_none,
                        n_heat=n_heat,
                    )

            print(f"[{ci}/{len(cats)}] {cat}: {len(samples)} -> {out}", flush=True)

    print("DONE", flush=True)


if __name__ == "__main__":
    main()
