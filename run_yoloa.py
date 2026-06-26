#!/usr/bin/env python
"""YOLOA driver — fit a memory bank per MVTec category, then predict or val with a prior.

One model load, per-category fit (disk-cached via ``YOLOA.fit``), then:
  --mode predict : save annotated predictions + heatmap overlays for sampled test images
  --mode val     : report image/pixel AUROC + mAP50 per category (+ AVERAGE), write a CSV

Both modes fit through ``YOLOA.fit``, so they share one bank cache (``--bank-cache``): a bank built
for predict is reused by val and vice versa.

All fit parameters (heatmap_mode, scorer settings, bank knobs, imgsz, etc.) are read exclusively
from the fit YAML (``--fit-cfg``). The CLI only carries operational args (mode, device, batch, etc.).

Usage:
  python run_yoloa.py --mode predict --cat bottle --fit-cfg yoloa_fit_imgsz640_texture.yaml
  python run_yoloa.py --mode val --cat texture --fit-cfg yoloa_fit_imgsz640_texture.yaml
  python run_yoloa.py --mode val --cat all --fit-cfg yoloa_fit_imgsz640_texture.yaml
"""

import argparse
import csv
import math
from pathlib import Path

import numpy as np
import torch

import cv2

from ultralytics.models.yolo.anomaly_v2.val import MVTEC_CATEGORIES, resolve_mvtec_root, run_mvtec_ood_eval
from ultralytics.utils import LOGGER
from ultralytics.yoloa import YOLOA
from yoloa_utils import (
    CAT_GROUPS, VAL_METRICS,
    collect_test_images, good_dir, model_id_from_ckpt, save_heatmap,
    load_fit_yaml, resolve_prior, resolve_scorer_kwargs, resolve_infer,
    txt_to_mask, load_mask_tensor, run_prior_viz, save_compare_grid,
)

DEFAULT_CKPT = ("/Users/louis/workspace/ultra_louis_work/expman/data/pulled/yoloa_v2/"
                "26m_yoloav2_softhint_maskonly_aug3_mixup_binary_v1/weights/best.pt")

# DEFAULT_CKPT="/Users/louis/workspace/ultra_louis_work/expman/data/pulled/yoloa_v2/26m_yoloav2_softhint_maskonly_aug3_mixup_binary_v2/weights/best.pt"
DEFAULT_CKPT="/Users/louis/workspace/ultra_louis_work/expman/data/pulled/yoloa_v2/26m_yoloav2_softhint_maskonly_aug3_mixup_ood_v3/weights/best.pt"
DEFAULT_CKPT="/Users/louis/workspace/ultra_louis_work/expman/data/pulled/yoloa_v2/26m_yoloav2_softhint_maskonly_aug3_mixup_instseg_v7_v1/weights/last.pt"

def main():
    ap = argparse.ArgumentParser(
        description="YOLOA driver — per-category fit, then predict or val. "
                    "All fit params come from --fit-cfg YAML.")
    ap.add_argument("--mode", choices=["predict", "val", "visualize", "vis"], default="predict")
    ap.add_argument("--ckpt", default=DEFAULT_CKPT)
    ap.add_argument("--cat", default="bottle",
                    help="MVTec category name, 'all' (14), 'object' (10), or 'texture' (5)")
    ap.add_argument("--prior", default="heatmap",
                    help="comma-separated prior modes: none, heatmap, mask (e.g. 'none,heatmap,mask'). "
                         "none = honest floor; heatmap = variant from fit YAML; mask = GT-bbox upper bound")
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--device", default=None, help="cpu / mps / 0 (default: auto mps-else-cpu)")
    ap.add_argument("--n-per-cat", type=int, default=0, help="predict: test images per category (0=all)")
    ap.add_argument("--n-train", type=int, default=0,
                    help="vis mode: also sample N training images per category (default 0)")
    ap.add_argument("--conf", type=float, default=0.1)
    ap.add_argument("--iou", type=float, default=0.1, help="NMS IoU")
    ap.add_argument("--e2e", action="store_true", help="end-to-end NMS-free head")
    ap.add_argument("--fit-cfg", default="yoloa_fit_default.yaml",
                    help="fit-config yaml (all fit params: heatmap_mode, scorer, bank, imgsz, etc.). "
                         "Its filename stem IS the fit_id.")
    ap.add_argument("--bank-cache", default=None,
                    help="bank cache dir (default: <out>/banks)")
    ap.add_argument("--mvtec-root", default=None, help="MVTec-YOLO root (default: auto-resolve)")
    ap.add_argument("--out", default=None,
                    help="output root (default: runs/temp/yoloa/<model_id>/<fit_id>)")
    args = ap.parse_args()

    # -- Load fit YAML (single source of truth for all fit params) -------------
    yaml, fit_cfg_path = load_fit_yaml(args.fit_cfg)

    # -- Resolve priors (comma-separated, each → val_mode) -----------------
    prior_list = [p.strip() for p in args.prior.split(",")]
    val_modes = []
    mode_to_prior = {}  # val_mode → prior_name (for column prefixes)
    needs_disc = False
    scorer_kwargs = {}
    scorer_fuse = "mean"

    for p in prior_list:
        if p == "none":
            mode = "mask_off"
        elif p == "heatmap":
            mode = resolve_prior(yaml)
            if mode in ("heatmap_learned", "heatmap_fused"):
                needs_disc = True
                scorer_kwargs = resolve_scorer_kwargs(yaml)
                scorer_fuse = yaml.get("scorer_fuse", "mean")
                scorer_weight = yaml.get("scorer_weight", 0.5)
                scorer_kwargs["scorer_weight"] = scorer_weight
        elif p == "mask":
            mode = "mask_on"
        else:
            mode = p  # "segment", etc.
        if mode not in val_modes:
            val_modes.append(mode)
            mode_to_prior[mode] = p
    val_modes = tuple(val_modes)
    prior_mode_display = ",".join(prior_list)

    # Single-prior name for predict/visualize mode (first prior in list)
    _first = prior_list[0]
    if _first == "none":
        predict_prior = "none"
    elif _first == "heatmap":
        predict_prior = resolve_prior(yaml)
    else:
        predict_prior = _first

    infer = resolve_infer(yaml)

    device = args.device or ("mps" if torch.backends.mps.is_available() else "cpu")
    root = resolve_mvtec_root(args.mvtec_root)
    assert root is not None, "MVTec root not found (pass --mvtec-root or set MVTEC_ROOT)"
    cat_arg = args.cat.lower()
    if cat_arg in CAT_GROUPS:
        cats = CAT_GROUPS[cat_arg]
    elif cat_arg == "all":
        cats = MVTEC_CATEGORIES
    else:
        cats = [args.cat]

    m = YOLOA(args.ckpt)

    fit_over = {}
    if "imgsz" in yaml:
        fit_over["imgsz"] = yaml["imgsz"]
    if "max_images" in yaml:
        fit_over["max_images"] = yaml["max_images"]
    fit_args = m.resolve_fit_args(cfg=fit_cfg_path, **fit_over)
    imgsz = int(fit_args["imgsz"])

    mid = model_id_from_ckpt(args.ckpt)
    fid = Path(fit_cfg_path).stem if fit_cfg_path else m.fit_id(cfg=fit_cfg_path, **fit_over)
    out_root = Path(args.out) if args.out else Path("runs/temp/yoloa") / mid / fid
    bank_cache = args.bank_cache or str(out_root / "banks")

    fit_disc = scorer_kwargs if needs_disc else False

    print(f"YOLOA {args.mode} | root: {root} | device: {device} | imgsz: {imgsz} | "
          f"priors: {prior_mode_display} | heat_edge: {infer.get('heat_edge', False)} "
          f"| cats({len(cats)}): {', '.join(cats)}", flush=True)
    print(f"  model: {type(m.model).__name__}, fusion_mode={getattr(m.model, 'fusion_mode', '?')}", flush=True)
    print(f"  out: {out_root}  |  bank-cache: {bank_cache}", flush=True)
    if needs_disc:
        print(f"  scorer: {scorer_kwargs}  fuse={scorer_fuse}", flush=True)

    rows = []
    for ci, cat in enumerate(cats, 1):
        gd = good_dir(root, cat)
        if not gd.is_dir():
            LOGGER.warning(f"[{ci}/{len(cats)}] {cat}: no train dir at {gd}; skipping")
            continue
        m.fit(str(gd), name=cat, cfg=fit_cfg_path, batch=args.batch, device=device,
              cache=bank_cache, fit_disc=fit_disc, **fit_over)

        if args.mode == "predict":
            out = out_root / "predict" / cat
            out.mkdir(parents=True, exist_ok=True)
            samples = collect_test_images(root / cat / "test", args.n_per_cat)
            for img, label in samples:
                r = m.predict(img, prior=predict_prior, imgsz=imgsz, conf=args.conf,
                              iou=args.iou, device=device, end2end=args.e2e,
                              verbose=False, **infer)[0]
                stem = f"{label}__{Path(img).stem}"
                cv2.imwrite(str(out / f"{stem}__pred.jpg"), r.plot())
                save_heatmap(m.model, img, out / f"{stem}__heat.jpg")
            print(f"[{ci}/{len(cats)}] {cat}: {len(samples)} predictions -> {out}", flush=True)
        elif args.mode in ("visualize", "vis"):
            out = out_root / "visualize" / cat
            out.mkdir(parents=True, exist_ok=True)
            # Collect test + train images
            test_samples = [(p, sub, "TEST") for p, sub in
                            collect_test_images(root / cat / "test", args.n_per_cat)]
            train_samples = []
            n_train = args.n_train or (20 if args.n_per_cat == 0 else 0)
            if n_train:
                train_dir = root / cat / "train" / "good"
                if train_dir.is_dir():
                    import random as _random
                    train_imgs = sorted(p for ext in ("*.jpg", "*.png")
                                        for p in train_dir.glob(ext))
                    _random.Random(0).shuffle(train_imgs)
                    train_samples = [(str(p), "good", "TRAIN") for p in train_imgs[:n_train]]
            samples = test_samples + train_samples
            pkw = dict(imgsz=imgsz, conf=args.conf, iou=args.iou, device=device, end2end=args.e2e)
            for img, label, source in samples:
                original = cv2.imread(img)
                # Overlay source label top-left
                color = (0, 165, 255) if source == "TRAIN" else (0, 255, 0)  # orange / green
                cv2.putText(original, source, (8, 28), cv2.FONT_HERSHEY_SIMPLEX,
                            0.8, color, 2, cv2.LINE_AA)
                h, w = original.shape[:2]
                gt_mask = txt_to_mask(str(Path(img).with_suffix(".txt")), h, w)
                mask_tensor = load_mask_tensor(gt_mask, imgsz)
                none_pred, n_none, _ = run_prior_viz(m, img, "none", **pkw)
                seg_pred, n_seg, seg_hmap = run_prior_viz(m, img, "segment", **pkw)
                heat_pred, n_heat, heat_hmap = run_prior_viz(m, img, "heatmap", **pkw, **infer)
                mask_pred, n_mask, _ = run_prior_viz(m, img, "mask", external_mask=mask_tensor, **pkw)
                pre = f"{source}_{label}__{Path(img).stem}"
                save_compare_grid(
                    original=original, none_pred=none_pred,
                    seg_heat=seg_hmap, seg_pred=seg_pred,
                    heat_heat=heat_hmap, heat_pred=heat_pred,
                    mask_img=gt_mask, mask_pred=mask_pred,
                    out_path=out / f"{pre}.jpg",
                    n_none=n_none, n_seg=n_seg, n_heat=n_heat, n_mask=n_mask,
                    original_title=f"original ({source})",
                )
            n_test, n_tr = len(test_samples), len(train_samples)
            print(f"[{ci}/{len(cats)}] {cat}: {n_test} test + {n_tr} train grids -> {out}", flush=True)
        else:  # val
            cat_rows = run_mvtec_ood_eval(
                m.model, root, categories=[cat], modes=val_modes,
                imgsz=imgsz, batch=args.batch, device=device, e2e=args.e2e, iou=args.iou,
                heatmap_norm=infer.get("heat_norm", "none"),
                heatmap_edge_weight=(True if infer.get("heat_edge") else None),
                heatmap_edge_sigma=infer.get("heat_edge_sigma", 1.0),
                scorer_kwargs=scorer_kwargs if needs_disc else None,
                scorer_fuse=scorer_fuse,
            )
            mode_rows = [x for x in cat_rows if x["category"] == cat and x.get("mode") in mode_to_prior]
            if not mode_rows:
                LOGGER.warning(f"[{ci}/{len(cats)}] {cat}: no val rows; skipping")
                continue
            row = {"category": cat}
            for mr in mode_rows:
                prefix = mode_to_prior[mr["mode"]]
                for k in VAL_METRICS:
                    row[f"{prefix}_{k}"] = float(mr.get(k, math.nan))
            rows.append(row)
            # Compact progress line: key metrics per prior
            prior_parts = []
            for p in prior_list:
                im = row.get(f"{p}_image_auroc", math.nan)
                px = row.get(f"{p}_pixel_auroc", math.nan)
                prior_parts.append(f"{p}(im={im:.4f} px={px:.4f})")
            print(f"[{ci}/{len(cats)}] {cat}: {'  '.join(prior_parts)}", flush=True)

    if args.mode == "val" and rows:
        # Short metric names for compact display
        SHORT = {"image_auroc": "im_auroc", "pixel_auroc": "px_auroc",
                 "mAP10": "mAP10", "mAP25": "mAP25", "mAP50": "mAP50", "mAP50_95": "mAP5095"}
        CW = 9  # column width per metric
        sep = " "  # column spacer
        gsep = " │"  # group separator (before each prior group after the first)
        # Group width = 6 cols × (sep + CW), matches data area under each prior name
        GP = len(VAL_METRICS) * (len(sep) + CW)

        val_cols = [f"{p}_{k}" for p in prior_list for k in VAL_METRICS]

        # AVERAGE row
        avg = {"category": "AVERAGE"}
        for c in val_cols:
            vals = [r[c] for r in rows if not math.isnan(r[c])]
            avg[c] = float(np.nanmean(vals)) if vals else math.nan
        rows.append(avg)

        # --- Two-line header ---
        line1 = f"{'':12s}"
        line2 = f"{'category':12s}"
        for i, p in enumerate(prior_list):
            if i > 0:
                line1 += gsep
                line2 += gsep
            line1 += f"{p:^{GP}s}"
            for k in VAL_METRICS:
                line2 += sep + f"{SHORT[k]:>{CW}s}"
        print("\n" + line1)
        print(line2)
        print("-" * len(line2))

        # --- Data rows ---
        for r in rows:
            line = f"{r['category']:12s}"
            for i, c in enumerate(val_cols):
                # Add group separator before each new prior group
                if i % len(VAL_METRICS) == 0 and i > 0:
                    line += gsep
                line += sep + f"{r[c]:{CW}.4f}"
            print(line, flush=True)

        # --- CSV (full metric names) ---
        out_csv = out_root / f"val_{prior_mode_display}_{args.cat}.csv"
        out_csv.parent.mkdir(parents=True, exist_ok=True)
        with open(out_csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["category", *val_cols])
            w.writeheader()
            w.writerows(rows)
        print(f"\nCSV -> {out_csv}", flush=True)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
