#!/usr/bin/env python
"""YOLOA driver — fit a memory bank per MVTec category, then predict or val with a prior.

One model load, per-category fit (disk-cached via ``YOLOA.fit``), then:
  --mode predict : save annotated predictions + heatmap overlays for sampled test images
  --mode val     : report image/pixel AUROC + mAP50 per category (+ AVERAGE), write a CSV

Both modes fit through ``YOLOA.fit``, so they share one bank cache (``--bank-cache``): a bank built
for predict is reused by val and vice versa.

Usage:
  python yoloa.py --mode predict --cat bottle --prior heatmap
  python yoloa.py --mode predict --cat object --prior heatmap   # 10 object categories
  python yoloa.py --mode val --cat texture --prior none         # 5 texture categories
  python yoloa.py --mode val --cat all --prior heatmap
  python yoloa.py --mode val --cat bottle --prior none          # honest floor (no prior)
"""

import argparse
import csv
import math
import random
from pathlib import Path

import cv2
import numpy as np
import torch

from ultralytics.models.yolo.anomaly_v2.val import MVTEC_CATEGORIES, resolve_mvtec_root, run_mvtec_ood_eval
from ultralytics.utils import LOGGER
from ultralytics.yoloa import YOLOA

MVTEC_OBJECT = ["bottle", "cable", "capsule", "hazelnut", "metal_nut", "pill", "screw",
                "toothbrush", "transistor", "zipper"]
MVTEC_TEXTURE = ["carpet", "grid", "leather", "tile", "wood"]

_CAT_GROUPS = {"object": MVTEC_OBJECT, "texture": MVTEC_TEXTURE}

DEFAULT_CKPT = ("/Users/louis/workspace/ultra_louis_work/expman/data/pulled/yoloa_v2/"
                "26m_yoloav2_softhint_maskonly_aug3_mixup_binary_v1/weights/best.pt")
# Anomaly localization is coarse, so the low-IoU mAP10/mAP25 are the informative operating points.
VAL_METRICS = ("image_auroc", "pixel_auroc", "mAP10", "mAP25", "mAP50", "mAP50_95")
_PRIOR_TO_MODE = {"none": "mask_off", "heatmap": "heatmap"}  # driver prior -> OOD-eval mode


def collect_test_images(test_root: Path, n: int, seed: int = 0) -> list[tuple[str, str]]:
    """Return up to ``n`` (path, defect_type) pairs sampled across a category's test subdirs."""
    pairs = [(str(p), sub.name) for sub in sorted(test_root.iterdir()) if sub.is_dir()
             for p in sorted(sub.glob("*.png"))]
    random.Random(seed).shuffle(pairs)
    return pairs[:n] if n and n > 0 else pairs


def good_dir(root: Path, cat: str) -> Path:
    """Resolve a category's normal-images dir for fitting (train/good, else train)."""
    d = root / cat / "train" / "good"
    return d if d.is_dir() else root / cat / "train"


def model_id_from_ckpt(ckpt: str) -> str:
    """Short model id from a ckpt path: <run>/weights/best.pt -> <run>, else the file stem."""
    p = Path(ckpt).resolve()
    return p.parents[1].name if p.parent.name == "weights" else p.stem


def save_heatmap(model, img_path: str, out_path: Path) -> None:
    """Save the model's last prior heatmap as a JET overlay on the original image."""
    hm = getattr(model, "_last_heatmap", None)
    if hm is None:
        return
    h = hm.detach().cpu().numpy().squeeze()
    h = (h - h.min()) / (np.ptp(h) + 1e-9)
    orig = cv2.imread(img_path)
    color = cv2.resize(cv2.applyColorMap((h * 255).astype(np.uint8), cv2.COLORMAP_JET),
                       (orig.shape[1], orig.shape[0]))
    cv2.imwrite(str(out_path), cv2.addWeighted(orig, 0.55, color, 0.45, 0))


def load_mask_tensor(mask_path, imgsz: int):
    """Load a GT mask as a (1, 1, imgsz, imgsz) float tensor, or None."""
    if mask_path is None:
        return None
    m = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if m is None:
        return None
    m = cv2.resize(m, (imgsz, imgsz), interpolation=cv2.INTER_NEAREST).astype(np.float32) / 255.0
    return torch.from_numpy(m).unsqueeze(0).unsqueeze(0)


def run_prior_viz(m, img, prior, imgsz, conf, iou, device, external_mask=None, **kw):
    """Predict with one prior; return (pred_rgb, n_det, heatmap_np) for the 8-panel grid."""
    res = m.predict(img, prior=prior, imgsz=imgsz, conf=conf, iou=iou, device=device,
                    external_mask=external_mask, verbose=False, **kw)[0]
    n = 0 if res.boxes is None else res.boxes.shape[0]
    pred_rgb = cv2.cvtColor(res.plot(), cv2.COLOR_BGR2RGB)
    hm = getattr(m.model, "_last_heatmap", None)
    hm_np = hm.detach().cpu().numpy().squeeze() if hm is not None else None
    return pred_rgb, n, hm_np


def main():
    ap = argparse.ArgumentParser(description="YOLOA driver — per-category fit, then predict or val")
    ap.add_argument("--mode", choices=["predict", "val", "visualize"], default="predict")
    ap.add_argument("--ckpt", default=DEFAULT_CKPT)
    ap.add_argument("--cat", default="bottle",
                    help="MVTec category name, 'all' (14), 'object' (10), or 'texture' (5)")
    ap.add_argument("--prior", default="heatmap", choices=["none", "heatmap"],
                    help="inference prior: heatmap (memory bank) or none (honest floor)")
    ap.add_argument("--imgsz", type=int, default=None,
                    help="override the fit yaml's imgsz (else from --fit-cfg, else model yaml default 640). "
                         "640 = native; 320 = fast preview.")
    ap.add_argument("--max-images", type=int, default=None,
                    help="override the fit yaml's max_images cap (else from --fit-cfg, else 0 = all normals)")
    ap.add_argument("--n-per-cat", type=int, default=0, help="predict: test images per category (0=all)")
    ap.add_argument("--conf", type=float, default=0.1)
    ap.add_argument("--iou", type=float, default=0.1, help="NMS IoU (val: matches mvtec_deploy_eval default)")
    ap.add_argument("--e2e", action="store_true", help="end-to-end NMS-free head (default off -> NMS + --iou)")
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--device", default=None, help="cpu / mps / 0 (default: auto mps-else-cpu)")
    ap.add_argument("--heat-edge", action="store_true", help="edge-suppression weight on the heatmap")
    ap.add_argument("--heat-edge-sigma", type=float, default=1.0, help="edge value (bigger = gentler)")
    ap.add_argument("--heat-norm", default="mean", choices=["none", "minmax", "gaussian", "mean"],
                    help="prior processing before fusion (affects fused prior -> mAP, not AUROC)")
    ap.add_argument("--fit-cfg", default="yoloa_fit_default.yaml",
                    help="fit-config yaml (imgsz + max_images + bb_layers/bb_K/bb_temperature/bb_calibrate/...). "
                         "Its filename stem IS the fit_id (the <fit_id> output-path segment). imgsz should live here.")
    ap.add_argument("--bank-cache", default=None,
                    help="bank cache dir (default: <out>/banks; per model+fit, so models never share a bank)")
    ap.add_argument("--mvtec-root", default=None, help="MVTec-YOLO root (default: auto-resolve)")
    ap.add_argument("--out", default=None,
                    help="output root (default: runs/temp/yoloa/<model_id>/<fit_id>)")
    args = ap.parse_args()

    # Resolve --fit-cfg: default looks inside the package; explicit paths are tried as-is first
    # (e.g. "yoloa_fit_default.yaml" from CWD), then fall back to the package cfg dir.
    if args.fit_cfg is not None:
        p = Path(args.fit_cfg)
        if not p.is_file() and not p.is_absolute():
            cfg_path = Path(__file__).resolve().parent / "ultralytics" / "cfg" / p.name
            if cfg_path.is_file():
                args.fit_cfg = str(cfg_path)

    device = args.device or ("mps" if torch.backends.mps.is_available() else "cpu")
    root = resolve_mvtec_root(args.mvtec_root)
    assert root is not None, "MVTec root not found (pass --mvtec-root or set MVTEC_ROOT)"
    cat_arg = args.cat.lower()
    if cat_arg in _CAT_GROUPS:
        cats = _CAT_GROUPS[cat_arg]
    elif cat_arg == "all":
        cats = MVTEC_CATEGORIES
    else:
        cats = [args.cat]

    m = YOLOA(args.ckpt)  # load once; bank is re-fit per category

    # The fit config (imgsz + bank knobs) lives in the fit yaml; --imgsz/--max-images are optional
    # overrides only. fit_over holds the explicit overrides (None = take from yaml / model default).
    fit_over = {k: v for k, v in (("imgsz", args.imgsz), ("max_images", args.max_images)) if v is not None}
    fit_args = m.resolve_fit_args(cfg=args.fit_cfg, **fit_over)
    imgsz = int(fit_args["imgsz"])  # effective imgsz for the whole run (fit + predict/val)

    # Output root = model identity (ckpt run name) + fit identity. fit_id = the fit yaml's stem (you
    # name it), or the config hash when no fit yaml. A different model OR fit gets its own dir and bank.
    mid = model_id_from_ckpt(args.ckpt)
    fid = Path(args.fit_cfg).stem if args.fit_cfg else m.fit_id(cfg=args.fit_cfg, **fit_over)
    out_root = Path(args.out) if args.out else Path("runs/temp/yoloa") / mid / fid
    bank_cache = args.bank_cache or str(out_root / "banks")

    infer = {}  # sticky prior-shaping knobs forwarded to predict/val
    if args.heat_edge:
        infer.update(heat_edge=True, heat_edge_sigma=args.heat_edge_sigma)
    if args.heat_norm:
        infer.update(heat_norm=args.heat_norm)

    edge_str = f"on(sigma={args.heat_edge_sigma})" if args.heat_edge else "off"
    print(f"YOLOA {args.mode} | root: {root} | device: {device} | imgsz: {imgsz} | "
          f"prior: {args.prior} | heat-edge: {edge_str} | cats({len(cats)}): {', '.join(cats)}", flush=True)
    print(f"  model: {type(m.model).__name__}, fusion_mode={getattr(m.model, 'fusion_mode', '?')}", flush=True)
    print(f"  out: {out_root}  |  bank-cache: {bank_cache}", flush=True)

    rows = []
    for ci, cat in enumerate(cats, 1):
        gd = good_dir(root, cat)
        if not gd.is_dir():
            LOGGER.warning(f"[{ci}/{len(cats)}] {cat}: no train dir at {gd}; skipping")
            continue
        m.fit(str(gd), name=cat, cfg=args.fit_cfg, batch=args.batch, device=device,
              cache=bank_cache, **fit_over)

        if args.mode == "predict":
            out = out_root / "predict" / cat
            out.mkdir(parents=True, exist_ok=True)
            samples = collect_test_images(root / cat / "test", args.n_per_cat)
            for img, label in samples:
                r = m.predict(img, prior=args.prior, imgsz=imgsz, conf=args.conf,
                              iou=args.iou, device=device, end2end=args.e2e,
                              verbose=False, **infer)[0]
                stem = f"{label}__{Path(img).stem}"
                cv2.imwrite(str(out / f"{stem}__pred.jpg"), r.plot())
                save_heatmap(m.model, img, out / f"{stem}__heat.jpg")
            print(f"[{ci}/{len(cats)}] {cat}: {len(samples)} predictions -> {out}", flush=True)
        elif args.mode == "visualize":  # 8-panel CompareGrid (4 priors + heatmaps + GT mask)
            from compare_grid import CompareGrid
            cg = CompareGrid()
            out = out_root / "visualize" / cat
            out.mkdir(parents=True, exist_ok=True)
            samples = collect_test_images(root / cat / "test", args.n_per_cat)
            for img, label in samples:
                original = cv2.cvtColor(cv2.imread(img), cv2.COLOR_BGR2RGB)
                mask_path = CompareGrid.find_mask(img)
                mask_tensor = load_mask_tensor(mask_path, imgsz)
                pkw = dict(imgsz=imgsz, conf=args.conf, iou=args.iou, device=device,
                           end2end=args.e2e)
                none_pred, n_none, _ = run_prior_viz(m, img, "none", **pkw)
                seg_pred, n_seg, seg_hmap = run_prior_viz(m, img, "segment", **pkw)
                heat_pred, n_heat, heat_hmap = run_prior_viz(m, img, "heatmap", **pkw, **infer)
                mask_pred, n_mask, _ = run_prior_viz(m, img, "mask", external_mask=mask_tensor, **pkw)
                cg.save(
                    original=original, none_pred=none_pred,
                    seg_heat=CompareGrid.heatmap_panel(original, seg_hmap), seg_pred=seg_pred,
                    heat_heat=CompareGrid.heatmap_panel(original, heat_hmap), heat_pred=heat_pred,
                    mask_img=CompareGrid.mask_panel(original, mask_path), mask_pred=mask_pred,
                    out_path=out / f"{label}__{Path(img).stem}.jpg",
                    n_none=n_none, n_seg=n_seg, n_heat=n_heat, n_mask=n_mask,
                )
            print(f"[{ci}/{len(cats)}] {cat}: {len(samples)} grids -> {out}", flush=True)
        else:  # val — delegate metrics to the tested OOD engine (reuses the fitted bank on the model)
            cat_rows = run_mvtec_ood_eval(
                m.model, root, categories=[cat], modes=(_PRIOR_TO_MODE[args.prior],),
                imgsz=imgsz, batch=args.batch, device=device, e2e=args.e2e, iou=args.iou,
                heatmap_norm=args.heat_norm,
                heatmap_edge_weight=(True if args.heat_edge else None),
                heatmap_edge_sigma=args.heat_edge_sigma,
            )
            r = next((x for x in cat_rows if x["category"] == cat), None)
            if r is None:
                LOGGER.warning(f"[{ci}/{len(cats)}] {cat}: no val row; skipping")
                continue
            row = {"category": cat, **{k: float(r.get(k, math.nan)) for k in VAL_METRICS}}
            rows.append(row)
            print(f"[{ci}/{len(cats)}] {cat}: " + "  ".join(f"{k}={row[k]:.4f}" for k in VAL_METRICS), flush=True)

    if args.mode == "val" and rows:
        avg = {"category": "AVERAGE", **{k: float(np.nanmean([r[k] for r in rows])) for k in VAL_METRICS}}
        rows.append(avg)
        hdr = f"{'category':12s} " + "".join(f"{k:>11s}" for k in VAL_METRICS)
        print("\n" + hdr)
        print("-" * len(hdr))
        for r in rows:
            print(f"{r['category']:12s} " + "".join(f"{r[k]:11.4f}" for k in VAL_METRICS), flush=True)
        out_csv = out_root / f"val_{args.prior}.csv"
        out_csv.parent.mkdir(parents=True, exist_ok=True)
        with open(out_csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["category", *VAL_METRICS])
            w.writeheader()
            w.writerows(rows)
        print(f"\nCSV -> {out_csv}", flush=True)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
