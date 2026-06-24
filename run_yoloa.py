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
import random
from pathlib import Path

import cv2
import numpy as np
import torch

from ultralytics.models.yolo.anomaly_v2.val import MVTEC_CATEGORIES, resolve_mvtec_root, run_mvtec_ood_eval
from ultralytics.utils import LOGGER, YAML
from ultralytics.yoloa import YOLOA

MVTEC_OBJECT = ["bottle", "cable", "capsule", "hazelnut", "metal_nut", "pill", "screw",
                "toothbrush", "transistor", "zipper"]
MVTEC_TEXTURE = ["carpet", "grid", "leather", "tile", "wood"]

_CAT_GROUPS = {"object": MVTEC_OBJECT, "texture": MVTEC_TEXTURE}

DEFAULT_CKPT = ("/Users/louis/workspace/ultra_louis_work/expman/data/pulled/yoloa_v2/"
                "26m_yoloav2_softhint_maskonly_aug3_mixup_binary_v1/weights/best.pt")
VAL_METRICS = ("image_auroc", "pixel_auroc", "mAP10", "mAP25", "mAP50", "mAP50_95")

# YAML heatmap_mode -> prior_mode (used by predict/val/run_mvtec_ood_eval)
_MODE_MAP = {"memory_bank": "heatmap", "learned": "heatmap_learned", "fused": "heatmap_fused"}

# YAML keys -> FeatureDiscriminatorScorer kwargs
_SCORER_YAML_KEYS = {
    "scorer_noise_std": "noise_std", "scorer_steps": "steps", "scorer_hidden": "hidden",
    "scorer_n_noise": "n_noise", "scorer_batch": "batch", "scorer_lr": "lr",
    "scorer_noise_mode": "noise_mode",
}


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


def load_fit_yaml(cfg_path: str | None) -> tuple[dict, str | None]:
    """Load and resolve the fit YAML; returns (dict, resolved_path)."""
    if not cfg_path:
        return {}, None
    p = Path(cfg_path)
    if not p.is_file() and not p.is_absolute():
        alt = Path(__file__).resolve().parent / "ultralytics" / "cfg" / p.name
        if alt.is_file():
            p = alt
    if p.is_file():
        return dict(YAML.load(str(p))), str(p)
    return {}, str(p)


def resolve_prior(yaml: dict) -> str:
    """Map YAML heatmap_mode to prior_mode string."""
    mode = yaml.get("heatmap_mode", "memory_bank")
    return _MODE_MAP.get(mode, "heatmap")


def resolve_scorer_kwargs(yaml: dict) -> dict:
    """Extract FeatureDiscriminatorScorer kwargs from YAML keys."""
    kw = {}
    for yk, kwk in _SCORER_YAML_KEYS.items():
        if yk in yaml:
            kw[kwk] = yaml[yk]
    kw.setdefault("adaptor", yaml.get("scorer_adaptor", True))
    return kw


def resolve_infer(yaml: dict) -> dict:
    """Extract prior-shaping inference knobs from YAML."""
    infer = {}
    if yaml.get("heat_edge"):
        infer["heat_edge"] = True
        infer["heat_edge_sigma"] = yaml.get("heat_edge_sigma", 1.0)
    if yaml.get("heat_norm"):
        infer["heat_norm"] = yaml["heat_norm"]
    return infer


def main():
    ap = argparse.ArgumentParser(
        description="YOLOA driver — per-category fit, then predict or val. "
                    "All fit params come from --fit-cfg YAML.")
    # -- Operational args only -------------------------------------------------
    ap.add_argument("--mode", choices=["predict", "val", "visualize"], default="predict")
    ap.add_argument("--ckpt", default=DEFAULT_CKPT)
    ap.add_argument("--cat", default="bottle",
                    help="MVTec category name, 'all' (14), 'object' (10), or 'texture' (5)")
    ap.add_argument("--prior", default="heatmap", choices=["none", "heatmap"],
                    help="none = honest floor (no prior); heatmap = use variant from fit YAML")
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--device", default=None, help="cpu / mps / 0 (default: auto mps-else-cpu)")
    ap.add_argument("--n-per-cat", type=int, default=0, help="predict: test images per category (0=all)")
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

    # Resolve prior: --prior selects none vs heatmap; YAML heatmap_mode picks the variant
    if args.prior == "none":
        prior_mode = "none"        # predict prior
        val_mode = "mask_off"       # run_mvtec_ood_eval mode
        needs_disc = False
        scorer_kwargs = {}
        scorer_fuse = "mean"
    else:
        prior_mode = resolve_prior(yaml)
        val_mode = prior_mode       # same string: heatmap / heatmap_learned / heatmap_fused
        needs_disc = prior_mode in ("heatmap_learned", "heatmap_fused")
        scorer_kwargs = resolve_scorer_kwargs(yaml)
        scorer_fuse = yaml.get("scorer_fuse", "mean")
        scorer_weight = yaml.get("scorer_weight", 0.5)
        scorer_kwargs["scorer_weight"] = scorer_weight

    infer = resolve_infer(yaml)

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

    m = YOLOA(args.ckpt)

    # fit_over: imgsz + max_images from YAML (no CLI overrides anymore)
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
          f"prior: {prior_mode} | heat_edge: {infer.get('heat_edge', False)} "
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
                r = m.predict(img, prior=prior_mode, imgsz=imgsz, conf=args.conf,
                              iou=args.iou, device=device, end2end=args.e2e,
                              verbose=False, **infer)[0]
                stem = f"{label}__{Path(img).stem}"
                cv2.imwrite(str(out / f"{stem}__pred.jpg"), r.plot())
                save_heatmap(m.model, img, out / f"{stem}__heat.jpg")
            print(f"[{ci}/{len(cats)}] {cat}: {len(samples)} predictions -> {out}", flush=True)
        elif args.mode == "visualize":
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
        else:  # val
            cat_rows = run_mvtec_ood_eval(
                m.model, root, categories=[cat], modes=(val_mode,),
                imgsz=imgsz, batch=args.batch, device=device, e2e=args.e2e, iou=args.iou,
                heatmap_norm=infer.get("heat_norm", "none"),
                heatmap_edge_weight=(True if infer.get("heat_edge") else None),
                heatmap_edge_sigma=infer.get("heat_edge_sigma", 1.0),
                scorer_kwargs=scorer_kwargs if needs_disc else None,
                scorer_fuse=scorer_fuse,
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
        out_csv = out_root / f"val_{prior_mode}.csv"
        out_csv.parent.mkdir(parents=True, exist_ok=True)
        with open(out_csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["category", *VAL_METRICS])
            w.writeheader()
            w.writerows(rows)
        print(f"\nCSV -> {out_csv}", flush=True)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
