#!/usr/bin/env python
"""MVTec deploy eval — 3-mode OOD sweep over N checkpoints, per-model AVERAGE table.

For each --ckpt, run the 3-mode OOD sweep over the chosen categories:
  mask_off = head_a honest (no prior, bare-detection floor)
  heatmap  = head + per-category memory-bank heatmap = the DEPLOY metric (no GT)
  mask_on  = GT-bbox upper bound (leaky)
then print a per-model AVERAGE table (image/pixel AUROC + mAP10/25/50/50-95). With exactly two
models it also prints Δ. Per-category rows + AVERAGE land in <out>/<label>/mvtec_ood.csv; the
table is mirrored to <out>/summary.json.

Usage:
  python mvtec_deploy_eval.py --ckpt runs/yoloa_v2/<run>/weights/best.pt
  python mvtec_deploy_eval.py --ckpt A/best.pt B/best.pt --categories bottle carpet --imgsz 640
  python mvtec_deploy_eval.py --ckpt <best.pt> --categories all          # all 15 MVTec categories
  python mvtec_deploy_eval.py --ckpt A/best.pt B/best.pt --name refiner norefiner
  python mvtec_deploy_eval.py --ckpt <best.pt> --heat-edge                 # edge-suppression weight on
  python mvtec_deploy_eval.py --ckpt <best.pt> --heat-edge --heat-edge-sigma 1.2   # gentler edges

To A/B the edge weight on one checkpoint, run twice (with and without --heat-edge) and compare the
heatmap(DEPLOY) rows; the weight only touches that prior, so mask_off / mask_on are unchanged.
"""
import argparse
import json
import math
from pathlib import Path

import torch

from ultralytics import YOLO
from ultralytics.models.yolo.anomaly_v2.val import MVTEC_CATEGORIES, resolve_mvtec_root, run_mvtec_ood_eval

DEFAULT_CATS = ["bottle", "hazelnut", "carpet", "grid", "leather"]  # object + texture mix
MODE_LABEL = {"mask_off": "mask_off(honest floor)", "heatmap": "heatmap(DEPLOY)", "mask_on": "mask_on(GT upper)"}
METRICS = ("image_auroc", "pixel_auroc", "mAP10", "mAP25", "mAP50", "mAP50_95")


def label_for(ckpt: str) -> str:
    """Short label from a ckpt path: <run>/weights/best.pt -> <run>, else the file stem."""
    p = Path(ckpt).resolve()
    return p.parents[1].name if p.parent.name == "weights" else p.stem


def main():
    ap = argparse.ArgumentParser(description="MVTec deploy eval — 3-mode OOD sweep, per-model AVERAGE table")
    ap.add_argument("--ckpt", type=str, nargs="+", required=True,
                    help="One or more best.pt paths (labels auto-derived from the run-dir name)")
    ap.add_argument("--name", type=str, nargs="+", default=None,
                    help="Run name per --ckpt (table column header); order matches --ckpt "
                         "(default: derived from each run-dir name)")
    ap.add_argument("--categories", type=str, nargs="+", default=DEFAULT_CATS,
                    help=f"MVTec categories, or 'all' for all 15 (default: {' '.join(DEFAULT_CATS)})")
    ap.add_argument("--imgsz", type=int, default=320, help="320 = fast verify; 640 matches a 640-trained model")
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--device", type=str, default=None, help="cpu / mps / 0 / cuda:0 (default: auto mps-else-cpu)")
    ap.add_argument("--bank-size", type=int, default=10000, help="memory-bank cap for the heatmap prior")
    ap.add_argument("--e2e", action="store_true",
                    help="Use the end-to-end NMS-free head. Default OFF -> one2many head + NMS, so --iou "
                         "merges/suppresses nearby boxes (the e2e head ignores --iou).")
    ap.add_argument("--iou", type=float, default=0.7,
                    help="NMS IoU threshold (only active when --e2e is off). Lower = more box merging.")
    ap.add_argument("--heat-norm", type=str, default=None, choices=["none", "minmax", "gaussian", "mean"],
                    help="Prior processing before fusion: minmax (stretch to [0,1]) | gaussian/mean (blur). "
                         "Affects the fused prior -> mAP; AUROC uses the raw heatmap so it is unchanged.")
    ap.add_argument("--heat-smooth-kernel", type=int, default=5,
                    help="Kernel size for --heat-norm gaussian/mean blur (odd; default 5)")
    ap.add_argument("--heat-edge", action="store_true",
                    help="Multiply the memory-bank heatmap by a fixed squircle-Gaussian center window "
                         "(1 at center, decaying to borders) to suppress peripheral noise. Applies to the "
                         "'heatmap' prior only; unlike --heat-norm it is applied BEFORE the AUROC stash, "
                         "so it moves AUROC too.")
    ap.add_argument("--heat-edge-sigma", type=float, default=1.0,
                    help="--heat-edge: edge value / transition (bigger = gentler; default 1.0 -> edge-mid ~0.61)")
    ap.add_argument("--heat-edge-m", type=float, default=4.4, help="--heat-edge: center plateau steepness")
    ap.add_argument("--heat-edge-p", type=float, default=4.0, help="--heat-edge: shape (2=circle, 4=squircle, >=8 square)")
    ap.add_argument("--modes", type=str, nargs="+", default=["mask_off", "heatmap", "mask_on"],
                    choices=["mask_off", "heatmap", "mask_on"])
    ap.add_argument("--mvtec-root", type=str, default=None, help="MVTec-YOLO root (default: auto-resolve)")
    ap.add_argument("--bank-cache", type=str, default=None,
                    help="Dir to cache+reuse per-category memory banks (disk), so re-runs and the "
                         "edge OFF/ON A/B skip the slow rebuild. Off by default (rebuild per run). "
                         "Keyed by imgsz+bank-size, so changing those is safe.")
    ap.add_argument("--out", type=str, default="runs/temp/mvtec_deploy_eval",
                    help="output dir for per-model CSV + summary.json")
    args = ap.parse_args()

    device = args.device or ("mps" if torch.backends.mps.is_available() else "cpu")
    cats = (MVTEC_CATEGORIES if len(args.categories) == 1 and args.categories[0].lower() == "all"
            else args.categories)

    # Resolve names (explicit --name or derived), de-duplicate so two runs sharing a stem stay distinct.
    labels = args.name or [label_for(c) for c in args.ckpt]
    if len(labels) != len(args.ckpt):
        ap.error(f"--name ({len(labels)}) must match --ckpt count ({len(args.ckpt)})")
    seen, uniq = {}, []
    for l in labels:
        seen[l] = seen.get(l, 0) + 1
        uniq.append(l if seen[l] == 1 else f"{l}#{seen[l]}")
    models = dict(zip(uniq, args.ckpt))

    root = resolve_mvtec_root(args.mvtec_root)
    assert root is not None, "MVTec root not found (pass --mvtec-root or set MVTEC_ROOT)"
    hn = args.heat_norm or "none"
    hn_str = f"{hn}(k={args.heat_smooth_kernel})" if hn in ("gaussian", "mean") else hn
    edge_str = f"on(sigma={args.heat_edge_sigma},m={args.heat_edge_m},p={args.heat_edge_p})" if args.heat_edge else "off"
    print(f"MVTEC root: {root} | device: {device} | imgsz: {args.imgsz} | "
          f"e2e: {args.e2e} | iou: {args.iou if not args.e2e else 'n/a (e2e ignores iou)'} | "
          f"heat-norm: {hn_str} | heat-edge: {edge_str} | cats({len(cats)}): {', '.join(cats)}",
          flush=True)

    results = {}
    for name, path in models.items():
        print(f"\n{'='*70}\n=== {name}: {path}\n{'='*70}", flush=True)
        try:
            y = YOLO(path)
            print(f"  model class: {type(y.model).__name__}, fusion_mode="
                  f"{getattr(y.model, 'fusion_mode', '?')}, two_head={getattr(y.model, 'two_head', '?')}",
                  flush=True)
            rows = run_mvtec_ood_eval(
                y.model, root, categories=cats, modes=tuple(args.modes),
                imgsz=args.imgsz, batch=args.batch, device=device, bank_size=args.bank_size,
                save_dir=f"{args.out}/{name}", e2e=args.e2e, iou=args.iou,
                heatmap_norm=args.heat_norm, heatmap_smooth_kernel=args.heat_smooth_kernel,
                heatmap_edge_weight=args.heat_edge or None,
                heatmap_edge_sigma=args.heat_edge_sigma,
                heatmap_edge_m=args.heat_edge_m, heatmap_edge_p=args.heat_edge_p,
                bank_cache_dir=args.bank_cache,
            )
            results[name] = {r["mode"]: r for r in rows if r["category"] == "AVERAGE"}
        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f"  {name} FAILED: {type(e).__name__}: {e}", flush=True)
            results[name] = {}

    # ---- comparison table (one column per model; Δ when exactly two) ----
    cols = list(results.keys())
    print(f"\n\n{'='*78}\nMVTec deploy AVERAGE ({len(cats)} cats: {', '.join(cats)})\n{'='*78}", flush=True)
    hdr = f"{'mode':10s} {'metric':12s} " + "".join(f"{l:>16s}" for l in cols)
    if len(cols) == 2:
        hdr += f"{'d(%s-%s)' % tuple(cols):>18s}"
    print(hdr); print("-" * len(hdr), flush=True)
    for mode in args.modes:
        print(f"[{MODE_LABEL.get(mode, mode)}]", flush=True)
        for metric in METRICS:
            vals = [results.get(l, {}).get(mode, {}).get(metric, math.nan) for l in cols]
            row = f"  {'':8s} {metric:12s} " + "".join(f"{v:16.4f}" for v in vals)
            if len(cols) == 2 and not any(math.isnan(v) for v in vals):
                row += f"{vals[0] - vals[1]:+18.4f}"
            print(row, flush=True)

    Path(args.out).mkdir(parents=True, exist_ok=True)
    out = f"{args.out}/summary.json"
    with open(out, "w") as f:
        json.dump({k: {m: dict(r) for m, r in v.items()} for k, v in results.items()}, f, indent=2)
    print(f"\nsummary -> {out}\nDONE", flush=True)


if __name__ == "__main__":
    main()
