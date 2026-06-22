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
    ap.add_argument("--modes", type=str, nargs="+", default=["mask_off", "heatmap", "mask_on"],
                    choices=["mask_off", "heatmap", "mask_on"])
    ap.add_argument("--mvtec-root", type=str, default=None, help="MVTec-YOLO root (default: auto-resolve)")
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
    print(f"MVTEC root: {root} | device: {device} | imgsz: {args.imgsz} | cats({len(cats)}): {', '.join(cats)}",
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
                save_dir=f"{args.out}/{name}",
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
