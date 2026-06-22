"""MVTec deploy eval — 3-mode OOD sweep over N checkpoints, per-model AVERAGE table.

For each best.pt in MODELS, run the 3-mode OOD sweep over CATS:
  mask_off = head_a honest (no prior, bare-detection floor)
  heatmap  = head + per-category memory-bank heatmap = the DEPLOY metric (no GT)
  mask_on  = GT-bbox upper bound (leaky)
then print a per-model AVERAGE table (image/pixel AUROC + mAP10/25/50/50-95). With exactly two
models it also prints Δ. Generalized from runs/temp/mvtec_deploy_eval.py (which hardcoded
refiner-vs-norefiner). imgsz=320 = fast verify; bump to 640 to match a 640-trained model.
"""
import math

import torch

from ultralytics import YOLO
from ultralytics.models.yolo.anomaly_v2.val import resolve_mvtec_root, run_mvtec_ood_eval

DEV = "mps" if torch.backends.mps.is_available() else "cpu"
IMGSZ = 320  # fast verify; use 640 to match a 640-trained model's feature resolution
CATS = ["bottle", "hazelnut", "carpet", "grid", "leather"]  # object + texture mix
OUT = "/Users/louis/workspace/ultra_louis_work/ultralytics/runs/temp/mvtec_deploy_eval_single"
MODELS = {
    "film_objcrop033": "/Users/louis/workspace/ultra_louis_work/expman/data/pulled/yoloa_v2/"
                       "26m_yoloav2_film_maskonly_aug3_mixup_objcrop033_binary_v1/weights/best.pt",
}

MVTEC = resolve_mvtec_root()
print(f"MVTEC root: {MVTEC} | device: {DEV} | imgsz: {IMGSZ}", flush=True)
assert MVTEC is not None, "MVTec root not found"

results = {}
for name, path in MODELS.items():
    print(f"\n{'='*70}\n=== {name}: {path}\n{'='*70}", flush=True)
    try:
        y = YOLO(path)
        print(f"  model class: {type(y.model).__name__}, fusion_mode="
              f"{getattr(y.model, 'fusion_mode', '?')}, two_head={getattr(y.model, 'two_head', '?')}",
              flush=True)
        rows = run_mvtec_ood_eval(
            y.model, MVTEC, categories=CATS,
            modes=("mask_off", "heatmap", "mask_on"),
            imgsz=IMGSZ, batch=8, device=DEV, bank_size=10000,
            save_dir=f"{OUT}/{name}",
        )
        results[name] = {r["mode"]: r for r in rows if r["category"] == "AVERAGE"}
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"  {name} FAILED: {type(e).__name__}: {e}", flush=True)
        results[name] = {}

# ---- comparison table (one column per model; Δ when exactly two) ----
labels = list(results.keys())
print(f"\n\n{'='*78}\nMVTec deploy AVERAGE ({len(CATS)} cats: {', '.join(CATS)})\n{'='*78}", flush=True)
MODE_LABEL = {"mask_off": "mask_off(honest floor)", "heatmap": "heatmap(DEPLOY)", "mask_on": "mask_on(GT upper)"}
hdr = f"{'mode':10s} {'metric':12s} " + "".join(f"{l:>16s}" for l in labels)
if len(labels) == 2:
    hdr += f"{'Δ(%s-%s)' % tuple(labels):>18s}"
print(hdr); print("-" * len(hdr), flush=True)
for mode in ("mask_off", "heatmap", "mask_on"):
    print(f"[{MODE_LABEL.get(mode, mode)}]", flush=True)
    for metric in ("image_auroc", "pixel_auroc", "mAP10", "mAP25", "mAP50", "mAP50_95"):
        vals = [results.get(l, {}).get(mode, {}).get(metric, math.nan) for l in labels]
        row = f"  {'':8s} {metric:12s} " + "".join(f"{v:16.4f}" for v in vals)
        if len(labels) == 2 and not any(math.isnan(v) for v in vals):
            row += f"{vals[0] - vals[1]:+18.4f}"
        print(row, flush=True)

import json
out = f"{OUT}/summary.json"
with open(out, "w") as f:
    json.dump({k: {m: dict(r) for m, r in v.items()} for k, v in results.items()}, f, indent=2)
print(f"\nsummary -> {out}\nDONE", flush=True)
