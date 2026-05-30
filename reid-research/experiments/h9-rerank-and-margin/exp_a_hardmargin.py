"""Experiment A: fine-tune champion with triplet_margin=0.05 to focus loss on near-miss pairs.

Resumes from arch31_imgsz384/weights/best.pt. Small triplet margin shifts loss pressure
onto exactly the borderline cosine band where 97% of recoverable failures live
(champion margin in [-0.10, 0]).

Trains 30 additional epochs; evals against champion baseline (R1=0.8996 no-TTA).
"""
from __future__ import annotations
import os, time, json, sys
from pathlib import Path
from ultralytics import YOLO

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "3")

CHAMP_CKPT = "/root/autodl-tmp/ultralytics_reid/runs/reid/runs/reid/runs/arch31_imgsz384/weights/best.pt"
DATA = "ultralytics/cfg/datasets/Market-1501.yaml"
OUT = Path("/root/expa")
OUT.mkdir(parents=True, exist_ok=True)

print(">>> loading champion from", CHAMP_CKPT)
m = YOLO(CHAMP_CKPT, task="reid")
print(f"    params: {sum(p.numel() for p in m.model.parameters())/1e6:.1f}M")

# Baseline (no rerank, no TTA) — should be ~0.8996
print("\n>>> baseline val")
base = m.val(data=DATA, imgsz=384, batch=64, reid_tta=False, reid_reranking=False)
print(f"    baseline R1={base.rank1:.4f} mAP={base.mAP:.4f}")

t0 = time.time()
print("\n>>> training with triplet_margin=0.05")
# Args mirror reid_best.yaml but with smaller triplet margin and shorter schedule.
m.train(
    data=DATA,
    imgsz=384,
    batch=64,
    epochs=30,
    name="expa_margin005",
    project="runs/reid/runs",
    exist_ok=True,
    # Optimizer + schedule
    optimizer="AdamW",
    lr0=5e-4,
    lrf=0.01,
    weight_decay=0.01,
    warmup_epochs=2,
    cos_lr=True,
    amp=False,
    # Sampler (P=8, K=8 = batch 64)
    reid_p=8,
    reid_k=8,
    # Loss weights
    triplet_margin=0.05,   # <-- the lever
    triplet_weight=1.0,
    ce_weight=2.0,
    center_weight=0.005,
    supcon_temp=0.02,
    label_smoothing=0.1,
    arcface=False,
    # Augmentation (same as champion)
    erasing=0.7,
    fliplr=0.5,
    hsv_h=0.015,
    hsv_s=0.7,
    hsv_v=0.4,
    auto_augment="randaugment",
    scale=0.5,
    val=False,  # val separately at the end
    seed=0,
    workers=4,
    dropout=0.0,
)
print(f"\n>>> training done in {(time.time()-t0)/60:.1f} min")

# Eval new model
print("\n>>> final val")
new = m.val(data=DATA, imgsz=384, batch=64, reid_tta=False, reid_reranking=False)
print(f"    new      R1={new.rank1:.4f} mAP={new.mAP:.4f}")
print(f"    baseline R1={base.rank1:.4f}  delta R1 = {new.rank1 - base.rank1:+.4f}")

# Also TTA+rerank for the standard headline number
print("\n>>> final val with flip TTA + k-reciprocal rerank")
new_tta = m.val(data=DATA, imgsz=384, batch=64, reid_tta=True, reid_reranking=True)
print(f"    new (TTA+rerank) R1={new_tta.rank1:.4f} mAP={new_tta.mAP:.4f}")

with open(OUT / "result.json", "w") as f:
    json.dump({
        "baseline_no_tta": {"r1": float(base.rank1), "mAP": float(base.mAP)},
        "new_no_tta": {"r1": float(new.rank1), "mAP": float(new.mAP)},
        "new_tta_rerank": {"r1": float(new_tta.rank1), "mAP": float(new_tta.mAP)},
        "delta_r1_no_tta": float(new.rank1 - base.rank1),
        "wall_clock_min": (time.time() - t0) / 60,
    }, f, indent=2)
