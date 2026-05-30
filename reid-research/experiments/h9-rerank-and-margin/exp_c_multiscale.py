"""Experiment C: multi-scale single-model training from champion init.

Subclasses ReidTrainer to randomly choose imgsz per batch from {384, 416, 448}.
Goal: internalize multi-scale geometry into one model (single-shot inference)
to recover the +2.47pp gain seen from the champion+tx2 concat ensemble.

Init from champion best.pt; 40 epochs at lr0=5e-4 cosine; same other args as champion recipe.
"""
from __future__ import annotations
import os, time, json, random, sys
from pathlib import Path
import torch
import torch.nn.functional as F

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "3")

from ultralytics import YOLO
from ultralytics.models.yolo.reid.train import ReidTrainer


class MultiScaleReidTrainer(ReidTrainer):
    """Per-batch random imgsz from a discrete set."""
    SCALES = [384, 416, 448]

    def preprocess_batch(self, batch):
        batch = super().preprocess_batch(batch)
        sz = random.choice(self.SCALES)
        if batch["img"].shape[-1] != sz:
            batch["img"] = F.interpolate(batch["img"], size=(sz, sz), mode="bilinear", align_corners=False)
        return batch


CHAMP_CKPT = "/root/autodl-tmp/ultralytics_reid/runs/reid/runs/reid/runs/arch31_imgsz384/weights/best.pt"
DATA = "ultralytics/cfg/datasets/Market-1501.yaml"
OUT = Path("/root/expc")
OUT.mkdir(parents=True, exist_ok=True)

print(">>> loading champion")
m = YOLO(CHAMP_CKPT, task="reid")
print(f"    params: {sum(p.numel() for p in m.model.parameters())/1e6:.1f}M")

print("\n>>> baseline val at imgsz=384 (no TTA/rerank)")
b384 = m.val(data=DATA, imgsz=384, batch=64, reid_tta=False, reid_reranking=False)
print(f"    @384  R1={b384.rank1:.4f} mAP={b384.mAP:.4f}")
print("\n>>> baseline val at imgsz=448")
b448 = m.val(data=DATA, imgsz=448, batch=64, reid_tta=False, reid_reranking=False)
print(f"    @448  R1={b448.rank1:.4f} mAP={b448.mAP:.4f}")

t0 = time.time()
print("\n>>> training multi-scale (scales=384,416,448) for 40 epochs from champion init")
m.train(
    trainer=MultiScaleReidTrainer,
    data=DATA,
    imgsz=448,                # max scale; the model will see 384/416/448 mix
    batch=64,
    epochs=40,
    name="expc_multiscale_384_416_448",
    project="runs/reid/runs",
    exist_ok=True,
    optimizer="AdamW",
    lr0=5e-4,
    lrf=0.01,
    weight_decay=0.01,
    warmup_epochs=2,
    cos_lr=True,
    amp=False,
    reid_p=8, reid_k=8,
    triplet_margin=0.5,
    triplet_weight=1.0,
    ce_weight=2.0,
    center_weight=0.005,
    supcon_temp=0.02,
    label_smoothing=0.1,
    arcface=False,
    erasing=0.7,
    fliplr=0.5,
    hsv_h=0.015, hsv_s=0.7, hsv_v=0.4,
    auto_augment="randaugment",
    scale=0.5,
    val=False,
    seed=0,
    workers=4,
    dropout=0.0,
)
elapsed_min = (time.time() - t0) / 60
print(f"\n>>> training done in {elapsed_min:.1f} min")

# Eval new model at both scales
print("\n>>> eval new model at imgsz=384")
n384 = m.val(data=DATA, imgsz=384, batch=64, reid_tta=False, reid_reranking=False)
print(f"    new @384 R1={n384.rank1:.4f} mAP={n384.mAP:.4f}")
print("\n>>> eval new model at imgsz=448")
n448 = m.val(data=DATA, imgsz=448, batch=64, reid_tta=False, reid_reranking=False)
print(f"    new @448 R1={n448.rank1:.4f} mAP={n448.mAP:.4f}")

# TTA+rerank for headline
print("\n>>> eval at imgsz=384 + TTA + rerank")
n384tta = m.val(data=DATA, imgsz=384, batch=64, reid_tta=True, reid_reranking=True)
print(f"    new @384 TTA+rerank R1={n384tta.rank1:.4f} mAP={n384tta.mAP:.4f}")
print("\n>>> eval at imgsz=448 + TTA + rerank")
n448tta = m.val(data=DATA, imgsz=448, batch=64, reid_tta=True, reid_reranking=True)
print(f"    new @448 TTA+rerank R1={n448tta.rank1:.4f} mAP={n448tta.mAP:.4f}")

result = {
    "baseline_champion_384": {"r1": float(b384.rank1), "mAP": float(b384.mAP)},
    "baseline_champion_448": {"r1": float(b448.rank1), "mAP": float(b448.mAP)},
    "new_no_tta_384":  {"r1": float(n384.rank1),  "mAP": float(n384.mAP)},
    "new_no_tta_448":  {"r1": float(n448.rank1),  "mAP": float(n448.mAP)},
    "new_tta_rerank_384": {"r1": float(n384tta.rank1), "mAP": float(n384tta.mAP)},
    "new_tta_rerank_448": {"r1": float(n448tta.rank1), "mAP": float(n448tta.mAP)},
    "wall_clock_min": elapsed_min,
}
with open(OUT / "result.json", "w") as f:
    json.dump(result, f, indent=2)
print("\n=== SUMMARY ===")
print(f"  champion @384 (baseline)  R1={b384.rank1:.4f}")
print(f"  champion @448 (baseline)  R1={b448.rank1:.4f}")
print(f"  TX2 @448 (reference)      R1=0.9044 (separately trained)")
print(f"  new @384                  R1={n384.rank1:.4f}  delta vs champion={n384.rank1-b384.rank1:+.4f}")
print(f"  new @448                  R1={n448.rank1:.4f}  delta vs champion={n448.rank1-b448.rank1:+.4f}")
print(f"  new @384 TTA+rerank       R1={n384tta.rank1:.4f}")
print(f"  new @448 TTA+rerank       R1={n448tta.rank1:.4f}")
