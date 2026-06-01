"""Experiment D v2: from-MSMT multi-scale, 100 epochs, save every 10.

Shorter schedule (champion-equivalent) + periodic saves. Earlier 635-epoch attempt
was at 70s/epoch (~10h ETA) with no intermediate checkpoints visible; this version
finishes in ~2h on 4-GPU DDP and gives eval-able snapshots throughout.
"""
import os, time, json
from pathlib import Path
from ultralytics import YOLO

YAML = "ultralytics/cfg/models/26/yolo26l-reid-2psa.yaml"
MSMT_PRETRAIN = "/root/autodl-tmp/ultralytics_reid/runs/reid/yolo26l_2psa_msmt/weights/best.pt"
DATA = "ultralytics/cfg/datasets/Market-1501.yaml"
OUT = Path("/root/expd_v2")
OUT.mkdir(parents=True, exist_ok=True)

m = YOLO(YAML, task="reid")
t0 = time.time()
m.train(
    data=DATA,
    imgsz=448,
    batch=192,
    epochs=100,
    name="expd_v2_multiscale",
    project="runs/reid/runs",
    exist_ok=True,
    pretrained=MSMT_PRETRAIN,
    device=[0, 1, 2, 3],
    optimizer="AdamW",
    lr0=3.5e-3, lrf=0.001, weight_decay=0.01,
    warmup_epochs=3, cos_lr=True, amp=False,
    reid_p=24, reid_k=8,
    triplet_margin=0.5, triplet_weight=1.0, ce_weight=2.0,
    center_weight=0.005, supcon_temp=0.02,
    label_smoothing=0.1, arcface=False,
    erasing=0.7, fliplr=0.5,
    hsv_h=0.015, hsv_s=0.7, hsv_v=0.4,
    auto_augment="randaugment", scale=0.5,
    val=False, seed=0, workers=4, dropout=0.0,
    save_period=10,                    # KEY: snapshot every 10 epochs
)
elapsed = (time.time() - t0) / 60
print(f"\n>>> training done in {elapsed:.1f} min")

os.environ.pop("REID_MS_SCALES", None)
results = {"wall_clock_min": elapsed}
for imgsz in (384, 448):
    r = m.val(data=DATA, imgsz=imgsz, batch=64, reid_tta=False, reid_reranking=False)
    results[f"no_tta_{imgsz}"] = {"r1": float(r.rank1), "mAP": float(r.mAP)}
    print(f"  @{imgsz} no-TTA  R1={r.rank1:.4f}  mAP={r.mAP:.4f}")
for imgsz in (384, 448):
    r = m.val(data=DATA, imgsz=imgsz, batch=64, reid_tta=True, reid_reranking=True)
    results[f"tta_rerank_{imgsz}"] = {"r1": float(r.rank1), "mAP": float(r.mAP)}
    print(f"  @{imgsz} TTA+rerank  R1={r.rank1:.4f}  mAP={r.mAP:.4f}")

with open(OUT / "result.json", "w") as f:
    json.dump(results, f, indent=2)
print("\n=== SUMMARY (vs reference) ===")
print(f"  champion @384 raw      R1=0.8996  (TTA+rerank=0.9267)")
print(f"  TX2 @448 raw           R1=0.9044  (TTA+rerank=0.9311)")
print(f"  champ+TX2 concat       R1=0.9243  (raw, 2-model — what we're chasing)")
print(f"  NEW @384               R1={results['no_tta_384']['r1']:.4f}  (TTA+rerank={results['tta_rerank_384']['r1']:.4f})")
print(f"  NEW @448               R1={results['no_tta_448']['r1']:.4f}  (TTA+rerank={results['tta_rerank_448']['r1']:.4f})")
