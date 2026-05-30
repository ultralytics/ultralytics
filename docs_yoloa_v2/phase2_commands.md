# Phase 2 (v2.2) — SegBranch ultra6 Training Commands

Run on ultra6 (`ssh ultra6`), repo at `~/ultra_louis_work/ultralytics/`.
Make sure: (1) `conda activate ultra`, (2) `set_wandb_true`, (3) branch `yoloa_v2`
checked out and synced (git bundle, **never push**), (4) git clean before launching.

## What's new vs Phase 0

A lightweight `SegBranch` (semantic-seg head adapted from upstream `SemanticSegment`)
predicts the anomaly heatmap from the P3/P4 PAN features. It is supervised by the
rect-rendered GT mask (BCE + Dice, `seg_gain=1.0`). An alpha curriculum blends the
fusion prior from GT → prediction:

```
seg_alpha = 1.0  at epoch 0   (pure GT mask  ≡ Phase 0)
          → 0.0  at epoch (epochs - close_mosaic) = 30   (pure prediction)
          = 0.0  thereafter
```

Because the **mask-on validation pass uses the current `seg_alpha`**, the
`metrics/mAP50-95(B)` column morphs over training:
- early epochs → GT-prior upper bound (Phase 0 semantics)
- **final epochs (alpha=0) → real prior-free inference** (what deploys)

The `mask_off/*` column stays a pure passthrough (vanilla) floor as before.
New loss column: `seg_loss` (BCE+Dice on the predicted heatmap).

Three runs form an α-curriculum ablation set; they differ ONLY in `model=`:

| YAML | `seg_alpha_mode` | Reads as |
|---|---|---|
| `yolo26m-anomaly-v2-seg.yaml`    | `curriculum` (default) | main: GT→pred anneal |
| `yolo26m-anomaly-v2-seg-a1.yaml` | `pinned_one`           | α=1 全程 (GT-only fusion; non-destructive check) |
| `yolo26m-anomaly-v2-seg-a0.yaml` | `pinned_zero`          | α=0 全程 (pred-only fusion; necessity check) |

## 1. Main — alpha curriculum (1.0 → 0.0)

```
nohupyolo train task=anomaly_v2 \
  model=yolo26m-anomaly-v2-seg.yaml \
  pretrained=yolo26m.pt \
  data=/home/louis/ultra_louis_work/datasets/AnomalyDataset/merge_data_v5_binary/data.yaml \
  epochs=50 batch=96 close_mosaic=20 device=0,1 \
  optimizer=MuSGD lr0=0.00125 lrf=0.5 momentum=0.9 weight_decay=0.0005 \
  scale=0.1 copy_paste=0.1 mixup=0.0 save_json=True \
  project=yoloa_v2 name=26m_yoloav2seg_v5_binary_cm20_rect_pd50_acur_v1
```

## 2. Ablation — alpha pinned at 1.0 (GT only)

Expected: mask-on column ≈ Phase 0 `rect_pd50` throughout; `seg_loss` still trains
(detached from detection). If this drops vs Phase 0, the SegBranch additions are
destructive and run 1 results are suspect.

```
nohupyolo train task=anomaly_v2 \
  model=yolo26m-anomaly-v2-seg-a1.yaml \
  pretrained=yolo26m.pt \
  data=/home/louis/ultra_louis_work/datasets/AnomalyDataset/merge_data_v5_binary/data.yaml \
  epochs=50 batch=96 close_mosaic=20 device=2,3 \
  optimizer=MuSGD lr0=0.00125 lrf=0.5 momentum=0.9 weight_decay=0.0005 \
  scale=0.1 copy_paste=0.1 mixup=0.0 save_json=True \
  project=yoloa_v2 name=26m_yoloav2seg_v5_binary_cm20_rect_pd50_a1_v1
```

## 3. Ablation — alpha pinned at 0.0 (prediction only)

Expected: mask-on column starts near mask-off floor (random heatmap), climbs as
`seg_loss` falls. If the final mask-on number matches run 1's, the curriculum
adds nothing; if run 1 is clearly higher, the curriculum earns its keep.

```
nohupyolo train task=anomaly_v2 \
  model=yolo26m-anomaly-v2-seg-a0.yaml \
  pretrained=yolo26m.pt \
  data=/home/louis/ultra_louis_work/datasets/AnomalyDataset/merge_data_v5_binary/data.yaml \
  epochs=50 batch=96 close_mosaic=20 device=4,5 \
  optimizer=MuSGD lr0=0.00125 lrf=0.5 momentum=0.9 weight_decay=0.0005 \
  scale=0.1 copy_paste=0.1 mixup=0.0 save_json=True \
  project=yoloa_v2 name=26m_yoloav2seg_v5_binary_cm20_rect_pd50_a0_v1
```

## How to read results

`runs/yoloa_v2/<name>/results.csv`:

| Column | Meaning |
|---|---|
| `metrics/mAP50-95(B)` | mask-on pass at current `seg_alpha`. **Final epoch = prior-free seg-pred inference.** |
| `mask_off/metrics/mAP50-95(B)` | passthrough (vanilla) floor. |
| `train/seg_loss`, `val/seg_loss` | SegBranch BCE+Dice. Should fall steadily; plateau = heatmap learned. |

### Success criteria for Phase 2

| Metric | Source | Expected |
|---|---|---|
| final mask-on mAP50-95 (alpha=0) | `metrics/mAP50-95(B)` | **> mask_off floor** — the predicted heatmap helps with no GT prior |
| final mask-off mAP50-95 | `mask_off/metrics/mAP50-95(B)` | ≈ vanilla baseline |
| early-epoch mask-on (alpha≈1) | `metrics/mAP50-95(B)` | ≈ Phase 0 `rect_pd50` mask-on (non-destructive check) |

If the final mask-on (alpha=0) sits well above the mask-off floor, the SegBranch
delivers a usable prior at inference time without any external mask — the v2.2 goal.

## 4. Extra — gauss render + alpha curriculum (GPU 4,7)

Same as run 1 but `mask_mode: gauss`. Tests whether the Phase 0 advantage of
gauss-rendered GT mask over rect carries into SegBranch. Curriculum default.

```
nohupyolo train task=anomaly_v2 \
  model=yolo26m-anomaly-v2-seg-gauss.yaml \
  pretrained=yolo26m.pt \
  data=/home/louis/ultra_louis_work/datasets/AnomalyDataset/merge_data_v5_binary/data.yaml \
  epochs=50 batch=96 close_mosaic=20 device=4,7 \
  optimizer=MuSGD lr0=0.00125 lrf=0.5 momentum=0.9 weight_decay=0.0005 \
  scale=0.1 copy_paste=0.1 mixup=0.0 save_json=True \
  project=yoloa_v2 name=26m_yoloav2seg_v5_binary_cm20_gauss_pd50_acur_v1
```

## 5. Diagnostic — remove detach, let det loss train SegBranch (GPU 0,1)

Phase 2 finished with prior-free inference failing (a0/acur ≈ off floor). The probe
(`scripts/probe_segbranch_pred.py`) showed sigmoid(pred) has full dynamic range
(max≈1, bg≈0) but localizes poorly — SegBranch learned to "match a rect", not to
"produce a heatmap that helps detection". Root cause: pred is detached before
fusion, so det loss never trains SegBranch.

This run flips `seg_detach: false` so det loss flows through fusion -> SegBranch.

```
nohupyolo train task=anomaly_v2 \
  model=yolo26m-anomaly-v2-seg-graddet.yaml \
  pretrained=yolo26m.pt \
  data=/home/louis/ultra_louis_work/datasets/AnomalyDataset/merge_data_v5_binary/data.yaml \
  epochs=50 batch=96 close_mosaic=20 device=0,1 \
  optimizer=MuSGD lr0=0.00125 lrf=0.5 momentum=0.9 weight_decay=0.0005 \
  scale=0.1 copy_paste=0.1 mixup=0.0 save_json=True \
  project=yoloa_v2 name=26m_yoloav2seg_v5_binary_cm20_rect_pd50_acur_gd_v1
```

**Read result:** compare final mask-on (alpha=0) mAP50-95 against `..._rect_pd50_acur_v1`
(0.6140). If clearly higher and approaches the a1 ceiling (0.6851), the detach was
the root cause and SegBranch can be trained end-to-end.

## 6. Diagnostic — graddet + lower seg_gain (GPU 4,5,6,7)

Companion ablation to run 5: same `seg_detach: false`, but `seg_gain: 0.3` instead
of 1.0. Tests whether seg_loss (match rect) was too dominant and blocking det_loss
from pulling pred toward detection-useful shapes.

```
nohupyolo train task=anomaly_v2 \
  model=yolo26m-anomaly-v2-seg-graddet-sg03.yaml \
  pretrained=yolo26m.pt \
  data=/home/louis/ultra_louis_work/datasets/AnomalyDataset/merge_data_v5_binary/data.yaml \
  epochs=50 batch=96 close_mosaic=20 device=4,5,6,7 \
  optimizer=MuSGD lr0=0.00125 lrf=0.5 momentum=0.9 weight_decay=0.0005 \
  scale=0.1 copy_paste=0.1 mixup=0.0 save_json=True \
  project=yoloa_v2 name=26m_yoloav2seg_v5_binary_cm20_rect_pd50_acur_gd_sg03_v1
```

**Read result:** direct comparison against `..._acur_gd_v1` (sg=1.0). If sg=0.3
mask-on > sg=1.0 mask-on → seg_loss was too dominant; if ≈ → seg_gain not the
bottleneck; if < → seg_loss anchoring matters, don't lower further.

## Monitor

```
python ~/.claude/skills/yoloa/scripts/yoloa_status.py        # project=yoloa_v2
tail -f runs/yoloa_v2/26m_yoloav2seg_v5_binary_cm20_rect_pd50_acur_v1.log
```
