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

## 1. Main run — rect supervision, p_drop=0.5, linear alpha curriculum

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

## 2. Sanity — alpha pinned at 1.0 (should ≈ Phase 0 rect_pd50)

Verifies that adding the SegBranch + its loss does not degrade the GT-prior path.
Pin alpha by setting `close_mosaic=0` is NOT enough (alpha still anneals over epochs);
instead this is a code-level check — easiest is to compare the **early-epoch** mask-on
numbers of run 1 against `26m_yoloav2_v5_binary_cm20_rect_pd50_v1`. If they track, the
seg additions are non-destructive. (Skip a dedicated run unless run 1 looks off.)

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

## Monitor

```
python ~/.claude/skills/yoloa/scripts/yoloa_status.py        # project=yoloa_v2
tail -f runs/yoloa_v2/26m_yoloav2seg_v5_binary_cm20_rect_pd50_acur_v1.log
```
