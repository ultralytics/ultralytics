# Phase 0 — ultra6 Training Commands

Run on ultra6 (`ssh ultra6`), repo at `~/ultra_louis_work/ultralytics/`.
Make sure: (1) `conda activate ultra`, (2) `set_wandb_true`, (3) the branch
`yoloa_v2` is checked out, (4) git is clean before launching.

Each run:
- 50 epochs, batch 96, 4 GPUs DDP
- Identical hparams to your existing baseline `26m_yolo_v5_binary_cm20_v1`
- Different `model=` YAML (controls mask_mode and p_drop)
- All saved to `project=yoloa_v2` so they're separate from your `yoloa` baseline

Baseline reference (already trained, NOT re-run):
```
26m_yolo_v5_binary_cm20_v1     (project=yoloa)
```

---

## 1. Primary B-on (rect mask, p_drop=0.5) — main result

```
nohupyolo train task=anomaly_v2 \
  model=yolo26m-anomaly-v2.yaml \
  pretrained=yolo26m.pt \
  data=/home/louis/ultra_louis_work/datasets/AnomalyDataset/merge_data_v5_binary/data.yaml \
  epochs=50 batch=96 close_mosaic=20 device=0,1,2,3 \
  optimizer=MuSGD lr0=0.00125 lrf=0.5 momentum=0.9 weight_decay=0.0005 \
  scale=0.1 copy_paste=0.1 mixup=0.0 save_json=True \
  project=yoloa_v2 name=26m_yoloav2_v5_binary_cm20_rect_pd50_v1
```

## 2. Primary B-on (gauss mask, p_drop=0.5) — render-mode ablation

```
nohupyolo train task=anomaly_v2 \
  model=yolo26m-anomaly-v2-gauss.yaml \
  pretrained=yolo26m.pt \
  data=/home/louis/ultra_louis_work/datasets/AnomalyDataset/merge_data_v5_binary/data.yaml \
  epochs=50 batch=96 close_mosaic=20 device=0,1,2,3 \
  optimizer=MuSGD lr0=0.00125 lrf=0.5 momentum=0.9 weight_decay=0.0005 \
  scale=0.1 copy_paste=0.1 mixup=0.0 save_json=True \
  project=yoloa_v2 name=26m_yoloav2_v5_binary_cm20_gauss_pd50_v1
```

## 3. Ablation: full shortcut (rect mask, p_drop=0)

If mAP here ≫ run 1, model is exploiting the GT-mask shortcut → mask dropout is necessary.
If mAP ≈ run 1, model is not shortcutting → dropout is a free regularizer.

```
nohupyolo train task=anomaly_v2 \
  model=yolo26m-anomaly-v2-pd0.yaml \
  pretrained=yolo26m.pt \
  data=/home/louis/ultra_louis_work/datasets/AnomalyDataset/merge_data_v5_binary/data.yaml \
  epochs=50 batch=96 close_mosaic=20 device=0,1,2,3 \
  optimizer=MuSGD lr0=0.00125 lrf=0.5 momentum=0.9 weight_decay=0.0005 \
  scale=0.1 copy_paste=0.1 mixup=0.0 save_json=True \
  project=yoloa_v2 name=26m_yoloav2_v5_binary_cm20_rect_pd0_v1
```

## 4. Sanity: full no-mask (p_drop=1.0) — should ≈ vanilla baseline

Mask always dropped → fusion is exact passthrough → model is functionally identical
to vanilla yolo26m. Final mAP should match `26m_yolo_v5_binary_cm20_v1` within noise.
**If it doesn't, something is broken in the v2 plumbing and must be fixed before
trusting results from runs 1–3.**

```
nohupyolo train task=anomaly_v2 \
  model=yolo26m-anomaly-v2-pd100.yaml \
  pretrained=yolo26m.pt \
  data=/home/louis/ultra_louis_work/datasets/AnomalyDataset/merge_data_v5_binary/data.yaml \
  epochs=50 batch=96 close_mosaic=20 device=0,1,2,3 \
  optimizer=MuSGD lr0=0.00125 lrf=0.5 momentum=0.9 weight_decay=0.0005 \
  scale=0.1 copy_paste=0.1 mixup=0.0 save_json=True \
  project=yoloa_v2 name=26m_yoloav2_v5_binary_cm20_rect_pd100_v1
```

---

## How to read results

Each run produces `runs/yoloa_v2/<name>/results.csv` with these key columns:

| Column | Meaning |
|---|---|
| `metrics/mAP50-95(B)` | **B-on**: mAP when forward is given the bbox-rendered mask. Upper bound. |
| `mask_off/metrics/mAP50-95(B)` | **B-off**: mAP when mask is disabled during the forward (fusion = passthrough). |
| `val/box_loss`, `val/cls_loss`, `val/dfl_loss` | Validation loss (mask-on pass). |
| `mask_off/val/*_loss` | Validation loss (mask-off pass). Will likely be higher. |

## Success criteria for Phase 0

Compare these four numbers from the run-1 final epoch + your existing baseline:

| Metric | Source | Expected vs baseline |
|---|---|---|
| baseline mAP50-95 | `runs/yoloa/26m_yolo_v5_binary_cm20_v1/results.csv` | reference |
| run 1 mAP50-95 (B-on) | `metrics/mAP50-95(B)` | **> baseline** (the goal — fusion helps when prior is perfect) |
| run 1 mAP50-95 (B-off) | `mask_off/metrics/mAP50-95(B)` | **≈ baseline** (model still works without mask — anti-shortcut succeeded) |
| run 4 mAP50-95 | `metrics/mAP50-95(B)` (mask is dropped anyway) | **≈ baseline** (sanity check — v2 plumbing doesn't break vanilla path) |

If all four hold, Phase 0 validates the fusion hypothesis and we move to v2.1 (MemoryBank inference path).

## Suggested launch order

1. **Run 4 first** (sanity). 50 epochs ≈ a few hours. If it doesn't ≈ baseline, debug before launching others.
2. **Run 1** (primary B-on rect). Main result.
3. **Run 2** (gauss ablation) in parallel if GPUs available.
4. **Run 3** (pd0 shortcut ablation) last — informational, not gating.
