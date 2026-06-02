# Softhint Fusion — ultra6 Training Commands

Run on ultra6 (`ssh ultra6`), repo at `~/ultra_louis_work/ultralytics/`.
Preconditions:
1. `conda activate ultra`
2. `set_wandb_true`
3. Branch `yoloa_v2_softhint` is checked out and clean
4. The Phase 0 mask-augment runs from `yoloa_v2` may still be running on other GPUs — coordinate device IDs to avoid conflict (`gpuu6` to check).

Each run:
- **20 epochs** (fast iteration), batch 96, 3 GPUs DDP
- Identical hparams to baseline `26m_yoloav2_v5_binary_cm20_rect_pd50_v1` except fusion mechanism and epochs.

Baseline reference (already trained on `yoloa_v2`):
```
26m_yoloav2_v5_binary_cm20_rect_pd50_v1     mask-on 0.6941 / off 0.6229 (e50)
```

---

## 1. Softhint main — `softhint_rect_pd50_v1`

```
nohuprun python -m ultralytics.cfg \
  train task=anomaly_v2 \
  model=yolo26m-anomaly-v2-softhint.yaml \
  pretrained=yolo26m.pt \
  data=/home/louis/ultra_louis_work/datasets/AnomalyDataset/merge_data_v5_binary/data.yaml \
  epochs=20 batch=96 close_mosaic=20 device=0,1,2 \
  optimizer=MuSGD lr0=0.00125 lrf=0.5 momentum=0.9 weight_decay=0.0005 \
  scale=0.1 copy_paste=0.1 mixup=0.0 save_json=True \
  project=yoloa_v2 name=26m_yoloav2_softhint_rect_pd50_v1
```

If `python -m ultralytics.cfg` is not your preferred entry point on ultra6, use whatever wrapper your shell aliases (`nohupyolo`, etc.) — preserve exactly these arg=values.

## 2. Softhint + SegBranch a1 — `softhint_rect_pd50_seg_a1_v1`

Use a different device set if 0–2 are taken.

```
nohuprun python -m ultralytics.cfg \
  train task=anomaly_v2 \
  model=yolo26m-anomaly-v2-softhint-seg-a1.yaml \
  pretrained=yolo26m.pt \
  data=/home/louis/ultra_louis_work/datasets/AnomalyDataset/merge_data_v5_binary/data.yaml \
  epochs=20 batch=96 close_mosaic=20 device=3,4,5 \
  optimizer=MuSGD lr0=0.00125 lrf=0.5 momentum=0.9 weight_decay=0.0005 \
  scale=0.1 copy_paste=0.1 mixup=0.0 save_json=True \
  project=yoloa_v2 name=26m_yoloav2_softhint_rect_pd50_seg_a1_v1
```

## 3. Monitor

```
tail -f runs/yoloa_v2/26m_yoloav2_softhint_rect_pd50_v1.log
lsta
lsddp
```

Inspect `beta` after epoch 20:

```
python -c "
import torch
ckpt = torch.load('runs/yoloa_v2/26m_yoloav2_softhint_rect_pd50_v1/weights/best.pt', map_location='cpu', weights_only=False)
m = ckpt['model'] if isinstance(ckpt, dict) and 'model' in ckpt else ckpt
print('beta:', m.heatmap_bias_fusion.beta.detach().tolist())
"
```

## 4. Evaluation

After both runs finish:

```
# False-prompt benchmark on each softhint run
python scripts/false_prompt_eval.py \
    --weights runs/yoloa_v2/26m_yoloav2_softhint_rect_pd50_v1/weights/best.pt \
    --data    /home/louis/ultra_louis_work/datasets/AnomalyDataset/merge_data_v5_binary/data.yaml \
    --out     runs/temp/false_prompt_softhint.json

python scripts/false_prompt_eval.py \
    --weights runs/yoloa_v2/26m_yoloav2_softhint_rect_pd50_seg_a1_v1/weights/best.pt \
    --data    /home/louis/ultra_louis_work/datasets/AnomalyDataset/merge_data_v5_binary/data.yaml \
    --out     runs/temp/false_prompt_softhint_seg_a1.json

# Baseline for comparison: checkout yoloa_v2 in a worktree so the script picks up
# that branch's HeatmapEncoder/HeatmapGuidedFusion code, then re-run.
git -C ~/ultra_louis_work/ultralytics worktree add /tmp/yoloa_v2_wt yoloa_v2
cd /tmp/yoloa_v2_wt && python scripts/false_prompt_eval.py \
    --weights runs/yoloa_v2/26m_yoloav2_v5_binary_cm20_rect_pd50_v1/weights/best.pt \
    --data    /home/louis/ultra_louis_work/datasets/AnomalyDataset/merge_data_v5_binary/data.yaml \
    --out     runs/temp/false_prompt_baseline.json
```

(The baseline branch doesn't have `scripts/false_prompt_eval.py`. Copy it across with `cp ~/ultra_louis_work/ultralytics/scripts/false_prompt_eval.py /tmp/yoloa_v2_wt/scripts/` before running, or run the eval from a worktree of `yoloa_v2_softhint` against the baseline's `best.pt` — the eval script only depends on the `anomaly_v2` predictor's `set_external_mask_once`, which exists on both branches.)

## 5. Pass criteria

- Softhint AUROC > baseline AUROC by ≥ 0.05
- Softhint mAP50-95 (mask-on, e20) within 0.02 of baseline (mAP50-95 at baseline e20 — read from `runs/yoloa_v2/26m_yoloav2_v5_binary_cm20_rect_pd50_v1/results.csv` row 20)
- `beta` final values are finite (printed at step 3)
