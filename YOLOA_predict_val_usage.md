# YOLOA-v2 — Predict & Val Quickstart

Hey! Here's how to run the two scripts on the weights + test data I sent you. You already have
the repo (branch `yoloa_v2_queryfilm`) and the conda env, so this is just "how to run them".

**What I sent you**
- `best.pt` — the trained YOLOA-v2 model (run `26m_yoloav2_film_maskonly_aug3_mixup_objcrop033_binary_v1`).
- a `MVTec-YOLO/` folder — test data in MVTec-YOLO layout (one folder per category, each with a
  `train/good/` for normal images, a `test/` for defect+good images, and a `<cat>_binary.yaml`).

**Two scripts**
- `test_predict_visual.py` — *visual*: per-image comparison grids, to eyeball what the model predicts.
- `mvtec_deploy_eval.py` — *numbers*: an AUROC + mAP table across categories.

---

## 1. What YOLOA-v2 is (30s read)

It's a normal YOLO detector, except its features get fused with an anomaly **prior** (a heatmap
that says "this area looks off") right before the detect head. The prior can come from different
places — that's the **prior mode**:

| prior mode | where the hint comes from | what it's for |
| --- | --- | --- |
| `none` | nothing | bare detector — the floor |
| `heatmap` | a memory bank of normal images | **the real deploy mode** (no labels needed) |
| `mask` | the ground-truth box | a "perfect prior" upper bound (reference only, it's cheating) |

The memory bank for `heatmap` mode is built on the fly from each category's `train/good/` images.

---

## 2. The model config (the `anomaly_v2:` block)

Baked into `best.pt` — you don't need to set any of this to run, it's just so you know what the
model is. Lives in `ultralytics/cfg/models/v2/yolo26-anomaly-v2-film-maskonly-aug3-mixup.yaml`.

| param | this model | meaning |
| --- | --- | --- |
| `fusion_mode` | `film` | how the prior is fused in — FiLM = per-channel scale+shift (vs `bias` = plain additive) |
| `mask_mode` | `gauss` | prior is a soft Gaussian blob (vs a hard box) |
| `bb_layers` | `[6]` | which backbone layer the memory bank reads features from → this is what enables `heatmap` mode |
| `bb_K` / `bb_temperature` | `9` / `5.0` | memory-bank nearest-neighbor count + softness |
| `seg_branch` | `false` | no extra segmentation branch on this one |
| `film_groups` | `16` | FiLM channel groups |

The `mask_*` knobs in that block (jitter / erase / mixup …) are **training-time augmentation only**
— irrelevant for running predict/val.

---

## 3. Predict — visual grids (`test_predict_visual.py`)

Saves an 8-panel grid per image: original + the model's prediction under each prior mode + the
heatmaps. Good for a gut check.

```bash
python test_predict_visual.py \
  --ckpt /Users/louis/workspace/ultra_louis_work/expman/data/pulled/yoloa_v2/26m_yoloav2_softhint_maskonly_aug3_mixup_binary_v1/weights/best.pt \
  --yaml yolo26m-anomaly-v2-softhint-maskonly-aug3-mixup.yaml \
  --mvtec-root /Users/louis/workspace/ultra_louis_work/buffer/AnomalyData/MVTEC/MVTec-YOLO \
  --imgsz 320 --device mps \
  --conf 0.1 --iou 0.1
```

- `--yaml` **must keep the `m`** (`yolo26m-...`) — drop it and it silently builds the wrong-size model.
- `--category all` runs all 15 categories; or name one. `--n-per-category 0` = every image.
- Output grids: `runs/temp/predict_visual/<run>/<category>/<type>__<image>.jpg`

---

## 4. Val — the numbers (`mvtec_deploy_eval.py`)

Runs the 3 prior modes over the categories and prints an AUROC + mAP table. You pass the data
folder with `--mvtec-root` (no file editing needed here).

```bash
python mvtec_deploy_eval.py \                         
  --ckpt /Users/louis/workspace/ultra_louis_work/expman/data/pulled/yoloa_v2/26m_yoloav2_softhint_maskonly_aug3_mixup_binary_v1/weights/best.pt \
  --name softhint \                                                                                                                              
  --mvtec-root /Users/louis/workspace/ultra_louis_work/buffer/AnomalyData/MVTEC/MVTec-YOLO \
  --categories bottle carpet --imgsz 320 --iou 0.1   
```

- `--categories all` = all 15; or list a few (`bottle carpet grid ...`). Missing categories auto-skip.
- `--imgsz 640` matches how it was trained (most faithful, but slower); `320` is the quick check.
- `--name` is just the column label in the printed table.
- Output: a table in the terminal, plus `runs/temp/mvtec_deploy_eval/objcrop033/mvtec_ood.csv` and
  `summary.json`.

The table has one block per prior mode — **the `heatmap` block is the one that matters** (that's
the real, label-free setup):

```
[heatmap(DEPLOY)]
            image_auroc          0.97     <- is the image anomalous?
            pixel_auroc          0.89     <- where exactly?
            mAP10 / mAP25 / ...           <- detection boxes
```

---

## 5. Optional: prior processing (`--heat-norm`)

Both scripts can tweak the heatmap prior *before* it's fused, via `--heat-norm` (default `none`):

| `--heat-norm` | what it does |
| --- | --- |
| `none` | raw memory-bank heatmap (default) |
| `minmax` | per-image stretch to [0,1] — boosts a weak / low-peak prior up to the GT-mask scale |
| `gaussian` / `mean` | blur the heatmap (kernel via `--heat-smooth-kernel`, default 5) — denoise toward a smooth blob |

**What we found** (15-cat deploy `heatmap` average, on a softhint model):

| metric | `none` | `mean` (k=5) | Δ |
| --- | --- | --- | --- |
| mAP10 | 0.4003 | 0.4125 | +0.012 |
| mAP50 | 0.0772 | 0.0743 | −0.003 |
| image_auroc | 0.9622 | 0.9622 | **0** |

Smoothing barely moves the box metrics (mAP10 up a hair, mAP50 down a hair — noise) and it *hurts*
the precise `mask_on` upper bound (the blur smears the GT box). **AUROC doesn't change at all** — it's
computed on the *raw* heatmap (before `--heat-norm`), so these knobs only affect the box/mAP numbers,
never the AUROC. Net: quick to try per model, but it didn't help here. (`spatial_softmax`, by the way,
collapses the prior to ~0 and kills detection entirely — it'd need to be trained in, not toggled at
inference.)

---

## Metrics cheat-sheet

- **image_auroc** — image-level: anomalous or not. Higher is better; 0.5 = coin flip.
- **pixel_auroc** — pixel-level: heatmap vs the defect mask (localization quality).
- **mAP10 / mAP25** — detection mAP at *loose* IoU. Defects are small/fuzzy, so the low-IoU mAP is
  the meaningful one; mAP50 is stricter.
- The `none` / `mask_off` rows show `nan` for AUROC — that's expected, not a bug (bare mode has no
  heatmap to score).


