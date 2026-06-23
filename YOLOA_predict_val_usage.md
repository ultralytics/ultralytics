# YOLOA — Predict, Val & Visualize Usage

Single CLI (`run_yoloa.py`) for all three modes. One model load, per-category memory-bank fit
(disk-cached), then predict / val / visualize with a chosen prior.

## Quick reference

```bash
python run_yoloa.py --mode predict   --cat texture --prior heatmap --n-per-cat 5
python run_yoloa.py --mode visualize --cat bottle  --prior heatmap --n-per-cat 3
python run_yoloa.py --mode val       --cat all     --prior heatmap
python run_yoloa.py --mode val       --cat object  --prior heatmap   # 10 object categories
python run_yoloa.py --mode val       --cat texture --prior none      # 5 texture, honest floor
```

---

## 1. What YOLOA is (30s)

A normal YOLO detector whose features get fused with an anomaly **prior** (a heatmap that says
"this area looks off") right before the detect head.

| prior | where the hint comes from | what it's for |
|-------|---------------------------|---------------|
| `none` | nothing | bare detector — the honest floor |
| `heatmap` | memory bank of normal images | **the real deploy mode** (no labels needed) |

The memory bank is built from each category's `train/good/` images during `fit()`. Once built,
it's disk-cached (keyed by category + fit config hash), so subsequent runs reuse it instantly.

---

## 2. Category groups

| `--cat` | Count | Categories |
|---------|-------|------------|
| `all` | 15 | every MVTec category |
| `object` | 10 | bottle, cable, capsule, hazelnut, metal_nut, pill, screw, toothbrush, transistor, zipper |
| `texture` | 5 | carpet, grid, leather, tile, wood |
| `<name>` | 1 | single category (e.g. `bottle`, `screw`) |

---

## 3. Fit config (`--fit-cfg`)

Controls the memory-bank build: image size, coreset cap, top-K, calibration, etc. Three-layer
precedence at `fit()`: **CLI overrides > fit yaml > model yaml v2_cfg defaults**.

`--fit-cfg` accepts:
- **A bare filename** (e.g. `yoloa_fit_default.yaml`) — resolved from `ultralytics/cfg/` first,
  then CWD as fallback.
- **A relative/absolute path** (e.g. `my_fit.yaml`) — used directly.

The fit yaml's filename stem becomes the output subdirectory (`<fit_id>`), so different fit
configs produce different output dirs and never share banks.

### Shipped fit configs

| File | Use for |
|------|---------|
| `yoloa_fit_default.yaml` | general (imgsz 640, 10K coreset) |
| `yoloa_fit_imgsz640_all.yaml` | all 14 categories @ 640 |
| `yoloa_fit_imgsz640_texture.yaml` | texture-only @ 640 |
| `yoloa_fit_imgsz320_object.yaml` | object categories @ 320 (fast) |
| `yoloa_fit_imgsz320_l4_texture.yaml` | texture @ 320 with 4 backbone layers |

### Fit config keys

| Key | Default | Role | Compactness? |
|-----|---------|------|--------------|
| `imgsz` | 640 | image size for bank build + predict/val | yes |
| `max_images` | 0 | cap on normal images (0 = all) | yes |
| `bb_layers` | `[6]` | backbone tap indices | yes |
| `bb_max_bank_size` | `null` | coreset cap (`null` = no cap, keep all features) | yes |
| `bb_calibrate` | `compactness` | `compactness` (coreset + local-density) or `auto` | — |
| `bb_K` | 5 | top-K for local-density neighbours in compactness; Noisy-OR K in scoring | yes |
| `bb_calibration_target_score` | 0.5 | target score for β calibration | yes |
| `bb_temperature` | 5.0 | initial β — **ignored in compactness mode** (always recalibrated) | no |
| `bb_auto_temperature` | `true` | auto-calibrate β in `auto` mode — **ignored in compactness mode** | no |

When `bb_calibrate: compactness` (the default), `bb_temperature` and `bb_auto_temperature` are
shadowed — compactness always measures local density on the coreset and recalibrates β from
`bb_calibration_target_score`. Set them if you switch to `bb_calibrate: auto`.

### Overriding fit values from CLI

```bash
# Override imgsz for a quick low-res sweep
python run_yoloa.py --mode val --cat object --prior heatmap --imgsz 320

# Cap normal images at 200
python run_yoloa.py --mode val --cat bottle --prior heatmap --max-images 200
```

---

## 4. Predict mode (`--mode predict`)

Runs detection + heatmap overlay on sampled test images. Good for quick visual inspection.

### Basic example

```bash
python run_yoloa.py --mode predict --cat texture --prior heatmap --n-per-cat 5
```

### What happens step-by-step

1. **Load model** — `YOLOA(ckpt)` loads the checkpoint once.
2. **Per category:**
   a. `m.fit(good_dir, name=cat)` — builds the memory bank from `train/good/` images
      (first run takes ~30-120s per category; subsequent runs hit disk cache, <1s).
   b. `m.predict(img, prior="heatmap")` — runs the detector with the heatmap prior fused in.
   c. Saves two files per image:
      - `<type>__<stem>__pred.jpg` — annotated detection boxes
      - `<type>__<stem>__heat.jpg` — heatmap overlay (JET colormap on original)

### Output location

```
runs/temp/yoloa/<model_id>/<fit_id>/predict/<cat>/
├── broken_large__000_pred.jpg
├── broken_large__000_heat.jpg
├── contamination__001_pred.jpg
├── contamination__001_heat.jpg
├── good__010_pred.jpg
├── good__010_heat.jpg
└── ...
```

### Variations

```bash
# All test images (not just 5)
python run_yoloa.py --mode predict --cat bottle --prior heatmap --n-per-cat 0

# Honest floor — bare detector, no prior
python run_yoloa.py --mode predict --cat bottle --prior none --n-per-cat 5

# Higher confidence threshold, fewer false positives
python run_yoloa.py --mode predict --cat cable --prior heatmap --n-per-cat 5 --conf 0.3

# Fast low-res preview
python run_yoloa.py --mode predict --cat texture --prior heatmap --n-per-cat 3 --imgsz 320
```

### Key flags

| Flag | Default | Effect |
|------|---------|--------|
| `--n-per-cat` | 0 | test images per category (0 = all) |
| `--conf` | 0.1 | detection confidence threshold |
| `--iou` | 0.1 | NMS IoU threshold (ignored if `--e2e`) |
| `--e2e` | off | NMS-free head (when on, `--iou` is ignored) |

---

## 5. Visualize mode (`--mode visualize`)

Produces an 8-panel comparison grid per image — the most useful mode for understanding what the
model is doing.

### Basic example

```bash
python run_yoloa.py --mode visualize --cat bottle --prior heatmap --n-per-cat 3
```

### What happens step-by-step

1. Same fit flow as predict mode (bank built or loaded from cache).
2. For each test image, runs **4 prior variants** and assembles them into one grid:
   - `none` — bare detector, no prior (honest floor)
   - `segment` — SegBranch output as prior (only if ckpt has SegBranch)
   - `heatmap` — memory-bank heatmap as prior (the real deploy path)
   - `mask` — GT mask as prior (cheating upper bound; what a perfect prior would give)
3. Saves one grid image per test sample.

### 8-panel grid layout

| # | Panel | Prior | Shows |
|---|-------|-------|-------|
| 1 | original image | — | raw input |
| 2 | none pred | bare | what the detector sees without help |
| 3 | seg heatmap | SegBranch | what the refiner branch produces |
| 4 | seg pred | SegBranch | detections guided by seg heatmap |
| 5 | heatmap overlay | memory bank | what the anomaly heatmap looks like |
| 6 | heatmap pred | memory bank | detections guided by memory-bank heatmap |
| 7 | GT mask overlay | ground truth | the actual defect mask (if exists) |
| 8 | mask pred | GT mask | upper bound — detections with perfect prior |

### Interpreting the grid

- **Panel 2 (none) vs Panel 6 (heatmap)** — the core comparison. If heatmap pred catches defects
  that none pred misses, the prior is adding real signal. If they're identical, the prior isn't
  helping (or the detector is already strong enough).
- **Panel 5 (heatmap)** — check if hot spots align with actual defects. If the heatmap fires on
  background texture but not the defect, the memory bank isn't capturing the right features.
- **Panel 8 (mask)** — the upper bound. If even mask pred misses the defect, the detector head
  itself can't locate that anomaly (capacity or training issue, not a prior issue).

### Output location

```
runs/temp/yoloa/<model_id>/<fit_id>/visualize/<cat>/
├── broken_large__000.jpg
├── contamination__001.jpg
├── good__010.jpg
└── ...
```

### Variations

```bash
# Honest floor only (compare none vs mask upper bound)
python run_yoloa.py --mode visualize --cat metal_nut --prior none --n-per-cat 3

# With edge suppression on the heatmap
python run_yoloa.py --mode visualize --cat leather --prior heatmap --n-per-cat 3 --heat-edge

# All test images, higher confidence
python run_yoloa.py --mode visualize --cat pill --prior heatmap --n-per-cat 0 --conf 0.2
```

---

## 6. Val mode (`--mode val`)

Reports per-category + AVERAGE metrics across 6 dimensions. This is the quantitative evaluation
mode — run it to get numbers, not pictures.

### Basic example

```bash
python run_yoloa.py --mode val --cat texture --prior heatmap
```

### What happens step-by-step

1. Same fit flow per category (bank built or loaded from cache).
2. Delegates to `run_mvtec_ood_eval()` which:
   - Runs the model on all test images in the category
   - Computes image-level AUROC (is this image anomalous?)
   - Computes pixel-level AUROC (how well does the heatmap localize?)
   - Computes detection mAP at IoU thresholds 0.10, 0.25, 0.50, 0.50:0.95
3. Prints a table and writes `val_<prior>.csv`.

### Understanding the metrics

| Metric | Range | What it means |
|--------|-------|---------------|
| `image_auroc` | 0–1 | Can the heatmap tell normal from anomalous images? >0.9 = usable |
| `pixel_auroc` | 0–1 | How well does the heatmap localize the defect? |
| `mAP10` | 0–1 | Detection mAP at IoU≥0.10 — **the key metric** for anomaly (boxes are coarse) |
| `mAP25` | 0–1 | Detection mAP at IoU≥0.25 |
| `mAP50` | 0–1 | Detection mAP at IoU≥0.50 — strict, often <0.1 for anomaly |
| `mAP50_95` | 0–1 | Average mAP across IoU 0.50–0.95 — very strict |

**Important:** `image_auroc` and `pixel_auroc` are `nan` for `--prior none` (bare mode has no
heatmap to score). The `none` prior still reports mAP values — these are your honest floor.

`--prior none` is always `nan` for AUROC because there's no heatmap. That's expected — use it
only for the mAP floor.

### Output location

```
runs/temp/yoloa/<model_id>/<fit_id>/val_<prior>.csv
```

Format:

```csv
category,image_auroc,pixel_auroc,mAP10,mAP25,mAP50,mAP50_95
bottle,0.9984,0.9103,0.8472,0.7183,0.1840,0.0586
...
AVERAGE,0.9681,0.8810,0.5855,0.4114,0.1409,0.0538
```

### Key val flags

| Flag | Default | Effect |
|------|---------|--------|
| `--iou` | 0.1 | NMS IoU threshold |
| `--e2e` | off | NMS-free head (when on, `--iou` is ignored) |
| `--batch` | 8 | batch size for inference |

### Variations

```bash
# Full 14-category sweep (the main evaluation)
python run_yoloa.py --mode val --cat all --prior heatmap

# Object categories only (10 of 14)
python run_yoloa.py --mode val --cat object --prior heatmap

# Honest floor — bare detector on texture
python run_yoloa.py --mode val --cat texture --prior none

# Single category quick check
python run_yoloa.py --mode val --cat screw --prior heatmap

# With a custom fit config
python run_yoloa.py --mode val --cat all --prior heatmap \
  --fit-cfg ultralytics/cfg/yoloa_fit_imgsz640_all.yaml

# Override fit imgsz for a fast rough sweep
python run_yoloa.py --mode val --cat object --prior heatmap --imgsz 320
```

### Comparing two prior modes

To compare `none` vs `heatmap`, run val twice and diff the CSVs:

```bash
python run_yoloa.py --mode val --cat all --prior none
python run_yoloa.py --mode val --cat all --prior heatmap
# Compare: runs/temp/yoloa/<model_id>/<fit_id>/val_none.csv
#      vs  runs/temp/yoloa/<model_id>/<fit_id>/val_heatmap.csv
```

The companion scripts `compare_chart.py` and `compare_lines.py` in the output dir generate
bar charts and line charts from these two CSVs.

---

## 7. Output directory structure

```
runs/temp/yoloa/
└── <model_id>/                     # ckpt run name (e.g. 26m_yoloav2_softhint_maskonly_aug3_mixup_binary_v1)
    └── <fit_id>/                   # fit yaml stem (e.g. yoloa_fit_default)
        ├── banks/                  # disk-cached memory banks (*.pt per category)
        │   ├── bottle_a1b2c3d4.pt
        │   ├── cable_e5f6g7h8.pt
        │   └── ...
        ├── val_none.csv            # val results with --prior none
        ├── val_heatmap.csv         # val results with --prior heatmap
        ├── predict/<cat>/          # predict mode output
        └── visualize/<cat>/        # visualize mode output
```

Different `--fit-cfg` values produce different `<fit_id>` directories, so experiments with
different bank parameters (imgsz, coreset size, layers, etc.) never interfere.

---

## 8. Bank caching

Memory banks are keyed by `{category}_{fit_hash}.pt` and stored in `<out>/banks/`. The hash
covers all fit parameters (imgsz, layers, K, temperature, calibration mode, max_images, etc.),
so changing any fit key produces a new cache file — you never accidentally reuse a stale bank.

Fit is the expensive step (feature extraction + coreset subsampling + calibration). Once cached,
subsequent predict/val/visualize runs on the same category + fit config load the bank instantly.

All three modes share the same cache, so you can fit once via val and immediately visualize
with the same bank.

---

## 9. Prior-shaping knobs

Optional flags that tweak the heatmap before fusion. These affect the fused prior → mAP but
**never change AUROC** (AUROC is computed on the raw heatmap).

| Flag | Default | Effect |
|------|---------|--------|
| `--heat-norm` | `mean` | `none` / `minmax` / `gaussian` / `mean` — rescale/blur the heatmap |
| `--heat-edge` | off | enable edge-suppression on the heatmap |
| `--heat-edge-sigma` | 1.0 | edge-sigma (bigger = gentler suppression) |

Example:

```bash
python run_yoloa.py --mode val --cat texture --prior heatmap --heat-edge --heat-edge-sigma 2.0
```
