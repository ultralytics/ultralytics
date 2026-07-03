# Calibrated Validation Images After Depth Auto-Calibration

**Date:** 2026-07-03
**Branch:** `depth-calibrated-val-plots` (off `origin/depth_anything` @ f6218ba5d)

## Problem

Depth training writes `val_batch{0,1,2}.jpg` (`RGB | GT | pred` panels) during validation, but
post-training auto-calibration (`cal_a`/`cal_b` written into best.pt/last.pt by
`DepthTrainer.final_eval` → `calibrate_checkpoint`) runs *after* the last plotting pass. No image
ever shows the calibrated output — the thing the model actually predicts at deploy time. The val
scoreboard can't show it either: the default metric protocol is scale-invariant
(`align="median"`), so calibration is invisible in scores by construction.

## Deliverable

After auto-calibration in `final_eval`, write `val_batch{0,1,2}_calibrated.jpg` to the run dir:
4-column panels `RGB | GT | raw pred | calibrated pred`, one row per image (up to 4 per batch),
over the first 3 val batches — the same batches BaseValidator plots (val loader is not shuffled),
so the new files are directly comparable to the existing `val_batch{ni}.jpg`.

## Design

### 1. Panel renderer refactor — `ultralytics/models/yolo/depth/val.py`

Extract the row-building logic of `DepthValidator.plot_predictions` into a module-level helper:

```python
def plot_depth_panels(imgs, gt, preds, fname, titles=None, max_images=4)
```

- `preds`: list of `(B,1,H,W)` depth tensors; each becomes one column after `RGB | GT`.
- Colormap range per row comes from GT's valid pixels (min/max), shared by all depth columns —
  scale errors show as color mismatch, same as today.
- `titles`: optional column labels rendered in a thin header strip (4 unlabeled columns are
  ambiguous). Existing 3-column plots pass no titles and stay byte-compatible in layout.
- `DepthValidator.plot_predictions` becomes a thin wrapper calling it with `[pred]` —
  behavior unchanged.

### 2. Calibrated plot pass — `ultralytics/models/yolo/depth/calibrate.py`

`calibrate_checkpoint(ckpt_path, dataloader, device, dist_power=0.0, plot_dir=None)`:

- When `plot_dir` is set and calibration succeeded, run the already-loaded float copy (`work`)
  over the first 3 batches of `dataloader` with cal buffers at identity to get the **raw**
  prediction, then compute the **calibrated** column as `exp(a·log(raw) + b)` — a deterministic
  affine of raw, no second forward.
- Write `plot_dir / f"val_batch{ni}_calibrated.jpg"` via `plot_depth_panels` with titles
  `["RGB", "GT", "raw", f"calibrated ({name} ×{exp(b):.2f})"]`.
- **Identity case:** when the "calibrate only if it helps" policy selects `identity`, plots are
  still written (raw == calibrated) with the header saying `calibrated (identity)` — informative,
  not a bug.

### 3. Hook — `ultralytics/models/yolo/depth/train.py`

In `DepthTrainer.final_eval`, pass `plot_dir=self.save_dir` for the first existing checkpoint in
`(best, last)` only (each checkpoint is fitted separately; best represents the run), and only when
`self.args.plots` is set. Already gated on `RANK in {-1, 0}` and `auto_calibrate`.

## Error handling

Plot pass wrapped in try/except logging a `LOGGER.warning` — a plot failure never breaks
calibration or training (same policy as every other plot path in the repo).

## Testing

- Unit tests in `tests/depth/` following the existing stub-model pattern:
  - `plot_depth_panels` writes a file with height/width = rows × h, (2 + len(preds)) × w
    (+ header strip).
  - `calibrate_checkpoint(..., plot_dir=tmp)` produces `val_batch0_calibrated.jpg`.
- End-to-end on depthx1 (seetacloud, `/root/ult`): 1-epoch `yolo26n-depth` train with
  `auto_calibrate=True`, confirm `val_batch*_calibrated.jpg` appear in the run dir alongside
  `val_batch*.jpg`.

## Out of scope

- Changing the val metric protocol or scoreboard (median alignment stays).
- Plotting for `Model.calibrate()` API calls (trainer-only for now).
- Any change to existing `val_batch{ni}.jpg` content.
