# Native High-Resolution Tiling — Final Report

## 1. What changed

- Added detection-only `tiles` and `tile_size=640` configuration. `tiles=True` and `tiles='auto'` enable deterministic fixed tiling; defaults preserve existing behavior.
- Added native fixed grid generation with 0.2 overlap, edge-anchored tiles, training-label clipping at 0.1 minimum visible-area fraction, empty-tile retention, and original-cache preservation.
- Added bounded tiled training, prediction, and validation through existing dataset, predictor, and validator owners.
- Added full-image cross-tile class-aware NMS at 0.5 IoU, native-coordinate `Results`, original-image metrics, JSON/TXT/plots, tile-local training-validation loss, and accumulated per-image profiling.
- Supported file/PIL/NumPy image and video prediction. Other tasks, tensor input, tracking, embeddings, and activation maps fail explicitly when tiling is requested. RAM caching is rejected for tiled training; disk caching deduplicates original images.
- Kept export graphs unchanged. Application-level tiled inference works with TorchScript. A checkpoint trained with `tiles=True` still defaults to ordinary non-tiled inference after export.
- Production documentation was intentionally unchanged for this under-the-surface implementation.

Branch: `native-high-resolution-tiling`

Commits:

- `b45a8b7da Add native detection tiling`
- `224e76844 Test native detection tiling lifecycle`
- `3ea1cb40f Polish native tiling validation and profiling`
- `0a20f87a3 Fix tiled validation output scaling`

## 2. Reproduction commands and environment

Environment: Linux, Python 3.14.7, PyTorch 2.13.0+cu130, local NVIDIA RTX 6000D GPUs. COCO8 was already present under `/data/shared-datasets/coco8`; no datasets or weights were downloaded.

Run from the feature worktree with existing weights:

```bash
export PYTHONPATH=/home/lq/codes/ultralytics-native-tiling
export YOLO_WEIGHTS_DIR=/home/lq/codes/ultralytics/weights
export YOLO_AUTOINSTALL=false

pytest -q tests/test_engine.py
pytest -q \
  tests/test_python.py::test_tile_geometry_and_merge \
  tests/test_python.py::test_tiled_dataset \
  tests/test_python.py::test_predict_tiles \
  tests/test_python.py::test_train_val_tiles \
  tests/test_python.py::test_build_yolo_dataset_hyp_isolated \
  tests/test_python.py::test_cfg_rejects_fuzzed_values

CUDA_VISIBLE_DEVICES=6,7 python - <<'PY'
from ultralytics import YOLO
YOLO('/home/lq/codes/ultralytics/yolo26n.pt').train(
    data='coco8.yaml', tiles=True, tile_size=256, imgsz=256,
    epochs=1, batch=2, workers=0, device=[0, 1], plots=False, verbose=False,
)
PY
```

Prediction and export smoke:

```python
import numpy as np
from ultralytics import YOLO

im = np.zeros((128, 192, 3), dtype=np.uint8)
model = YOLO('yolo26n.pt')
result = model.predict(im, tiles='auto', tile_size=64, imgsz=64, batch=2)[0]
assert result.orig_shape == (128, 192)

exported = model.export(format='torchscript', imgsz=64)
result = YOLO(exported).predict(im, tiles=True, tile_size=64, imgsz=64)[0]
assert result.orig_shape == (128, 192)
```

Benchmark settings: `yolo26n.pt`, four locally available COCO8 validation images, 320 native tiles, `imgsz=320`, batch 2, overlap 0.2, merge IoU 0.5, default confidence/NMS, three warm-up repetitions and ten measured repetitions on `CUDA_VISIBLE_DEVICES=6`. Raw machine-readable results are in `runs/plans/native_tiling_benchmark.json`.

## 3. Tests added and results

Added to `tests/test_python.py`:

- `test_tile_geometry_and_merge`: small/non-square grids, edge anchoring, label clipping, three-tile duplicates, adjacent-object preservation.
- `test_tiled_dataset`: real native image contents, negative tiles, label expansion, and RAM-cache error.
- `test_predict_tiles`: fixed/auto public prediction, native shape and box bounds, full-image save, tensor-input rejection.
- `test_train_val_tiles`: one-epoch tiled train, training validation, standalone full-image validation, JSON and four original-image TXT outputs.

Verified:

- 31/31 `tests/test_engine.py` tests passed in 205.28 s.
- 6/6 focused new and existing config/dataset tests passed in 25.53 s.
- Updated `test_train_val_tiles` passed again after JSON/TXT assertions were added.
- Two-process tiled DDP train plus training validation and final standalone validation passed on GPUs 6 and 7: 27 train tiles, four full validation images, 17 instances.
- TorchScript export and tiled/ordinary exported prediction passed for ordinary and tiled-trained checkpoints.
- Tiled validation with plots, JSON, and TXT produced 1,007 predictions, four label files, and full-image plots.
- `ruff format .` left 825 files unchanged; `git diff --check` passed.
- `ruff check --fix .` reported 145 existing repository findings, including pre-existing BLE001/S110/RUF012 rules. No unrelated cleanup was attempted.

Not run: complete `test_python.py`/`test_cli.py`/xdist suites (other task weights were not locally cached and local weight downloads were prohibited), documentation build (production docs unchanged), and multi-GPU standalone validation outside the completed trainer-driven DDP validation.

## 4. Metrics, baselines, hardware, and failed runs

Native COCO8 tiled validation with `save_json=True` (four images, 17 original-image instances):

- Precision 0.520; recall 0.680; mAP50 0.669; mAP50-95 0.274.
- COCO evaluation mAP50-95 0.277; small-object AP 0.099; small-object AR 0.250.
- These measurements establish native full-image behavior only; they are not SAHI parity evidence.

Native latency/memory on one idle RTX 6000D, 40 measured full-image predictions:

- p50 12.351 ms; p95 14.016 ms; mean 11.906 ms.
- Peak GPU allocated 58.736 MiB; peak GPU reserved 92.0 MiB.
- Process peak RSS 1,595.699 MiB.

SAHI/VisDrone baseline: **not reproduced**. Host aliases `ultra9`–`ultra12` did not resolve and `/home/lq/.ssh/config` was absent. No approved reachable remote fleet or local SAHI/VisDrone artifacts were available. The pre-registered parity gate (mAP50-95 within 0.005 and small-object AR within 0.01) is therefore **unverified**. No accuracy-parity or acceleration claim is made.

Development failures fixed before the final tests:

- Full-image loader resizing caused zero-sized native crops; fixed by preserving native validation pixels.
- Retained full-image metric tensors were on CPU; fixed by moving them to the validator device without renormalizing images.
- Native no-letterbox `ratio_pad` lacked padding metadata for JSON/TXT scaling; fixed at dataset sample construction and covered by regression assertions.
- Initial inspection commands used a non-existent `model.validator` attribute and a relative output path different from the configured runs directory; validation itself had completed in both cases.

## 5. Limitations and unavailable measurements

- Detection only; no segment, semantic, depth, classify, pose, or OBB tiling.
- No adaptive/activity routing and no additional overlap/merge configuration arguments.
- No SAHI baseline, VisDrone benchmark, side-by-side SAHI/native visual comparison, or parity verdict.
- Native timings use four COCO8 images and must not be represented as VisDrone or large-benchmark results.
- RAM caching is unsupported for tiled training; use disk caching or disable caching.
- Full repository tests and documentation build were not completed for the reasons above.

## 6. Decisions requiring owner review

- Whether fixed internal constants (0.2 overlap, 0.1 visible-area fraction, 0.5 merge IoU) are appropriate before exposing tiling publicly.
- Whether `tiles='auto'` should remain an alias for deterministic fixed tiling or gain a separate policy later.
- Whether RAM caching and currently rejected prediction modes need future support.
- Whether the SAHI gate can be run on an approved reachable host before any public release or accuracy claim.

## 7. Branch, artifacts, and experiment links

- Worktree: `/home/lq/codes/ultralytics-native-tiling`
- Branch: `native-high-resolution-tiling`
- Benchmark: `runs/plans/native_tiling_benchmark.json`
- Validation artifacts: `/home/lq/codes/ultralytics/runs/detect/artifacts/val-full-image`
- DDP artifacts: `/home/lq/codes/ultralytics/runs/detect/train-3`
- Remote artifacts: none; remote hosts were unreachable.
- PR: none, as requested.

Visual inspection covered `val_batch0_labels.jpg`, `val_batch0_pred.jpg`, and `confusion_matrix.png`. Predictions and labels align to the original full image with no visible tile-coordinate offset. The small COCO8 model produces additional false-positive boxes; no SAHI visual comparison is claimed.
