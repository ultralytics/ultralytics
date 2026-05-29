---
comments: true
description: Use BNNR with Ultralytics YOLO for bbox-aware XAI-guided augmentations, branch search, and structured training reports via UltralyticsDetectionAdapter.
keywords: BNNR, YOLO, YOLO26, XAI, data augmentation, object detection, Ultralytics, open source, saliency, ICD, AICD
---

# BNNR with Ultralytics YOLO

[BNNR](https://github.com/bnnr-team/bnnr) (Bulletproof Neural Network Recipe) is an external MIT project that adds **XAI-guided augmentations**, optional **augmentation branch search**, and **structured training reports** on top of Ultralytics YOLO weights.

BNNR uses the [`UltralyticsDetectionAdapter`](https://github.com/bnnr-team/bnnr/blob/v0.4.10/src/bnnr/detection_adapter.py) and the BNNR Python API. It does **not** replace the Ultralytics `yolo` CLI or `YOLO.train()`. Use BNNR when you want bbox-aware **DetectionICD** / **DetectionAICD** augmentations and detection XAI reports alongside a YOLO model you already load through Ultralytics.

## What BNNR adds

- **UltralyticsDetectionAdapter** wraps a YOLO model for BNNR's training loop.
- **DetectionICD / DetectionAICD** apply tile-based masking guided by bounding-box regions (see below).
- **BNNRTrainer** can search augmentation branches and write `report.json` with metrics and detection XAI.

For classification-only workflows (no YOLO adapter), see BNNR's [`bnnr analyze`](https://github.com/bnnr-team/bnnr/blob/v0.4.10/docs/analyze.md) separately.

## Detection ICD and AICD

On the detection path, BNNR builds a saliency map from **ground-truth boxes**: pixels inside boxes are treated as important, pixels outside as background. That map drives the same tile masking idea as classification ICD/AICD, but stays **bbox-aware** (boxes move with flips and scales).

| Augmentation | Effect on training |
|--------------|-------------------|
| **DetectionICD** | Masks high-importance (object) tiles so the model must use context around objects |
| **DetectionAICD** | Masks low-importance (background) tiles so training focuses on object regions |

<p align="center">
  <img width="720" src="https://cdn.jsdelivr.net/gh/bnnr-team/bnnr@v0.4.10/docs/assets/icd-panel.png" alt="DetectionICD masks salient object tiles on a sample image">
</p>
<p align="center"><strong>DetectionICD</strong> — masks regions tied to objects (high bbox saliency), encouraging learning from context.</p>

<p align="center">
  <img width="720" src="https://cdn.jsdelivr.net/gh/bnnr-team/bnnr@v0.4.10/docs/assets/aicd-panel.png" alt="DetectionAICD masks background tiles on a sample image">
</p>
<p align="center"><strong>DetectionAICD</strong> — masks background tiles, sharpening focus on discriminative object areas.</p>

Pass augmentations into `BNNRTrainer` together with the adapter (same pattern as the [quickstart script](https://github.com/bnnr-team/bnnr/blob/v0.4.10/examples/integrations/ultralytics_yolo_quickstart.py)):

```python
from bnnr.detection_augmentations import DetectionHorizontalFlip
from bnnr.detection_icd import DetectionAICD, DetectionICD

augmentations = [
    DetectionHorizontalFlip(probability=0.5, random_state=42),
    DetectionICD(
        probability=0.5,
        threshold_percentile=70,
        tile_size=8,
        fill_strategy="gaussian_blur",
        random_state=52,
    ),
    DetectionAICD(
        probability=0.5,
        threshold_percentile=70,
        tile_size=8,
        fill_strategy="gaussian_blur",
        random_state=53,
    ),
]

trainer = BNNRTrainer(adapter, train_loader, val_loader, augmentations, config)
```

Classification ICD with Grad-CAM caches is documented separately in BNNR's [plugin_icd.md](https://github.com/bnnr-team/bnnr/blob/v0.4.10/docs/plugin_icd.md).

## When to use BNNR

| Use BNNR | Use Ultralytics alone |
|----------|------------------------|
| Experiment with XAI-guided aug search and HTML/JSON reports | Standard `yolo train` / `model.train()` is enough |
| Need bbox-aware ICD on detection batches | Default augmentations in `train()` meet your needs |

## Installation

```bash
pip install "bnnr[ultralytics]==0.4.10"
```

This extra installs **`ultralytics`** as a dependency of BNNR. You do not need a separate `pip install ultralytics` unless you want to pin a specific Ultralytics version yourself.

BNNR is not vendored inside the Ultralytics package. Install it from PyPI or from the [BNNR repository](https://github.com/bnnr-team/bnnr).

## Quickstart

The runnable COCO128 example (data loaders, augmentations, `BNNRTrainer`, and `report.json`) is in the BNNR repository:

[ultralytics_yolo_quickstart.py](https://github.com/bnnr-team/bnnr/blob/v0.4.10/examples/integrations/ultralytics_yolo_quickstart.py)

```bash
git clone https://github.com/bnnr-team/bnnr.git
cd bnnr
pip install "bnnr[ultralytics]==0.4.10"
PYTHONPATH=src:examples/integrations python examples/integrations/ultralytics_yolo_quickstart.py --quick --device cpu
```

The training loop requires COCO128 loaders from the quickstart script. To verify the adapter loads a YOLO model:

```python
from bnnr.detection_adapter import UltralyticsDetectionAdapter

adapter = UltralyticsDetectionAdapter(
    model_name="yolov8n.pt",
    device="cpu",
    num_classes=80,
)
```

## Dataset and image format

Detection datasets for BNNR must provide:

- **Images**: `torch.Tensor` `(C, H, W)` in **[0, 1]** float (do not pass 0-255 float tensors to YOLO through the adapter).
- **Targets**: Adapter-specific dicts with `boxes` and `labels` (these are not the same keys Ultralytics uses internally).
- **Index**: third element per sample for XAI cache keys.

Details: [BNNR detection guide](https://github.com/bnnr-team/bnnr/blob/v0.4.10/docs/detection.md).

## Limitations

- First run may download COCO128 and YOLO weights (`yolov8n.pt` by default in the quickstart).
- Keep Ultralytics and BNNR versions compatible; the quickstart targets BNNR **0.4.10**.
- Do not call `model.fuse()` on a fused head before `train_step` when using the adapter (see BNNR detection docs).
- BNNR's path uses **fp32 loss** and plain backward on the adapter; it is not the same code path as `YOLO.train()`.

## FAQ

### Can I still use `yolo train`?

Yes. BNNR is an optional Python API layer. Many projects use Ultralytics for training and deployment, and BNNR only for aug-search experiments.

### Adapter vs `YOLO()` directly?

`UltralyticsDetectionAdapter` implements BNNR's detection adapter protocol (`train_step`, XAI hooks, report integration). Use raw `YOLO` when you do not need BNNR augmentations or reports.

### Licenses

BNNR is [MIT](https://github.com/bnnr-team/bnnr/blob/main/LICENSE). Ultralytics is [AGPL-3.0](https://github.com/ultralytics/ultralytics/blob/main/LICENSE). Using `pip install ultralytics` and `pip install bnnr` in your project is separate from forking either repository. This page describes a third-party integration, not an official partnership.

## Learn more

- [BNNR integrations hub](https://github.com/bnnr-team/bnnr/blob/v0.4.10/docs/integrations.md)
- [Citation (BNNR and Ultralytics)](https://github.com/bnnr-team/bnnr/blob/v0.4.10/docs/citation.md)
