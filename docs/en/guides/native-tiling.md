---
title: Native High-Resolution Tiling
comments: true
description: Train, validate, and predict with native YOLO tiling for high-resolution images across detection, segmentation, pose, OBB, classification, semantic segmentation, and depth.
keywords: Ultralytics, YOLO26, Native Tiling, High-resolution, Small Object Detection, Training, Validation, Predict, SAHI
---

# Native High-Resolution Tiling

!!! tip "Looking for the external SAHI integration?"

    Native tiling is built into Ultralytics. If you need SAHI's external slicing API and its additional annotation exports, see [YOLO26 Sliced Inference with SAHI](sahi-tiled-inference.md).

Native tiling processes high-resolution images as fixed pixel windows while keeping every output in the original image coordinate system. The `tile` argument controls [training](../modes/train.md), [validation](../modes/val.md), and [prediction](../modes/predict.md) for detect, segment, semantic, depth, classify, pose, and OBB tasks, using `imgsz` as the native window size.

## Native Tiling and SAHI

Both approaches slice large images so small objects retain more pixels at the model input. Native tiling is part of the Ultralytics lifecycle: training labels, full-image metrics, and `Results` objects use the same geometry and require no external package. SAHI remains useful for external sliced inference, deployment integrations, and its annotation conversion APIs. See the [SAHI guide](sahi-tiled-inference.md) for those workflows.

| Capability        | Native tiling                                   | SAHI sliced inference                      |
| ----------------- | ----------------------------------------------- | ------------------------------------------ |
| Supported outputs | All Ultralytics vision tasks                    | Detection-oriented external postprocessing |
| Training labels   | Clipped and restored automatically              | External label preparation                 |
| Validation        | Metrics against original full-image annotations | Usually inference-only                     |
| Results           | Original-image coordinates                      | SAHI `PredictionResult` objects            |
| Dependencies      | Ultralytics only                                | `ultralytics` and `sahi`                   |

## How Tiling Works

`imgsz` is the native tile height and width in pixels. Windows cover the complete image with 20% overlap between adjacent tiles, and edge windows align to the image boundary. Images smaller than `imgsz` use one full-image window.

During tiled training, each source image expands into its fixed windows. Boxes, polygons, and keypoints are clipped to their windows, while depth maps and semantic masks are cropped natively. Instances retain at least 10% of their original visible area, and windows without objects remain as negative samples.

During tiled validation and prediction, each window is resized to the model `imgsz`, predicted independently, and merged:

- Boxes, keypoints, instances, and oriented boxes return as full-image [`Results`](../reference/engine/results.md) objects. Cross-tile duplicates are suppressed with full-image NMS at IoU 0.5 and `max_det`.
- Classification predictions average the tile outputs.
- Semantic and depth outputs stitch native-resolution maps, using distance-weighted averaging for depth and the most central tile in semantic overlaps.

## Train with Native Tiling

```python
from ultralytics import YOLO

model = YOLO("yolo26n.pt")

model.train(data="aerial.yaml", tile=True, imgsz=640)
```

```bash
yolo detect train model=yolo26n.pt data=aerial.yaml tile=True imgsz=640
```

Use `cache="disk"` when tiled images need caching; `cache=True` and `cache="ram"` are rejected because each tile already references its full-resolution source.

## Validate and Predict

Both modes process tiles but report original-image outputs and metrics:

```python
from ultralytics import YOLO

model = YOLO("runs/detect/train/weights/best.pt")

metrics = model.val(data="aerial.yaml", tile=True, imgsz=640)
results = model.predict("large-image.tif", tile=True, imgsz=640, save=True)
```

```bash
yolo detect val model=best.pt data=aerial.yaml tile=True imgsz=640
yolo predict model=best.pt source=large-image.tif tile=True imgsz=640 save=True
```

## Configuration

| Argument | Type             | Default | Description                                                                    |
| -------- | ---------------- | ------- | ------------------------------------------------------------------------------ |
| `tile`   | `bool` or `str`  | `False` | Enables fixed native tiling. `"auto"` currently selects the same fixed policy. |
| `imgsz`  | `int` or `tuple` | `640`   | Native tile height and width in pixels, also used as the model input size.     |

Tiling is application-level orchestration: [exported models](../modes/export.md) retain their normal single-input graph and standard export behavior is unchanged. Tile windows, batch processing, and result merging wrap the normal prediction path at runtime.

## Limitations

- Tile overlap and cross-tile merge thresholds are fixed at 20% and IoU 0.5.
- `tile=True` does not support tensor sources, `embed`, `visualize`, or tracking.
- Very large outputs still require memory proportional to the complete original image during stitching.
- Objects split across windows are restored by duplicate suppression, not by reconstructing one prediction from multiple visible fragments.

For external sliced inference and SAHI result conversions, continue with [YOLO26 Sliced Inference with SAHI](sahi-tiled-inference.md).
