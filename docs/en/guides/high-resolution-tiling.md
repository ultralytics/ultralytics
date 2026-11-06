---
title: High Resolution Tiling
description: Train, validate, and predict on image tiles with Ultralytics, using shared data loading and full-image results.
keywords: YOLO, tiling, high resolution, small objects, sliced inference
---

# High Resolution Tiling

Use `tile=True` to process high-resolution images in overlapping windows before resizing them to the model input. This can preserve small-object detail that would be lost by resizing the whole image. No additional package is required.

Tiling uses the normal training, validation, and prediction pipelines. Prediction returns one [`Results`](../reference/engine/results.md) object per original image. Validation merges predictions before evaluating them against the original annotations; training loss is computed on the tiles.

## Usage

```python
from ultralytics import YOLO

model = YOLO("yolo26n.pt")
model.train(data="aerial.yaml", tile=True, imgsz=640)
metrics = model.val(data="aerial.yaml", tile=True, imgsz=640)
results = model.predict("large-image.tif", tile=640, imgsz=640, batch=8)
```

```bash
yolo predict model=yolo26n.pt source=large-image.tif tile=640 imgsz=640 batch=8
```

`tile=True` uses `imgsz` as the window size. A positive integer, such as `tile=1024`, sets the window size independently of `imgsz`. `tile=False` is the default. Windows overlap by 20%, with the final windows aligned to the image boundary. Smaller images use a single full-image window.

`batch` controls how many tiles the model processes together during prediction. The existing backend handles fixed export batch sizes, including the last partial batch. [Exported models](../modes/export.md) retain their normal input graph; tiling runs in the Python prediction pipeline.

For video, stream results to keep memory bounded and use the existing prediction options to save or display them:

```python
for result in model.predict("video.mp4", tile=640, stream=True, save=True):
    print(result.boxes)
```

```bash
yolo predict model=yolo26n.pt source=video.mp4 tile=640 save=True
```

## Tasks and Models

| Task     | Full-image output                                                                                      |
| -------- | ------------------------------------------------------------------------------------------------------ |
| Detect   | Restore boxes to source coordinates and suppress duplicates with `iou`, `agnostic_nms`, and `max_det`. |
| Segment  | Apply the same box selection and place the selected masks in the source image.                         |
| Semantic | Stitch class maps, choosing the most central tile in overlapping regions.                              |
| Depth    | Blend overlapping depth maps with weights increasing toward each tile's center.                        |
| Classify | Average tile probabilities.                                                                            |
| Pose     | Apply box selection and restore the corresponding keypoints.                                           |
| OBB      | Restore rotated boxes and use rotated NMS.                                                             |

The shared pipeline supports YOLO task models, YOLO-World, YOLOE, RT-DETR, and FastSAM prediction. YOLOE visual prompts are embedded from the full source image and reused by its tiles. FastSAM selects prompted masks after merging them into source coordinates. SAM predictors retain their own prompt and crop-generation pipeline; they do not accept `tile`.

During training, boxes and polygons are cropped to their windows, OBB clipping preserves orientation, and out-of-window keypoints lose visibility. Instances with less than 10% of their original bounding-box area remaining are removed. Empty windows remain as negative examples. Semantic and depth targets use the same windows as their images.

## Performance and Quality

- Tiling runs more model forwards than whole-image inference. Choose the largest tile and batch sizes that preserve the detail you need within available memory.
- Consecutive tiles reuse the decoded source. With `cache="ram"`, training stores source images once; `cache="disk"` uses the existing image cache. Training shuffles tiles, so caching is especially useful for large compressed source images.
- Source-sized masks and dense maps still require memory proportional to the original image. `max_det` limits the number of retained instance masks.
- Tiling removes global context. Validate accuracy on your own data, especially for classification and depth, whose tile outputs may not agree globally.
- Duplicate suppression selects whole predictions; it does not reconstruct an object from separate visible fragments. Overlap reduces boundary truncation but cannot eliminate it for objects larger than a tile.
- Tensor sources, `embed`, and `visualize` are unsupported. Tracking runs on the merged predictions once per source frame, including ReID features.
