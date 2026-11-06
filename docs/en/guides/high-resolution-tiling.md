---
title: YOLO26 High Resolution Tiling for Tiny Objects
comments: true
description: Detect small objects in large images with the Ultralytics YOLO tile argument. Train, validate, and predict on overlapping tiles without installing SAHI.
keywords: Ultralytics, YOLO26, high resolution tiling, tile argument, tiled inference, sliced inference, SAHI alternative, small object detection, aerial imagery, satellite imagery, drone imagery
---

# High Resolution Tiling with Ultralytics YOLO

Use `tile=True` to process high-resolution images in overlapping windows before resizing them to the model input. This can preserve small-object detail that would be lost by resizing the whole image. No additional package is required, and `tile` replaces the deprecated [SAHI integration](sahi-tiled-inference.md) for Ultralytics models.

|                                                         Whole image, `imgsz=640`: 35 detections                                                          |                                                                                `tile=640`: 170 detections                                                                                |
| :------------------------------------------------------------------------------------------------------------------------------------------------------: | :--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: |
| ![YOLO26s whole-image prediction on a drone view of a city intersection, with 35 detections](https://cdn.ul.run/i/197756a7765c387e6b3ecef2da707ef2.avif) | ![YOLO26s tiled prediction on the same intersection, with 170 detections, including distant pedestrians, cyclists, and cars](https://cdn.ul.run/i/f24115497afe748017bcf54d69aa5511.avif) |

YOLO26s on a 1920 × 1080 [VisDrone](../datasets/detect/visdrone.md) frame at the default confidence threshold. With `tile=640`, the model runs eight overlapping 640-pixel windows at native resolution instead of one pass over the frame shrunk to 640 pixels.

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

## FAQ

### What does the `tile` argument do in Ultralytics YOLO?

`tile` splits each image into overlapping windows, runs the model on every window, and merges the outputs back into one result per original image. It works in training, validation, prediction, and tracking. With `tile=True`, windows match `imgsz`, so small objects keep the pixels they would lose if the whole image were resized to `imgsz`.

### How do I choose the tile size?

`tile=True` uses `imgsz` as the window size, and an integer such as `tile=1024` sets it independently. Windows overlap by 20%, so a 1920 × 1080 image at `tile=640` runs eight windows and a 3840 × 2160 image at `tile=1280` also runs eight. Smaller windows keep more detail but add model passes and remove more global context, so compare a few sizes on your own validation set.

### Do I still need SAHI for sliced inference with YOLO?

No. The `tile` argument handles sliced inference without an extra package, and it also tiles training and validation. Prediction returns standard [`Results`](../reference/engine/results.md) objects, and validation scores the merged predictions against the original annotations. The [SAHI guide](sahi-tiled-inference.md) is deprecated.

### Should I train with tiling as well as predict with it?

Use the same `tile` setting for training and prediction when your images are much larger than `imgsz`. Training on tiles shows the model objects at the scale it sees during tiled prediction, clips labels to each window, and keeps empty windows as negative examples. Validate with `tile` too, since validation merges tile predictions before computing metrics on the full images.

### Does tiling work with exported models such as ONNX or TensorRT?

Yes. Tiling runs in the Python prediction pipeline, so [exported models](../modes/export.md) keep their normal input graph. Load the exported file with `YOLO("yolo26n.onnx")` and pass `tile` to `predict` as usual. `batch` sets how many tiles run together, and fixed export batch sizes are handled, including the last partial batch.

### Which tasks and models support tiling?

Detection, segmentation, semantic segmentation, depth, classification, pose, and OBB all support `tile`, as do YOLO-World, YOLOE, RT-DETR, and FastSAM prediction. SAM keeps its own prompt and crop pipeline and does not accept `tile`. See [Tasks and Models](#tasks-and-models) for how each task merges tile outputs.
