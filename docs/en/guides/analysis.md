---
comments: true
description: Per-image property extraction for object detection datasets. Augment YOLO dataset labels in place with 27 pixel, object-size, and crowdedness properties as JSON-ready per-image metadata.
keywords: Ultralytics, image property analysis, dataset analysis, brightness, blurriness, crowdedness, object size, data-centric
---

# Image Property Analysis

The [`ImagePropertyExtractor`](../reference/utils/analysis.md) turns a `YOLODataset` into per-image properties with no model, no metrics, and no I/O. It augments each `dataset.labels[i]` in place with a single `im_properties` dict of **27 properties** across three groups: 8 pixel-reading (brightness, contrast, entropy, edge density, ...), 17 cache-derived (object counts in COCO size buckets, class entropy, edge proximity, ...), and 2 annotation-interaction (max/mean pairwise IoU).

The design is deliberately model-free: because the extractor needs no predictions, you can compute properties once and reuse them across many model evaluations, or explore a dataset's characteristics before training anything. The `im_properties` dict is all-scalar, so it serializes straight to JSON for a JS/TS front-end or the [Ultralytics Platform](https://platform.ultralytics.com/).

## Quick start

```python
from ultralytics.data.build import build_yolo_dataset
from ultralytics.data.utils import check_det_dataset
from ultralytics.utils.analysis import ImagePropertyExtractor

# Dataset-only, no model. Labels are augmented in place and returned for chaining.
data = check_det_dataset("coco128.yaml")
dataset = build_yolo_dataset(None, data["val"], 1, data, mode="val", rect=False, stride=32)
labels = ImagePropertyExtractor(dataset).labels  # list[dict], each with an "im_properties" entry
```

Each label keeps its original fields (`im_file`, `cls`, `bboxes`, ...) and gains a single `im_properties` sub-dict. For one 42-object `coco128` image:

```json
{
  "im_file": "000000000196.jpg",
  "im_properties": {
    "brightness": 0.3564,
    "blurriness": 0.0005,
    "contrast": 0.1976,
    "dark_pixel_ratio": 0.1134,
    "bright_pixel_ratio": 0.0009,
    "entropy": 7.5621,
    "edge_density": 0.1452,
    "sharpness": 84.9387,
    "width": 640,
    "height": 480,
    "aspect_ratio": 1.3333,
    "total_pixels": 307200,
    "num_objects": 42,
    "num_small": 15,
    "num_medium": 18,
    "num_large": 9,
    "small_object_ratio": 0.3571,
    "num_near_edge": 10,
    "mean_center_x": 0.5615,
    "mean_center_y": 0.6068,
    "center_spread": 0.3384,
    "box_area_std_norm": 0.1379,
    "object_scale_variance": 3.6724,
    "num_classes_present": 6,
    "class_entropy": 2.1833,
    "max_pairwise_iou": 0.5004,
    "mean_pairwise_iou": 0.0063
  }
}
```

`ImagePropertyExtractor` writes no files. To export the properties for a front-end, serialize the `im_properties` dicts directly (`json.dumps([lbl["im_properties"] for lbl in labels])`) — they hold only scalars, so no numpy-array dropping is needed.

## Property catalog and references

| Per-image field                                                       | Source                                                                                                  |
| --------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------- |
| `brightness` (HSP perceptual)                                         | [Hendrycks & Dietterich, ICLR 2019](https://arxiv.org/abs/1903.12261)                                   |
| `dark_pixel_ratio`, `bright_pixel_ratio`                              | [Hendrycks & Dietterich, ICLR 2019](https://arxiv.org/abs/1903.12261)                                   |
| `contrast` (grayscale std)                                            | [Hendrycks & Dietterich, ICLR 2019](https://arxiv.org/abs/1903.12261)                                   |
| `blurriness` (variance-of-Laplacian)                                  | [Pech-Pacheco et al., ICPR 2000](https://doi.org/10.1109/ICPR.2000.903548)                              |
| `entropy` (Shannon over grayscale histogram), `class_entropy`         | [Shannon, BSTJ 1948](https://doi.org/10.1002/j.1538-7305.1948.tb01338.x)                                |
| `edge_density` (Canny edge mean)                                      | [Canny, IEEE TPAMI 1986](https://doi.org/10.1109/TPAMI.1986.4767851)                                    |
| `sharpness` (Tenengrad gradient)                                      | [Krotkov, IJCV 1988](https://doi.org/10.1007/BF00127822)                                                |
| `width`, `height`, `aspect_ratio`, `total_pixels`, `num_objects`      | trivial                                                                                                 |
| `num_small` / `num_medium` / `num_large` (COCO area buckets 32², 96²) | [Lin et al., COCO, ECCV 2014](https://arxiv.org/abs/1405.0312)                                           |
| `small_object_ratio`, `box_area_std_norm`, `object_scale_variance`    | trivial                                                                                                 |
| `num_classes_present`                                                 | trivial                                                                                                 |
| `mean_center_x`, `mean_center_y`, `center_spread`                     | trivial                                                                                                 |
| `num_near_edge` (boundary-truncated objects)                          | [Everingham et al., Pascal VOC, IJCV 2010](https://link.springer.com/article/10.1007/s11263-009-0275-4) |
| `max_pairwise_iou`, `mean_pairwise_iou` (per-image crowdedness)       | [Shao et al., CrowdHuman, 2018](https://arxiv.org/abs/1805.00123)                                       |

## Caveats

- **Empty-label images**: zero-box images have undefined per-image box statistics (means over no boxes), so the extractor emits `NaN` for the affected object-size, center, and pairwise-IoU fields.
- **Tasks supported**: the 27 fields read pixels and boxes only, so they work for detection, segmentation, pose, and OBB datasets alike.
