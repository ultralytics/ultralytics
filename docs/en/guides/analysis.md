---
comments: true
description: Per-image property extraction for object detection datasets. Augment YOLO dataset labels in place with six object-count, scale, layout, and crowdedness properties.
keywords: Ultralytics, image property analysis, dataset analysis, crowdedness, object size, data-centric
---

# Image Property Analysis

The [`ImagePropertyExtractor`](../reference/utils/analysis.md) turns a `YOLODataset` into six per-image properties with no model or metrics. It augments each `dataset.labels[i]` in place with an `im_properties` dict containing object count, small-object ratio, object-scale variation, class count, center spread, and maximum pairwise IoU.

The extractor uses only image headers and annotations, so it does not decode pixel data. Because it needs no predictions, you can compute properties once and reuse them across many model evaluations. The `im_properties` dict is all-scalar, so it serializes directly to JSON for a JS/TS front-end or the [Ultralytics Platform](https://platform.ultralytics.com/).

## Quick start

```python
from ultralytics.data.build import build_yolo_dataset
from ultralytics.data.utils import check_det_dataset
from ultralytics.utils import DEFAULT_CFG
from ultralytics.utils.analysis import ImagePropertyExtractor

# Dataset-only, no model or pixel decoding. Labels are augmented in place and returned for chaining.
data = check_det_dataset("coco128.yaml")
dataset = build_yolo_dataset(DEFAULT_CFG, data["val"], 1, data, mode="val", rect=False, stride=32)
labels = ImagePropertyExtractor(dataset).labels  # list[dict], each with an "im_properties" entry
```

Each label keeps its original fields (`im_file`, `cls`, `bboxes`, ...) and gains a single `im_properties` sub-dict. For one 42-object `coco128` image:

```json
{
  "im_file": "000000000196.jpg",
  "im_properties": {
    "num_objects": 42,
    "small_object_ratio": 0.3571,
    "object_scale_variance": 3.6724,
    "num_classes_present": 6,
    "center_spread": 0.3384,
    "max_pairwise_iou": 0.5004
  }
}
```

`ImagePropertyExtractor` writes no files. To export the properties for a front-end, serialize the `im_properties` dicts directly (`json.dumps([lbl["im_properties"] for lbl in labels])`) — they hold only scalars, so no numpy-array dropping is needed.

## Property catalog and references

| Per-image field         | Meaning                                                                                            |
| ----------------------- | -------------------------------------------------------------------------------------------------- |
| `num_objects`           | Number of labeled objects                                                                          |
| `small_object_ratio`    | Fraction of boxes below the [COCO](https://arxiv.org/abs/1405.0312) 32²-pixel small-area threshold |
| `object_scale_variance` | Coefficient of variation of normalized box areas                                                   |
| `num_classes_present`   | Number of distinct labeled classes                                                                 |
| `center_spread`         | Root-sum variance of normalized box-center coordinates                                             |
| `max_pairwise_iou`      | Maximum box overlap as a [CrowdHuman](https://arxiv.org/abs/1805.00123)-style crowdedness proxy    |

## Caveats

- **Empty-label images**: zero-box images have undefined scale, center, and pairwise-IoU statistics, so the extractor emits `NaN` for those fields.
- **Tasks supported**: the six fields use image headers and boxes, so they work for detection, segmentation, pose, and OBB datasets alike.
