---
comments: true
description: Correlate image properties and find possible detection label issues.
keywords: Ultralytics, image property analysis, label quality, dataset quality, detection
---

# Image Property Analysis

[`analyze_correlations`](../reference/utils/analysis.md) derives six scalar properties from each `YOLODataset` label and rank-correlates them against the per-image F1 recorded during [validation](../modes/val.md). Properties come from the cached label shape and annotations, so no image pixels are decoded and no files are written.

```python
from ultralytics import YOLO
from ultralytics.data import YOLODataset
from ultralytics.data.utils import check_det_dataset
from ultralytics.utils.analysis import analyze_correlations

data = check_det_dataset("coco128.yaml")
dataset = YOLODataset(data["val"], data=data, augment=False)
metrics = YOLO("yolo26n.pt").val(data="coco128.yaml", conf=0.25)
report = analyze_correlations(dataset, metrics)
print(report.summary())
```

The following scalar properties are derived per image:

| Property                | Meaning                                                |
| ----------------------- | ------------------------------------------------------ |
| `num_objects`           | labeled object count                                   |
| `small_object_ratio`    | fraction below the COCO 32²-pixel small-area threshold |
| `object_scale_variance` | coefficient of variation of normalized box areas       |
| `num_classes_present`   | distinct labeled class count                           |
| `center_spread`         | spread of normalized box centers                       |
| `max_pairwise_iou`      | maximum box overlap as a crowdedness proxy             |

`report.summary()` returns `property`, Spearman `spearman_r`, and sample count `n`. `report.per_image` and `report.correlations` retain the source values, and `report.to_csv()` and `report.to_json()` return data without writing files.

## Label issue scores

Detection validation also records three per-image scores alongside `precision`, `recall` and `f1` in `results.box.image_metrics`, and they are repeated in `report.per_image`. Rank by whichever one matches the problem you are looking for:

| Score                      | Meaning                                                   |
| -------------------------- | --------------------------------------------------------- |
| `possible_fp`              | confident prediction with little label overlap            |
| `possible_fn`              | average label lacking a matching same-class prediction    |
| `possible_label_confusion` | overlapping prediction and label with different class IDs |

```python
worst = sorted(report.per_image.items(), key=lambda kv: kv[1]["possible_fn"], reverse=True)[:3]
```

A low score does not prove a label is correct, and a high score does not prove it is wrong; treat these as a review queue rather than a verdict.
