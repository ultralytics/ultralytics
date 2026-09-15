---
comments: true
description: Correlate image properties and find possible detection label issues.
keywords: Ultralytics, image property analysis, label quality, dataset quality, detection
---

# Image Property Analysis

[`ImagePropertyExtractor`](../reference/utils/analysis.md) adds six scalar properties to each `YOLODataset` label using image headers and annotations. [`analyze_correlations`](../reference/utils/analysis.md) compares those properties with per-image F1 and ranks possible label issues when validation uses `score_labels=True`.

```python
from ultralytics import YOLO
from ultralytics.data import YOLODataset
from ultralytics.data.utils import check_det_dataset
from ultralytics.utils.analysis import ImagePropertyExtractor, analyze_correlations

data = check_det_dataset("coco128.yaml")
dataset = YOLODataset(data["val"], data=data, augment=False)
metrics = YOLO("yolo26n.pt").val(data="coco128.yaml", conf=0.25, score_labels=True)
report = analyze_correlations(ImagePropertyExtractor(dataset).labels, metrics)
print(report.summary())
print(report.label_issues)
plot = report.plot()  # RGB numpy array
```

The extractor writes the following scalar dictionary to each label:

| Field                   | Meaning                                                |
| ----------------------- | ------------------------------------------------------ |
| `num_objects`           | labeled object count                                   |
| `small_object_ratio`    | fraction below the COCO 32²-pixel small-area threshold |
| `object_scale_variance` | coefficient of variation of normalized box areas       |
| `num_classes_present`   | distinct labeled class count                           |
| `center_spread`         | spread of normalized box centers                       |
| `max_pairwise_iou`      | maximum box overlap as a crowdedness proxy             |

`report.summary()` returns `property`, Spearman `spearman_r`, and sample count `n`. `report.per_image` and `report.correlations` retain the source values, while `report.to_csv()`, `report.to_json()`, and `report.plot()` return data without writing files.

`report.label_issues` returns the three highest-scoring image candidates:

| Field                      | Meaning                                                   |
| -------------------------- | --------------------------------------------------------- |
| `possible_fp`              | confident prediction with little label overlap            |
| `possible_fn`              | label without a matching same-class prediction            |
| `possible_label_confusion` | overlapping prediction and label with different class IDs |
