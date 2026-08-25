---
comments: true
description: Extract image properties and turn model correlations into dataset actions.
keywords: Ultralytics, image property analysis, correlation, dataset quality, detection
---

# Image Property Analysis

[`ImagePropertyExtractor`](../reference/utils/analysis.md) adds six scalar properties to each `YOLODataset` label using only image headers and annotations. [`CorrelationAnalysis`](../reference/utils/analysis.md) joins those properties with per-image F1 and returns up to three dataset actions.

```python
from ultralytics import YOLO
from ultralytics.utils.analysis import CorrelationAnalysis, ImagePropertyExtractor

model = YOLO("yolo11n.pt")
metrics = model.val(data="coco128.yaml", conf=0.25)
labels = ImagePropertyExtractor(model.validator.dataloader.dataset).labels
report = CorrelationAnalysis(labels, metrics).run()
print(report.summary())
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

`report.summary()` contains `target`, `issue`, Spearman `score`, numeric `evidence`, and `action`. Only the three strongest F1-lowering correlations at or below -0.1 become actions. `report.per_image` and `report.correlations` retain the supporting values, while `report.to_csv()`, `report.to_json()`, and `report.plot()` return data without writing files.

Use `conf=0.25` for useful per-image F1. The analyzer warns when median F1 is below 0.1, ignores undefined values and properties with fewer than 30 samples, and warns if duplicate image basenames would collide.
