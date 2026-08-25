---
comments: true
description: Turn image properties and model evidence into dataset and label-review actions.
keywords: Ultralytics, image property analysis, label quality, correlation, detection
---

# Image Property Analysis

[`ImagePropertyExtractor`](../reference/utils/analysis.md) adds six scalar properties to each `YOLODataset` label using only image headers and annotations. [`CorrelationAnalysis`](../reference/utils/analysis.md) joins them with model evidence and returns dataset actions plus up to three images to review for possible missing labels, incorrect boxes, or incorrect classes.

```python
from ultralytics.utils.analysis import CorrelationAnalysis, ImagePropertyExtractor

from ultralytics import YOLO

model = YOLO("yolo11n.pt")
metrics = model.val(data="coco128.yaml", conf=0.25, score_labels=True)
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

`report.summary()` contains `target`, `issue`, numeric `score`, `evidence`, and `action`. Only the three strongest F1-lowering correlations at or below -0.1 and the three lowest label-review scores below 0.5 become actions.

| Label evidence     | Review question                                |
| ------------------ | ---------------------------------------------- |
| `overlooked_score` | is a predicted object missing from the labels? |
| `badloc_score`     | is a label box misplaced?                      |
| `swap_score`       | does a label use the wrong class?              |

These scores rank review priority. They are not probabilities or calibrated error rates. `report.per_image` retains the prediction and ground-truth boxes needed to draw an overlay. `report.correlations`, `report.to_csv()`, `report.to_json()`, and `report.plot()` return supporting data without writing files.

Use `conf=0.25` for useful per-image F1. Label scoring is opt-in and detection-only. The analyzer ignores undefined values and properties with fewer than 30 samples and warns about low median F1 or duplicate image basenames.
