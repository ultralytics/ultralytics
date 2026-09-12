---
comments: true
description: Extract six per-image properties from object detection datasets.
keywords: Ultralytics, image property analysis, dataset analysis, object size, crowdedness
---

# Image Property Analysis

[`ImagePropertyExtractor`](../reference/utils/analysis.md) adds six scalar properties to each `YOLODataset` label using only image headers and annotations. It does not need a model, decode pixels, or write files.

## Quick start

```python
from ultralytics.data import YOLODataset
from ultralytics.data.utils import check_det_dataset
from ultralytics.utils.analysis import ImagePropertyExtractor

data = check_det_dataset("coco128.yaml")
dataset = YOLODataset(data["val"], data=data, augment=False)
labels = ImagePropertyExtractor(dataset).labels
```

Each label keeps its existing fields and gains an `im_properties` dictionary:

| Field                   | Meaning                                                |
| ----------------------- | ------------------------------------------------------ |
| `num_objects`           | labeled object count                                   |
| `small_object_ratio`    | fraction below the COCO 32²-pixel small-area threshold |
| `object_scale_variance` | coefficient of variation of normalized box areas       |
| `num_classes_present`   | distinct labeled class count                           |
| `center_spread`         | spread of normalized box centers                       |
| `max_pairwise_iou`      | maximum box overlap as a crowdedness proxy             |

Undefined box statistics are `NaN` for empty-label images. The scalar dictionary can be serialized directly for API or Platform use.
