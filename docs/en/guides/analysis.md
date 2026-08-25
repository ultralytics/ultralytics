---
comments: true
description: Actionable model and label-quality analysis for object detection. Map image properties and ObjectLab scores to specific training and label-review actions.
keywords: Ultralytics, image property analysis, ObjectLab, label quality, actionable insights, detection
---

# Image Property Analysis

The [`ImagePropertyExtractor`](../reference/utils/analysis.md) turns a `YOLODataset` into six per-image properties with no model or metrics. After validation, [`CorrelationAnalysis`](../reference/utils/analysis.md) returns dataset-performance actions and, with ObjectLab scoring enabled, image-specific label-review actions.

The extractor uses only image headers and annotations, so it does not decode pixel data. You can compute the properties once and reuse them across many model evaluations. The `im_properties` dict is all-scalar, so it serializes directly to JSON for a JS/TS front-end or the [Ultralytics Platform](https://platform.ultralytics.com/).

## Quick start

Extract properties directly, or join them with validation metrics for actionable analysis:

```python
from ultralytics import YOLO
from ultralytics.data.build import build_yolo_dataset
from ultralytics.data.utils import check_det_dataset
from ultralytics.utils import DEFAULT_CFG
from ultralytics.utils.analysis import CorrelationAnalysis, ImagePropertyExtractor

# Dataset-only properties, no model or pixel decoding.
data = check_det_dataset("coco128.yaml")
dataset = build_yolo_dataset(DEFAULT_CFG, data["val"], 1, data, mode="val", rect=False, stride=32)
labels = ImagePropertyExtractor(dataset).labels  # list[dict], each with an "im_properties" entry

# Performance and label-quality analysis after model.val().
model = YOLO("yolo11n.pt")
metrics = model.val(data="coco128.yaml", score_labels=True)
labels = ImagePropertyExtractor(model.validator.dataloader.dataset).labels
report = CorrelationAnalysis(labels, metrics).run()
print(report.summary())  # target, issue, score, evidence, action
plot = report.plot()  # RGB numpy array, no file written
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

Neither class writes files. `report.summary()` is the default actionable output, while `report.per_image` and `report.correlations` retain raw evidence for Platform integrations. Use `report.to_csv()` or `report.to_json()` when an export is needed. `report.plot()` returns one compact RGB image and saves it only with `report.plot(save=True, filename="analysis.png")`.

ObjectLab quality scores follow a low-is-worse convention. A subtype below 0.5 adds one review action for that image:

| Evidence           | Issue                                         | Action                                             |
| ------------------ | --------------------------------------------- | -------------------------------------------------- |
| `overlooked_score` | missing label or model false positive         | add the box or add the image as a hard negative    |
| `badloc_score`     | incorrect box or model localization error     | correct the box or add a localization example      |
| `swap_score`       | incorrect class or model classification error | correct the class or add a confusing-class example |

## Ultralytics Platform integration (`ul://`)

`ul://` URIs from the [Ultralytics Platform](https://platform.ultralytics.com/) are resolved by the underlying `YOLO()` constructor and `model.val()`, not by the analyzer. The API key must be set **before** `YOLO("ul://...")` is constructed (the URI is resolved at load time), via the `ULTRALYTICS_API_KEY` environment variable or `settings.update({"api_key": ...})`. Once that's in place, use the URIs as you would with any standard validation call, then pass the metrics and the validator's dataset through to the two analysis classes:

```python
import os

os.environ["ULTRALYTICS_API_KEY"] = "ul_xxx_40hex"  # or set in shell, or use settings.update(...)

from ultralytics import YOLO
from ultralytics.utils.analysis import CorrelationAnalysis, ImagePropertyExtractor

model = YOLO("ul://owner/project/model-name")
metrics = model.val(data="ul://owner/datasets/slug", score_labels=True)
labels = ImagePropertyExtractor(model.validator.dataloader.dataset).labels
CorrelationAnalysis(labels, metrics).run()
```

See the [Platform API docs](https://docs.ultralytics.com/platform/api/) for URI details.

## Property catalog and references

| Feature / per-image field                                              | Source                                                                                                                                            |
| ---------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------- |
| `num_objects`                                                          | in-tree dataset labels                                                                                                                            |
| `small_object_ratio`                                                   | [Lin et al., COCO, ECCV 2014](https://arxiv.org/abs/1405.0312)                                                                                    |
| `object_scale_variance`, `num_classes_present`, `center_spread`        | in-tree dataset labels                                                                                                                            |
| `max_pairwise_iou` (per-image crowdedness)                             | [Shao et al., CrowdHuman, 2018](https://arxiv.org/abs/1805.00123)                                                                                 |
| ObjectLab scores                                                       | [Tkachenko, Thyagarajan & Mueller, 2023](https://arxiv.org/abs/2309.00832)                                                                        |
| Per-image P/R/F1/TP/FP/FN                                              | in-tree validator                                                                                                                                 |
| Pearson + Spearman correlation per property × F1 with effect-size band | [Pearson, Proc. Royal Society 1895](https://doi.org/10.1098/rspl.1895.0041) / [Spearman, Am. J. Psychology 1904](https://doi.org/10.2307/1412159) |
| Actionable issue and next-step mapping                                 | in-tree                                                                                                                                           |
| `ul://` platform-URI resolution for model + dataset inputs             | [Ultralytics Platform API docs](https://docs.ultralytics.com/platform/api/)                                                                       |

## Actionable output

`report.summary()` returns at most three F1-lowering dataset drivers, ordered by Spearman correlation. Each row is ready for an API, table, or automated training decision:

```json
[
    {
        "target": "dataset",
        "issue": "dense scenes reduce F1",
        "score": -0.455,
        "evidence": "num_objects Spearman correlation, n=5000",
        "action": "add crowded-scene training images or use tiled crops"
    },
    {
        "target": "000000000196.jpg",
        "issue": "possible missing label or model false positive",
        "score": 0.03,
        "evidence": "overlooked_score",
        "action": "review the overlay; add a box if correct, otherwise add the image as a hard negative"
    }
]
```

Only negative correlations with `|spearman_r| >= 0.1` become dataset actions. The three lowest images below 0.5 for each ObjectLab subtype become the default review queue. Full evidence remains available in `report.per_image` and `report.correlations`.

## Caveats

- **Filename collisions**: `Metric.image_metrics` is keyed by image basename. If your dataset has duplicate basenames across subdirectories they collide silently. The analyzer emits a single `LOGGER.warning` listing the count and a few examples.
- **Empty-label images**: zero-box images break per-image-box stats (mean undefined). The analyzer emits `NaN` for those properties and excludes them from correlations.
- **Tasks supported**: the six image-property fields work for detection, segmentation, pose, and OBB. ObjectLab actions are detection-only.
- **DDP**: the validator-side retention path is rank-0 safe, the existing `dist.gather_object` plumbing pickles numpy arrays cleanly without new logic.
