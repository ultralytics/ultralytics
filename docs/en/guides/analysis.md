---
comments: true
description: Per-image property extraction and correlation analysis for object detection. Augment YOLO dataset labels with 27 pixel, object-size, and crowdedness properties, correlate them with per-image F1, and add ObjectLab label-quality scores to surface which properties drive bad model performance.
keywords: Ultralytics, image property analysis, dataset analysis, label quality, ObjectLab, correlation, worst images, brightness, blurriness, crowdedness, object size, data-centric
---

# Image Property Analysis

The [`ImagePropertyExtractor`](../reference/utils/analysis.md) turns a `YOLODataset` into per-image properties with no model, no metrics, and no I/O. It augments each `dataset.labels[i]` in place with a single `im_properties` dict of **27 properties** across three groups: 8 pixel-reading (brightness, contrast, entropy, edge density, ...), 17 cache-derived (object counts in COCO size buckets, class entropy, edge proximity, ...), and 2 annotation-interaction (max/mean pairwise IoU). After training, [`CorrelationAnalysis`](../reference/utils/analysis.md) joins those properties with per-image F1 scores from validation, computes Pearson and Spearman correlations, and ranks the worst-performing images so you can feed them into a curation or synthetic-data pipeline.

With `score_labels=True`, `model.val()` also computes 4 [ObjectLab](https://arxiv.org/abs/2309.00832) label-quality scores per image (overlooked, badloc, swap, and their `label_quality_score` geometric mean), which `CorrelationAnalysis` joins alongside the 27 properties for **31 analyzed columns**. The two-piece split is deliberate: `ImagePropertyExtractor` needs no model or metrics, so you can compute properties once and reuse them across many model evaluations, or explore a dataset's characteristics before training anything. The `im_properties` dict is all-scalar, so it serializes straight to JSON for a JS/TS front-end or the [Ultralytics Platform](https://platform.ultralytics.com/).

## Quick start

Three composition patterns cover the common use cases:

```python
from ultralytics import YOLO
from ultralytics.data.build import build_yolo_dataset
from ultralytics.data.utils import check_det_dataset
from ultralytics.utils.analysis import CorrelationAnalysis, ImagePropertyExtractor

# Path 1: dataset-only, no model. Labels are augmented in place (platform-friendly, consume in JS/TS).
data = check_det_dataset("coco128.yaml")
dataset = build_yolo_dataset(None, data["val"], 1, data, mode="val", rect=False, stride=32)
labels = ImagePropertyExtractor(dataset).labels  # list[dict], each with an "im_properties" entry

# Path 2: full analysis after model.val(). Use score_labels=True to enable ObjectLab.
model = YOLO("yolo11n.pt")
metrics = model.val(data="coco128.yaml", score_labels=True)
labels = ImagePropertyExtractor(model.validator.dataloader.dataset).labels
report = CorrelationAnalysis(labels, metrics).run()

# Path 3: reuse one extraction across many models. Property compute happens once.
labels = ImagePropertyExtractor(dataset).labels
for ckpt in ("yolo11n.pt", "yolo11s.pt", "yolo11m.pt"):
    metrics = YOLO(ckpt).val(data="coco128.yaml", score_labels=True)
    CorrelationAnalysis(labels, metrics).run(save_dir=f"runs/analyze-{ckpt[:-3]}")
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

`CorrelationAnalysis.run()` writes the following to an auto-incremented `runs/analyze/` directory (`runs/analyze`, `runs/analyze-2`, ...), following the same `increment_path` convention used for `runs/detect/train`, `runs/detect/val`, etc.:

| File                      | Purpose                                                                               |
| ------------------------- | ------------------------------------------------------------------------------------- |
| `per_image_analysis.csv`  | One row per image, sorted ascending by F1                                             |
| `correlations.json`       | Pearson + Spearman r and p-values per property, with effect-size band and direction   |
| `worst_images.json`       | Top 100 worst-performing images plus their top 3 problematic properties               |
| `summary.md`              | Human-readable summary with top correlations and worst-image table                    |
| `correlation_scatter.png` | Per-property scatter against F1 with regression line and Pearson r                    |
| `correlation_heatmap.png` | Property × property Pearson r matrix (self-correlations blanked)                      |
| `worst_images_strip.png`  | Thumbnails of bottom 20 by F1 with green ground-truth and red dashed prediction boxes |

`ImagePropertyExtractor` writes no files. To export the properties for a front-end, serialize the `im_properties` dicts directly (`json.dumps([lbl["im_properties"] for lbl in labels])`) — they hold only scalars, so no numpy-array dropping is needed.

## Example outputs

Rendered on COCO val2017 (5000 images) with `yolo11n.pt` at `conf=0.25`. `summary.md` reports the strongest correlates (object count, object size variation, small-object count) in plain English and links the three plots:

![F1 vs each property](https://cdn.jsdelivr.net/gh/ultralytics/assets@main/docs/image-property-correlation-scatter.avif)

![Property correlation heatmap](https://cdn.jsdelivr.net/gh/ultralytics/assets@main/docs/image-property-correlation-heatmap.avif)

![Worst 20 images](https://cdn.jsdelivr.net/gh/ultralytics/assets@main/docs/image-property-worst-images-strip.avif)

## Enabling label-quality scores

The 4 ObjectLab fields (`overlooked_score`, `badloc_score`, `swap_score`, `label_quality_score`) require the validator to compute them inline during validation. Pass `score_labels=True` to `model.val()` before constructing `CorrelationAnalysis`:

```python
metrics = model.val(data="coco128.yaml", score_labels=True)
```

The validator stores ~32 bytes/image extra in `metrics.box.image_metrics` (4 float scores per image). Raw IoU matrices and pred/GT arrays are not retained. Without the flag, ObjectLab columns are populated as `NaN`.

All 4 ObjectLab scores follow the same convention. **Low score = model behavior that suggests a label issue. 1.0 = no such behavior observed, not a clean bill of health.**

| Score                 | Low means                                                                     |
| --------------------- | ----------------------------------------------------------------------------- |
| `overlooked_score`    | Model confidently predicts an object the annotator did not label.             |
| `badloc_score`        | Model predicts the right class but disagrees with the annotated box location. |
| `swap_score`          | Model confidently predicts a different class than the annotated one.          |
| `label_quality_score` | Geometric mean of the above (1/3 each).                                       |

`overlooked_score` needs preds with `conf ≥ 0.95` _and_ zero IoU with every GT, so it saturates at 1.0 on clean datasets or small models. Look for **variance** in these columns, not absolute values.

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

| Feature / per-image field                                                             | Source                                                                                                                                            |
| ------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------- |
| `brightness` (HSP perceptual)                                                         | [Hendrycks & Dietterich, ICLR 2019](https://arxiv.org/abs/1903.12261)                                                                             |
| `dark_pixel_ratio`, `bright_pixel_ratio`                                              | [Hendrycks & Dietterich, ICLR 2019](https://arxiv.org/abs/1903.12261)                                                                             |
| `contrast` (grayscale std)                                                            | [Hendrycks & Dietterich, ICLR 2019](https://arxiv.org/abs/1903.12261)                                                                             |
| `blurriness` (variance-of-Laplacian)                                                  | [Pech-Pacheco et al., ICPR 2000](https://doi.org/10.1109/ICPR.2000.903548)                                                                        |
| `entropy` (Shannon over grayscale histogram), `class_entropy`                         | [Shannon, BSTJ 1948](https://doi.org/10.1002/j.1538-7305.1948.tb01338.x)                                                                          |
| `edge_density` (Canny edge mean)                                                      | [Canny, IEEE TPAMI 1986](https://doi.org/10.1109/TPAMI.1986.4767851)                                                                              |
| `sharpness` (Tenengrad gradient)                                                      | [Krotkov, IJCV 1988](https://doi.org/10.1007/BF00127822)                                                                                          |
| `width`, `height`, `aspect_ratio`, `total_pixels`, `num_objects`                      | trivial                                                                                                                                           |
| `num_small` / `num_medium` / `num_large` (COCO area buckets 32², 96²)                 | [Lin et al., COCO, ECCV 2014](https://arxiv.org/abs/1405.0312)                                                                                    |
| `small_object_ratio`, `box_area_std_norm`, `object_scale_variance`                    | trivial                                                                                                                                           |
| `num_classes_present`                                                                 | trivial                                                                                                                                           |
| `mean_center_x`, `mean_center_y`, `center_spread`                                     | trivial                                                                                                                                           |
| `num_near_edge` (boundary-truncated objects)                                          | [Everingham et al., Pascal VOC, IJCV 2010](https://link.springer.com/article/10.1007/s11263-009-0275-4)                                           |
| `max_pairwise_iou`, `mean_pairwise_iou` (per-image crowdedness)                       | [Shao et al., CrowdHuman, 2018](https://arxiv.org/abs/1805.00123)                                                                                 |
| `overlooked_score`, `badloc_score`, `swap_score`, `label_quality_score` (ObjectLab)   | [Tkachenko, Thyagarajan & Mueller, ICML Workshop 2023](https://arxiv.org/abs/2309.00832)                                                          |
| Per-image P/R/F1/TP/FP/FN                                                             | in-tree validator                                                                                                                                 |
| Pearson + Spearman correlation per property × F1 with effect-size band                | [Pearson, Proc. Royal Society 1895](https://doi.org/10.1098/rspl.1895.0041) / [Spearman, Am. J. Psychology 1904](https://doi.org/10.2307/1412159) |
| Worst-image ranking + scatter grid + heatmap + worst-image strip plots + `summary.md` | in-tree                                                                                                                                           |
| `ul://` platform-URI resolution for model + dataset inputs                            | [Ultralytics Platform API docs](https://docs.ultralytics.com/platform/api/)                                                                       |

## Output schema

`per_image_analysis.csv` columns: `im_name`, `im_file`, then validator-supplied prediction-quality fields (`precision`, `recall`, `f1`, `tp`, `fp`, `fn`), then all 31 property fields (27 image properties + 4 ObjectLab scores) plus `anomaly_score`. The CSV is always fully sorted ascending by F1.

`correlations.json` entries:

```json
{
    "brightness": {
        "pearson_r": -0.34,
        "pearson_p": 1.2e-5,
        "spearman_r": -0.31,
        "spearman_p": 3.4e-5,
        "n": 458,
        "effect_band": "moderate",
        "direction": "higher brightness -> lower F1"
    }
}
```

`worst_images.json` entries:

```json
[
    {
        "im_name": "img_0042.jpg",
        "f1": 0.12,
        "anomaly_score": 2.31,
        "top_3_problematic": ["blurriness", "num_small", "num_near_edge"]
    }
]
```

## Acting on the results

`summary.md` lists the top correlated properties with a strength band (`strong`, `moderate`, `weak`, `negligible`) and a direction (`higher X -> lower F1` or `higher X -> higher F1`). The full per-property breakdown is in `correlations.json`. Focus on the `strong` and `moderate` entries, those are where a training-data change will move the needle.

- **Crowdedness / object count**: if `num_objects`, `max_pairwise_iou`, or `small_object_ratio` correlate with low F1, your model struggles in dense scenes. Consider raising `imgsz`, training with more crowded-scene augmentation (mosaic, copy-paste), or generating synthetic crowded scenes targeting the worst images.
- **Object scale spread**: if `object_scale_variance` or `num_small` correlate with low F1, multi-scale predictions are weak. Tune anchor-free head capacity or add tiled-inference for small targets.
- **Pixel-level corruptions**: brightness/contrast/blurriness/`dark_pixel_ratio` correlations point at exposure or motion-blur issues. Augment with the corresponding [Albumentations](../integrations/albumentations.md) transforms, or retrain after curating examples with similar properties.
- **Label-quality scores** (`overlooked_score`, `badloc_score`, `swap_score`, `label_quality_score`): low scores flag specific annotation issues per image. Review the listed worst images, fix labels, and retrain.
- **Worst-image triage**: the listed worst images are direct candidates for synthetic-data targets. Generate variants with the highlighted properties amplified, label them, and add to the training set.

The `anomaly_score` per image is a signed z-score average across all properties, weighted so positive = unusual in an F1-degrading direction. Treat large positive values as "this image is statistically the kind of input your model struggles with."

## Reading the correlation values

If you want to read the raw `pearson_r` / `spearman_r` numbers in `correlations.json` directly instead of leaning on the summary bands:

- **Focus on `spearman_r`** ([Wikipedia](https://en.wikipedia.org/wiki/Spearman%27s_rank_correlation_coefficient)). Per-image F1 distributions are often non-normal (skewed toward the extremes after NMS + IoU=0.5 matching), and image properties like `blurriness` have heavy tails, so the property-vs-F1 relationship is typically monotonic but not strictly linear. Spearman ranks values before correlating, which handles both, and it is what `effect_band` and `direction` are derived from.
- **Pearson r** ([Wikipedia](https://en.wikipedia.org/wiki/Pearson_correlation_coefficient)) measures a strictly linear fit. It is reported in parallel mainly as a cross-check. When `pearson_r` and `spearman_r` differ by a lot, the relationship is non-linear or a few extreme images are dominating, so open `correlation_scatter.png` for that property and decide visually.
- **Sign of r**: negative = _higher property -> lower F1_ (property looks like it hurts the model, candidate for augmentation, curation, or relabeling). Positive = _higher property -> higher F1_ (looks like it helps, you probably want more of it in training). These are heuristics from the correlation alone, always sanity-check on the scatter plot before changing your pipeline.
- **Effect band thresholds** are `|spearman_r| >= 0.5` (strong), `>= 0.3` (moderate), `>= 0.1` (weak), otherwise `negligible`. These are the common Cohen-style conventions and match what `summary.md` shows.
- **`pearson_p`, `spearman_p`, `n`**: standard p-values from `scipy.stats`, and `n` is the image count after dropping NaNs (used both for the correlation and for the significance test). Lower p is more significant. The bands are already a coarser version of this.

## Caveats

- **Filename collisions**: `Metric.image_metrics` is keyed by image basename. If your dataset has duplicate basenames across subdirectories they collide silently. The analyzer emits a single `LOGGER.warning` listing the count and a few examples.
- **Empty-label images**: zero-box images break per-image-box stats (mean undefined). The analyzer emits `NaN` for those properties and excludes them from correlations.
- **Tasks supported**: the 27 image-property fields read pixels and boxes only, so they work for any of detection / segmentation / pose / OBB. The 4 ObjectLab fields ship for **detection only** (segmentation, pose, and OBB extensions via mask-IoU, OKS, and rotated-box similarity are deferred to a follow-up release).
- **DDP**: the validator-side retention path is rank-0 safe, the existing `dist.gather_object` plumbing pickles numpy arrays cleanly without new logic.
