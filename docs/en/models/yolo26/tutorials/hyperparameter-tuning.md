---
comments: true
description: Tune Ultralytics YOLO26 hyperparameters automatically with Ray Tune integration. Search space, trials, and result analysis in Python and CLI.
keywords: YOLO26, hyperparameter tuning, Ray Tune, Ultralytics, tune(), automated search, mAP optimisation
canonical: https://docs.ultralytics.com/models/yolo26/tutorials/hyperparameter-tuning/
---

<script type="application/ld+json">
{
  "@context": "https://schema.org",
  "@type": "TechArticle",
  "headline": "YOLO26 Hyperparameter Tuning with Ray Tune",
  "description": "Tune Ultralytics YOLO26 hyperparameters automatically with Ray Tune integration. Search space, trials, and result analysis in Python and CLI.",
  "url": "https://docs.ultralytics.com/models/yolo26/tutorials/hyperparameter-tuning/",
  "image": "https://cdn.jsdelivr.net/gh/ultralytics/assets@main/docs/ultralytics-active-learning-loop.avif",
  "datePublished": "2026-06-04",
  "dateModified": "2026-06-04",
  "author": {"@type": "Organization", "name": "Ultralytics", "url": "https://www.ultralytics.com"},
  "publisher": {"@type": "Organization", "name": "Ultralytics", "url": "https://www.ultralytics.com"},
  "mainEntityOfPage": "https://docs.ultralytics.com/models/yolo26/tutorials/hyperparameter-tuning/"
}
</script>

# YOLO26 Hyperparameter Tuning

<!-- NOTE FOR MURAT: Please verify the tune() API works correctly with YOLO26 (especially with the E2E NMS-free head — some loss hyperparameters may have different names or ranges). Add a concrete before/after example showing mAP improvement on a real dataset after tuning (e.g. +2.1 mAP on a custom 5-class dataset). Confirm `iterations` and `epochs` interaction, and whether Ray Tune's ASHA scheduler is applied by default. Add recommended `space` values specifically validated for YOLO26. -->

Finding the right hyperparameters for your specific dataset and hardware can dramatically improve [Ultralytics YOLO26](../../../models/yolo26.md) model accuracy. Ultralytics provides a built-in `tune()` method powered by [Ray Tune](https://docs.ray.io/en/latest/tune/index.html) that automates the search over a configurable hyperparameter space.

## Install Ray Tune

```bash
pip install "ultralytics>=26.0.0" "ray[tune]"
```

## Quick Start

=== "Python"

    ```python
    from ultralytics import YOLO

    model = YOLO("yolo26n.pt")

    # Run 30 trials, 10 epochs each
    result_grid = model.tune(
        data="coco8.yaml",
        epochs=10,
        iterations=30,
        optimizer="AdamW",
        plots=False,
        save=False,
        val=True,
    )
    ```

=== "CLI"

    ```bash
    # Hyperparameter tuning via CLI
    yolo train model=yolo26n.pt data=coco8.yaml epochs=10 \
        tune=True iterations=30
    ```

## Key `tune()` Arguments

| Argument | Default | Description |
|---|---|---|
| `data` | required | Dataset YAML path |
| `epochs` | `10` | Training epochs per trial (keep short for faster search) |
| `iterations` | `10` | Number of hyperparameter trials to run |
| `optimizer` | `"AdamW"` | Optimizer to use during tuning |
| `space` | `None` | Custom hyperparameter search space (dict) |
| `gpu_per_trial` | `1` | GPUs allocated per trial |
| `plots` | `True` | Generate tuning result plots |

## Defining a Custom Search Space

You can narrow the search space to only the parameters most relevant to your task:

```python
from ultralytics import YOLO
from ray import tune

model = YOLO("yolo26n.pt")

custom_space = {
    "lr0": tune.loguniform(1e-5, 1e-1),      # initial learning rate
    "lrf": tune.uniform(0.01, 1.0),           # final LR fraction
    "momentum": tune.uniform(0.6, 0.98),      # SGD momentum / Adam beta1
    "weight_decay": tune.uniform(0.0, 0.001), # optimizer weight decay
    "warmup_epochs": tune.uniform(0.0, 5.0),  # warmup epochs
    "warmup_momentum": tune.uniform(0.0, 0.95),
    "box": tune.uniform(0.02, 0.2),           # box loss gain
    "cls": tune.uniform(0.2, 4.0),            # classification loss gain
    "hsv_h": tune.uniform(0.0, 0.1),          # HSV-Hue augmentation
    "hsv_s": tune.uniform(0.0, 0.9),          # HSV-Saturation augmentation
    "hsv_v": tune.uniform(0.0, 0.9),          # HSV-Value augmentation
    "degrees": tune.uniform(0.0, 45.0),       # rotation degrees
    "translate": tune.uniform(0.0, 0.9),      # translation fraction
    "scale": tune.uniform(0.0, 0.9),          # scale gain
    "mosaic": tune.uniform(0.0, 1.0),         # mosaic probability
    "mixup": tune.uniform(0.0, 1.0),          # mixup probability
}

result_grid = model.tune(
    data="my_dataset.yaml",
    space=custom_space,
    epochs=15,
    iterations=50,
)
```

## Interpreting Results

After tuning completes, Ray Tune saves results to a `tune/` directory. The best hyperparameters are printed to the console and written to `best_hyperparameters.yaml`:

```yaml
# best_hyperparameters.yaml (example output)
lr0: 0.00812
lrf: 0.143
momentum: 0.934
weight_decay: 0.000341
box: 0.072
cls: 1.43
mosaic: 0.85
```

Use these values in your full training run:

```bash
yolo train model=yolo26n.pt data=my_dataset.yaml \
    lr0=0.00812 lrf=0.143 momentum=0.934 \
    box=0.072 cls=1.43 mosaic=0.85 \
    epochs=200
```

## Tips for Effective Tuning

!!! tip "Recommended workflow"

    1. **Start narrow** — run 20–30 short trials (10 epochs) to identify the most sensitive parameters.
    2. **Refine** — fix insensitive parameters at their best values and run 50+ longer trials on the remaining search space.
    3. **Validate** — train once from scratch with the best hyperparameters using full epochs to confirm improvement.

!!! warning "Compute cost"

    Each trial is a full training run. With `iterations=30` and `epochs=10`, expect 30× the cost of a 10-epoch training run. Use a small model variant (`yolo26n`) and a subset of your dataset for the search phase.

## See Also

- [YOLO26 Model Overview](../../../models/yolo26.md)
- [Train YOLO26 on a Custom Dataset](train-custom-dataset.md)
- [Tips for Best Training Results](tips-for-best-training-results.md)
- [Ultralytics Hyperparameter Tuning Docs](../../../integrations/ray-tune.md)
