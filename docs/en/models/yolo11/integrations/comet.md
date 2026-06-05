---
comments: true
description: Log YOLO11 training to Comet ML. Track metrics, images, confusion matrices, and model checkpoints with Ultralytics' built-in Comet integration.
keywords: YOLO11, Comet ML, experiment tracking, MLOps, Ultralytics, confusion matrix, metrics logging, model checkpoints
canonical: https://docs.ultralytics.com/models/yolo11/integrations/comet/
---

<script type="application/ld+json">
{
  "@context": "https://schema.org",
  "@type": "TechArticle",
  "headline": "YOLO11 Comet ML Integration",
  "description": "Log YOLO11 training to Comet ML. Track metrics, images, confusion matrices, and model checkpoints automatically with Ultralytics' built-in Comet integration.",
  "url": "https://docs.ultralytics.com/models/yolo11/integrations/comet/",
  "image": "https://cdn.jsdelivr.net/gh/ultralytics/assets@main/docs/ultralytics-active-learning-loop.avif",
  "datePublished": "2026-06-05",
  "dateModified": "2026-06-05",
  "author": {"@type": "Organization", "name": "Ultralytics", "url": "https://www.ultralytics.com"},
  "publisher": {"@type": "Organization", "name": "Ultralytics", "url": "https://www.ultralytics.com"},
  "mainEntityOfPage": "https://docs.ultralytics.com/models/yolo11/integrations/comet/"
}
</script>

# YOLO11 + Comet ML Integration

<!-- NOTE FOR MURAT: Please verify comet_ml integration works with YOLO11 (pip install ultralytics comet_ml), add a screenshot of the Comet UI showing a YOLO11 run with at minimum the Metrics panel and the image/confusion matrix panel populated, confirm which panels auto-populate vs. require manual instrumentation, and verify the COMET_API_KEY env var approach is still the recommended authentication method. -->

[Comet ML](https://www.comet.com/) is a cloud-based experiment tracking and model monitoring platform built for machine-learning teams. [Ultralytics YOLO11](../../../models/yolo11.md) logs to Comet automatically when `comet_ml` is installed and a valid API key is set — no callbacks to register.

## Install

```bash
pip install ultralytics comet_ml
```

## Authenticate

Set your Comet API key as an environment variable:

```bash
export COMET_API_KEY=<your-api-key>
```

Find your API key at [comet.com](https://www.comet.com/) under **Account → API Keys**. Alternatively, run `comet login` for an interactive setup.

!!! tip "Project naming"
    Set `COMET_PROJECT_NAME` to control which Comet project receives the run:
    ```bash
    export COMET_PROJECT_NAME=yolo11-experiments
    ```

## Train YOLO11 with Comet Logging

=== "CLI"

    ```bash
    yolo detect train \
        model=yolo11n.pt \
        data=coco8.yaml \
        epochs=100 \
        project=yolo11-comet \
        name=exp1
    ```

=== "Python"

    ```python
    from ultralytics import YOLO

    model = YOLO("yolo11n.pt")

    results = model.train(
        data="coco8.yaml",
        epochs=100,
        project="yolo11-comet",
        name="exp1",
    )
    ```

Navigate to [comet.com](https://www.comet.com/) to see the experiment appear in your project workspace.

## What Gets Logged

| Category | Details |
|---|---|
| **Metrics** | Train/val box loss, cls loss, dfl loss; `mAP50`, `mAP50-95` — logged per epoch |
| **Parameters** | Full training configuration (learning rate schedule, augmentation flags, batch size, etc.) |
| **Images** | Sample validation images with predicted and ground-truth bounding boxes |
| **Confusion Matrix** | Per-class confusion matrix captured at the end of training |
| **Model Checkpoints** | `best.pt` and `last.pt` uploaded as Comet Assets |

## Comet Panels Walkthrough

### Metrics Panel
The **Charts** tab displays all scalar metrics as time-series plots. Use the panel controls to toggle smoothing or switch between log and linear scale. Pin the most important metrics (e.g., `metrics/mAP50-95`) to the top of the dashboard.

### Image Panel
The **Graphics** tab shows sample validation images per epoch. This lets you visually inspect how the model's predictions on held-out data improve over training.

### Confusion Matrix Panel
Comet renders the confusion matrix uploaded at the end of training. Use it to identify which classes are most often confused, then refine your dataset or class definitions accordingly.

### Hyperparameters Panel
The **Other** tab lists every hyperparameter. Use **Parallel Coordinates** in the project view to visualise which parameter combinations produce the best `mAP50-95` across runs.

## Comparing Experiments

In the Comet project view, select two or more experiments and click **Compare**. The comparison view shows:

- Side-by-side metric charts
- Diff view of hyperparameters
- Image grids showing predicted outputs per model

This is especially useful when evaluating different YOLO11 model sizes (`yolo11n`, `yolo11s`, `yolo11m`) against each other.

## Disabling Comet Logging

To run a training session without Comet logging, set:

```bash
export COMET_MODE=disabled
```

Or pass `comet_ml.init(disabled=True)` before calling `model.train()`.

## Resuming a Run

Comet supports resuming interrupted experiments. Pass the existing experiment key to continue logging to the same run:

```python
import comet_ml

comet_ml.init(experiment_key="<existing-experiment-key>")

from ultralytics import YOLO
model = YOLO("yolo11n.pt")
model.train(data="coco8.yaml", epochs=100, resume=True)
```

## Frequently Asked Questions

**Does Comet work in an air-gapped environment?**
Yes. Use [Comet On-Premises](https://www.comet.com/site/products/on-premises/) and point `COMET_URL_OVERRIDE` to your internal server.

**Can I log custom metrics alongside the built-in ones?**
Yes. Use a custom Ultralytics callback:

```python
import comet_ml
from ultralytics import YOLO

experiment = comet_ml.get_global_experiment()

def on_val_end(validator):
    experiment.log_metric("custom_precision", validator.metrics.mp)

model = YOLO("yolo11n.pt")
model.add_callback("on_val_end", on_val_end)
model.train(data="coco8.yaml", epochs=50)
```
