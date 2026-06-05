---
comments: true
description: Track YOLO11 experiments with ClearML. Auto-log metrics, datasets, and models, then run hyperparameter optimization with ClearML's HPO and tune().
keywords: YOLO11, ClearML, experiment tracking, HPO, hyperparameter optimization, MLOps, Ultralytics, dataset management
canonical: https://docs.ultralytics.com/models/yolo11/integrations/clearml/
---

<script type="application/ld+json">
{
  "@context": "https://schema.org",
  "@type": "TechArticle",
  "headline": "YOLO11 ClearML Integration",
  "description": "Track YOLO11 experiments with ClearML. Auto-log metrics, datasets, and models, then run hyperparameter optimization with ClearML's HPO and tune().",
  "url": "https://docs.ultralytics.com/models/yolo11/integrations/clearml/",
  "image": "https://cdn.jsdelivr.net/gh/ultralytics/assets@main/docs/ultralytics-active-learning-loop.avif",
  "datePublished": "2026-06-05",
  "dateModified": "2026-06-05",
  "author": {"@type": "Organization", "name": "Ultralytics", "url": "https://www.ultralytics.com"},
  "publisher": {"@type": "Organization", "name": "Ultralytics", "url": "https://www.ultralytics.com"},
  "mainEntityOfPage": "https://docs.ultralytics.com/models/yolo11/integrations/clearml/"
}
</script>

# YOLO11 + ClearML Integration

<!-- NOTE FOR MURAT: Please verify clearml auto-integration works with YOLO11 (pip install ultralytics clearml, clearml-init), add a screenshot of the ClearML UI showing a YOLO11 experiment (scalars, plots, artifacts, and hyper-parameters tabs), confirm the HPO integration using model.tune() works as described, and verify the ClearML Dataset approach for dataset versioning is current. -->

[ClearML](https://clear.ml/) is an open-source MLOps platform providing experiment tracking, dataset management, and automated hyperparameter optimisation. [Ultralytics YOLO11](../../../models/yolo11.md) integrates with ClearML automatically — installing the `clearml` package and running `clearml-init` is all that's needed.

## Install

```bash
pip install ultralytics clearml
```

Initialise ClearML credentials once (generates `~/clearml.conf`):

```bash
clearml-init
```

Follow the prompts to enter your ClearML server URL and API credentials. If you use the hosted ClearML server at [app.clear.ml](https://app.clear.ml/), copy the credentials from **Settings → Workspace → Create new credentials**.

!!! note "Self-hosted ClearML"
    For a self-hosted server set `api.web_server`, `api.api_server`, and `api.files_server` in `clearml.conf` to your instance URLs.

## Train YOLO11 with ClearML Logging

=== "CLI"

    ```bash
    yolo detect train \
        model=yolo11n.pt \
        data=coco8.yaml \
        epochs=100 \
        project=yolo11-clearml \
        name=exp1
    ```

=== "Python"

    ```python
    from ultralytics import YOLO

    model = YOLO("yolo11n.pt")

    results = model.train(
        data="coco8.yaml",
        epochs=100,
        project="yolo11-clearml",
        name="exp1",
    )
    ```

A ClearML Task is created automatically under the project specified by `project`. Open [app.clear.ml](https://app.clear.ml/) to see the live run.

## What Gets Tracked

| Category | Details |
|---|---|
| **Scalars** | Train/val losses and mAP metrics, logged per epoch |
| **Plots** | Confusion matrix, PR curve, F1 curve, results plots |
| **Hyper-Parameters** | Complete training configuration stored in the Task |
| **Artifacts** | `best.pt`, `last.pt`, `results.csv` uploaded to ClearML file server |
| **Debug Images** | Sample validation predictions shown in the **Debug Samples** panel |

## ClearML UI Walkthrough

After training, navigate to your experiment in the ClearML UI:

### Scalars Tab
All loss and mAP metrics are plotted interactively. Hover over any point to see the exact epoch value. Select multiple experiments in the project view to overlay their scalar curves for comparison.

### Plots Tab
Static plots (confusion matrix, PR curve) are captured automatically and stored here as images.

### Hyper-Parameters Tab
Every argument passed to `model.train()` is stored here. When cloning an experiment (see below), these parameters are pre-populated for easy configuration changes.

### Artifacts Tab
Model weights and CSV logs are available for download or can be consumed by downstream ClearML Pipeline steps.

## Dataset Versioning with ClearML Datasets

Version your training datasets alongside experiments:

```python
from clearml import Dataset

dataset = Dataset.create(
    dataset_name="my-detection-dataset",
    dataset_project="yolo11-data",
)
dataset.add_files("datasets/my_dataset/")
dataset.upload()
dataset.finalize()
print(dataset.id)  # Use this ID to reference the dataset in experiments
```

Link the dataset to a training run:

```python
from clearml import Task
from ultralytics import YOLO

task = Task.init(project_name="yolo11-clearml", task_name="train-with-dataset")
task.connect_configuration({"dataset_id": "<your-dataset-id>"})

model = YOLO("yolo11n.pt")
model.train(data="coco8.yaml", epochs=100)
```

## Hyperparameter Optimisation with tune()

ClearML's HyperParameter Optimisation controller works seamlessly with YOLO11's built-in `tune()` method:

```python
from ultralytics import YOLO

model = YOLO("yolo11n.pt")

# tune() runs multiple training trials; ClearML logs each as a child Task
result_grid = model.tune(
    data="coco8.yaml",
    epochs=50,
    iterations=30,
    optimizer="AdamW",
    plots=False,
    save=False,
    val=False,
)
```

Each `tune()` iteration creates a separate ClearML Task linked to the parent optimisation task. View the full sweep in the **Hyper-Parameter Optimization** section of your ClearML project.

## Cloning and Re-running Experiments

ClearML makes it trivial to reproduce any past experiment:

1. In the ClearML UI, right-click a Task and select **Clone**.
2. Modify any hyper-parameters in the cloned Task.
3. Click **Enqueue** to send it to a ClearML Agent for execution.

This enables reproducible experimentation without modifying local code.

## Frequently Asked Questions

**Does ClearML work offline?**
Yes. Set `CLEARML_OFFLINE_MODE=1` and runs are cached locally. Upload them later with `clearml-task import`.

**Can I track multiple GPUs?**
ClearML records GPU stats per device automatically. For multi-GPU runs the metrics from all devices are aggregated.
