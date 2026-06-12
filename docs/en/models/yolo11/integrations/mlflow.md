---
comments: true
description: Track YOLO11 experiments with MLflow. Log parameters, metrics, and artifacts, then register trained models in the MLflow Model Registry for deployment.
keywords: YOLO11, MLflow, experiment tracking, model registry, MLOps, Ultralytics, training metrics, artifact logging
canonical: https://docs.ultralytics.com/models/yolo11/integrations/mlflow/
---

<script type="application/ld+json">
{
  "@context": "https://schema.org",
  "@type": "TechArticle",
  "headline": "YOLO11 MLflow Integration",
  "description": "Track YOLO11 experiments with MLflow. Log parameters, metrics, and artifacts, then register trained models in the MLflow Model Registry for deployment.",
  "url": "https://docs.ultralytics.com/models/yolo11/integrations/mlflow/",
  "image": "https://cdn.jsdelivr.net/gh/ultralytics/assets@main/docs/ultralytics-active-learning-loop.avif",
  "datePublished": "2026-06-05",
  "dateModified": "2026-06-05",
  "author": {"@type": "Organization", "name": "Ultralytics", "url": "https://www.ultralytics.com"},
  "publisher": {"@type": "Organization", "name": "Ultralytics", "url": "https://www.ultralytics.com"},
  "mainEntityOfPage": "https://docs.ultralytics.com/models/yolo11/integrations/mlflow/"
}
</script>

# YOLO11 + MLflow Integration

<!-- NOTE FOR MURAT: Please verify mlflow integration works with YOLO11 (pip install ultralytics mlflow), add a screenshot of the MLflow UI showing a YOLO11 experiment run (params, metrics, artifacts tabs), confirm the MLFLOW_TRACKING_URI env var approach still works, verify model registry steps (mlflow.register_model path) are accurate, and confirm the mlflow ui command launches correctly. -->

[MLflow](https://mlflow.org/) is an open-source platform for managing the full machine-learning lifecycle — experiment tracking, reproducible runs, model packaging, and a model registry. [Ultralytics YOLO11](../../../models/yolo11.md) logs to MLflow automatically when the `mlflow` package is installed.

## Install

```bash
pip install ultralytics mlflow
```

## Configure the Tracking Server

By default MLflow logs to a local `./mlruns` directory. To use a remote tracking server set the `MLFLOW_TRACKING_URI` environment variable before training:

=== "Local"

    ```bash
    # Starts a local UI at http://127.0.0.1:5000
    mlflow ui
    export MLFLOW_TRACKING_URI=http://127.0.0.1:5000
    ```

=== "Remote"

    ```bash
    export MLFLOW_TRACKING_URI=https://my-mlflow-server.example.com
    export MLFLOW_TRACKING_USERNAME=<user>
    export MLFLOW_TRACKING_PASSWORD=<password>
    ```

!!! tip "Databricks Managed MLflow"
    If you use Databricks, set `MLFLOW_TRACKING_URI=databricks` and configure `DATABRICKS_HOST` / `DATABRICKS_TOKEN`.

## Train YOLO11 with MLflow Logging

=== "CLI"

    ```bash
    yolo detect train \
        model=yolo11n.pt \
        data=coco8.yaml \
        epochs=100 \
        project=yolo11-mlflow \
        name=exp1
    ```

=== "Python"

    ```python
    from ultralytics import YOLO

    model = YOLO("yolo11n.pt")

    results = model.train(
        data="coco8.yaml",
        epochs=100,
        project="yolo11-mlflow",
        name="exp1",
    )
    ```

Ultralytics detects MLflow at startup and creates or resumes an MLflow experiment named after the `project` argument. Each run corresponds to one training invocation.

## What Gets Logged

| Category | Examples |
|---|---|
| **Parameters** | `epochs`, `batch`, `imgsz`, `lr0`, `lrf`, `optimizer`, `augment` flags |
| **Metrics** | `train/box_loss`, `train/cls_loss`, `val/mAP50`, `val/mAP50-95` (logged per epoch) |
| **Artifacts** | `best.pt`, `last.pt`, `results.csv`, confusion matrix PNG, `args.yaml` |

## Viewing Results in the MLflow UI

Start the MLflow UI if using local tracking:

```bash
mlflow ui --port 5000
```

Navigate to `http://127.0.0.1:5000` and select your experiment. The **Runs** table shows all training runs with their metrics. Click a run to drill into:

- **Parameters tab** — full hyperparameter config
- **Metrics tab** — interactive loss and mAP plots per epoch
- **Artifacts tab** — model weights and charts available for download

## Model Registry

Register a trained YOLO11 model for versioned deployment:

```python
import mlflow

# Log and register in one step
mlflow.register_model(
    model_uri=f"runs:/<run_id>/best.pt",
    name="yolo11-coco",
)
```

Transition a version to **Staging** or **Production**:

```python
from mlflow.tracking import MlflowClient

client = MlflowClient()
client.transition_model_version_stage(
    name="yolo11-coco",
    version=1,
    stage="Production",
)
```

Load a registered model for inference:

```python
from ultralytics import YOLO

# Download the artifact and load with Ultralytics
client = MlflowClient()
model_uri = client.get_model_version_download_uri("yolo11-coco", "1")
model = YOLO(model_uri)
results = model("image.jpg")
```

## Comparing Runs

Use the MLflow Compare view to benchmark multiple configurations side-by-side. Select two or more runs in the Experiments table and click **Compare**. The parallel coordinates plot is especially useful for identifying which hyperparameters drive `mAP50-95`.

## Frequently Asked Questions

**Can I use MLflow with the Ultralytics Platform?**
Yes. Set `MLFLOW_TRACKING_URI` to your server and train from the Platform; experiments are mirrored to MLflow automatically.

**How do I log custom tags to a run?**
Use a custom callback:

```python
import mlflow
from ultralytics import YOLO
from ultralytics.utils import callbacks

def on_train_start(trainer):
    mlflow.set_tag("dataset_version", "v2.1")

model = YOLO("yolo11n.pt")
model.add_callback("on_train_start", on_train_start)
model.train(data="coco8.yaml", epochs=50)
```
