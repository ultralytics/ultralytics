---
comments: true
description: Log YOLO26 training runs to Weights & Biases. Track loss curves, mAP, images, and model artifacts automatically with Ultralytics' built-in W&B integration.
keywords: YOLO26, Weights and Biases, wandb, experiment tracking, MLOps, Ultralytics, loss curves, mAP, model artifacts
canonical: https://docs.ultralytics.com/models/yolo26/integrations/weights-and-biases/
---

<script type="application/ld+json">
{
  "@context": "https://schema.org",
  "@type": "TechArticle",
  "headline": "YOLO26 Weights & Biases Integration",
  "description": "Log YOLO26 training runs to Weights & Biases. Track loss curves, mAP, images, and model artifacts automatically with Ultralytics' built-in W&B integration.",
  "url": "https://docs.ultralytics.com/models/yolo26/integrations/weights-and-biases/",
  "image": "https://cdn.jsdelivr.net/gh/ultralytics/assets@main/docs/ultralytics-active-learning-loop.avif",
  "datePublished": "2026-06-05",
  "dateModified": "2026-06-05",
  "author": {"@type": "Organization", "name": "Ultralytics", "url": "https://www.ultralytics.com"},
  "publisher": {"@type": "Organization", "name": "Ultralytics", "url": "https://www.ultralytics.com"},
  "mainEntityOfPage": "https://docs.ultralytics.com/models/yolo26/integrations/weights-and-biases/"
}
</script>

# YOLO26 + Weights & Biases Integration

<!-- NOTE FOR MURAT: Please verify wandb auto-integration works correctly with YOLO26 (pip install ultralytics>=26.0.0 wandb), add a screenshot of a real W&B dashboard showing a YOLO26 run (loss curves, mAP plot, sample images panel, artifact list), confirm artifact logging includes best.pt and last.pt, and verify the wandb.login() / WANDB_API_KEY flow is current. -->

[Weights & Biases (W&B)](https://wandb.ai/) is a machine-learning experiment-tracking platform that records metrics, hyperparameters, model artifacts, and media in real time. [Ultralytics YOLO26](../../../models/yolo26.md) automatically logs to W&B whenever the `wandb` package is installed — no extra code required.

## Install

```bash
pip install "ultralytics>=26.0.0" wandb
```

Authenticate once with your W&B account:

```bash
wandb login
```

Or set the environment variable:

```bash
export WANDB_API_KEY=<your-api-key>
```

## How Auto-Detection Works

Ultralytics checks for installed integrations at the start of every training run. When `wandb` is importable, a W&B run is initialised automatically using the training `project` and `name` arguments as the W&B project and run name. No callback registration is needed.

!!! tip "Disabling W&B"
    Set `WANDB_MODE=disabled` in your environment to turn off W&B logging without uninstalling the package.

## Train YOLO26 with W&B Logging

=== "CLI"

    ```bash
    yolo detect train \
        model=yolo26n.pt \
        data=coco8.yaml \
        epochs=100 \
        project=yolo26-runs \
        name=exp1
    ```

=== "Python"

    ```python
    from ultralytics import YOLO

    model = YOLO("yolo26n.pt")

    results = model.train(
        data="coco8.yaml",
        epochs=100,
        project="yolo26-runs",
        name="exp1",
    )
    ```

After training starts, open your [W&B workspace](https://wandb.ai/) to see the live run.

## What Gets Logged

| Category | Details |
|---|---|
| **Scalars** | `train/box_loss`, `train/cls_loss`, `train/dfl_loss`, `val/box_loss`, `val/cls_loss`, `val/dfl_loss`, `metrics/mAP50`, `metrics/mAP50-95` |
| **Hyperparameters** | Full training config (learning rate, augmentation settings, batch size, etc.) |
| **Media** | Validation batch images with predicted and ground-truth bounding boxes |
| **Artifacts** | `best.pt` and `last.pt` model weights logged as versioned W&B Artifacts |

## W&B Dashboard Walkthrough

Once training completes, navigate to your W&B project to explore these panels:

### Metrics Panel
The default **Charts** tab shows all scalar metrics over epochs. Compare multiple runs side-by-side by selecting them in the left sidebar.

### System Panel
GPU utilisation, memory, and CPU usage are recorded automatically. Use this to spot training bottlenecks.

### Media Panel
The **Media** tab shows sample validation images per epoch, letting you visually track how predictions improve.

### Artifacts
Under the **Artifacts** tab, find `best.pt` linked to the run that produced it. Download directly or reference it in downstream pipelines:

```python
import wandb

run = wandb.init()
artifact = run.use_artifact("entity/project/run-<id>-best:latest", type="model")
artifact_dir = artifact.download()
```

### Hyperparameter Table
The **Overview** tab lists every hyperparameter passed to training. Use the **Table** view to compare configurations across sweeps.

## Hyperparameter Sweeps

Combine W&B Sweeps with YOLO26 for automated hyperparameter optimisation:

```python
import wandb
from ultralytics import YOLO

sweep_config = {
    "method": "bayes",
    "metric": {"name": "metrics/mAP50-95", "goal": "maximize"},
    "parameters": {
        "lr0": {"min": 1e-5, "max": 1e-1},
        "batch": {"values": [16, 32, 64]},
    },
}

def train():
    with wandb.init() as run:
        cfg = run.config
        model = YOLO("yolo26n.pt")
        model.train(
            data="coco8.yaml",
            epochs=50,
            lr0=cfg.lr0,
            batch=cfg.batch,
            project="yolo26-sweep",
            name=run.name,
        )

sweep_id = wandb.sweep(sweep_config, project="yolo26-sweep")
wandb.agent(sweep_id, train, count=20)
```

## Frequently Asked Questions

**Does W&B logging slow down training?**
The overhead is negligible. Metrics are written to a local buffer and uploaded asynchronously.

**Can I log custom metrics?**
Yes. Call `wandb.log({"my_metric": value})` anywhere inside a custom callback and it will appear alongside the built-in metrics.

**How do I share a run with my team?**
Runs are visible to all members of your W&B team by default. Share a direct link from the run page.
