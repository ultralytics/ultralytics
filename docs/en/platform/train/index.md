---
comments: true
description: Learn about model training in Ultralytics Platform including project organization, cloud training, and real-time metrics streaming.
keywords: Ultralytics Platform, model training, cloud training, YOLO, GPU training, machine learning, deep learning
---

# Model Training

[Ultralytics Platform](https://platform.ultralytics.com) provides comprehensive tools for training YOLO models, from organizing experiments to running cloud training jobs with real-time metrics streaming.

## Overview

The Training section helps you:

- **Organize** models into projects for easier management
- **Train** on cloud GPUs with a single click
- **Monitor** real-time metrics during training
- **Compare** model performance across experiments

<!-- Screenshot: platform-train-overview.avif -->

## Workflow

```mermaid
graph LR
    A[📁 Project] --> B[⚙️ Configure]
    B --> C[🚀 Train]
    C --> D[📈 Monitor]
    D --> E[📦 Export]

    style A fill:#4CAF50,color:#fff
    style B fill:#2196F3,color:#fff
    style C fill:#FF9800,color:#fff
    style D fill:#9C27B0,color:#fff
    style E fill:#00BCD4,color:#fff
```

| Stage         | Description                                         |
| ------------- | --------------------------------------------------- |
| **Project**   | Create a workspace to organize related models       |
| **Configure** | Select dataset, base model, and training parameters |
| **Train**     | Run on cloud GPUs or your local hardware            |
| **Monitor**   | View real-time loss curves and metrics              |
| **Export**    | Convert to 17 deployment formats                    |

## Training Options

Ultralytics Platform supports multiple training approaches:

| Method              | Description                                | Best For                   |
| ------------------- | ------------------------------------------ | -------------------------- |
| **Cloud Training**  | Train on Platform cloud GPUs               | No local GPU, scalability  |
| **Remote Training** | Train locally, stream metrics to Platform  | Existing hardware, privacy |
| **Colab Training**  | Use Google Colab with Platform integration | Free GPU access            |

## GPU Options

Available GPUs for cloud training:

| Tier       | GPU          | VRAM   | Cost/Hour | Best For                   |
| ---------- | ------------ | ------ | --------- | -------------------------- |
| Budget     | RTX A2000    | 6 GB   | $0.12     | Small datasets, testing    |
| Budget     | RTX 3080     | 10 GB  | $0.25     | Medium datasets            |
| Budget     | RTX 3080 Ti  | 12 GB  | $0.30     | Medium datasets            |
| Budget     | A30          | 24 GB  | $0.44     | Larger batch sizes         |
| Mid        | L4           | 24 GB  | $0.54     | Inference optimized        |
| Mid        | RTX 4090     | 24 GB  | $0.60     | Great price/performance    |
| Mid        | A6000        | 48 GB  | $0.90     | Large models               |
| Mid        | L40S         | 48 GB  | $1.72     | Large batch training       |
| Pro        | A100 40GB    | 40 GB  | $2.78     | Production training        |
| Pro        | A100 80GB    | 80 GB  | $3.44     | Very large models          |
| Pro        | RTX PRO 6000 | 48 GB  | $3.68     | Ultralytics infrastructure |
| Pro        | H100         | 80 GB  | $5.38     | Fastest training           |
| Enterprise | H200         | 141 GB | $5.38     | Maximum performance        |
| Enterprise | B200         | 192 GB | $10.38    | Largest models             |

!!! tip "Signup Credits"

    New accounts receive signup credits for training. Check [Billing](../account/billing.md) for details.

## Real-Time Metrics

During training, view live metrics:

- **Loss Curves**: Box, class, and DFL loss
- **Performance**: mAP50, mAP50-95, precision, recall
- **System Stats**: GPU utilization, memory usage
- **Checkpoints**: Automatic saving of best weights

## Quick Links

- [**Projects**](projects.md): Organize your models and experiments
- [**Models**](models.md): Manage trained checkpoints
- [**Cloud Training**](cloud-training.md): Train on cloud GPUs

## FAQ

### How long does training take?

Training time depends on:

- Dataset size (number of images)
- Model size (n, s, m, l, x)
- Number of epochs
- GPU type selected

A typical training run with 1000 images, YOLO26n, 100 epochs on RTX 4090 takes about 30-60 minutes.

### Can I train multiple models simultaneously?

Cloud training currently supports one concurrent training job per account. For parallel training, use remote training from multiple machines.

### What happens if training fails?

If training fails:

1. Checkpoints are saved at each epoch
2. You can resume from the last checkpoint
3. Credits are only charged for completed compute time

### How do I choose the right GPU?

| Scenario                            | Recommended GPU   |
| ----------------------------------- | ----------------- |
| Small datasets (<5000 images)       | RTX 4090          |
| Medium datasets (5000-50000 images) | A100 40GB         |
| Large datasets or batch sizes       | A100 80GB or H100 |
| Budget-conscious                    | RTX 3090          |
