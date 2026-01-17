---
comments: true
description: Learn how to train YOLO models on cloud GPUs with Ultralytics Platform, including remote training and real-time metrics streaming.
keywords: Ultralytics Platform, cloud training, GPU training, remote training, YOLO, model training, machine learning
---

# Cloud Training

[Ultralytics Platform](https://platform.ultralytics.com) Cloud Training offers single-click training on cloud GPUs, making model training accessible without complex setup. Train YOLO models with real-time metrics streaming and automatic checkpoint saving.

<p align="center">
  <iframe loading="lazy" width="720" height="405" src="https://www.youtube.com/embed/ie3vLUDNYZo"
    title="YouTube video player" frameborder="0"
    allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share"
    allowfullscreen>
  </iframe>
  <br>
  <strong>Watch:</strong> Cloud Training with Ultralytics Platform
</p>

## Train from UI

Start cloud training directly from the Platform:

1. Navigate to your project
2. Click **Train Model**
3. Configure training parameters
4. Click **Start Training**

### Step 1: Select Dataset

Choose a dataset from your uploads:

<!-- Screenshot: platform-training-start.avif -->

| Option              | Description                  |
| ------------------- | ---------------------------- |
| **Your Datasets**   | Datasets you've uploaded     |
| **Public Datasets** | Public datasets from Explore |

### Step 2: Configure Model

Select base model and parameters:

| Parameter      | Description                             | Default |
| -------------- | --------------------------------------- | ------- |
| **Model**      | Base architecture (YOLO26n, s, m, l, x) | YOLO26n |
| **Epochs**     | Number of training iterations           | 100     |
| **Image Size** | Input resolution                        | 640     |
| **Batch Size** | Samples per iteration                   | Auto    |

<!-- Screenshot: platform-training-config.avif -->

### Step 3: Select GPU

Choose your compute resources:

<!-- Screenshot: platform-training-gpu.avif -->

| GPU            | VRAM  | Speed     | Cost/Hour |
| -------------- | ----- | --------- | --------- |
| RTX 2000 Ada   | 16GB  | Good      | $0.48     |
| RTX A4000      | 16GB  | Good      | $0.50     |
| RTX 3090       | 24GB  | Good      | $0.92     |
| RTX A6000      | 48GB  | Fast      | $0.98     |
| RTX 4090       | 24GB  | Fast      | $1.18     |
| L40S           | 48GB  | Fast      | $1.72     |
| RTX 5090       | 32GB  | Very Fast | $1.78     |
| A100 80GB PCIe | 80GB  | Very Fast | $2.78     |
| H100 PCIe      | 80GB  | Fastest   | $4.78     |
| H100 SXM       | 80GB  | Fastest   | $5.38     |
| B200           | 180GB | Fastest   | $10.38    |

!!! tip "GPU Selection"

    - **RTX 4090**: Best value with excellent performance for most jobs
    - **A100 80GB**: Required for large batch sizes or big models
    - **H100**: Maximum performance for time-sensitive training
    - **B200**: NVIDIA Blackwell architecture for cutting-edge workloads

### Step 4: Start Training

Click **Start Training** to launch your job. The Platform:

1. Provisions a GPU instance
2. Downloads your dataset
3. Begins training
4. Streams metrics in real-time

!!! success "Free Credits"

    New accounts receive $5 in signup credits ($25 for company emails) - enough for several training runs. [Check your balance](../account/billing.md) in Settings > Billing.

<!-- Screenshot: platform-training-progress.avif -->

## Monitor Training

View real-time training progress:

### Live Metrics

<!-- Screenshot: platform-training-realtime.avif -->

| Metric        | Description                  |
| ------------- | ---------------------------- |
| **Loss**      | Training and validation loss |
| **mAP**       | Mean Average Precision       |
| **Precision** | Correct positive predictions |
| **Recall**    | Detected ground truths       |
| **GPU Util**  | GPU utilization percentage   |
| **Memory**    | GPU memory usage             |

### Checkpoints

Checkpoints are saved automatically:

- **Every epoch**: Latest weights saved
- **Best model**: Highest mAP checkpoint preserved
- **Final model**: Weights at training completion

## Stop and Resume

### Stop Training

Click **Stop Training** to pause your job:

- Current checkpoint is saved
- GPU instance is released
- Credits stop being charged

### Resume Training

Continue from your last checkpoint:

1. Navigate to the model
2. Click **Resume Training**
3. Confirm continuation

!!! note "Resume Limitations"

    You can only resume training that was explicitly stopped. Failed training jobs may need to restart from scratch.

## Remote Training

Train on your own hardware while streaming metrics to the Platform.

!!! warning "Package Version Requirement"

    Platform integration requires **ultralytics>=8.4.0**. Lower versions will NOT work with Platform.

    ```bash
    pip install "ultralytics>=8.4.0"
    ```

### Setup API Key

1. Go to Settings > API Keys
2. Create a new key with training scope
3. Set the environment variable:

```bash
export ULTRALYTICS_API_KEY="your_api_key"
```

### Train with Streaming

Use the `project` and `name` parameters to stream metrics:

=== "CLI"

    ```bash
    yolo train model=yolo26n.pt data=coco.yaml epochs=100 \
      project=username/my-project name=experiment-1
    ```

=== "Python"

    ```python
    from ultralytics import YOLO

    model = YOLO("yolo26n.pt")
    model.train(
        data="coco.yaml",
        epochs=100,
        project="username/my-project",
        name="experiment-1",
    )
    ```

### Using Platform Datasets

Train with datasets stored on the Platform:

```bash
yolo train model=yolo26n.pt data=ul://username/datasets/my-dataset epochs=100
```

The `ul://` URI format automatically downloads and configures your dataset.

## Billing

Training costs are based on GPU usage:

### Cost Calculation

```
Total Cost = GPU Rate × Training Time (hours)
```

| Example    | GPU       | Time    | Cost   |
| ---------- | --------- | ------- | ------ |
| Small job  | RTX 4090  | 1 hour  | $1.18  |
| Medium job | A100 80GB | 4 hours | $11.12 |
| Large job  | H100 PCIe | 8 hours | $38.24 |

### Payment Methods

| Method              | Description              |
| ------------------- | ------------------------ |
| **Account Balance** | Pre-loaded credits       |
| **Pay Per Job**     | Charge at job completion |

!!! note "Minimum Balance"

    A minimum balance of $5.00 is required to start epoch-based training.

### View Training Costs

After training, view detailed costs in the **Billing** tab:

- Per-epoch cost breakdown
- Total GPU time
- Download cost report

<!-- Screenshot: platform-training-complete.avif -->

## Training Tips

### Choose the Right Model Size

| Model   | Parameters | Best For                |
| ------- | ---------- | ----------------------- |
| YOLO26n | 2.4M       | Real-time, edge devices |
| YOLO26s | 9.5M       | Balanced speed/accuracy |
| YOLO26m | 20.4M      | Higher accuracy         |
| YOLO26l | 24.8M      | Production accuracy     |
| YOLO26x | 55.7M      | Maximum accuracy        |

### Optimize Training Time

1. **Start small**: Test with fewer epochs first
2. **Use appropriate GPU**: Match GPU to model/batch size
3. **Validate dataset**: Ensure quality before training
4. **Monitor early**: Stop if metrics plateau

### Troubleshooting

| Issue                | Solution                            |
| -------------------- | ----------------------------------- |
| Training stuck at 0% | Check dataset format, retry         |
| Out of memory        | Reduce batch size or use larger GPU |
| Poor accuracy        | Increase epochs, check data quality |
| Training slow        | Consider faster GPU                 |

## FAQ

### How long does training take?

Training time depends on:

- Dataset size
- Model size
- Number of epochs
- GPU selected

Typical times (1000 images, 100 epochs):

| Model   | RTX 4090 | A100   |
| ------- | -------- | ------ |
| YOLO26n | 30 min   | 20 min |
| YOLO26m | 60 min   | 40 min |
| YOLO26x | 120 min  | 80 min |

### Can I train overnight?

Yes, training continues until completion. You'll receive a notification when training finishes. Make sure your account has sufficient balance for epoch-based training.

### What happens if I run out of credits?

Training pauses at the end of the current epoch. Your checkpoint is saved, and you can resume after adding credits.

### Can I use custom training arguments?

Yes, advanced users can specify additional arguments in the training configuration.

## Training Parameters Reference

### Core Parameters

| Parameter  | Type | Default | Range     | Description               |
| ---------- | ---- | ------- | --------- | ------------------------- |
| `epochs`   | int  | 100     | 1+        | Number of training epochs |
| `batch`    | int  | 16      | -1 = auto | Batch size (-1 for auto)  |
| `imgsz`    | int  | 640     | 32+       | Input image size          |
| `patience` | int  | 100     | 0+        | Early stopping patience   |
| `workers`  | int  | 8       | 0+        | Dataloader workers        |
| `cache`    | bool | False   | -         | Cache images (ram/disk)   |

### Learning Rate Parameters

| Parameter       | Type  | Default | Range   | Description           |
| --------------- | ----- | ------- | ------- | --------------------- |
| `lr0`           | float | 0.01    | 0.0-1.0 | Initial learning rate |
| `lrf`           | float | 0.01    | 0.0-1.0 | Final LR factor       |
| `momentum`      | float | 0.937   | 0.0-1.0 | SGD momentum          |
| `weight_decay`  | float | 0.0005  | 0.0-1.0 | L2 regularization     |
| `warmup_epochs` | float | 3.0     | 0+      | Warmup epochs         |
| `cos_lr`        | bool  | False   | -       | Cosine LR scheduler   |

### Augmentation Parameters

| Parameter    | Type  | Default | Range   | Description          |
| ------------ | ----- | ------- | ------- | -------------------- |
| `hsv_h`      | float | 0.015   | 0.0-1.0 | HSV hue augmentation |
| `hsv_s`      | float | 0.7     | 0.0-1.0 | HSV saturation       |
| `hsv_v`      | float | 0.4     | 0.0-1.0 | HSV value            |
| `degrees`    | float | 0.0     | -       | Rotation degrees     |
| `translate`  | float | 0.1     | 0.0-1.0 | Translation fraction |
| `scale`      | float | 0.5     | 0.0-1.0 | Scale factor         |
| `fliplr`     | float | 0.5     | 0.0-1.0 | Horizontal flip prob |
| `flipud`     | float | 0.0     | 0.0-1.0 | Vertical flip prob   |
| `mosaic`     | float | 1.0     | 0.0-1.0 | Mosaic augmentation  |
| `mixup`      | float | 0.0     | 0.0-1.0 | Mixup augmentation   |
| `copy_paste` | float | 0.0     | 0.0-1.0 | Copy-paste (segment) |

### Optimizer Selection

| Value   | Description                   |
| ------- | ----------------------------- |
| `auto`  | Automatic selection (default) |
| `SGD`   | Stochastic Gradient Descent   |
| `Adam`  | Adam optimizer                |
| `AdamW` | Adam with weight decay        |

!!! tip "Task-Specific Parameters"

    Some parameters only apply to specific tasks:

    - **Segment**: `overlap_mask`, `mask_ratio`, `copy_paste`
    - **Pose**: `pose` (loss weight), `kobj` (keypoint objectness)
    - **Classify**: `dropout`, `erasing`, `auto_augment`
