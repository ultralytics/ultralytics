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

| Option | Description |
|--------|-------------|
| **Your Datasets** | Datasets you've uploaded |
| **Public Datasets** | Shared datasets from Explore |

### Step 2: Configure Model

Select base model and parameters:

| Parameter | Description | Default |
|-----------|-------------|---------|
| **Model** | Base architecture (YOLO11n, s, m, l, x) | YOLO11n |
| **Epochs** | Number of training iterations | 100 |
| **Image Size** | Input resolution | 640 |
| **Batch Size** | Samples per iteration | Auto |

<!-- Screenshot: platform-training-config.avif -->

### Step 3: Select GPU

Choose your compute resources:

<!-- Screenshot: platform-training-gpu.avif -->

| GPU | VRAM | Speed | Cost/Hour |
|-----|------|-------|-----------|
| RTX 3090 | 24GB | Good | $0.44 |
| RTX 4090 | 24GB | Fast | $0.74 |
| L40S | 48GB | Fast | $1.14 |
| A100 40GB | 40GB | Very Fast | $1.29 |
| A100 80GB | 80GB | Very Fast | $1.99 |
| H100 80GB | 80GB | Fastest | $3.99 |

!!! tip "GPU Selection"

    - **RTX 4090**: Best value for most training jobs
    - **A100 80GB**: Required for large batch sizes or big models
    - **H100**: Maximum performance for time-sensitive training

### Step 4: Start Training

Click **Start Training** to launch your job. The Platform:

1. Provisions a GPU instance
2. Downloads your dataset
3. Begins training
4. Streams metrics in real-time

<!-- Screenshot: platform-training-progress.avif -->

## Monitor Training

View real-time training progress:

### Live Metrics

<!-- Screenshot: platform-training-realtime.avif -->

| Metric | Description |
|--------|-------------|
| **Loss** | Training and validation loss |
| **mAP** | Mean Average Precision |
| **Precision** | Correct positive predictions |
| **Recall** | Detected ground truths |
| **GPU Util** | GPU utilization percentage |
| **Memory** | GPU memory usage |

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

Train on your own hardware while streaming metrics to the Platform:

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
    yolo train model=yolo11n.pt data=coco.yaml epochs=100 \
        project=username/my-project name=experiment-1
    ```

=== "Python"

    ```python
    from ultralytics import YOLO

    model = YOLO("yolo11n.pt")
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
yolo train model=yolo11n.pt data=ul://username/datasets/my-dataset epochs=100
```

The `ul://` URI format automatically downloads and configures your dataset.

## Billing

Training costs are based on GPU usage:

### Cost Calculation

```
Total Cost = GPU Rate × Training Time (hours)
```

| Example | GPU | Time | Cost |
|---------|-----|------|------|
| Small job | RTX 4090 | 1 hour | $0.74 |
| Medium job | A100 40GB | 4 hours | $5.16 |
| Large job | H100 | 8 hours | $31.92 |

### Payment Methods

| Method | Description |
|--------|-------------|
| **Account Balance** | Pre-loaded credits |
| **Pay Per Job** | Charge at job completion |

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

| Model | Parameters | Best For |
|-------|------------|----------|
| YOLO11n | 2.6M | Real-time, edge devices |
| YOLO11s | 9.4M | Balanced speed/accuracy |
| YOLO11m | 20.1M | Higher accuracy |
| YOLO11l | 25.3M | Production accuracy |
| YOLO11x | 56.9M | Maximum accuracy |

### Optimize Training Time

1. **Start small**: Test with fewer epochs first
2. **Use appropriate GPU**: Match GPU to model/batch size
3. **Validate dataset**: Ensure quality before training
4. **Monitor early**: Stop if metrics plateau

### Troubleshooting

| Issue | Solution |
|-------|----------|
| Training stuck at 0% | Check dataset format, retry |
| Out of memory | Reduce batch size or use larger GPU |
| Poor accuracy | Increase epochs, check data quality |
| Training slow | Consider faster GPU |

## FAQ

### How long does training take?

Training time depends on:

- Dataset size
- Model size
- Number of epochs
- GPU selected

Typical times (1000 images, 100 epochs):

| Model | RTX 4090 | A100 |
|-------|----------|------|
| YOLO11n | 30 min | 20 min |
| YOLO11m | 60 min | 40 min |
| YOLO11x | 120 min | 80 min |

### Can I train overnight?

Yes, training continues until completion. You'll receive a notification when training finishes. Make sure your account has sufficient balance for epoch-based training.

### What happens if I run out of credits?

Training pauses at the end of the current epoch. Your checkpoint is saved, and you can resume after adding credits.

### Can I use custom training arguments?

Yes, advanced users can specify additional arguments in the training configuration. Common options:

| Argument | Description |
|----------|-------------|
| `patience` | Early stopping patience |
| `lr0` | Initial learning rate |
| `augment` | Enable augmentation |
| `cache` | Cache images in RAM |
