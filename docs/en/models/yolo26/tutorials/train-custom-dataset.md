---
comments: true
description: Train Ultralytics YOLO26 on your own custom dataset. Covers dataset format, YAML config, training commands, results analysis, and tips for better accuracy.
keywords: YOLO26, custom dataset, train YOLO26, object detection, Ultralytics, machine learning, dataset preparation, YOLO format, transfer learning
canonical: https://docs.ultralytics.com/models/yolo26/tutorials/train-custom-dataset/
---

<script type="application/ld+json">
{
  "@context": "https://schema.org",
  "@type": "TechArticle",
  "headline": "Train YOLO26 on a Custom Dataset",
  "description": "Train Ultralytics YOLO26 on your own custom dataset. Covers dataset format, YAML config, training commands, results analysis, and tips for better accuracy.",
  "url": "https://docs.ultralytics.com/models/yolo26/tutorials/train-custom-dataset/",
  "image": "https://cdn.jsdelivr.net/gh/ultralytics/assets@main/docs/ultralytics-active-learning-loop.avif",
  "datePublished": "2026-06-04",
  "dateModified": "2026-06-04",
  "author": {"@type": "Organization", "name": "Ultralytics", "url": "https://www.ultralytics.com"},
  "publisher": {"@type": "Organization", "name": "Ultralytics", "url": "https://www.ultralytics.com"},
  "mainEntityOfPage": "https://docs.ultralytics.com/models/yolo26/tutorials/train-custom-dataset/"
}
</script>

# Train YOLO26 on a Custom Dataset

<!-- NOTE FOR MURAT: Please verify all training commands, add benchmark numbers for YOLO26 on custom datasets where available, confirm the dataset YAML format hasn't changed, and add representative screenshots of training output and results charts. Also confirm the Ultralytics Platform CTA is accurate for the current platform state. -->

This guide walks you through training [Ultralytics YOLO26](../../../models/yolo26.md) on your own custom dataset. YOLO26 is the latest Ultralytics model, delivering **40.9–57.5 mAP on COCO** with native end-to-end NMS-free inference. Training on custom data lets you apply this performance to your specific objects and use cases.

For a no-code alternative, [Ultralytics Platform](../../../platform/index.md) handles dataset management, annotation, training, and deployment in one place.

## Before You Start

Install the Ultralytics package:

```bash
pip install "ultralytics>=26.0.0"
yolo checks
```

## Step 1: Prepare Your Dataset

YOLO26 uses the standard Ultralytics dataset format: one `.txt` label file per image, containing one row per object in `class x_center y_center width height` format (all values normalised 0–1).

```
datasets/
└── my_dataset/
    ├── images/
    │   ├── train/
    │   └── val/
    └── labels/
        ├── train/
        └── val/
```

Each label file row:
```
0 0.512 0.623 0.240 0.380
```

Where `0` is the class index, followed by bounding box centre coordinates and dimensions.

!!! tip "Use Ultralytics Platform for annotation"

    [Ultralytics Platform](https://platform.ultralytics.com) provides built-in annotation tools and auto-annotation to speed up dataset creation.

## Step 2: Create a Dataset YAML

Create a `my_dataset.yaml` file defining paths, number of classes, and class names:

```yaml
# my_dataset.yaml
path: ./datasets/my_dataset  # dataset root
train: images/train
val: images/val

nc: 3  # number of classes
names: ["cat", "dog", "person"]  # class names
```

See the [datasets overview](../../../datasets/index.md) for more examples.

## Step 3: Train YOLO26

=== "CLI"

    ```bash
    # Train from a pretrained YOLO26 small checkpoint
    yolo train model=yolo26s.pt data=my_dataset.yaml epochs=100 imgsz=640

    # Train from scratch (not recommended — use pretrained weights)
    yolo train model=yolo26s.yaml data=my_dataset.yaml epochs=100 imgsz=640
    ```

=== "Python"

    ```python
    from ultralytics import YOLO

    # Load a pretrained YOLO26 model
    model = YOLO("yolo26s.pt")

    # Train on your dataset
    results = model.train(
        data="my_dataset.yaml",
        epochs=100,
        imgsz=640,
        batch=16,
    )
    ```

### Choosing a Model Scale

| Model | Size | mAP COCO | Speed (T4 TRT) | Use when |
|-------|------|-----------|----------------|----------|
| `yolo26n.pt` | Nano | 40.9 | 1.7 ms | Edge devices, speed-critical |
| `yolo26s.pt` | Small | 48.0 | 2.6 ms | Balanced — good starting point |
| `yolo26m.pt` | Medium | 52.0 | 4.4 ms | Higher accuracy needed |
| `yolo26l.pt` | Large | 55.0 | 7.0 ms | Large datasets |
| `yolo26x.pt` | XLarge | 57.5 | 11.8 ms | Maximum accuracy |

Start with `yolo26s.pt` for most custom datasets.

## Step 4: Key Training Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `epochs` | 100 | Number of training epochs |
| `imgsz` | 640 | Input image size |
| `batch` | 16 | Batch size (-1 for auto) |
| `lr0` | 0.01 | Initial learning rate |
| `patience` | 50 | Epochs before early stopping |
| `device` | — | Device: `0`, `0,1`, `cpu` |
| `pretrained` | True | Start from pretrained weights |
| `resume` | False | Resume from last checkpoint |

```bash
# Example with common overrides
yolo train model=yolo26s.pt data=my_dataset.yaml epochs=200 imgsz=640 batch=32 patience=30
```

See the full [configuration reference](../../../usage/cfg.md) for all options.

## Step 5: Validate and Analyse Results

After training, validate the best checkpoint:

```bash
yolo val model=runs/train/exp/weights/best.pt data=my_dataset.yaml
```

Training results are saved to `runs/train/exp/` and include:
- `results.csv` — per-epoch metrics (mAP, precision, recall, loss)
- `results.png` — training curves
- `confusion_matrix.png` — class-level performance
- `weights/best.pt` — best checkpoint by mAP
- `weights/last.pt` — final epoch checkpoint

## Step 6: Run Inference

```bash
# Predict on images
yolo predict model=runs/train/exp/weights/best.pt source=path/to/images

# Predict on video
yolo predict model=runs/train/exp/weights/best.pt source=path/to/video.mp4
```

## Tips for Better Results

- **More data beats more epochs.** If accuracy is low, add more training images before tuning hyperparameters.
- **Use pretrained weights.** `yolo26s.pt` transfers well even to very different domains.
- **Check your labels.** Most training failures come from label errors, not model issues. Visualise a sample with `yolo val ... plots=True`.
- **Match image size.** Train and infer at the same `imgsz` for best results.
- **Early stopping.** Set `patience=30` to avoid overfitting on small datasets.

## What's Next

- Run [Hyperparameter Tuning](./hyperparameter-tuning.md) to find optimal training settings automatically.
- Export your trained model with the [Model Export Guide](./model-export.md).
- Scale up with [Multi-GPU Training](./multi-gpu-training.md).
- Deploy via [Ultralytics Platform](../../../platform/index.md) for managed inference and monitoring.
