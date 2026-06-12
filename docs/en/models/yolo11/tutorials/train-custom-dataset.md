---
comments: true
description: Train Ultralytics YOLO11 on your own custom dataset. Covers dataset format, YAML config, training commands, results analysis, and tips for better accuracy.
keywords: YOLO11, custom dataset, train YOLO11, object detection, Ultralytics, machine learning, dataset preparation, YOLO format, transfer learning
canonical: https://docs.ultralytics.com/models/yolo11/tutorials/train-custom-dataset/
---

<script type="application/ld+json">
{
  "@context": "https://schema.org",
  "@type": "TechArticle",
  "headline": "Train YOLO11 on a Custom Dataset",
  "description": "Train Ultralytics YOLO11 on your own custom dataset. Covers dataset format, YAML config, training commands, results analysis, and tips for better accuracy.",
  "url": "https://docs.ultralytics.com/models/yolo11/tutorials/train-custom-dataset/",
  "image": "https://cdn.jsdelivr.net/gh/ultralytics/assets@main/docs/ultralytics-active-learning-loop.avif",
  "datePublished": "2026-06-04",
  "dateModified": "2026-06-04",
  "author": {"@type": "Organization", "name": "Ultralytics", "url": "https://www.ultralytics.com"},
  "publisher": {"@type": "Organization", "name": "Ultralytics", "url": "https://www.ultralytics.com"},
  "mainEntityOfPage": "https://docs.ultralytics.com/models/yolo11/tutorials/train-custom-dataset/"
}
</script>

# Train YOLO11 on a Custom Dataset

<!-- NOTE FOR MURAT: Please verify all training commands, add benchmark numbers for YOLO11 on custom datasets where available, confirm the dataset YAML format hasn't changed, and add representative screenshots of training output and results charts. Also confirm the Ultralytics Platform CTA is accurate for the current platform state. -->

This guide walks you through training [Ultralytics YOLO11](../../../models/yolo11.md) on your own custom dataset. YOLO11 delivers **39.5–54.7 mAP on COCO** across model sizes, with an improved architecture for efficiency and accuracy. Training on custom data lets you apply this performance to your specific objects and use cases.

For a no-code alternative, [Ultralytics Platform](../../../platform/index.md) handles dataset management, annotation, training, and deployment in one place.

## Before You Start

Install the Ultralytics package:

```bash
pip install ultralytics
yolo checks
```

## Step 1: Prepare Your Dataset

YOLO11 uses the standard Ultralytics dataset format: one `.txt` label file per image, containing one row per object in `class x_center y_center width height` format (all values normalised 0–1).

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

## Step 3: Train YOLO11

=== "CLI"

    ```bash
    # Fine-tune from pretrained YOLO11n weights
    yolo train model=yolo11n.pt data=my_dataset.yaml epochs=100 imgsz=640

    # Larger model for higher accuracy
    yolo train model=yolo11m.pt data=my_dataset.yaml epochs=100 imgsz=640
    ```

=== "Python"

    ```python
    from ultralytics import YOLO

    model = YOLO("yolo11n.pt")  # load pretrained

    results = model.train(
        data="my_dataset.yaml",
        epochs=100,
        imgsz=640,
        batch=16,
    )
    ```

## Step 4: Choose a Model Size

| Model | mAP<sup>val</sup> 50-95 | Speed CPU (ms) | Params |
|---|---|---|---|
| `yolo11n.pt` | 39.5 | ~56 | ~2.6M |
| `yolo11s.pt` | 47.0 | ~90 | ~9.4M |
| `yolo11m.pt` | 51.5 | ~183 | ~20.1M |
| `yolo11l.pt` | 53.4 | ~238 | ~25.3M |
| `yolo11x.pt` | 54.7 | ~462 | ~56.9M |

Start with `yolo11n.pt` for fast iteration, then scale up once your data pipeline is validated.

## Step 5: Evaluate Results

After training, evaluate on your validation set:

```bash
yolo val model=runs/train/exp/weights/best.pt data=my_dataset.yaml
```

Review the output in `runs/train/exp/`:

| File | What to check |
|---|---|
| `results.png` | Loss and mAP curves |
| `confusion_matrix.png` | Per-class performance |
| `PR_curve.png` | Precision-Recall trade-off |
| `val_batch*.jpg` | Qualitative predictions |

## Step 6: Run Inference

```bash
yolo predict model=runs/train/exp/weights/best.pt source=path/to/images/
```

## Next Steps

- [Export YOLO11 for deployment](model-export.md)
- [Tune hyperparameters](hyperparameter-tuning.md)
- [Multi-GPU training](multi-gpu-training.md)
- [Tips for best results](tips-for-best-training-results.md)
