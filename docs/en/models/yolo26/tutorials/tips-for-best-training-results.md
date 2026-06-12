---
comments: true
description: Practical tips to get the best detection accuracy when training Ultralytics YOLO26. Data quality, augmentation, batch size, learning rate, and more.
keywords: YOLO26, training tips, best results, data quality, augmentation, learning rate, batch size, pretrained weights, Ultralytics
canonical: https://docs.ultralytics.com/models/yolo26/tutorials/tips-for-best-training-results/
---

<script type="application/ld+json">
{
  "@context": "https://schema.org",
  "@type": "TechArticle",
  "headline": "Tips for Best YOLO26 Training Results",
  "description": "Practical tips to get the best detection accuracy when training Ultralytics YOLO26. Data quality, augmentation, batch size, learning rate, and more.",
  "url": "https://docs.ultralytics.com/models/yolo26/tutorials/tips-for-best-training-results/",
  "image": "https://cdn.jsdelivr.net/gh/ultralytics/assets@main/docs/ultralytics-active-learning-loop.avif",
  "datePublished": "2026-06-04",
  "dateModified": "2026-06-04",
  "author": {"@type": "Organization", "name": "Ultralytics", "url": "https://www.ultralytics.com"},
  "publisher": {"@type": "Organization", "name": "Ultralytics", "url": "https://www.ultralytics.com"},
  "mainEntityOfPage": "https://docs.ultralytics.com/models/yolo26/tutorials/tips-for-best-training-results/"
}
</script>

# Tips for Best YOLO26 Training Results

<!-- NOTE FOR MURAT: Please verify all tips apply correctly to YOLO26's architecture — in particular, confirm whether YOLO26's end-to-end head changes the recommended anchor/NMS-related settings, whether mosaic augmentation interacts differently with the E2E head, and whether the image size guidance holds for YOLO26's specific backbone. Add any YOLO26-specific flags or defaults that differ from previous YOLO versions. Add representative before/after mAP numbers showing the impact of key tips (e.g. pretrained vs scratch, with/without mosaic). -->

Getting the best accuracy from [Ultralytics YOLO26](../../../models/yolo26.md) on your custom dataset depends on more than just training duration. This guide consolidates practical advice on data quality, model configuration, augmentation, and training hygiene.

## 1. Prioritise Data Quality

Model accuracy is bounded by data quality. No training trick compensates for poorly labelled or unrepresentative data.

!!! tip "Data quality checklist"

    - **Consistent labelling** — all annotators should use the same bounding box conventions (tight vs loose).
    - **Class balance** — aim for roughly equal numbers of instances per class, or use weighted sampling.
    - **Representative diversity** — include images from different lighting conditions, viewpoints, scales, and backgrounds that mirror real deployment conditions.
    - **Remove duplicate images** — duplicates in the training set cause overfitting without adding signal.
    - **Minimum instances** — target at least **1,500 labelled instances per class**; below 500 per class, results become unreliable.

## 2. Use Pretrained Weights

Always start from a pretrained YOLO26 checkpoint rather than training from scratch. Pretrained weights encode rich visual features from large-scale datasets and dramatically reduce the data and compute needed to reach peak accuracy.

```bash
# Good — fine-tune from pretrained weights
yolo train model=yolo26n.pt data=my_dataset.yaml epochs=100

# Avoid — training from scratch requires far more data and epochs
yolo train model=yolo26n.yaml data=my_dataset.yaml epochs=300
```

| Starting point | Epochs to convergence | Typical mAP improvement |
|---|---|---|
| `yolo26n.pt` (pretrained) | 100–150 | Baseline |
| `yolo26n.yaml` (scratch) | 300+ | −5 to −15 mAP |

## 3. Match Image Size to Your Data

YOLO26 trains at `imgsz=640` by default. If your objects are small, consider training at a larger resolution.

```bash
# Default
yolo train model=yolo26n.pt data=my_dataset.yaml imgsz=640

# Larger resolution for small objects
yolo train model=yolo26n.pt data=my_dataset.yaml imgsz=1280
```

!!! warning "Memory trade-off"

    Doubling `imgsz` roughly quadruples GPU memory usage. Use a smaller model variant or reduce batch size if you run out of memory.

## 4. Choose the Right Batch Size

Use the largest batch size your GPU memory allows. Larger batches produce more stable gradient estimates.

```bash
# Let Ultralytics auto-detect the optimal batch size
yolo train model=yolo26n.pt data=my_dataset.yaml batch=-1

# Or set explicitly
yolo train model=yolo26n.pt data=my_dataset.yaml batch=32
```

## 5. Enable Mosaic and Mixup Augmentation

Mosaic augmentation (enabled by default) combines four training images into one, giving the model exposure to diverse contexts and scales. Keep `mosaic=1.0` for most datasets; reduce it for very high-resolution or aerial imagery.

```bash
# Mosaic + Mixup (good for general datasets)
yolo train model=yolo26n.pt data=my_dataset.yaml mosaic=1.0 mixup=0.1

# Close mosaic in the final epochs to let the model stabilise
yolo train model=yolo26n.pt data=my_dataset.yaml mosaic=1.0 close_mosaic=10
```

## 6. Set an Appropriate Learning Rate

The default learning rate schedule works well for most datasets. If you are fine-tuning on a very small dataset, reduce the initial learning rate:

```bash
# Standard fine-tuning
yolo train model=yolo26n.pt data=my_dataset.yaml lr0=0.01

# Small dataset fine-tuning — lower LR reduces catastrophic forgetting
yolo train model=yolo26n.pt data=my_dataset.yaml lr0=0.001 lrf=0.1
```

## 7. Use Early Stopping

Enable patience-based early stopping to avoid wasting compute once the model has stopped improving:

```bash
yolo train model=yolo26n.pt data=my_dataset.yaml patience=50
```

Set `patience` to the number of epochs without improvement before training halts. A value of 50 is a sensible default for 200-epoch runs.

## 8. Monitor Training Results

Review the following metrics in `runs/train/expN/` after training:

| File | What to check |
|---|---|
| `results.png` | mAP, box loss, cls loss curves — all should decrease/increase smoothly |
| `PR_curve.png` | Precision-Recall curve — higher area-under-curve is better |
| `confusion_matrix.png` | Identify confused class pairs |
| `val_batch*.jpg` | Qualitative check of predictions on validation images |

!!! tip "Common issues"

    - **Loss not decreasing** — try a smaller learning rate or check for label errors.
    - **Validation loss diverging from training loss** — classic overfitting; add augmentation, use a smaller model, or collect more data.
    - **One class always missed** — check class balance and ensure enough labelled instances.

## 9. Scale Up Gradually

When exploring a new dataset, start with the nano (`yolo26n`) model for fast iteration. Only scale to larger variants once you have a validated data pipeline and know the task difficulty.

| Model | Parameters | Recommended use |
|---|---|---|
| `yolo26n.pt` | ~3M | Rapid prototyping, edge deployment |
| `yolo26s.pt` | ~12M | Balanced speed/accuracy |
| `yolo26m.pt` | ~25M | Most custom datasets |
| `yolo26l.pt` | ~43M | High-accuracy production |
| `yolo26x.pt` | ~68M | Maximum accuracy (benchmark) |

## See Also

- [YOLO26 Model Overview](../../../models/yolo26.md)
- [Train YOLO26 on a Custom Dataset](train-custom-dataset.md)
- [Hyperparameter Tuning for YOLO26](hyperparameter-tuning.md)
- [Multi-GPU Training](multi-gpu-training.md)
