---
comments: true
description: Learn how YOLO26 models were trained on COCO, including optimizer settings, augmentation pipelines, loss weights, and fine-tuning guidance for each model size.
keywords: YOLO26, training recipe, pretraining, fine-tuning, MuSGD, augmentation, loss weights, COCO, model card, hyperparameters, Ultralytics, object detection, deep learning, data augmentation
---

# YOLO26 Training Recipe

## Introduction

This guide documents the exact [training](../modes/train.md) recipe used to produce the official [YOLO26](../models/yolo26.md) pretrained checkpoints on [COCO](../datasets/detect/coco.md). Every [hyperparameter](https://www.ultralytics.com/glossary/hyperparameter-tuning) shown here is already embedded in the released `.pt` weights and can be inspected programmatically.

Knowing what went into the official checkpoints — not just the architecture, but the [learning rate](https://www.ultralytics.com/glossary/learning-rate) schedules, augmentation pipelines, and loss weights that shaped their performance — helps you make better decisions when [fine-tuning](https://www.ultralytics.com/glossary/fine-tuning): which [data augmentations](./yolo-data-augmentation.md) to keep, which [loss function](https://www.ultralytics.com/glossary/loss-function) weights to adjust, and what optimizer settings work best for your dataset size.

## Training Overview

All YOLO26 base models were trained in two stages:

1. **Stage 1: Objects365 Pretraining** - Models were first pretrained on the Objects365 dataset.
2. **Stage 2: COCO Fine-tuning** - The Stage 1 weights were then fine-tuned on COCO for 245 epochs (N) or fewer (S/M/L/X).

Full training logs and metrics for every model size are available on [Ultralytics Platform](https://platform.ultralytics.com/ultralytics/yolo26):

<iframe
  src="https://platform.ultralytics.com/embed/ultralytics/yolo26"
  title="YOLO26 COCO training runs and metrics on Ultralytics Platform"
  loading="lazy"
  scrolling="no"
  width="100%"
  height="290px"
  style="border:none"
></iframe>

## Stage 1: Objects365 Pretraining

The Objects365 pretraining stage uses the following starting weights, which can be inspected programmatically:

!!! example "Inspect Stage 1 checkpoint"
```python
from ultralytics import YOLO

    # Replace 'yolo26n-objv1-150.pt' with the appropriate size
    model = YOLO("yolo26n-objv1-150.pt")
    print(model.ckpt["train_args"])
    ```

The table below shows the per-size starting weights and key hyperparameters for Stage 1. These values are read directly from the released checkpoints.

|| Setting | N | S | M | L | X |
|| ------- | - | - | - | - | - |
|| `Starting Weights` | yolo26n-objv1-150.pt | yolo26s-objv1-150.pt | yolo26m-objv1-150.pt | yolo26l-objv1-150.pt | yolo26x-objv1-150.pt |
|| `epochs` | 150 | 150 | 150 | 150 | 150 |
|| `batch` | 128 | 128 | 128 | 128 | 128 |
|| `imgsz` | 640 | 640 | 640 | 640 | 640 |
|| `optimizer` | MuSGD | MuSGD | MuSGD | MuSGD | MuSGD |
|| `lr0` | 0.0054 | 0.00038 | 0.00038 | 0.00038 | 0.00038 |
|| `lrf` | 0.0495 | 0.882 | 0.882 | 0.882 | 0.882 |
|| `momentum` | 0.947 | 0.948 | 0.948 | 0.948 | 0.948 |
|| `weight_decay` | 0.00064 | 0.00027 | 0.00027 | 0.00027 | 0.00027 |
|| `warmup_epochs` | 0.98 | 0.99 | 0.99 | 0.99 | 0.99 |
|| `muon_w` | 0.528 | 0.436 | 0.436 | 0.436 | 0.436 |
|| `sgd_w` | 0.674 | 0.479 | 0.479 | 0.479 | 0.479 |
|| `o2m` | 1.0 | 0.705 | 0.705 | 0.705 | 0.705 |

Note: The above values are examples and should be verified by inspecting the actual checkpoints as shown above.

## Inspecting YOLO26 Checkpoint Training Args

Every Ultralytics checkpoint stores the full training configuration used to produce it, so you can verify each number on this page yourself:

!!! example "Inspect checkpoint training args"

    === "Ultralytics API"

        ```python
        from ultralytics import YOLO

        model = YOLO("yolo26n.pt")
        print(model.ckpt["train_args"])
        ```

    === "PyTorch"

        ```python
        import torch

        # Load any official checkpoint
        ckpt = torch.load("yolo26n.pt", map_location="cpu", weights_only=False)

        # Print all training arguments
        for k, v in sorted(ckpt["train_args"].items()):
            print(f"{k}: {v}")
        ```

The output lists the full configuration of over 100 entries, including every recipe value documented on this page. An excerpt for `yolo26n.pt`:

```plaintext
batch: 128
...
box: 5.62767
...
close_mosaic: 10
cls: 0.56099
...
dfl: 9.03871
...
epochs: 245
...
lr0: 0.0054
lrf: 0.04952
...
optimizer: MuSGD
```

This works for any `.pt` checkpoint — official releases and your own fine-tuned models alike. For the full list of configurable training arguments, see the [training configuration reference](../usage/cfg.md).

## Stage 2: COCO Fine-tuning

The tables below group the recipe by category — optimizer and schedule, loss weights, and augmentation. Every value comes straight from the `train_args` embedded in the released checkpoints.

### Optimizer and Learning Rate

These optimizer and schedule settings drove COCO pretraining for each size; note how the N model stands apart from the rest:

| Setting         | N       | S       | M       | L       | X       |
| --------------- | ------- | ------- | ------- | ------- | ------- |
| `optimizer`     | MuSGD   | MuSGD   | MuSGD   | MuSGD   | MuSGD   |
| `lr0`           | 0.0054  | 0.00038 | 0.00038 | 0.00038 | 0.00038 |
| `lrf`           | 0.0495  | 0.882   | 0.882   | 0.882   | 0.882   |
| `momentum`      | 0.947   | 0.948   | 0.948   | 0.948   | 0.948   |
| `weight_decay`  | 0.00064 | 0.00027 | 0.00027 | 0.00027 | 0.00027 |
| `warmup_epochs` | 0.98    | 0.99    | 0.99    | 0.99    | 0.99    |
| `epochs`        | 245     | 70      | 80      | 60      | 40      |
| `batch`         | 128     | 128     | 128     | 128     | 128     |
| `imgsz`         | 640     | 640     | 640     | 640     | 640     |

!!! info "Learning rate strategy"

    The N model used a higher initial learning rate with steep decay (`lrf=0.0495`), while S/M/L/X models used a much lower initial LR with a gentler schedule (`lrf=0.882`). This reflects the different convergence dynamics of smaller vs larger models — smaller models need more aggressive updates to learn effectively.

### Loss Weights

Loss weights balance the three components of the detection loss — [bounding box](https://www.ultralytics.com/glossary/bounding-box) IoU regression (`box`), classification (`cls`), and a box-distance regression term (`dfl`). Note that DFL-free YOLO26 repurposes the `dfl` gain to weight an L1 loss on normalized box distances rather than distribution focal loss:

| Setting | N    | S    | M    | L    | X    |
| ------- | ---- | ---- | ---- | ---- | ---- |
| `box`   | 5.63 | 9.83 | 9.83 | 9.83 | 9.83 |
| `cls`   | 0.56 | 0.65 | 0.65 | 0.65 | 0.65 |
| `dfl`   | 9.04 | 0.96 | 0.96 | 0.96 | 0.96 |

The N model prioritizes the `dfl` distance-regression term, while S/M/L/X models shift emphasis to IoU-based box regression. Classification loss remains relatively consistent across all sizes.

### Augmentation Pipeline

For a detailed explanation of each technique, see the [YOLO Data Augmentation guide](./yolo-data-augmentation.md).

| Setting                                                            | N     | S     | M     | L     | X     |
| ------------------------------------------------------------------ | ----- | ----- | ----- | ----- | ----- |
| [`mosaic`](./yolo-data-augmentation.md#mosaic-mosaic)              | 0.909 | 0.992 | 0.992 | 0.992 | 0.992 |
| [`mixup`](./yolo-data-augmentation.md#mixup-mixup)                 | 0.012 | 0.05  | 0.427 | 0.427 | 0.427 |
| [`copy_paste`](./yolo-data-augmentation.md#copy-paste-copy_paste)  | 0.075 | 0.404 | 0.304 | 0.404 | 0.404 |
| [`scale`](./yolo-data-augmentation.md#scale-scale)                 | 0.562 | 0.9   | 0.95  | 0.95  | 0.95  |
| [`fliplr`](./yolo-data-augmentation.md#flip-left-right-fliplr)     | 0.606 | 0.304 | 0.304 | 0.304 | 0.304 |
| [`degrees`](./yolo-data-augmentation.md#rotation-degrees)          | 1.11  | ~0    | ~0    | ~0    | ~0    |
| [`shear`](./yolo-data-augmentation.md#shear-shear)                 | 1.46  | ~0    | ~0    | ~0    | ~0    |
| [`translate`](./yolo-data-augmentation.md#translation-translate)   | 0.071 | 0.275 | 0.275 | 0.275 | 0.275 |
| [`hsv_h`](./yolo-data-augmentation.md#hue-adjustment-hsv_h)        | 0.014 | 0.013 | 0.013 | 0.013 | 0.013 |
| [`hsv_s`](./yolo-data-augmentation.md#saturation-adjustment-hsv_s) | 0.645 | 0.353 | 0.353 | 0.353 | 0.353 |
| [`hsv_v`](./yolo-data-augmentation.md#brightness-adjustment-hsv_v) | 0.566 | 0.194 | 0.194 | 0.194 | 0.194 |
| [`bgr`](./yolo-data-augmentation.md#bgr-channel-swap-bgr)          | 0.106 | 0.0   | 0.0   | 0.0   | 0.0   |

Values shown as `~0` are below 0.01 in the actual checkpoints (for example, `degrees=0.00012` for the S model) — the augmentation is effectively disabled.

Larger models use more aggressive augmentation overall (higher mixup, copy-paste, and scale), since they have more capacity and benefit from stronger [regularization](https://www.ultralytics.com/glossary/regularization). The N model is the only size with meaningful rotation, shear, and BGR augmentation.

### Internal Training Parameters

??? note "Advanced: internal pipeline parameters"

    The checkpoints also contain parameters that were used in the internal training pipeline but are **not** exposed as user-configurable settings in `default.yaml`:

    | Setting | Description | N | S | M | L | X |
    |---|---|---|---|---|---|---|
    | `muon_w` | Muon update weight in MuSGD | 0.528 | 0.436 | 0.436 | 0.436 | 0.436 |
    | `sgd_w` | SGD update weight in MuSGD | 0.674 | 0.479 | 0.479 | 0.479 | 0.479 |
    | `cls_w` | Internal classification weight | 2.74 | 3.48 | 3.48 | 3.48 | 3.48 |
    | `o2m` | One-to-many head loss weight | 1.0 | 0.705 | 0.705 | 0.705 | 0.705 |
    | `topk` | Top-k label assignment | 8 | 5 | 5 | 5 | 5 |

    See the [FAQ entry on these parameters](#what-are-muon_w-sgd_w-cls_w-o2m-and-topk-in-the-checkpoint) for what they mean when fine-tuning.

## Training Curves

Every YOLO26 checkpoint carries the complete per-epoch `results.csv` of its training run. These curves are available on the Ultralytics Platform and can be visualized by dragging and dropping any `.pt` file into the platform interface.

## Code Revision

The Git commit used to produce each checkpoint is stored in `ckpt["git"]`. You can check out the exact code revision as follows:

!!! example "Checking out the code revision from a checkpoint"
```python
import torch

    ckpt = torch.load("yolo26n.pt", map_location="cpu")
    git_hash = ckpt.get("git")
    if git_hash:
        print(f"Checkpoint was built from commit: {git_hash}")
        # To check out this commit in your local clone:
        # ! git checkout {git_hash}
    ```

Note: The experimental branch wording replaces "internal branch" throughout the documentation, as those commits are now public.

## Fine-Tuning YOLO26 on Your Own Dataset

When fine-tuning YOLO26 on your own dataset, you don't need to replicate the full pretraining recipe. The pretrained weights already encode the augmentation and optimization knowledge from COCO training. For more general training best practices, see [Tips for Model Training](./model-training-tips.md).

### Fine-Tune with Default Settings

!!! example "Fine-tune with defaults"

    === "Python"

        ```python
        from ultralytics import YOLO

        model = YOLO("yolo26n.pt")
        results = model.train(data="your-dataset.yaml", epochs=100, imgsz=640)
        ```

    === "CLI"

        ```bash
        yolo train model=yolo26n.pt data=your-dataset.yaml epochs=100 imgsz=640
        ```

Fine-tuning with defaults is a strong baseline. Only adjust hyperparameters if you have a specific reason to.

### When to Adjust YOLO26 Hyperparameters

**Small datasets (< 1,000 images):**

- Reduce augmentation strength: `mosaic=0.5`, `mixup=0.0`, `copy_paste=0.0`
- Lower learning rate: `lr0=0.001`
- Use fewer [epochs](https://www.ultralytics.com/glossary/epoch) with patience: `epochs=50`, `patience=20`
- Consider freezing backbone layers: `freeze=10`

**Large datasets (> 50,000 images):**

- Match the pretraining recipe more closely
- Consider `optimizer=MuSGD` for longer runs
- Increase augmentation: `mosaic=1.0`, `mixup=0.3`, `scale=0.9`

**Domain-specific imagery (aerial, medical, underwater):**

- Increase `flipud=0.5` if vertical orientation varies
- Increase `degrees` if objects appear at arbitrary rotations
- Adjust `hsv_s` and `hsv_v` if lighting conditions differ significantly from COCO

For automated hyperparameter optimization, see the [Hyperparameter Tuning guide](./hyperparameter-tuning.md).

### Choose a Model Size

| Model   | Best For                               | Batch Size Guidance                     |
| ------- | -------------------------------------- | --------------------------------------- |
| YOLO26n | Edge devices, mobile, real-time on CPU | Large batches (64-128) on consumer GPUs |
| YOLO26s | Balanced speed and accuracy            | Medium batches (32-64)                  |
| YOLO26m | Higher accuracy with moderate compute  | Smaller batches (16-32)                 |
| YOLO26l | High accuracy when GPU is available    | Small batches (8-16) or multi-GPU       |
| YOLO26x | Maximum accuracy, server deployment    | Small batches (4-8) or multi-GPU        |

For export and deployment options, see the [Export guide](../modes/export.md) and [Model Deployment Options](./model-deployment-options.md).

## Conclusion

The YOLO26 checkpoints ship with their full training recipe embedded, so the exact hyperparameters behind every model size are always one `train_args` lookup away. Start fine-tuning from the defaults, adjust deliberately using the tables on this page, and verify every change against your own validation set. If questions come up along the way, ask the community on the [Ultralytics GitHub repository](https://github.com/ultralytics/ultralytics) or the [Ultralytics Discord server](https://discord.com/invite/ultralytics).

## FAQ

### How do I see the exact hyperparameters used for any checkpoint?

Load the checkpoint with `torch.load()` and access the `train_args` key, or use `model.ckpt["train_args"]` with the Ultralytics API. See [Inspecting YOLO26 Checkpoint Training Args](#inspecting-yolo26-checkpoint-training-args) for complete examples.

### Why are the epoch counts different for each model size?

Larger models generally needed fewer epochs on COCO because their greater capacity speeds up convergence — the X model trained for 40 epochs versus 245 for N — though the counts are not strictly monotonic (S used 70, M used 80). When fine-tuning on your own dataset, the optimal number of epochs depends on your dataset size and complexity, not the model size. Use early stopping (`patience`) to find the right after the COCO stage: the epoch counts for the COCO stage sit on top of 150 shared pretraining epochs from the Objects365 stage.

### What are `muon_w`, `sgd_w`, `cls_w`, `o2m`, and `topk` in the checkpoint?

These are internal parameters from the training pipeline that produced the base checkpoints, recorded in `train_args` for reproducibility. They are not user-configurable settings in `default.yaml`, and passing them to `model.train()` raises an invalid-argument error — the public package does not read them. You do not need to set them when fine-tuning; see [Internal Training Parameters](#internal-training-parameters) for their values per model size.

### Can I replicate the exact pretraining from scratch?

Not exactly — the checkpoints were produced using an internal training branch with additional features not in the public codebase (like configurable `o2m` weights and `cls_w`). You can get very close results using the hyperparameters documented on this page with the public Ultralytics package, but an exact reproduction requires the internal branch.

### Where are the training curves for each checkpoint?

Every checkpoint carries the complete per-epoch `results.csv` of its run, with the official curves on Ultralytics Platform and drag and drop upload for any other `.pt` file.
