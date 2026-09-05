---
comments: true
description: Learn how YOLO26 models were trained, from Objects365 pretraining to COCO fine-tuning, including optimizer settings, augmentation pipelines, loss weights, embedded training logs, and fine-tuning guidance for each model size.
keywords: YOLO26, training recipe, pretraining, fine-tuning, MuSGD, augmentation, loss weights, COCO, Objects365, training logs, reproducibility, model card, hyperparameters, Ultralytics, object detection, deep learning, data augmentation
---

# YOLO26 Training Recipe

## Introduction

This guide documents the exact [training](../modes/train.md) recipe used to produce the official [YOLO26](../models/yolo26.md) pretrained checkpoints on [COCO](../datasets/detect/coco.md). Every [hyperparameter](https://www.ultralytics.com/glossary/hyperparameter-tuning) shown here is already embedded in the released `.pt` weights, together with the per-epoch training log and the code revision of the run, and all of it can be inspected programmatically.

Knowing what went into the official checkpoints — not just the architecture, but the [learning rate](https://www.ultralytics.com/glossary/learning-rate) schedules, augmentation pipelines, and loss weights that shaped their performance — helps you make better decisions when [fine-tuning](https://www.ultralytics.com/glossary/fine-tuning): which [data augmentations](./yolo-data-augmentation.md) to keep, which [loss function](https://www.ultralytics.com/glossary/loss-function) weights to adjust, and what optimizer settings work best for your dataset size.

!!! note "Read the paper for the full picture"

    This page covers the hyperparameters recorded in the released checkpoints. For the reasoning behind them, including the architecture, the loss and label assignment changes, and the ablations, read [Ultralytics YOLO26: Unified Real-Time End-to-End Vision Models](https://arxiv.org/abs/2606.03748).

## Training Overview

All YOLO26 base models were trained in two stages: **[Objects365v1](../datasets/detect/objects365.md) pretraining** for 150 epochs, followed by **COCO fine-tuning**. Both stages ran at **640x640** resolution with the **MuSGD** optimizer and **[batch size](https://www.ultralytics.com/glossary/batch-size) 128**. No YOLO26 checkpoint was trained on COCO from random weights, which is why the COCO stage is short for most sizes, and the COCO-stage hyperparameters were found via [evolutionary search](./hyperparameter-tuning.md#genetic-evolution-and-mutation). Full training logs and metrics for every model size are stored inside the released checkpoints and rendered as charts on [Ultralytics Platform](https://platform.ultralytics.com/ultralytics/yolo26).

Key design choices across all sizes:

- **[Objects365](../datasets/detect/objects365.md) pretraining** for every size before the COCO stage
- **[Dual-head training](end2end-detection.md)** with both one-to-many and one-to-one supervision; inference uses NMS by default, with `nms=False` selecting the NMS-free head
- **[MuSGD](../modes/train.md#musgd-optimizer) optimizer** combining SGD with Muon-style orthogonalized updates for weight matrices (2D linear weights and 4D conv filters, which are reshaped to 2D)
- **Heavy mosaic augmentation** (~0.9-1.0 probability) disabled for the final epochs (`close_mosaic=8` in pretraining, `close_mosaic=10` on COCO)
- **Aggressive scale augmentation** (0.5-0.95) to handle objects at different sizes
- **Minimal rotation/shear** for most sizes, keeping geometric distortion low

## Stage 1: Objects365 Pretraining

Every YOLO26 COCO checkpoint was fine-tuned from an Objects365v1 checkpoint of the same size. Those pretrained weights are published too, and are documented on the [Objects365 dataset page](../datasets/detect/objects365.md). Each COCO checkpoint names its starting weights in the training configuration embedded in the `.pt` file, which [Inspecting YOLO26 Checkpoint Training Args](#inspecting-yolo26-checkpoint-training-args) below shows how to read:

| COCO checkpoint | Starting weights       |
| --------------- | ---------------------- |
| `yolo26n.pt`    | `yolo26n-objv1-150.pt` |
| `yolo26s.pt`    | `yolo26s-objv1-150.pt` |
| `yolo26m.pt`    | `yolo26m-objv1-150.pt` |
| `yolo26l.pt`    | `yolo26l-objv1-150.pt` |
| `yolo26x.pt`    | `yolo26x-objv1-150.pt` |

Pretraining used mostly default settings rather than searched values. `lr0`, `lrf`, `momentum`, `weight_decay`, `box`, and `cls` match `default.yaml`, while `warmup_epochs`, `close_mosaic`, and `dfl` were overridden. Settings are shared across sizes apart from `warmup_epochs` on X, the augmentation strengths, and the internal MuSGD and head weights listed after the tables:

| Setting               | Value           |
| --------------------- | --------------- |
| `data`                | Objects365v1    |
| `epochs`              | 150             |
| `imgsz`               | 640             |
| `batch`               | 128             |
| `optimizer`           | MuSGD           |
| `lr0` / `lrf`         | 0.01 / 0.01     |
| `momentum`            | 0.937           |
| `weight_decay`        | 0.0005          |
| `warmup_epochs`       | 1 (2 for X)     |
| `close_mosaic`        | 8               |
| `box` / `cls` / `dfl` | 7.5 / 0.5 / 6.0 |

| Augmentation | N   | S    | M    | L    | X   |
| ------------ | --- | ---- | ---- | ---- | --- |
| `mosaic`     | 1.0 | 1.0  | 1.0  | 1.0  | 1.0 |
| `mixup`      | 0.0 | 0.05 | 0.15 | 0.15 | 0.2 |
| `copy_paste` | 0.1 | 0.15 | 0.4  | 0.5  | 0.6 |
| `scale`      | 0.5 | 0.9  | 0.9  | 0.9  | 0.9 |

These tables cover the settings that shape pretraining, not the full configuration. The checkpoints record over 100 arguments, so print `train_args` as shown below for the authoritative list.

??? note "Advanced: internal pretraining parameters"

    Pretraining also varied the same kind of experimental-branch parameters described in [Internal Training Parameters](#internal-training-parameters). `cls_w` was 1.0 for every size:

    | Setting  | N    | S   | M    | L    | X   |
    | -------- | ---- | --- | ---- | ---- | --- |
    | `muon_w` | 0.45 | 0.5 | 0.45 | 0.45 | 0.5 |
    | `sgd_w`  | 0.55 | 0.5 | 0.55 | 0.55 | 0.6 |
    | `o2m`    | 0.1  | 0.1 | 0.1  | 1.0  | 1.0 |

!!! tip "Start from the Objects365 weights"

    You do not need the Objects365 dataset to reuse stage 1. The pretrained checkpoints download automatically like any other Ultralytics asset, so you can fine-tune them on your own dataset:

    === "Python"

        ```python
        from ultralytics import YOLO

        model = YOLO("yolo26s-objv1-150.pt")
        results = model.train(data="your-dataset.yaml", epochs=100, imgsz=640)
        ```

    === "CLI"

        ```bash
        yolo train model=yolo26s-objv1-150.pt data=your-dataset.yaml epochs=100 imgsz=640
        ```

    To rerun the COCO stage instead, start from the same weights and pass the stage 2 optimizer, loss, and augmentation values for that size from the tables below. Those tables leave out smaller non-default arguments such as `warmup_momentum`, `warmup_bias_lr`, `perspective`, `flipud`, and `cutmix`, so print `train_args` from the checkpoint for the exact configuration. The internal parameters are rejected by the released package and need the experimental branch.

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

### Viewing the Training Curves

`train_args` is not the only thing stored. Every checkpoint also carries the complete per-epoch `results.csv` of the run that produced it, along with its final validation metrics. The curves for the official YOLO26 checkpoints are published on [Ultralytics Platform](https://platform.ultralytics.com/ultralytics/yolo26), and any `.pt` file can be [dragged and dropped onto a project](../platform/train/models.md#upload-model) to chart the same data, with the metadata parsed out of the file automatically.

Each checkpoint covers the stage that produced it, so the COCO curves are in `yolo26s.pt` and the 150-epoch Objects365 curves are in `yolo26s-objv1-150.pt`.

### Checking the Code Revision

`ckpt["git"]` records the commit that produced the checkpoint, and those commits live on public experimental branches of the Ultralytics repository, so you can check out the exact training code:

```python
from ultralytics import YOLO

print(YOLO("yolo26n.pt").ckpt["git"])
# {'root': ..., 'branch': 'exp-main', 'commit': 'cb13d5f9cfbd6f299da3620c625f81d721dc2849', ...}
```

```bash
git fetch origin cb13d5f9cfbd6f299da3620c625f81d721dc2849
git checkout cb13d5f9cfbd6f299da3620c625f81d721dc2849
```

Experimental branches carry work that never landed on `main`, such as configurable `o2m` and `cls_w`. Training on `main` with the hyperparameters documented below will not be bit-identical, but it lands within a negligible distance of the published metrics.

## YOLO26 Training Hyperparameters per Model Size

These are the stage 2 values, applied on top of the Objects365 weights above. The tables below group the recipe by category: optimizer and schedule, loss weights, and augmentation. Every value comes straight from the `train_args` embedded in the released checkpoints.

### Optimizer and Learning Rate

These optimizer and schedule settings drove COCO fine-tuning for each size; note how the N model stands apart from the rest:

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

Larger models use more aggressive augmentation overall (higher mixup and scale), since they have more capacity and benefit from stronger [regularization](https://www.ultralytics.com/glossary/regularization). The N model is the only size with meaningful rotation, shear, and BGR augmentation.

### Internal Training Parameters

??? note "Advanced: internal pipeline parameters"

    The checkpoints also contain parameters that were used on the experimental training branch but are **not** exposed as user-configurable settings in `default.yaml`:

    | Setting  | Description                    | N     | S     | M     | L     | X     |
    | -------- | ------------------------------ | ----- | ----- | ----- | ----- | ----- |
    | `muon_w` | Muon update weight in MuSGD    | 0.528 | 0.436 | 0.436 | 0.436 | 0.436 |
    | `sgd_w`  | SGD update weight in MuSGD     | 0.674 | 0.479 | 0.479 | 0.479 | 0.479 |
    | `cls_w`  | Internal classification weight | 2.74  | 3.48  | 3.48  | 3.48  | 3.48  |
    | `o2m`    | One-to-many head loss weight   | 1.0   | 0.705 | 0.705 | 0.705 | 0.705 |
    | `topk`   | Top-k label assignment         | 8     | 5     | 5     | 5     | 5     |

    See the [FAQ entry on these parameters](#what-are-muon_w-sgd_w-cls_w-o2m-and-topk-in-the-checkpoint) for what they mean when fine-tuning.

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
- Lower learning rate with an explicit optimizer: `optimizer=AdamW`, `lr0=0.001`
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

Every size received the same 150 epochs of Objects365 pretraining, so the COCO counts only cover the fine-tuning stage. Larger models converge on COCO in fewer of those epochs, 40 for X versus 245 for N. The counts are not strictly monotonic (S used 70, M used 80) because they came out of the per-size hyperparameter search. When fine-tuning on your own dataset, the optimal number of epochs depends on your dataset size and complexity, not the model size. Use early stopping (`patience`) to find the right stopping point automatically.

### What are `muon_w`, `sgd_w`, `cls_w`, `o2m`, and `topk` in the checkpoint?

These come from the experimental branch that produced the base checkpoints, recorded in `train_args` for reproducibility. They are not user-configurable settings in `default.yaml`, and passing them to `model.train()` raises an invalid-argument error because the released package does not read them. You do not need to set them when fine-tuning; see [Internal Training Parameters](#internal-training-parameters) for their values per model size.

### Were the YOLO26 models trained on COCO from scratch?

No. Each COCO checkpoint was fine-tuned from an Objects365v1 checkpoint of the same size that had already trained for 150 epochs, as described in [Stage 1: Objects365 Pretraining](#stage-1-objects365-pretraining) and in the [YOLO26 paper](https://arxiv.org/abs/2606.03748). There is no from-scratch COCO run behind the published numbers, so a from-scratch comparison against those numbers is not like for like.

### Where are the full training logs and loss curves?

Inside the checkpoints, and charted on [Ultralytics Platform](https://platform.ultralytics.com/ultralytics/yolo26). Every checkpoint stores the complete per-epoch `results.csv` of its run, so dropping a `.pt` file onto a Platform project plots the losses, mAP progression, and learning rates without any code. See [Viewing the Training Curves](#viewing-the-training-curves). The Objects365 stage has its own log in the `yolo26*-objv1-150.pt` checkpoints.

### Can I reproduce the published COCO metrics with the released package?

You will land close to the published metrics, but not identical to them. For an identical setup, check out the commit recorded in the checkpoint and train on that branch. See [Checking the Code Revision](#checking-the-code-revision).
