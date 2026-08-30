---
comments: true
description: Learn how to customize the Ultralytics YOLO trainer with custom metrics, class-weighted loss, custom model saving, backbone freezing, per-layer learning rates, SyncBatchNorm, and gradient clipping.
keywords: Ultralytics, YOLO, Custom Trainer, DetectionTrainer, BaseTrainer, Custom Metrics, F1 Score, Class Weights, Backbone Freezing, Per-Layer Learning Rate, SyncBatchNorm, Gradient Clipping, Multi-GPU Training, Fine-Tuning, Transfer Learning
---

# Customizing Trainer

The Ultralytics training pipeline is built around `BaseTrainer` and task-specific trainers like `DetectionTrainer`. These classes handle the training loop, validation, checkpointing, and logging out of the box. When you need more control — tracking custom metrics, adjusting loss weighting, or implementing learning rate schedules — you can subclass the trainer and override specific methods.

This guide walks through seven common customizations:

1. [Logging custom metrics (F1 score)](#logging-custom-metrics) at the end of each [epoch](https://www.ultralytics.com/glossary/epoch)
2. [Adding class weights](#adding-class-weights) to handle class imbalance
3. [Saving the best model](#saving-the-best-model-by-custom-metric) based on a different metric
4. [Freezing the backbone](#freezing-and-unfreezing-the-backbone) for the first N epochs, then unfreezing
5. [Specifying per-layer learning rates](#per-layer-learning-rates)
6. [Synchronizing BatchNorm across GPUs](#synchronized-batchnorm-for-multi-gpu-training) for multi-GPU training
7. [Configuring gradient clipping](#configurable-gradient-clipping) for stability tuning

!!! tip "Prerequisites"

    Before reading this guide, make sure you're familiar with the basics of [training YOLO models](../modes/train.md) and the [Advanced Customization](../usage/engine.md) page, which covers the `BaseTrainer` architecture.

## How Custom Trainers Work

The `YOLO` model class accepts a `trainer` parameter in the `train()` method. This allows you to pass your own trainer class that extends the default behavior:

```python
from ultralytics import YOLO
from ultralytics.models.yolo.detect import DetectionTrainer


class CustomTrainer(DetectionTrainer):
    """A custom trainer that extends DetectionTrainer with additional functionality."""

    # Add your customizations here


model = YOLO("yolo26n.pt")
model.train(data="coco8.yaml", epochs=10, trainer=CustomTrainer)
```

Your custom trainer inherits all functionality from `DetectionTrainer`, so you only need to override the specific methods you want to customize.

## Logging Custom Metrics

The [validation](../modes/val.md) step computes [precision](https://www.ultralytics.com/glossary/precision), [recall](https://www.ultralytics.com/glossary/recall), and [mAP](https://www.ultralytics.com/glossary/mean-average-precision-map). If you need additional metrics like per-class [F1 score](https://www.ultralytics.com/glossary/f1-score), override `validate()`:

```python
import numpy as np

from ultralytics import YOLO
from ultralytics.models.yolo.detect import DetectionTrainer
from ultralytics.utils import LOGGER


class MetricsTrainer(DetectionTrainer):
    """Custom trainer that computes and logs F1 score at the end of each epoch."""

    def validate(self):
        """Run validation and compute per-class F1 scores."""
        metrics, fitness = super().validate()
        if metrics is None:
            return metrics, fitness

        if hasattr(self.validator, "metrics") and hasattr(self.validator.metrics, "box"):
            box = self.validator.metrics.box
            f1_per_class = box.f1
            class_indices = box.ap_class_index
            names = self.validator.names

            mean_f1 = float(np.mean(f1_per_class)) if len(f1_per_class) else 0.0

            LOGGER.info(f"Mean F1 Score: {mean_f1:.4f}")
            per_class_str = [f"{names[i]}: {f1_per_class[j]:.3f}" for j, i in enumerate(class_indices)]
            LOGGER.info(f"Per-class F1: {per_class_str}")

        return metrics, fitness


model = YOLO("yolo26n.pt")
model.train(data="coco8.yaml", epochs=5, trainer=MetricsTrainer)
```

This logs the mean F1 score across all classes represented in validation and a per-class breakdown after each validation run.

!!! note "Available Metrics"

    The validator provides access to many metrics through `self.validator.metrics.box`:

    | Attribute | Description |
    |---|---|
    | `f1` | F1 score per class |
    | `image_metrics` | Per-image metrics dictionary with precision, recall, F1, TP, FP, and FN |
    | `p` | Precision per class |
    | `r` | Recall per class |
    | `ap50` | AP at IoU 0.5 per class |
    | `ap` | AP at IoU 0.5:0.95 per class |
    | `mp`, `mr` | Mean precision and recall |
    | `map50`, `map` | Mean AP metrics |

## Adding Class Weights

Set `cls_pw` between `0.0` and `1.0` to apply normalized inverse-frequency weights to the classification loss. Override the existing weight computation only when you need hand-picked ratios:

```python
import numpy as np

from ultralytics import YOLO
from ultralytics.models.yolo.detect import DetectionTrainer


class WeightedTrainer(DetectionTrainer):
    """Detection trainer with hand-picked class-weight ratios."""

    def compute_class_weights(self, class_counts):
        """Return custom per-class weights for the production loss owner."""
        weights = np.ones_like(class_counts)
        weights[0] = 2.0
        weights[1] = 3.0
        return weights


model = YOLO("yolo26n.pt")
model.train(data="custom.yaml", epochs=10, cls_pw=1.0, trainer=WeightedTrainer)
```

`set_class_weights()` normalizes these values to a mean of 1.0 and stores them on the model, where the existing detection loss applies them. The indices above require a dataset with at least two classes.

## Saving the Best Model by Custom Metric

The trainer saves `best.pt` based on fitness, which for detection defaults to `mAP@0.5:0.95` (weights `[0.0, 0.0, 0.0, 1.0]` for [P, R, mAP@0.5, mAP@0.5:0.95]). To use a different metric (like `mAP@0.5` or recall), override `validate()` and return your chosen metric as the fitness value. The built-in `save_model()` will then use it automatically:

```python
from ultralytics import YOLO
from ultralytics.models.yolo.detect import DetectionTrainer


class CustomSaveTrainer(DetectionTrainer):
    """Trainer that saves the best model based on mAP@0.5 instead of default fitness."""

    def validate(self):
        """Override fitness to use mAP@0.5 for best model selection."""
        previous_best = self.best_fitness
        metrics, fitness = super().validate()
        if metrics is None:
            return metrics, fitness
        fitness = metrics["metrics/mAP50(B)"]
        self.best_fitness = fitness if previous_best is None else max(previous_best, fitness)
        return metrics, fitness


model = YOLO("yolo26n.pt")
model.train(data="coco8.yaml", epochs=20, trainer=CustomSaveTrainer)
```

`BaseTrainer.validate()` updates `best_fitness` using the default metric, so capture its previous value before calling it.

!!! note "Available Metrics"

    Common metrics available in `self.metrics` after validation include:

    | Key | Description |
    |---|---|
    | `metrics/precision(B)` | Precision |
    | `metrics/recall(B)` | Recall |
    | `metrics/mAP50(B)` | mAP at IoU 0.5 |
    | `metrics/mAP50-95(B)` | mAP at IoU 0.5:0.95 |

## Freezing and Unfreezing the Backbone

[Transfer learning](https://www.ultralytics.com/glossary/transfer-learning) workflows often benefit from freezing the pretrained backbone for the first N epochs, allowing the detection head to adapt before [fine-tuning](https://www.ultralytics.com/glossary/fine-tuning) the entire network. Ultralytics provides a `freeze` parameter to freeze layers at the start of training, and you can use a [callback](../usage/callbacks.md) to unfreeze them after N epochs:

```python
from ultralytics import YOLO
from ultralytics.models.yolo.detect import DetectionTrainer
from ultralytics.utils import LOGGER

FREEZE_EPOCHS = 5


def unfreeze_backbone(trainer):
    """Callback to unfreeze the user-requested layers after FREEZE_EPOCHS."""
    if trainer.epoch == FREEZE_EPOCHS:
        user_freeze = [x for x in trainer.freeze_layer_names if x not in {".dfl", "teacher_model."}]
        LOGGER.info(f"Epoch {trainer.epoch}: Unfreezing requested layers for fine-tuning")
        for name, param in trainer.model.named_parameters():
            if (
                not param.requires_grad
                and ".dfl" not in name
                and "teacher_model." not in name
                and any(x in name for x in user_freeze)
            ):
                param.requires_grad = True
                LOGGER.info(f"  Unfroze: {name}")
        trainer.freeze_layer_names = [x for x in trainer.freeze_layer_names if x not in user_freeze]


class FreezingTrainer(DetectionTrainer):
    """Trainer with backbone freezing for first N epochs."""

    def __init__(self, *args, **kwargs):
        """Initialize and register the unfreeze callback."""
        super().__init__(*args, **kwargs)
        self.add_callback("on_train_epoch_start", unfreeze_backbone)


model = YOLO("yolo26n.pt")
model.train(data="coco8.yaml", epochs=20, freeze=10, trainer=FreezingTrainer)
```

The `freeze=10` parameter freezes the first 10 layers (indices 0-9) at training start, which covers most of the YOLO26 backbone. The backbone spans layers 0-10, so `freeze=10` leaves the final C2PSA block (layer 10) trainable; use `freeze=11` to freeze the entire backbone. The `on_train_epoch_start` callback fires at the beginning of each epoch and unfreezes those requested layers once the freeze period is complete, while preserving permanently frozen DFL and distillation-teacher parameters.

!!! tip "Choosing What to Freeze"

    - `freeze=10` freezes the first 10 layers, indices 0-9 (most of the YOLO26 backbone; use `freeze=11` to include the final C2PSA block at layer 10)
    - `freeze=[0, 1, 2, 3]` freezes specific layers by index
    - Higher `FREEZE_EPOCHS` values give the head more time to adapt before the backbone changes

## Per-Layer Learning Rates

Different parts of the network can benefit from different [learning rates](https://www.ultralytics.com/glossary/learning-rate). A common strategy is to use a lower learning rate for the pretrained backbone to preserve learned features, while allowing the detection head to adapt more quickly with a higher rate:

```python
from ultralytics import YOLO
from ultralytics.models.yolo.detect import DetectionTrainer
from ultralytics.utils import LOGGER
from ultralytics.utils.torch_utils import unwrap_model


class PerLayerLRTrainer(DetectionTrainer):
    """Trainer with different learning rates for backbone and head."""

    backbone_lr_ratio = 0.1

    def build_optimizer(self, model, name="auto", lr=0.001, momentum=0.9, decay=1e-5, iterations=1e5):
        """Reuse the trainer optimizer and lower its backbone parameter-group rates."""
        optimizer = super().build_optimizer(model, name, lr, momentum, decay, iterations)
        unwrapped = unwrap_model(model)
        backbone_len = len(unwrapped.yaml["backbone"])
        backbone = {
            id(p)
            for name, p in unwrapped.named_parameters()
            if any(name.startswith(f"model.{i}.") for i in range(backbone_len))
        }

        groups = []
        for group in optimizer.param_groups:
            head_params = [p for p in group["params"] if id(p) not in backbone]
            backbone_params = [p for p in group["params"] if id(p) in backbone]
            if head_params:
                groups.append({**group, "params": head_params})
            if backbone_params:
                groups.append({**group, "params": backbone_params, "lr": group["lr"] * self.backbone_lr_ratio})
        optimizer.param_groups = groups

        LOGGER.info(f"PerLayerLR: {len(backbone)} backbone params at {self.backbone_lr_ratio}x the head rate")
        return optimizer


model = YOLO("yolo26n.pt")
model.train(data="coco8.yaml", epochs=20, trainer=PerLayerLRTrainer)
```

### RT-DETR Variant

For RT-DETR, use the same override with `RTDETRTrainer` as the parent and load the checkpoint with `RTDETR("rtdetr-l.pt")`.

## Synchronized BatchNorm for Multi-GPU Training

When training on multiple GPUs with DistributedDataParallel, the default `BatchNorm2d` layers compute statistics independently on each GPU. For RT-DETR fine-tuning and other recipes that use small per-GPU batch sizes, per-GPU batch statistics can be noisy. PyTorch's `SyncBatchNorm` synchronizes mean and variance across all ranks for a single global batch statistic, which often improves convergence at the cost of a small inter-GPU communication overhead.

The conversion has to happen after the model is on the GPU but before DDP wraps it. The cleanest hook for this is `set_model_attributes()`, which `BaseTrainer` calls in exactly that window:

```python
from torch import nn

from ultralytics import RTDETR
from ultralytics.models.rtdetr.train import RTDETRTrainer


class SyncBNTrainer(RTDETRTrainer):
    """RT-DETR trainer that converts BatchNorm to SyncBatchNorm for multi-GPU training."""

    def set_model_attributes(self):
        """Run the parent setup, then convert BN to SyncBatchNorm when training on multiple GPUs."""
        super().set_model_attributes()
        if self.world_size > 1:
            self.model = nn.SyncBatchNorm.convert_sync_batchnorm(self.model)


model = RTDETR("rtdetr-l.pt")
model.train(data="coco8.yaml", epochs=20, device=[0, 1], trainer=SyncBNTrainer)
```

The `world_size > 1` guard ensures the trainer is safe to use in single-GPU runs as well; on a single GPU the conversion is skipped and training proceeds with regular `BatchNorm2d`. The same pattern works for YOLO by switching the parent class to `DetectionTrainer`.

!!! tip "When to use SyncBatchNorm"

    | Scenario                                       | Recommendation           |
    | ---------------------------------------------- | ------------------------ |
    | Multi-GPU training, small per-GPU batch (≤ 16) | Enable                   |
    | Multi-GPU training, large per-GPU batch (≥ 32) | Optional; minor benefit  |
    | Single-GPU training                            | Not applicable (skipped) |

## Configurable Gradient Clipping

The default trainer clips gradients to `max_norm=10.0` in `optimizer_step()`, a loose value tuned for YOLO models where gradients rarely exceed it. DETR-family detectors (RT-DETR, DEIM, DINO) typically use much tighter values such as `0.1` to stabilize the decoder's cross-attention layers, where gradient magnitudes can spike. To override the clip value, subclass the trainer and override `optimizer_step()`:

```python
import torch

from ultralytics import RTDETR
from ultralytics.models.rtdetr.train import RTDETRTrainer
from ultralytics.utils.torch_utils import TORCH_2_0


class CustomClipTrainer(RTDETRTrainer):
    """RT-DETR trainer with configurable gradient clipping."""

    clip_grad_norm = 0.1  # max gradient norm; set to 0 to disable clipping

    def optimizer_step(self):
        """Run an optimizer step with a configurable gradient-norm clip."""
        self.scaler.unscale_(self.optimizer)
        if self.clip_grad_norm > 0:
            kwargs = {"foreach": False} if self.device.type == "npu" and TORCH_2_0 else {}
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.clip_grad_norm, **kwargs)
        self.scaler.step(self.optimizer)
        self.scaler.update()
        self.optimizer.zero_grad()
        if self.ema:
            self.ema.update(self.model)


model = RTDETR("rtdetr-l.pt")
model.train(data="coco8.yaml", epochs=20, trainer=CustomClipTrainer)
```

The same trainer works for YOLO by switching the parent class to `DetectionTrainer` (`from ultralytics.models.yolo.detect import DetectionTrainer`) and loading a YOLO checkpoint with `YOLO("yolo26n.pt")`. The `optimizer_step` body is unchanged.

!!! tip "Typical `clip_grad_norm` values"

    | Architecture family          | Typical `max_norm` |
    | ---------------------------- | ------------------ |
    | RT-DETR / DEIM / DETR family | `0.1`              |
    | YOLO (Ultralytics default)   | `10.0`             |
    | Disable clipping             | `0`                |

## FAQ

### How do I pass a custom trainer to YOLO?

Pass your custom trainer class (not an instance) to the `trainer` parameter in `model.train()`:

```python
from ultralytics import YOLO
from ultralytics.models.yolo.detect import DetectionTrainer


class MyCustomTrainer(DetectionTrainer):
    """A custom trainer that extends DetectionTrainer."""


model = YOLO("yolo26n.pt")
model.train(data="coco8.yaml", trainer=MyCustomTrainer)
```

The `YOLO` class handles trainer instantiation internally. See the [Advanced Customization](../usage/engine.md) page for more details on the trainer architecture.

### Which BaseTrainer methods can I override?

Key methods available for customization:

| Method               | Purpose                           |
| -------------------- | --------------------------------- |
| `validate()`         | Run validation and return metrics |
| `build_optimizer()`  | Construct the optimizer           |
| `save_model()`       | Save training checkpoints         |
| `get_model()`        | Return the model instance         |
| `get_validator()`    | Return the validator instance     |
| `get_dataloader()`   | Build the dataloader              |
| `preprocess_batch()` | Preprocess input batch            |
| `label_loss_items()` | Format loss items for logging     |

For the full API reference, see the [`BaseTrainer` documentation](../reference/engine/trainer.md).

### Can I use callbacks instead of subclassing the trainer?

Yes, for simpler customizations, [callbacks](../usage/callbacks.md) are often sufficient. Available callback events include `on_train_start`, `on_train_epoch_start`, `on_train_epoch_end`, `on_fit_epoch_end`, and `on_model_save`. These allow you to hook into the training loop without subclassing. The backbone freezing example above demonstrates this approach.

### How do I customize the loss function without subclassing the model?

If your change is simpler (such as adjusting loss gains), you can modify the [hyperparameters](https://www.ultralytics.com/glossary/hyperparameter-tuning) directly:

```python
from ultralytics import YOLO

model = YOLO("yolo26n.pt")
model.train(data="coco8.yaml", box=10.0, cls=1.5, dfl=2.0)
```

On YOLO26, `dfl` scales the logged `l1_loss` because its detection head uses `reg_max: 1`; on models with `reg_max > 1` it scales `dfl_loss`.
