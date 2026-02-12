---
comments: true
description: Learn how to customize the Ultralytics YOLO trainer with custom metrics, class-weighted loss, custom model saving, backbone freezing, and per-layer learning rates.
keywords: Ultralytics, YOLO, Custom Trainer, DetectionTrainer, BaseTrainer, Custom Metrics, F1 Score, Class Weights, Backbone Freezing, Per-Layer Learning Rate, Fine-Tuning, Transfer Learning
---

# Customizing Trainer

The Ultralytics training pipeline is built around `BaseTrainer` and task-specific trainers like `DetectionTrainer`. These classes handle the training loop, validation, checkpointing, and logging out of the box. When you need more control — tracking custom metrics, adjusting loss weighting, or implementing learning rate schedules — you can subclass the trainer and override specific methods.

This guide walks through five common customizations:

1. [Logging custom metrics (F1 score)](#logging-custom-metrics) at the end of each [epoch](https://www.ultralytics.com/glossary/epoch)
2. [Adding class weights](#adding-class-weights) to handle class imbalance
3. [Saving the best model](#saving-the-best-model-by-custom-metric) based on a different metric
4. [Freezing the backbone](#freezing-and-unfreezing-the-backbone) for the first N epochs, then unfreezing
5. [Specifying per-layer learning rates](#per-layer-learning-rates)

!!! tip "Prerequisites"

    Before reading this guide, make sure you're familiar with the basics of [training YOLO models](../modes/train.md) and the [Advanced Customization](../usage/engine.md) page, which covers the `BaseTrainer` architecture.

## How Custom Trainers Work

The `YOLO` model class accepts a `trainer` parameter in the `train()` method. This allows you to pass your own trainer class that extends the default behavior:

```python
from ultralytics import YOLO
from ultralytics.models.yolo.detect import DetectionTrainer


class CustomTrainer(DetectionTrainer):
    """A custom trainer that extends DetectionTrainer with additional functionality."""

    pass  # Add your customizations here


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

            valid_f1 = f1_per_class[f1_per_class > 0]
            mean_f1 = np.mean(valid_f1) if len(valid_f1) > 0 else 0.0

            LOGGER.info(f"Mean F1 Score: {mean_f1:.4f}")
            per_class_str = [
                f"{names[i]}: {f1_per_class[j]:.3f}" for j, i in enumerate(class_indices) if f1_per_class[j] > 0
            ]
            LOGGER.info(f"Per-class F1: {per_class_str}")

        return metrics, fitness


model = YOLO("yolo26n.pt")
model.train(data="coco8.yaml", epochs=5, trainer=MetricsTrainer)
```

This logs the mean F1 score across all classes and a per-class breakdown after each validation run.

!!! note "Available Metrics"

    The validator provides access to many metrics through `self.validator.metrics.box`:

    | Attribute | Description |
    |---|---|
    | `f1` | F1 score per class |
    | `p` | Precision per class |
    | `r` | Recall per class |
    | `ap50` | AP at IoU 0.5 per class |
    | `ap` | AP at IoU 0.5:0.95 per class |
    | `mp`, `mr` | Mean precision and recall |
    | `map50`, `map` | Mean AP metrics |

## Adding Class Weights

If your dataset has imbalanced classes (e.g., a rare defect in manufacturing inspection), you can upweight underrepresented classes in the [loss function](https://www.ultralytics.com/glossary/loss-function). This makes the model penalize misclassifications on rare classes more heavily.

To customize the loss, subclass the loss classes, model, and trainer:

```python
import torch
from torch import nn

from ultralytics import YOLO
from ultralytics.models.yolo.detect import DetectionTrainer
from ultralytics.nn.tasks import DetectionModel
from ultralytics.utils import RANK
from ultralytics.utils.loss import E2ELoss, v8DetectionLoss


class WeightedDetectionLoss(v8DetectionLoss):
    """Detection loss with class weights applied to BCE classification loss."""

    def __init__(self, model, class_weights=None, tal_topk=10, tal_topk2=None):
        """Initialize loss with optional per-class weights for BCE."""
        super().__init__(model, tal_topk=tal_topk, tal_topk2=tal_topk2)
        if class_weights is not None:
            self.bce = nn.BCEWithLogitsLoss(
                pos_weight=class_weights.to(self.device),
                reduction="none",
            )


class WeightedE2ELoss(E2ELoss):
    """E2E Loss with class weights for YOLO26."""

    def __init__(self, model, class_weights=None):
        """Initialize E2E loss with weighted detection loss."""

        def weighted_loss_fn(model, tal_topk=10, tal_topk2=None):
            return WeightedDetectionLoss(model, class_weights=class_weights, tal_topk=tal_topk, tal_topk2=tal_topk2)

        super().__init__(model, loss_fn=weighted_loss_fn)


class WeightedDetectionModel(DetectionModel):
    """Detection model that uses class-weighted loss."""

    def init_criterion(self):
        """Initialize weighted loss criterion with per-class weights."""
        class_weights = torch.ones(self.nc)
        class_weights[0] = 2.0  # upweight class 0
        class_weights[1] = 3.0  # upweight rare class 1
        return WeightedE2ELoss(self, class_weights=class_weights)


class WeightedTrainer(DetectionTrainer):
    """Trainer that returns a WeightedDetectionModel."""

    def get_model(self, cfg=None, weights=None, verbose=True):
        """Return a WeightedDetectionModel."""
        model = WeightedDetectionModel(cfg, nc=self.data["nc"], verbose=verbose and RANK == -1)
        if weights:
            model.load(weights)
        return model


model = YOLO("yolo26n.pt")
model.train(data="coco8.yaml", epochs=10, trainer=WeightedTrainer)
```

!!! tip "Computing Weights from Dataset"

    You can compute class weights automatically from your dataset's label distribution. A common approach is inverse frequency weighting:

    ```python
    import numpy as np

    # class_counts: number of instances per class
    class_counts = np.array([5000, 200, 3000])
    # Inverse frequency: rarer classes get higher weight
    class_weights = max(class_counts) / class_counts
    # Result: [1.0, 25.0, 1.67]
    ```

## Saving the Best Model by Custom Metric

The trainer saves `best.pt` based on fitness, which uses `mAP@0.5:0.95`. To use a different metric (like `mAP@0.5` or recall), override `save_model()`:

```python
import torch

from ultralytics import YOLO
from ultralytics.models.yolo.detect import DetectionTrainer
from ultralytics.utils import LOGGER


class CustomSaveTrainer(DetectionTrainer):
    """Trainer that saves the best model based on mAP@0.5 instead of default fitness."""

    def __init__(self, *args, **kwargs):
        """Initialize with custom best metric tracking."""
        super().__init__(*args, **kwargs)
        self.best_map50 = 0.0

    def save_model(self):
        """Save best model based on mAP@0.5 instead of default fitness (mAP@0.5:0.95)."""
        import io
        from copy import deepcopy
        from datetime import datetime

        from ultralytics import __version__
        from ultralytics.utils.torch_utils import convert_optimizer_state_dict_to_fp16, unwrap_model

        current_map50 = self.metrics.get("metrics/mAP50(B)", 0.0)

        buffer = io.BytesIO()
        torch.save(
            {
                "epoch": self.epoch,
                "best_fitness": self.best_fitness,
                "model": None,
                "ema": deepcopy(unwrap_model(self.ema.ema)).half(),
                "updates": self.ema.updates,
                "optimizer": convert_optimizer_state_dict_to_fp16(deepcopy(self.optimizer.state_dict())),
                "scaler": self.scaler.state_dict(),
                "train_args": vars(self.args),
                "train_metrics": {**self.metrics, **{"fitness": self.fitness}},
                "train_results": self.read_results_csv(),
                "date": datetime.now().isoformat(),
                "version": __version__,
                "license": "AGPL-3.0 (https://ultralytics.com/license)",
                "docs": "https://docs.ultralytics.com",
            },
            buffer,
        )
        serialized_ckpt = buffer.getvalue()

        self.wdir.mkdir(parents=True, exist_ok=True)
        self.last.write_bytes(serialized_ckpt)

        if current_map50 > self.best_map50:
            self.best_map50 = current_map50
            self.best.write_bytes(serialized_ckpt)
            LOGGER.info(f"New best mAP@0.5: {current_map50:.4f} - saving best.pt")

        if (self.save_period > 0) and (self.epoch % self.save_period == 0):
            (self.wdir / f"epoch{self.epoch}.pt").write_bytes(serialized_ckpt)


model = YOLO("yolo26n.pt")
model.train(data="coco8.yaml", epochs=20, trainer=CustomSaveTrainer)
```

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
    """Callback to unfreeze all layers after FREEZE_EPOCHS."""
    if trainer.epoch == FREEZE_EPOCHS:
        LOGGER.info(f"Epoch {trainer.epoch}: Unfreezing all layers for fine-tuning")
        for name, param in trainer.model.named_parameters():
            if not param.requires_grad:
                param.requires_grad = True
                LOGGER.info(f"  Unfroze: {name}")
        trainer.freeze_layer_names = [".dfl"]


class FreezingTrainer(DetectionTrainer):
    """Trainer with backbone freezing for first N epochs."""

    def __init__(self, *args, **kwargs):
        """Initialize and register the unfreeze callback."""
        super().__init__(*args, **kwargs)
        self.add_callback("on_train_epoch_start", unfreeze_backbone)


model = YOLO("yolo26n.pt")
model.train(data="coco8.yaml", epochs=20, freeze=10, trainer=FreezingTrainer)
```

The `freeze=10` parameter freezes the first 10 layers (the backbone) at training start. The `on_train_epoch_start` callback fires at the beginning of each epoch and unfreezes all parameters once the freeze period is complete.

!!! tip "Choosing What to Freeze"

    - `freeze=10` freezes the first 10 layers (typically the backbone in YOLO architectures)
    - `freeze=[0, 1, 2, 3]` freezes specific layers by index
    - Higher `FREEZE_EPOCHS` values give the head more time to adapt before the backbone changes

## Per-Layer Learning Rates

Different parts of the network can benefit from different [learning rates](https://www.ultralytics.com/glossary/learning-rate). A common strategy is to use a lower learning rate for the pretrained backbone to preserve learned features, while allowing the detection head to adapt more quickly with a higher rate:

```python
import torch

from ultralytics import YOLO
from ultralytics.models.yolo.detect import DetectionTrainer
from ultralytics.utils import LOGGER
from ultralytics.utils.torch_utils import unwrap_model


class PerLayerLRTrainer(DetectionTrainer):
    """Trainer with different learning rates for backbone and head."""

    def build_optimizer(self, model, name="auto", lr=0.001, momentum=0.9, decay=1e-5, iterations=1e5):
        """Build optimizer with separate learning rates for backbone and head."""
        backbone_params = []
        head_params = []

        for k, v in unwrap_model(model).named_parameters():
            if not v.requires_grad:
                continue
            is_backbone = any(k.startswith(f"model.{i}.") for i in range(10))
            if is_backbone:
                backbone_params.append(v)
            else:
                head_params.append(v)

        backbone_lr = lr * 0.1

        optimizer = torch.optim.AdamW(
            [
                {"params": backbone_params, "lr": backbone_lr, "weight_decay": decay},
                {"params": head_params, "lr": lr, "weight_decay": decay},
            ],
        )

        LOGGER.info(
            f"PerLayerLR optimizer: backbone ({len(backbone_params)} params, lr={backbone_lr}) "
            f"| head ({len(head_params)} params, lr={lr})"
        )
        return optimizer


model = YOLO("yolo26n.pt")
model.train(data="coco8.yaml", epochs=20, trainer=PerLayerLRTrainer)
```

!!! note "Learning Rate Scheduler"

    The built-in learning rate scheduler (`cosine` or `linear`) still applies on top of the per-group base learning rates. Both the backbone and head learning rates will follow the same decay schedule, maintaining the ratio between them throughout training.

!!! tip "Combining Techniques"

    These customizations can be combined into a single trainer class by overriding multiple methods and adding callbacks as needed.

## FAQ

### How do I pass a custom trainer to YOLO?

Pass your custom trainer class (not an instance) to the `trainer` parameter in `model.train()`:

```python
from ultralytics import YOLO

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
model.train(data="coco8.yaml", box=10.0, cls=1.5, dfl=2.0)
```

For structural changes to the loss (such as adding class weights), you need to subclass the loss and model as shown in the [class weights section](#adding-class-weights).
