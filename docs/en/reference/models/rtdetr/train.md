---
description: Explore RTDETRTrainer for efficient real-time object detection leveraging Vision Transformers. Learn configuration, dataset handling, and advanced model training.
keywords: RTDETRTrainer, real-time object detection, Vision Transformers, YOLO, RT-DETR model, model training, dataset handling
---

# Reference for `ultralytics/models/rtdetr/train.py`

!!! success "Improvements"

    This page is sourced from [https://github.com/ultralytics/ultralytics/blob/main/ultralytics/models/rtdetr/train.py](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/models/rtdetr/train.py). Have an improvement or example to add? Open a [Pull Request](https://docs.ultralytics.com/help/contributing/) — thank you! 🙏

<br>

!!! abstract "Summary"

    === "<span class="doc-kind doc-kind-class">Classes</span>"

        - [`RTDETRTrainer`](#ultralytics.models.rtdetr.train.RTDETRTrainer)

    === "<span class="doc-kind doc-kind-method">Methods</span>"

        - [`RTDETRTrainer.get_model`](#ultralytics.models.rtdetr.train.RTDETRTrainer.get_model)
        - [`RTDETRTrainer.build_dataset`](#ultralytics.models.rtdetr.train.RTDETRTrainer.build_dataset)
        - [`RTDETRTrainer.get_validator`](#ultralytics.models.rtdetr.train.RTDETRTrainer.get_validator)


## Class `ultralytics.models.rtdetr.train.RTDETRTrainer` {#ultralytics.models.rtdetr.train.RTDETRTrainer}

```python
RTDETRTrainer()
```

**Bases:** `DetectionTrainer`

Trainer class for the RT-DETR model developed by Baidu for real-time object detection.

This class extends the DetectionTrainer class for YOLO to adapt to the specific features and architecture of RT-DETR. The model leverages Vision Transformers and has capabilities like IoU-aware query selection and adaptable inference speed.

**Attributes**

| Name | Type | Description |
| --- | --- | --- |
| `loss_names` | `tuple` | Names of the loss components used for training. |
| `data` | `dict` | Dataset configuration containing class count and other parameters. |
| `args` | `dict` | Training arguments and hyperparameters. |
| `save_dir` | `Path` | Directory to save training results. |
| `test_loader` | `DataLoader` | DataLoader for validation/testing data. |

**Methods**

| Name | Description |
| --- | --- |
| [`build_dataset`](#ultralytics.models.rtdetr.train.RTDETRTrainer.build_dataset) | Build and return an RT-DETR dataset for training or validation. |
| [`get_model`](#ultralytics.models.rtdetr.train.RTDETRTrainer.get_model) | Initialize and return an RT-DETR model for object detection tasks. |
| [`get_validator`](#ultralytics.models.rtdetr.train.RTDETRTrainer.get_validator) | Return an RTDETRValidator suitable for RT-DETR model validation. |

**Examples**

```python
>>> from ultralytics.models.rtdetr.train import RTDETRTrainer
>>> args = dict(model="rtdetr-l.yaml", data="coco8.yaml", imgsz=640, epochs=3)
>>> trainer = RTDETRTrainer(overrides=args)
>>> trainer.train()
```

!!! note "Notes"

    - F.grid_sample used in RT-DETR does not support the `deterministic=True` argument.
    - AMP training can lead to NaN outputs and may produce errors during bipartite graph matching.

<details>
<summary>Source code in <code>ultralytics/models/rtdetr/train.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/models/rtdetr/train.py#L14-L89"><i class="fa-brands fa-github" aria-hidden="true" style="margin-right:6px;"></i>View on GitHub</a>
```python
class RTDETRTrainer(DetectionTrainer):
```
</details>

<br>

### Method `ultralytics.models.rtdetr.train.RTDETRTrainer.build_dataset` {#ultralytics.models.rtdetr.train.RTDETRTrainer.build\_dataset}

```python
def build_dataset(self, img_path: str, mode: str = "val", batch: int | None = None)
```

Build and return an RT-DETR dataset for training or validation.

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `img_path` | `str` | Path to the folder containing images. | *required* |
| `mode` | `str` | Dataset mode, either 'train' or 'val'. | `"val"` |
| `batch` | `int, optional` | Batch size for rectangle training. | `None` |

**Returns**

| Type | Description |
| --- | --- |
| `RTDETRDataset` | Dataset object for the specific mode. |

<details>
<summary>Source code in <code>ultralytics/models/rtdetr/train.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/models/rtdetr/train.py#L60-L84"><i class="fa-brands fa-github" aria-hidden="true" style="margin-right:6px;"></i>View on GitHub</a>
```python
def build_dataset(self, img_path: str, mode: str = "val", batch: int | None = None):
    """Build and return an RT-DETR dataset for training or validation.

    Args:
        img_path (str): Path to the folder containing images.
        mode (str): Dataset mode, either 'train' or 'val'.
        batch (int, optional): Batch size for rectangle training.

    Returns:
        (RTDETRDataset): Dataset object for the specific mode.
    """
    return RTDETRDataset(
        img_path=img_path,
        imgsz=self.args.imgsz,
        batch_size=batch,
        augment=mode == "train",
        hyp=self.args,
        rect=False,
        cache=self.args.cache or None,
        single_cls=self.args.single_cls or False,
        prefix=colorstr(f"{mode}: "),
        classes=self.args.classes,
        data=self.data,
        fraction=self.args.fraction if mode == "train" else 1.0,
    )
```
</details>

<br>

### Method `ultralytics.models.rtdetr.train.RTDETRTrainer.get_model` {#ultralytics.models.rtdetr.train.RTDETRTrainer.get\_model}

```python
def get_model(self, cfg: dict | None = None, weights: str | None = None, verbose: bool = True)
```

Initialize and return an RT-DETR model for object detection tasks.

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `cfg` | `dict, optional` | Model configuration. | `None` |
| `weights` | `str, optional` | Path to pre-trained model weights. | `None` |
| `verbose` | `bool` | Verbose logging if True. | `True` |

**Returns**

| Type | Description |
| --- | --- |
| `RTDETRDetectionModel` | Initialized model. |

<details>
<summary>Source code in <code>ultralytics/models/rtdetr/train.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/models/rtdetr/train.py#L44-L58"><i class="fa-brands fa-github" aria-hidden="true" style="margin-right:6px;"></i>View on GitHub</a>
```python
def get_model(self, cfg: dict | None = None, weights: str | None = None, verbose: bool = True):
    """Initialize and return an RT-DETR model for object detection tasks.

    Args:
        cfg (dict, optional): Model configuration.
        weights (str, optional): Path to pre-trained model weights.
        verbose (bool): Verbose logging if True.

    Returns:
        (RTDETRDetectionModel): Initialized model.
    """
    model = RTDETRDetectionModel(cfg, nc=self.data["nc"], ch=self.data["channels"], verbose=verbose and RANK == -1)
    if weights:
        model.load(weights)
    return model
```
</details>

<br>

### Method `ultralytics.models.rtdetr.train.RTDETRTrainer.get_validator` {#ultralytics.models.rtdetr.train.RTDETRTrainer.get\_validator}

```python
def get_validator(self)
```

Return an RTDETRValidator suitable for RT-DETR model validation.

<details>
<summary>Source code in <code>ultralytics/models/rtdetr/train.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/models/rtdetr/train.py#L86-L89"><i class="fa-brands fa-github" aria-hidden="true" style="margin-right:6px;"></i>View on GitHub</a>
```python
def get_validator(self):
    """Return an RTDETRValidator suitable for RT-DETR model validation."""
    self.loss_names = "giou_loss", "cls_loss", "l1_loss"
    return RTDETRValidator(self.test_loader, save_dir=self.save_dir, args=copy(self.args))
```
</details>

<br><br>
