---
description: Documentation for YOLOE segmentation trainer classes in Ultralytics, supporting different training approaches including standard training, linear probing, training from scratch, and visual prompt training.
keywords: YOLOE, segmentation, trainers, YOLOESegTrainer, YOLOEPESegTrainer, YOLOESegTrainerFromScratch, YOLOESegVPTrainer, linear probing, visual prompts, Ultralytics, deep learning
---

# Reference for `ultralytics/models/yolo/yoloe/train_seg.py`

!!! success "Improvements"

    This page is sourced from [https://github.com/ultralytics/ultralytics/blob/main/ultralytics/models/yolo/yoloe/train_seg.py](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/models/yolo/yoloe/train_seg.py). Have an improvement or example to add? Open a [Pull Request](https://docs.ultralytics.com/help/contributing/) — thank you! 🙏

<br>

!!! abstract "Summary"

    === "<span class="doc-kind doc-kind-class">Classes</span>"

        - [`YOLOESegTrainer`](#ultralytics.models.yolo.yoloe.train_seg.YOLOESegTrainer)
        - [`YOLOEPESegTrainer`](#ultralytics.models.yolo.yoloe.train_seg.YOLOEPESegTrainer)
        - [`YOLOESegTrainerFromScratch`](#ultralytics.models.yolo.yoloe.train_seg.YOLOESegTrainerFromScratch)
        - [`YOLOESegVPTrainer`](#ultralytics.models.yolo.yoloe.train_seg.YOLOESegVPTrainer)

    === "<span class="doc-kind doc-kind-method">Methods</span>"

        - [`YOLOESegTrainer.get_model`](#ultralytics.models.yolo.yoloe.train_seg.YOLOESegTrainer.get_model)
        - [`YOLOESegTrainer.get_validator`](#ultralytics.models.yolo.yoloe.train_seg.YOLOESegTrainer.get_validator)
        - [`YOLOEPESegTrainer.get_model`](#ultralytics.models.yolo.yoloe.train_seg.YOLOEPESegTrainer.get_model)


## Class `ultralytics.models.yolo.yoloe.train_seg.YOLOESegTrainer` {#ultralytics.models.yolo.yoloe.train\_seg.YOLOESegTrainer}

```python
YOLOESegTrainer()
```

**Bases:** `YOLOETrainer`, `SegmentationTrainer`

Trainer class for YOLOE segmentation models.

This class combines YOLOETrainer and SegmentationTrainer to provide training functionality specifically for YOLOE segmentation models, enabling both object detection and instance segmentation capabilities.

**Attributes**

| Name | Type | Description |
| --- | --- | --- |
| `cfg` | `dict` | Configuration dictionary with training parameters. |
| `overrides` | `dict` | Dictionary with parameter overrides. |
| `_callbacks` | `list` | List of callback functions for training events. |

**Methods**

| Name | Description |
| --- | --- |
| [`get_model`](#ultralytics.models.yolo.yoloe.train_seg.YOLOESegTrainer.get_model) | Return YOLOESegModel initialized with specified config and weights. |
| [`get_validator`](#ultralytics.models.yolo.yoloe.train_seg.YOLOESegTrainer.get_validator) | Create and return a validator for YOLOE segmentation model evaluation. |

<details>
<summary>Source code in <code>ultralytics/models/yolo/yoloe/train_seg.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/models/yolo/yoloe/train_seg.py#L13-L58"><i class="fa-brands fa-github" aria-hidden="true" style="margin-right:6px;"></i>View on GitHub</a>
```python
class YOLOESegTrainer(YOLOETrainer, SegmentationTrainer):
```
</details>

<br>

### Method `ultralytics.models.yolo.yoloe.train_seg.YOLOESegTrainer.get_model` {#ultralytics.models.yolo.yoloe.train\_seg.YOLOESegTrainer.get\_model}

```python
def get_model(self, cfg = None, weights = None, verbose = True)
```

Return YOLOESegModel initialized with specified config and weights.

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `cfg` | `dict | str, optional` | Model configuration dictionary or YAML file path. | `None` |
| `weights` | `str, optional` | Path to pretrained weights file. | `None` |
| `verbose` | `bool` | Whether to display model information. | `True` |

**Returns**

| Type | Description |
| --- | --- |
| `YOLOESegModel` | Initialized YOLOE segmentation model. |

<details>
<summary>Source code in <code>ultralytics/models/yolo/yoloe/train_seg.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/models/yolo/yoloe/train_seg.py#L25-L47"><i class="fa-brands fa-github" aria-hidden="true" style="margin-right:6px;"></i>View on GitHub</a>
```python
def get_model(self, cfg=None, weights=None, verbose=True):
    """Return YOLOESegModel initialized with specified config and weights.

    Args:
        cfg (dict | str, optional): Model configuration dictionary or YAML file path.
        weights (str, optional): Path to pretrained weights file.
        verbose (bool): Whether to display model information.

    Returns:
        (YOLOESegModel): Initialized YOLOE segmentation model.
    """
    # NOTE: This `nc` here is the max number of different text samples in one image, rather than the actual `nc`.
    # NOTE: Following the official config, nc hard-coded to 80 for now.
    model = YOLOESegModel(
        cfg["yaml_file"] if isinstance(cfg, dict) else cfg,
        ch=self.data["channels"],
        nc=min(self.data["nc"], 80),
        verbose=verbose and RANK == -1,
    )
    if weights:
        model.load(weights)

    return model
```
</details>

<br>

### Method `ultralytics.models.yolo.yoloe.train_seg.YOLOESegTrainer.get_validator` {#ultralytics.models.yolo.yoloe.train\_seg.YOLOESegTrainer.get\_validator}

```python
def get_validator(self)
```

Create and return a validator for YOLOE segmentation model evaluation.

**Returns**

| Type | Description |
| --- | --- |
| `YOLOESegValidator` | Validator for YOLOE segmentation models. |

<details>
<summary>Source code in <code>ultralytics/models/yolo/yoloe/train_seg.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/models/yolo/yoloe/train_seg.py#L49-L58"><i class="fa-brands fa-github" aria-hidden="true" style="margin-right:6px;"></i>View on GitHub</a>
```python
def get_validator(self):
    """Create and return a validator for YOLOE segmentation model evaluation.

    Returns:
        (YOLOESegValidator): Validator for YOLOE segmentation models.
    """
    self.loss_names = "box", "seg", "cls", "dfl"
    return YOLOESegValidator(
        self.test_loader, save_dir=self.save_dir, args=copy(self.args), _callbacks=self.callbacks
    )
```
</details>


<br><br><hr><br>

## Class `ultralytics.models.yolo.yoloe.train_seg.YOLOEPESegTrainer` {#ultralytics.models.yolo.yoloe.train\_seg.YOLOEPESegTrainer}

```python
YOLOEPESegTrainer()
```

**Bases:** `SegmentationTrainer`

Fine-tune YOLOESeg model in linear probing way.

This trainer specializes in fine-tuning YOLOESeg models using a linear probing approach, which involves freezing most of the model and only training specific layers for efficient adaptation to new tasks.

**Attributes**

| Name | Type | Description |
| --- | --- | --- |
| `data` | `dict` | Dataset configuration containing channels, class names, and number of classes. |

**Methods**

| Name | Description |
| --- | --- |
| [`get_model`](#ultralytics.models.yolo.yoloe.train_seg.YOLOEPESegTrainer.get_model) | Return YOLOESegModel initialized with specified config and weights for linear probing. |

<details>
<summary>Source code in <code>ultralytics/models/yolo/yoloe/train_seg.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/models/yolo/yoloe/train_seg.py#L61-L115"><i class="fa-brands fa-github" aria-hidden="true" style="margin-right:6px;"></i>View on GitHub</a>
```python
class YOLOEPESegTrainer(SegmentationTrainer):
```
</details>

<br>

### Method `ultralytics.models.yolo.yoloe.train_seg.YOLOEPESegTrainer.get_model` {#ultralytics.models.yolo.yoloe.train\_seg.YOLOEPESegTrainer.get\_model}

```python
def get_model(self, cfg = None, weights = None, verbose = True)
```

Return YOLOESegModel initialized with specified config and weights for linear probing.

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `cfg` | `dict | str, optional` | Model configuration dictionary or YAML file path. | `None` |
| `weights` | `str, optional` | Path to pretrained weights file. | `None` |
| `verbose` | `bool` | Whether to display model information. | `True` |

**Returns**

| Type | Description |
| --- | --- |
| `YOLOESegModel` | Initialized YOLOE segmentation model configured for linear probing. |

<details>
<summary>Source code in <code>ultralytics/models/yolo/yoloe/train_seg.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/models/yolo/yoloe/train_seg.py#L71-L115"><i class="fa-brands fa-github" aria-hidden="true" style="margin-right:6px;"></i>View on GitHub</a>
```python
def get_model(self, cfg=None, weights=None, verbose=True):
    """Return YOLOESegModel initialized with specified config and weights for linear probing.

    Args:
        cfg (dict | str, optional): Model configuration dictionary or YAML file path.
        weights (str, optional): Path to pretrained weights file.
        verbose (bool): Whether to display model information.

    Returns:
        (YOLOESegModel): Initialized YOLOE segmentation model configured for linear probing.
    """
    # NOTE: This `nc` here is the max number of different text samples in one image, rather than the actual `nc`.
    # NOTE: Following the official config, nc hard-coded to 80 for now.
    model = YOLOESegModel(
        cfg["yaml_file"] if isinstance(cfg, dict) else cfg,
        ch=self.data["channels"],
        nc=self.data["nc"],
        verbose=verbose and RANK == -1,
    )

    del model.model[-1].savpe

    assert weights is not None, "Pretrained weights must be provided for linear probing."
    if weights:
        model.load(weights)

    model.eval()
    names = list(self.data["names"].values())
    # NOTE: `get_text_pe` related to text model and YOLOEDetect.reprta,
    # it'd get correct results as long as loading proper pretrained weights.
    tpe = model.get_text_pe(names)
    model.set_classes(names, tpe)
    model.model[-1].fuse(model.pe)
    model.model[-1].cv3[0][2] = deepcopy(model.model[-1].cv3[0][2]).requires_grad_(True)
    model.model[-1].cv3[1][2] = deepcopy(model.model[-1].cv3[1][2]).requires_grad_(True)
    model.model[-1].cv3[2][2] = deepcopy(model.model[-1].cv3[2][2]).requires_grad_(True)

    if getattr(model.model[-1], "one2one_cv3", None) is not None:
        model.model[-1].one2one_cv3[0][2] = deepcopy(model.model[-1].cv3[0][2]).requires_grad_(True)
        model.model[-1].one2one_cv3[1][2] = deepcopy(model.model[-1].cv3[1][2]).requires_grad_(True)
        model.model[-1].one2one_cv3[2][2] = deepcopy(model.model[-1].cv3[2][2]).requires_grad_(True)

    model.train()

    return model
```
</details>


<br><br><hr><br>

## Class `ultralytics.models.yolo.yoloe.train_seg.YOLOESegTrainerFromScratch` {#ultralytics.models.yolo.yoloe.train\_seg.YOLOESegTrainerFromScratch}

```python
YOLOESegTrainerFromScratch()
```

**Bases:** `YOLOETrainerFromScratch`, `YOLOESegTrainer`

Trainer for YOLOE segmentation models trained from scratch without pretrained weights.

<details>
<summary>Source code in <code>ultralytics/models/yolo/yoloe/train_seg.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/models/yolo/yoloe/train_seg.py#L118-L121"><i class="fa-brands fa-github" aria-hidden="true" style="margin-right:6px;"></i>View on GitHub</a>
```python
class YOLOESegTrainerFromScratch(YOLOETrainerFromScratch, YOLOESegTrainer):
```
</details>


<br><br><hr><br>

## Class `ultralytics.models.yolo.yoloe.train_seg.YOLOESegVPTrainer` {#ultralytics.models.yolo.yoloe.train\_seg.YOLOESegVPTrainer}

```python
YOLOESegVPTrainer()
```

**Bases:** `YOLOEVPTrainer`, `YOLOESegTrainerFromScratch`

Trainer for YOLOE segmentation models with Vision Prompt (VP) capabilities.

<details>
<summary>Source code in <code>ultralytics/models/yolo/yoloe/train_seg.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/models/yolo/yoloe/train_seg.py#L124-L127"><i class="fa-brands fa-github" aria-hidden="true" style="margin-right:6px;"></i>View on GitHub</a>
```python
class YOLOESegVPTrainer(YOLOEVPTrainer, YOLOESegTrainerFromScratch):
```
</details>

<br><br>
