---
description: Learn how to train YOLO models for segmentation tasks with Ultralytics. Explore the SegmentationTrainer class and its functionalities.
keywords: YOLO, segmentation, train, Ultralytics, SegmentationTrainer, Python, machine learning, deep learning, tutorials
---

# Reference for `ultralytics/models/yolo/segment/train.py`

!!! success "Improvements"

    This page is sourced from [https://github.com/ultralytics/ultralytics/blob/main/ultralytics/models/yolo/segment/train.py](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/models/yolo/segment/train.py). Have an improvement or example to add? Open a [Pull Request](https://docs.ultralytics.com/help/contributing/) — thank you! 🙏

<br>

!!! abstract "Summary"

    === "<span class="doc-kind doc-kind-class">Classes</span>"

        - [`SegmentationTrainer`](#ultralytics.models.yolo.segment.train.SegmentationTrainer)

    === "<span class="doc-kind doc-kind-method">Methods</span>"

        - [`SegmentationTrainer.get_model`](#ultralytics.models.yolo.segment.train.SegmentationTrainer.get_model)
        - [`SegmentationTrainer.get_validator`](#ultralytics.models.yolo.segment.train.SegmentationTrainer.get_validator)


## Class `ultralytics.models.yolo.segment.train.SegmentationTrainer` {#ultralytics.models.yolo.segment.train.SegmentationTrainer}

```python
SegmentationTrainer(self, cfg = DEFAULT_CFG, overrides: dict | None = None, _callbacks = None)
```

**Bases:** `yolo.detect.DetectionTrainer`

A class extending the DetectionTrainer class for training based on a segmentation model.

This trainer specializes in handling segmentation tasks, extending the detection trainer with segmentation-specific functionality including model initialization, validation, and visualization.

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `cfg` | `dict` | Configuration dictionary with default training settings. | `DEFAULT_CFG` |
| `overrides` | `dict, optional` | Dictionary of parameter overrides for the default configuration. | `None` |
| `_callbacks` | `list, optional` | List of callback functions to be executed during training. | `None` |

**Attributes**

| Name | Type | Description |
| --- | --- | --- |
| `loss_names` | `tuple[str]` | Names of the loss components used during training. |

**Methods**

| Name | Description |
| --- | --- |
| [`get_model`](#ultralytics.models.yolo.segment.train.SegmentationTrainer.get_model) | Initialize and return a SegmentationModel with specified configuration and weights. |
| [`get_validator`](#ultralytics.models.yolo.segment.train.SegmentationTrainer.get_validator) | Return an instance of SegmentationValidator for validation of YOLO model. |

**Examples**

```python
>>> from ultralytics.models.yolo.segment import SegmentationTrainer
>>> args = dict(model="yolo26n-seg.pt", data="coco8-seg.yaml", epochs=3)
>>> trainer = SegmentationTrainer(overrides=args)
>>> trainer.train()
```

<details>
<summary>Source code in <code>ultralytics/models/yolo/segment/train.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/models/yolo/segment/train.py#L13-L69"><i class="fa-brands fa-github" aria-hidden="true" style="margin-right:6px;"></i>View on GitHub</a>
```python
class SegmentationTrainer(yolo.detect.DetectionTrainer):
    """A class extending the DetectionTrainer class for training based on a segmentation model.

    This trainer specializes in handling segmentation tasks, extending the detection trainer with segmentation-specific
    functionality including model initialization, validation, and visualization.

    Attributes:
        loss_names (tuple[str]): Names of the loss components used during training.

    Examples:
        >>> from ultralytics.models.yolo.segment import SegmentationTrainer
        >>> args = dict(model="yolo26n-seg.pt", data="coco8-seg.yaml", epochs=3)
        >>> trainer = SegmentationTrainer(overrides=args)
        >>> trainer.train()
    """

    def __init__(self, cfg=DEFAULT_CFG, overrides: dict | None = None, _callbacks=None):
        """Initialize a SegmentationTrainer object.

        Args:
            cfg (dict): Configuration dictionary with default training settings.
            overrides (dict, optional): Dictionary of parameter overrides for the default configuration.
            _callbacks (list, optional): List of callback functions to be executed during training.
        """
        if overrides is None:
            overrides = {}
        overrides["task"] = "segment"
        super().__init__(cfg, overrides, _callbacks)
```
</details>

<br>

### Method `ultralytics.models.yolo.segment.train.SegmentationTrainer.get_model` {#ultralytics.models.yolo.segment.train.SegmentationTrainer.get\_model}

```python
def get_model(self, cfg: dict | str | None = None, weights: str | Path | None = None, verbose: bool = True)
```

Initialize and return a SegmentationModel with specified configuration and weights.

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `cfg` | `dict | str, optional` | Model configuration. Can be a dictionary, a path to a YAML file, or None. | `None` |
| `weights` | `str | Path, optional` | Path to pretrained weights file. | `None` |
| `verbose` | `bool` | Whether to display model information during initialization. | `True` |

**Returns**

| Type | Description |
| --- | --- |
| `SegmentationModel` | Initialized segmentation model with loaded weights if specified. |

**Examples**

```python
>>> trainer = SegmentationTrainer()
>>> model = trainer.get_model(cfg="yolo26n-seg.yaml")
>>> model = trainer.get_model(weights="yolo26n-seg.pt", verbose=False)
```

<details>
<summary>Source code in <code>ultralytics/models/yolo/segment/train.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/models/yolo/segment/train.py#L42-L62"><i class="fa-brands fa-github" aria-hidden="true" style="margin-right:6px;"></i>View on GitHub</a>
```python
def get_model(self, cfg: dict | str | None = None, weights: str | Path | None = None, verbose: bool = True):
    """Initialize and return a SegmentationModel with specified configuration and weights.

    Args:
        cfg (dict | str, optional): Model configuration. Can be a dictionary, a path to a YAML file, or None.
        weights (str | Path, optional): Path to pretrained weights file.
        verbose (bool): Whether to display model information during initialization.

    Returns:
        (SegmentationModel): Initialized segmentation model with loaded weights if specified.

    Examples:
        >>> trainer = SegmentationTrainer()
        >>> model = trainer.get_model(cfg="yolo26n-seg.yaml")
        >>> model = trainer.get_model(weights="yolo26n-seg.pt", verbose=False)
    """
    model = SegmentationModel(cfg, nc=self.data["nc"], ch=self.data["channels"], verbose=verbose and RANK == -1)
    if weights:
        model.load(weights)

    return model
```
</details>

<br>

### Method `ultralytics.models.yolo.segment.train.SegmentationTrainer.get_validator` {#ultralytics.models.yolo.segment.train.SegmentationTrainer.get\_validator}

```python
def get_validator(self)
```

Return an instance of SegmentationValidator for validation of YOLO model.

<details>
<summary>Source code in <code>ultralytics/models/yolo/segment/train.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/models/yolo/segment/train.py#L64-L69"><i class="fa-brands fa-github" aria-hidden="true" style="margin-right:6px;"></i>View on GitHub</a>
```python
def get_validator(self):
    """Return an instance of SegmentationValidator for validation of YOLO model."""
    self.loss_names = "box_loss", "seg_loss", "cls_loss", "dfl_loss", "sem_loss"
    return yolo.segment.SegmentationValidator(
        self.test_loader, save_dir=self.save_dir, args=copy(self.args), _callbacks=self.callbacks
    )
```
</details>

<br><br>
