---
description: Explore the Ultralytics YOLO OBB Trainer class for efficient training with Oriented Bounding Box models. Learn with examples and method details.
keywords: Ultralytics, YOLO, OBB Trainer, Oriented Bounding Box, Machine Learning, Training, AI
---

# Reference for `ultralytics/models/yolo/obb/train.py`

!!! success "Improvements"

    This page is sourced from [https://github.com/ultralytics/ultralytics/blob/main/ultralytics/models/yolo/obb/train.py](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/models/yolo/obb/train.py). Have an improvement or example to add? Open a [Pull Request](https://docs.ultralytics.com/help/contributing/) — thank you! 🙏

<br>

!!! abstract "Summary"

    === "<span class="doc-kind doc-kind-class">Classes</span>"

        - [`OBBTrainer`](#ultralytics.models.yolo.obb.train.OBBTrainer)

    === "<span class="doc-kind doc-kind-method">Methods</span>"

        - [`OBBTrainer.get_model`](#ultralytics.models.yolo.obb.train.OBBTrainer.get_model)
        - [`OBBTrainer.get_validator`](#ultralytics.models.yolo.obb.train.OBBTrainer.get_validator)


## Class `ultralytics.models.yolo.obb.train.OBBTrainer` {#ultralytics.models.yolo.obb.train.OBBTrainer}

```python
OBBTrainer(self, cfg = DEFAULT_CFG, overrides: dict | None = None, _callbacks: list[Any] | None = None)
```

**Bases:** `yolo.detect.DetectionTrainer`

A class extending the DetectionTrainer class for training based on an Oriented Bounding Box (OBB) model.

This trainer specializes in training YOLO models that detect oriented bounding boxes, which are useful for detecting objects at arbitrary angles rather than just axis-aligned rectangles.

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `cfg` | `dict, optional` | Configuration dictionary for the trainer. Contains training parameters and model<br>    configuration. | `DEFAULT_CFG` |
| `overrides` | `dict, optional` | Dictionary of parameter overrides for the configuration. Any values here will<br>    take precedence over those in cfg. | `None` |
| `_callbacks` | `list[Any], optional` | List of callback functions to be invoked during training. | `None` |

**Attributes**

| Name | Type | Description |
| --- | --- | --- |
| `loss_names` | `tuple` | Names of the loss components used during training including box_loss, cls_loss, dfl_loss,<br>    and angle_loss. |

**Methods**

| Name | Description |
| --- | --- |
| [`get_model`](#ultralytics.models.yolo.obb.train.OBBTrainer.get_model) | Return OBBModel initialized with specified config and weights. |
| [`get_validator`](#ultralytics.models.yolo.obb.train.OBBTrainer.get_validator) | Return an instance of OBBValidator for validation of YOLO model. |

**Examples**

```python
>>> from ultralytics.models.yolo.obb import OBBTrainer
>>> args = dict(model="yolo26n-obb.pt", data="dota8.yaml", epochs=3)
>>> trainer = OBBTrainer(overrides=args)
>>> trainer.train()
```

<details>
<summary>Source code in <code>ultralytics/models/yolo/obb/train.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/models/yolo/obb/train.py#L14-L79"><i class="fa-brands fa-github" aria-hidden="true" style="margin-right:6px;"></i>View on GitHub</a>
```python
class OBBTrainer(yolo.detect.DetectionTrainer):
    """A class extending the DetectionTrainer class for training based on an Oriented Bounding Box (OBB) model.

    This trainer specializes in training YOLO models that detect oriented bounding boxes, which are useful for detecting
    objects at arbitrary angles rather than just axis-aligned rectangles.

    Attributes:
        loss_names (tuple): Names of the loss components used during training including box_loss, cls_loss, dfl_loss,
            and angle_loss.

    Methods:
        get_model: Return OBBModel initialized with specified config and weights.
        get_validator: Return an instance of OBBValidator for validation of YOLO model.

    Examples:
        >>> from ultralytics.models.yolo.obb import OBBTrainer
        >>> args = dict(model="yolo26n-obb.pt", data="dota8.yaml", epochs=3)
        >>> trainer = OBBTrainer(overrides=args)
        >>> trainer.train()
    """

    def __init__(self, cfg=DEFAULT_CFG, overrides: dict | None = None, _callbacks: list[Any] | None = None):
        """Initialize an OBBTrainer object for training Oriented Bounding Box (OBB) models.

        Args:
            cfg (dict, optional): Configuration dictionary for the trainer. Contains training parameters and model
                configuration.
            overrides (dict, optional): Dictionary of parameter overrides for the configuration. Any values here will
                take precedence over those in cfg.
            _callbacks (list[Any], optional): List of callback functions to be invoked during training.
        """
        if overrides is None:
            overrides = {}
        overrides["task"] = "obb"
        super().__init__(cfg, overrides, _callbacks)
```
</details>

<br>

### Method `ultralytics.models.yolo.obb.train.OBBTrainer.get_model` {#ultralytics.models.yolo.obb.train.OBBTrainer.get\_model}

```python
def get_model(self, cfg: str | dict | None = None, weights: str | Path | None = None, verbose: bool = True) -> OBBModel
```

Return OBBModel initialized with specified config and weights.

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `cfg` | `str | dict, optional` | Model configuration. Can be a path to a YAML config file, a dictionary<br>    containing configuration parameters, or None to use default configuration. | `None` |
| `weights` | `str | Path, optional` | Path to pretrained weights file. If None, random initialization is used. | `None` |
| `verbose` | `bool` | Whether to display model information during initialization. | `True` |

**Returns**

| Type | Description |
| --- | --- |
| `OBBModel` | Initialized OBBModel with the specified configuration and weights. |

**Examples**

```python
>>> trainer = OBBTrainer()
>>> model = trainer.get_model(cfg="yolo26n-obb.yaml", weights="yolo26n-obb.pt")
```

<details>
<summary>Source code in <code>ultralytics/models/yolo/obb/train.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/models/yolo/obb/train.py#L50-L72"><i class="fa-brands fa-github" aria-hidden="true" style="margin-right:6px;"></i>View on GitHub</a>
```python
def get_model(
    self, cfg: str | dict | None = None, weights: str | Path | None = None, verbose: bool = True
) -> OBBModel:
    """Return OBBModel initialized with specified config and weights.

    Args:
        cfg (str | dict, optional): Model configuration. Can be a path to a YAML config file, a dictionary
            containing configuration parameters, or None to use default configuration.
        weights (str | Path, optional): Path to pretrained weights file. If None, random initialization is used.
        verbose (bool): Whether to display model information during initialization.

    Returns:
        (OBBModel): Initialized OBBModel with the specified configuration and weights.

    Examples:
        >>> trainer = OBBTrainer()
        >>> model = trainer.get_model(cfg="yolo26n-obb.yaml", weights="yolo26n-obb.pt")
    """
    model = OBBModel(cfg, nc=self.data["nc"], ch=self.data["channels"], verbose=verbose and RANK == -1)
    if weights:
        model.load(weights)

    return model
```
</details>

<br>

### Method `ultralytics.models.yolo.obb.train.OBBTrainer.get_validator` {#ultralytics.models.yolo.obb.train.OBBTrainer.get\_validator}

```python
def get_validator(self)
```

Return an instance of OBBValidator for validation of YOLO model.

<details>
<summary>Source code in <code>ultralytics/models/yolo/obb/train.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/models/yolo/obb/train.py#L74-L79"><i class="fa-brands fa-github" aria-hidden="true" style="margin-right:6px;"></i>View on GitHub</a>
```python
def get_validator(self):
    """Return an instance of OBBValidator for validation of YOLO model."""
    self.loss_names = "box_loss", "cls_loss", "dfl_loss", "angle_loss"
    return yolo.obb.OBBValidator(
        self.test_loader, save_dir=self.save_dir, args=copy(self.args), _callbacks=self.callbacks
    )
```
</details>

<br><br>
