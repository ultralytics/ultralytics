---
description: Explore the PoseTrainer class for training pose models using YOLO from Ultralytics. Includes initialization, model configuration, and plotting methods.
keywords: PoseTrainer, YOLO, Ultralytics, pose models, training, model configuration, deep learning, machine learning, pose estimation
---

# Reference for `ultralytics/models/yolo/pose/train.py`

!!! success "Improvements"

    This page is sourced from [https://github.com/ultralytics/ultralytics/blob/main/ultralytics/models/yolo/pose/train.py](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/models/yolo/pose/train.py). Have an improvement or example to add? Open a [Pull Request](https://docs.ultralytics.com/help/contributing/) — thank you! 🙏

<br>

!!! abstract "Summary"

    === "<span class="doc-kind doc-kind-class">Classes</span>"

        - [`PoseTrainer`](#ultralytics.models.yolo.pose.train.PoseTrainer)

    === "<span class="doc-kind doc-kind-method">Methods</span>"

        - [`PoseTrainer.get_model`](#ultralytics.models.yolo.pose.train.PoseTrainer.get_model)
        - [`PoseTrainer.set_model_attributes`](#ultralytics.models.yolo.pose.train.PoseTrainer.set_model_attributes)
        - [`PoseTrainer.get_validator`](#ultralytics.models.yolo.pose.train.PoseTrainer.get_validator)
        - [`PoseTrainer.get_dataset`](#ultralytics.models.yolo.pose.train.PoseTrainer.get_dataset)


## Class `ultralytics.models.yolo.pose.train.PoseTrainer` {#ultralytics.models.yolo.pose.train.PoseTrainer}

```python
PoseTrainer(self, cfg = DEFAULT_CFG, overrides: dict[str, Any] | None = None, _callbacks = None)
```

**Bases:** `yolo.detect.DetectionTrainer`

A class extending the DetectionTrainer class for training YOLO pose estimation models.

This trainer specializes in handling pose estimation tasks, managing model training, validation, and visualization of pose keypoints alongside bounding boxes.

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `cfg` | `dict, optional` | Default configuration dictionary containing training parameters. | `DEFAULT_CFG` |
| `overrides` | `dict, optional` | Dictionary of parameter overrides for the default configuration. | `None` |
| `_callbacks` | `list, optional` | List of callback functions to be executed during training. | `None` |

**Attributes**

| Name | Type | Description |
| --- | --- | --- |
| `args` | `dict` | Configuration arguments for training. |
| `model` | `PoseModel` | The pose estimation model being trained. |
| `data` | `dict` | Dataset configuration including keypoint shape information. |
| `loss_names` | `tuple` | Names of the loss components used in training. |

**Methods**

| Name | Description |
| --- | --- |
| [`get_dataset`](#ultralytics.models.yolo.pose.train.PoseTrainer.get_dataset) | Retrieve the dataset and ensure it contains the required `kpt_shape` key. |
| [`get_model`](#ultralytics.models.yolo.pose.train.PoseTrainer.get_model) | Get pose estimation model with specified configuration and weights. |
| [`get_validator`](#ultralytics.models.yolo.pose.train.PoseTrainer.get_validator) | Return an instance of the PoseValidator class for validation. |
| [`set_model_attributes`](#ultralytics.models.yolo.pose.train.PoseTrainer.set_model_attributes) | Set keypoints shape attribute of PoseModel. |

**Examples**

```python
>>> from ultralytics.models.yolo.pose import PoseTrainer
>>> args = dict(model="yolo26n-pose.pt", data="coco8-pose.yaml", epochs=3)
>>> trainer = PoseTrainer(overrides=args)
>>> trainer.train()
```

!!! note "Notes"

    This trainer will automatically set the task to 'pose' regardless of what is provided in overrides.
    A warning is issued when using Apple MPS device due to known bugs with pose models.

<details>
<summary>Source code in <code>ultralytics/models/yolo/pose/train.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/models/yolo/pose/train.py#L15-L113"><i class="fa-brands fa-github" aria-hidden="true" style="margin-right:6px;"></i>View on GitHub</a>
```python
class PoseTrainer(yolo.detect.DetectionTrainer):
    """A class extending the DetectionTrainer class for training YOLO pose estimation models.

    This trainer specializes in handling pose estimation tasks, managing model training, validation, and visualization
    of pose keypoints alongside bounding boxes.

    Attributes:
        args (dict): Configuration arguments for training.
        model (PoseModel): The pose estimation model being trained.
        data (dict): Dataset configuration including keypoint shape information.
        loss_names (tuple): Names of the loss components used in training.

    Methods:
        get_model: Retrieve a pose estimation model with specified configuration.
        set_model_attributes: Set keypoints shape attribute on the model.
        get_validator: Create a validator instance for model evaluation.
        plot_training_samples: Visualize training samples with keypoints.
        get_dataset: Retrieve the dataset and ensure it contains required kpt_shape key.

    Examples:
        >>> from ultralytics.models.yolo.pose import PoseTrainer
        >>> args = dict(model="yolo26n-pose.pt", data="coco8-pose.yaml", epochs=3)
        >>> trainer = PoseTrainer(overrides=args)
        >>> trainer.train()
    """

    def __init__(self, cfg=DEFAULT_CFG, overrides: dict[str, Any] | None = None, _callbacks=None):
        """Initialize a PoseTrainer object for training YOLO pose estimation models.

        Args:
            cfg (dict, optional): Default configuration dictionary containing training parameters.
            overrides (dict, optional): Dictionary of parameter overrides for the default configuration.
            _callbacks (list, optional): List of callback functions to be executed during training.

        Notes:
            This trainer will automatically set the task to 'pose' regardless of what is provided in overrides.
            A warning is issued when using Apple MPS device due to known bugs with pose models.
        """
        if overrides is None:
            overrides = {}
        overrides["task"] = "pose"
        super().__init__(cfg, overrides, _callbacks)
```
</details>

<br>

### Method `ultralytics.models.yolo.pose.train.PoseTrainer.get_dataset` {#ultralytics.models.yolo.pose.train.PoseTrainer.get\_dataset}

```python
def get_dataset(self) -> dict[str, Any]
```

Retrieve the dataset and ensure it contains the required `kpt_shape` key.

**Returns**

| Type | Description |
| --- | --- |
| `dict` | A dictionary containing the training/validation/test dataset and category names. |

**Raises**

| Type | Description |
| --- | --- |
| `KeyError` | If the `kpt_shape` key is not present in the dataset. |

<details>
<summary>Source code in <code>ultralytics/models/yolo/pose/train.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/models/yolo/pose/train.py#L101-L113"><i class="fa-brands fa-github" aria-hidden="true" style="margin-right:6px;"></i>View on GitHub</a>
```python
def get_dataset(self) -> dict[str, Any]:
    """Retrieve the dataset and ensure it contains the required `kpt_shape` key.

    Returns:
        (dict): A dictionary containing the training/validation/test dataset and category names.

    Raises:
        KeyError: If the `kpt_shape` key is not present in the dataset.
    """
    data = super().get_dataset()
    if "kpt_shape" not in data:
        raise KeyError(f"No `kpt_shape` in the {self.args.data}. See https://docs.ultralytics.com/datasets/pose/")
    return data
```
</details>

<br>

### Method `ultralytics.models.yolo.pose.train.PoseTrainer.get_model` {#ultralytics.models.yolo.pose.train.PoseTrainer.get\_model}

```python
def get_model(
    self,
    cfg: str | Path | dict[str, Any] | None = None,
    weights: str | Path | None = None,
    verbose: bool = True,
) -> PoseModel
```

Get pose estimation model with specified configuration and weights.

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `cfg` | `str | Path | dict, optional` | Model configuration file path or dictionary. | `None` |
| `weights` | `str | Path, optional` | Path to the model weights file. | `None` |
| `verbose` | `bool` | Whether to display model information. | `True` |

**Returns**

| Type | Description |
| --- | --- |
| `PoseModel` | Initialized pose estimation model. |

<details>
<summary>Source code in <code>ultralytics/models/yolo/pose/train.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/models/yolo/pose/train.py#L58-L80"><i class="fa-brands fa-github" aria-hidden="true" style="margin-right:6px;"></i>View on GitHub</a>
```python
def get_model(
    self,
    cfg: str | Path | dict[str, Any] | None = None,
    weights: str | Path | None = None,
    verbose: bool = True,
) -> PoseModel:
    """Get pose estimation model with specified configuration and weights.

    Args:
        cfg (str | Path | dict, optional): Model configuration file path or dictionary.
        weights (str | Path, optional): Path to the model weights file.
        verbose (bool): Whether to display model information.

    Returns:
        (PoseModel): Initialized pose estimation model.
    """
    model = PoseModel(
        cfg, nc=self.data["nc"], ch=self.data["channels"], data_kpt_shape=self.data["kpt_shape"], verbose=verbose
    )
    if weights:
        model.load(weights)

    return model
```
</details>

<br>

### Method `ultralytics.models.yolo.pose.train.PoseTrainer.get_validator` {#ultralytics.models.yolo.pose.train.PoseTrainer.get\_validator}

```python
def get_validator(self)
```

Return an instance of the PoseValidator class for validation.

<details>
<summary>Source code in <code>ultralytics/models/yolo/pose/train.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/models/yolo/pose/train.py#L92-L99"><i class="fa-brands fa-github" aria-hidden="true" style="margin-right:6px;"></i>View on GitHub</a>
```python
def get_validator(self):
    """Return an instance of the PoseValidator class for validation."""
    self.loss_names = "box_loss", "pose_loss", "kobj_loss", "cls_loss", "dfl_loss"
    if getattr(unwrap_model(self.model).model[-1], "flow_model", None) is not None:
        self.loss_names += ("rle_loss",)
    return yolo.pose.PoseValidator(
        self.test_loader, save_dir=self.save_dir, args=copy(self.args), _callbacks=self.callbacks
    )
```
</details>

<br>

### Method `ultralytics.models.yolo.pose.train.PoseTrainer.set_model_attributes` {#ultralytics.models.yolo.pose.train.PoseTrainer.set\_model\_attributes}

```python
def set_model_attributes(self)
```

Set keypoints shape attribute of PoseModel.

<details>
<summary>Source code in <code>ultralytics/models/yolo/pose/train.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/models/yolo/pose/train.py#L82-L90"><i class="fa-brands fa-github" aria-hidden="true" style="margin-right:6px;"></i>View on GitHub</a>
```python
def set_model_attributes(self):
    """Set keypoints shape attribute of PoseModel."""
    super().set_model_attributes()
    self.model.kpt_shape = self.data["kpt_shape"]
    kpt_names = self.data.get("kpt_names")
    if not kpt_names:
        names = list(map(str, range(self.model.kpt_shape[0])))
        kpt_names = {i: names for i in range(self.model.nc)}
    self.model.kpt_names = kpt_names
```
</details>

<br><br>
