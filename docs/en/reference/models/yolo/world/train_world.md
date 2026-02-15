---
description: Explore the WorldTrainerFromScratch in YOLO for open-set datasets. Learn how to build, train, and evaluate models efficiently.
keywords: YOLO, WorldTrainer, open-set datasets, training, evaluation, build dataset, YOLO World, machine learning
---

# Reference for `ultralytics/models/yolo/world/train_world.py`

!!! success "Improvements"

    This page is sourced from [https://github.com/ultralytics/ultralytics/blob/main/ultralytics/models/yolo/world/train_world.py](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/models/yolo/world/train_world.py). Have an improvement or example to add? Open a [Pull Request](https://docs.ultralytics.com/help/contributing/) — thank you! 🙏

<br>

!!! abstract "Summary"

    === "<span class="doc-kind doc-kind-class">Classes</span>"

        - [`WorldTrainerFromScratch`](#ultralytics.models.yolo.world.train_world.WorldTrainerFromScratch)

    === "<span class="doc-kind doc-kind-method">Methods</span>"

        - [`WorldTrainerFromScratch.build_dataset`](#ultralytics.models.yolo.world.train_world.WorldTrainerFromScratch.build_dataset)
        - [`WorldTrainerFromScratch.check_data_config`](#ultralytics.models.yolo.world.train_world.WorldTrainerFromScratch.check_data_config)
        - [`WorldTrainerFromScratch.get_dataset`](#ultralytics.models.yolo.world.train_world.WorldTrainerFromScratch.get_dataset)
        - [`WorldTrainerFromScratch.plot_training_labels`](#ultralytics.models.yolo.world.train_world.WorldTrainerFromScratch.plot_training_labels)
        - [`WorldTrainerFromScratch.final_eval`](#ultralytics.models.yolo.world.train_world.WorldTrainerFromScratch.final_eval)


## Class `ultralytics.models.yolo.world.train_world.WorldTrainerFromScratch` {#ultralytics.models.yolo.world.train\_world.WorldTrainerFromScratch}

```python
WorldTrainerFromScratch(self, cfg = DEFAULT_CFG, overrides = None, _callbacks = None)
```

**Bases:** `WorldTrainer`

A class extending the WorldTrainer for training a world model from scratch on open-set datasets.

This trainer specializes in handling mixed datasets including both object detection and grounding datasets, supporting training YOLO-World models with combined vision-language capabilities.

This initializes a trainer for YOLO-World models from scratch, supporting mixed datasets including both object detection and grounding datasets for vision-language capabilities.

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `cfg` | `dict` | Configuration dictionary with default parameters for model training. | `DEFAULT_CFG` |
| `overrides` | `dict, optional` | Dictionary of parameter overrides to customize the configuration. | `None` |
| `_callbacks` | `list, optional` | List of callback functions to be executed during different stages of training. | `None` |

**Attributes**

| Name | Type | Description |
| --- | --- | --- |
| `cfg` | `dict` | Configuration dictionary with default parameters for model training. |
| `overrides` | `dict` | Dictionary of parameter overrides to customize the configuration. |
| `_callbacks` | `list` | List of callback functions to be executed during different stages of training. |
| `data` | `dict` | Final processed data configuration containing train/val paths and metadata. |
| `training_data` | `dict` | Dictionary mapping training dataset paths to their configurations. |

**Methods**

| Name | Description |
| --- | --- |
| [`build_dataset`](#ultralytics.models.yolo.world.train_world.WorldTrainerFromScratch.build_dataset) | Build YOLO Dataset for training or validation. |
| [`check_data_config`](#ultralytics.models.yolo.world.train_world.WorldTrainerFromScratch.check_data_config) | Check and load the data configuration from a YAML file or dictionary. |
| [`final_eval`](#ultralytics.models.yolo.world.train_world.WorldTrainerFromScratch.final_eval) | Perform final evaluation and validation for the YOLO-World model. |
| [`get_dataset`](#ultralytics.models.yolo.world.train_world.WorldTrainerFromScratch.get_dataset) | Get train and validation paths from data dictionary. |
| [`plot_training_labels`](#ultralytics.models.yolo.world.train_world.WorldTrainerFromScratch.plot_training_labels) | Skip label plotting for YOLO-World training. |

**Examples**

```python
>>> from ultralytics.models.yolo.world.train_world import WorldTrainerFromScratch
>>> from ultralytics import YOLOWorld
>>> data = dict(
...     train=dict(
...         yolo_data=["Objects365.yaml"],
...         grounding_data=[
...             dict(
...                 img_path="flickr30k/images",
...                 json_file="flickr30k/final_flickr_separateGT_train.json",
...             ),
...             dict(
...                 img_path="GQA/images",
...                 json_file="GQA/final_mixed_train_no_coco.json",
...             ),
...         ],
...     ),
...     val=dict(yolo_data=["lvis.yaml"]),
... )
>>> model = YOLOWorld("yolov8s-worldv2.yaml")
>>> model.train(data=data, trainer=WorldTrainerFromScratch)
```

<details>
<summary>Source code in <code>ultralytics/models/yolo/world/train_world.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/models/yolo/world/train_world.py#L15-L197"><i class="fa-brands fa-github" aria-hidden="true" style="margin-right:6px;"></i>View on GitHub</a>
```python
class WorldTrainerFromScratch(WorldTrainer):
    """A class extending the WorldTrainer for training a world model from scratch on open-set datasets.

    This trainer specializes in handling mixed datasets including both object detection and grounding datasets,
    supporting training YOLO-World models with combined vision-language capabilities.

    Attributes:
        cfg (dict): Configuration dictionary with default parameters for model training.
        overrides (dict): Dictionary of parameter overrides to customize the configuration.
        _callbacks (list): List of callback functions to be executed during different stages of training.
        data (dict): Final processed data configuration containing train/val paths and metadata.
        training_data (dict): Dictionary mapping training dataset paths to their configurations.

    Methods:
        build_dataset: Build YOLO Dataset for training or validation with mixed dataset support.
        get_dataset: Get train and validation paths from data dictionary.
        plot_training_labels: Skip label plotting for YOLO-World training.
        final_eval: Perform final evaluation and validation for the YOLO-World model.

    Examples:
        >>> from ultralytics.models.yolo.world.train_world import WorldTrainerFromScratch
        >>> from ultralytics import YOLOWorld
        >>> data = dict(
        ...     train=dict(
        ...         yolo_data=["Objects365.yaml"],
        ...         grounding_data=[
        ...             dict(
        ...                 img_path="flickr30k/images",
        ...                 json_file="flickr30k/final_flickr_separateGT_train.json",
        ...             ),
        ...             dict(
        ...                 img_path="GQA/images",
        ...                 json_file="GQA/final_mixed_train_no_coco.json",
        ...             ),
        ...         ],
        ...     ),
        ...     val=dict(yolo_data=["lvis.yaml"]),
        ... )
        >>> model = YOLOWorld("yolov8s-worldv2.yaml")
        >>> model.train(data=data, trainer=WorldTrainerFromScratch)
    """

    def __init__(self, cfg=DEFAULT_CFG, overrides=None, _callbacks=None):
        """Initialize a WorldTrainerFromScratch object.

        This initializes a trainer for YOLO-World models from scratch, supporting mixed datasets including both object
        detection and grounding datasets for vision-language capabilities.

        Args:
            cfg (dict): Configuration dictionary with default parameters for model training.
            overrides (dict, optional): Dictionary of parameter overrides to customize the configuration.
            _callbacks (list, optional): List of callback functions to be executed during different stages of training.
        """
        if overrides is None:
            overrides = {}
        super().__init__(cfg, overrides, _callbacks)
```
</details>

<br>

### Method `ultralytics.models.yolo.world.train_world.WorldTrainerFromScratch.build_dataset` {#ultralytics.models.yolo.world.train\_world.WorldTrainerFromScratch.build\_dataset}

```python
def build_dataset(self, img_path, mode = "train", batch = None)
```

Build YOLO Dataset for training or validation.

This method constructs appropriate datasets based on the mode and input paths, handling both standard YOLO datasets and grounding datasets with different formats.

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `img_path` | `list[str] | str` | Path to the folder containing images or list of paths. | *required* |
| `mode` | `str` | 'train' mode or 'val' mode, allowing customized augmentations for each mode. | `"train"` |
| `batch` | `int, optional` | Size of batches, used for rectangular training/validation. | `None` |

**Returns**

| Type | Description |
| --- | --- |
| `YOLOConcatDataset | Dataset` | The constructed dataset for training or validation. |

<details>
<summary>Source code in <code>ultralytics/models/yolo/world/train_world.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/models/yolo/world/train_world.py#L72-L104"><i class="fa-brands fa-github" aria-hidden="true" style="margin-right:6px;"></i>View on GitHub</a>
```python
def build_dataset(self, img_path, mode="train", batch=None):
    """Build YOLO Dataset for training or validation.

    This method constructs appropriate datasets based on the mode and input paths, handling both standard YOLO
    datasets and grounding datasets with different formats.

    Args:
        img_path (list[str] | str): Path to the folder containing images or list of paths.
        mode (str): 'train' mode or 'val' mode, allowing customized augmentations for each mode.
        batch (int, optional): Size of batches, used for rectangular training/validation.

    Returns:
        (YOLOConcatDataset | Dataset): The constructed dataset for training or validation.
    """
    gs = max(int(unwrap_model(self.model).stride.max() if self.model else 0), 32)
    if mode != "train":
        return build_yolo_dataset(self.args, img_path, batch, self.data, mode=mode, rect=False, stride=gs)
    datasets = [
        build_yolo_dataset(self.args, im_path, batch, self.training_data[im_path], stride=gs, multi_modal=True)
        if isinstance(im_path, str)
        else build_grounding(
            # assign `nc` from validation set to max number of text samples for training consistency
            self.args,
            im_path["img_path"],
            im_path["json_file"],
            batch,
            stride=gs,
            max_samples=self.data["nc"],
        )
        for im_path in img_path
    ]
    self.set_text_embeddings(datasets, batch)  # cache text embeddings to accelerate training
    return YOLOConcatDataset(datasets) if len(datasets) > 1 else datasets[0]
```
</details>

<br>

### Method `ultralytics.models.yolo.world.train_world.WorldTrainerFromScratch.check_data_config` {#ultralytics.models.yolo.world.train\_world.WorldTrainerFromScratch.check\_data\_config}

```python
def check_data_config(data: dict | str | Path) -> dict
```

Check and load the data configuration from a YAML file or dictionary.

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `data` | `dict | str | Path` | Data configuration as a dictionary or path to a YAML file. | *required* |

**Returns**

| Type | Description |
| --- | --- |
| `dict` | Data configuration dictionary loaded from YAML file or passed directly. |

<details>
<summary>Source code in <code>ultralytics/models/yolo/world/train_world.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/models/yolo/world/train_world.py#L107-L121"><i class="fa-brands fa-github" aria-hidden="true" style="margin-right:6px;"></i>View on GitHub</a>
```python
@staticmethod
def check_data_config(data: dict | str | Path) -> dict:
    """Check and load the data configuration from a YAML file or dictionary.

    Args:
        data (dict | str | Path): Data configuration as a dictionary or path to a YAML file.

    Returns:
        (dict): Data configuration dictionary loaded from YAML file or passed directly.
    """
    # If string, load from YAML file
    if not isinstance(data, dict):
        from ultralytics.utils import YAML

        return YAML.load(check_file(data))
    return data
```
</details>

<br>

### Method `ultralytics.models.yolo.world.train_world.WorldTrainerFromScratch.final_eval` {#ultralytics.models.yolo.world.train\_world.WorldTrainerFromScratch.final\_eval}

```python
def final_eval(self)
```

Perform final evaluation and validation for the YOLO-World model.

Configures the validator with appropriate dataset and split information before running evaluation.

**Returns**

| Type | Description |
| --- | --- |
| `dict` | Dictionary containing evaluation metrics and results. |

<details>
<summary>Source code in <code>ultralytics/models/yolo/world/train_world.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/models/yolo/world/train_world.py#L186-L197"><i class="fa-brands fa-github" aria-hidden="true" style="margin-right:6px;"></i>View on GitHub</a>
```python
def final_eval(self):
    """Perform final evaluation and validation for the YOLO-World model.

    Configures the validator with appropriate dataset and split information before running evaluation.

    Returns:
        (dict): Dictionary containing evaluation metrics and results.
    """
    val = self.args.data["val"]["yolo_data"][0]
    self.validator.args.data = val
    self.validator.args.split = "minival" if isinstance(val, str) and "lvis" in val else "val"
    return super().final_eval()
```
</details>

<br>

### Method `ultralytics.models.yolo.world.train_world.WorldTrainerFromScratch.get_dataset` {#ultralytics.models.yolo.world.train\_world.WorldTrainerFromScratch.get\_dataset}

```python
def get_dataset(self)
```

Get train and validation paths from data dictionary.

Processes the data configuration to extract paths for training and validation datasets, handling both YOLO detection datasets and grounding datasets.

**Returns**

| Type | Description |
| --- | --- |
| `dict` | Final processed data configuration containing train/val paths and metadata. |

**Raises**

| Type | Description |
| --- | --- |
| `AssertionError` | If train or validation datasets are not found, or if validation has multiple datasets. |

<details>
<summary>Source code in <code>ultralytics/models/yolo/world/train_world.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/models/yolo/world/train_world.py#L123-L180"><i class="fa-brands fa-github" aria-hidden="true" style="margin-right:6px;"></i>View on GitHub</a>
```python
def get_dataset(self):
    """Get train and validation paths from data dictionary.

    Processes the data configuration to extract paths for training and validation datasets, handling both YOLO
    detection datasets and grounding datasets.

    Returns:
        (dict): Final processed data configuration containing train/val paths and metadata.

    Raises:
        AssertionError: If train or validation datasets are not found, or if validation has multiple datasets.
    """
    final_data = {}
    self.args.data = data_yaml = self.check_data_config(self.args.data)
    assert data_yaml.get("train", False), "train dataset not found"  # object365.yaml
    assert data_yaml.get("val", False), "validation dataset not found"  # lvis.yaml
    data = {k: [check_det_dataset(d) for d in v.get("yolo_data", [])] for k, v in data_yaml.items()}
    assert len(data["val"]) == 1, f"Only support validating on 1 dataset for now, but got {len(data['val'])}."
    val_split = "minival" if "lvis" in data["val"][0]["val"] else "val"
    for d in data["val"]:
        if d.get("minival") is None:  # for lvis dataset
            continue
        d["minival"] = str(d["path"] / d["minival"])
    for s in {"train", "val"}:
        final_data[s] = [d["train" if s == "train" else val_split] for d in data[s]]
        # save grounding data if there's one
        grounding_data = data_yaml[s].get("grounding_data")
        if grounding_data is None:
            continue
        grounding_data = grounding_data if isinstance(grounding_data, list) else [grounding_data]
        for g in grounding_data:
            assert isinstance(g, dict), f"Grounding data should be provided in dict format, but got {type(g)}"
            for k in {"img_path", "json_file"}:
                path = Path(g[k])
                if not path.exists() and not path.is_absolute():
                    g[k] = str((DATASETS_DIR / g[k]).resolve())  # path relative to DATASETS_DIR
        final_data[s] += grounding_data
    # assign the first val dataset as currently only one validation set is supported
    data["val"] = data["val"][0]
    final_data["val"] = final_data["val"][0]
    # NOTE: to make training work properly, set `nc` and `names`
    final_data["nc"] = data["val"]["nc"]
    final_data["names"] = data["val"]["names"]
    # NOTE: add path with lvis path
    final_data["path"] = data["val"]["path"]
    final_data["channels"] = data["val"]["channels"]
    self.data = final_data
    if self.args.single_cls:  # consistent with base trainer
        LOGGER.info("Overriding class names with single class.")
        self.data["names"] = {0: "object"}
        self.data["nc"] = 1
    self.training_data = {}
    for d in data["train"]:
        if self.args.single_cls:
            d["names"] = {0: "object"}
            d["nc"] = 1
        self.training_data[d["train"]] = d
    return final_data
```
</details>

<br>

### Method `ultralytics.models.yolo.world.train_world.WorldTrainerFromScratch.plot_training_labels` {#ultralytics.models.yolo.world.train\_world.WorldTrainerFromScratch.plot\_training\_labels}

```python
def plot_training_labels(self)
```

Skip label plotting for YOLO-World training.

<details>
<summary>Source code in <code>ultralytics/models/yolo/world/train_world.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/models/yolo/world/train_world.py#L182-L184"><i class="fa-brands fa-github" aria-hidden="true" style="margin-right:6px;"></i>View on GitHub</a>
```python
def plot_training_labels(self):
    """Skip label plotting for YOLO-World training."""
    pass
```
</details>

<br><br>
