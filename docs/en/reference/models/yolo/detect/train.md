---
description: Learn about the DetectionTrainer class for training YOLO models on custom datasets. Discover methods, examples, and more.
keywords: Ultralytics, YOLO, DetectionTrainer, training, object detection, machine learning, build dataset, dataloader, detection model
---

# Reference for `ultralytics/models/yolo/detect/train.py`

!!! success "Improvements"

    This page is sourced from [https://github.com/ultralytics/ultralytics/blob/main/ultralytics/models/yolo/detect/train.py](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/models/yolo/detect/train.py). Have an improvement or example to add? Open a [Pull Request](https://docs.ultralytics.com/help/contributing/) — thank you! 🙏

<br>

!!! abstract "Summary"

    === "<span class="doc-kind doc-kind-class">Classes</span>"

        - [`DetectionTrainer`](#ultralytics.models.yolo.detect.train.DetectionTrainer)

    === "<span class="doc-kind doc-kind-method">Methods</span>"

        - [`DetectionTrainer.build_dataset`](#ultralytics.models.yolo.detect.train.DetectionTrainer.build_dataset)
        - [`DetectionTrainer.get_dataloader`](#ultralytics.models.yolo.detect.train.DetectionTrainer.get_dataloader)
        - [`DetectionTrainer.preprocess_batch`](#ultralytics.models.yolo.detect.train.DetectionTrainer.preprocess_batch)
        - [`DetectionTrainer.set_model_attributes`](#ultralytics.models.yolo.detect.train.DetectionTrainer.set_model_attributes)
        - [`DetectionTrainer.get_model`](#ultralytics.models.yolo.detect.train.DetectionTrainer.get_model)
        - [`DetectionTrainer.get_validator`](#ultralytics.models.yolo.detect.train.DetectionTrainer.get_validator)
        - [`DetectionTrainer.label_loss_items`](#ultralytics.models.yolo.detect.train.DetectionTrainer.label_loss_items)
        - [`DetectionTrainer.progress_string`](#ultralytics.models.yolo.detect.train.DetectionTrainer.progress_string)
        - [`DetectionTrainer.plot_training_samples`](#ultralytics.models.yolo.detect.train.DetectionTrainer.plot_training_samples)
        - [`DetectionTrainer.plot_training_labels`](#ultralytics.models.yolo.detect.train.DetectionTrainer.plot_training_labels)
        - [`DetectionTrainer.auto_batch`](#ultralytics.models.yolo.detect.train.DetectionTrainer.auto_batch)


## Class `ultralytics.models.yolo.detect.train.DetectionTrainer` {#ultralytics.models.yolo.detect.train.DetectionTrainer}

```python
DetectionTrainer(self, cfg = DEFAULT_CFG, overrides: dict[str, Any] | None = None, _callbacks = None)
```

**Bases:** `BaseTrainer`

A class extending the BaseTrainer class for training based on a detection model.

This trainer specializes in object detection tasks, handling the specific requirements for training YOLO models for object detection including dataset building, data loading, preprocessing, and model configuration.

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `cfg` | `dict, optional` | Default configuration dictionary containing training parameters. | `DEFAULT_CFG` |
| `overrides` | `dict, optional` | Dictionary of parameter overrides for the default configuration. | `None` |
| `_callbacks` | `list, optional` | List of callback functions to be executed during training. | `None` |

**Attributes**

| Name | Type | Description |
| --- | --- | --- |
| `model` | `DetectionModel` | The YOLO detection model being trained. |
| `data` | `dict` | Dictionary containing dataset information including class names and number of classes. |
| `loss_names` | `tuple` | Names of the loss components used in training (box_loss, cls_loss, dfl_loss). |

**Methods**

| Name | Description |
| --- | --- |
| [`auto_batch`](#ultralytics.models.yolo.detect.train.DetectionTrainer.auto_batch) | Get optimal batch size by calculating memory occupation of model. |
| [`build_dataset`](#ultralytics.models.yolo.detect.train.DetectionTrainer.build_dataset) | Build YOLO Dataset for training or validation. |
| [`get_dataloader`](#ultralytics.models.yolo.detect.train.DetectionTrainer.get_dataloader) | Construct and return dataloader for the specified mode. |
| [`get_model`](#ultralytics.models.yolo.detect.train.DetectionTrainer.get_model) | Return a YOLO detection model. |
| [`get_validator`](#ultralytics.models.yolo.detect.train.DetectionTrainer.get_validator) | Return a DetectionValidator for YOLO model validation. |
| [`label_loss_items`](#ultralytics.models.yolo.detect.train.DetectionTrainer.label_loss_items) | Return a loss dict with labeled training loss items tensor. |
| [`plot_training_labels`](#ultralytics.models.yolo.detect.train.DetectionTrainer.plot_training_labels) | Create a labeled training plot of the YOLO model. |
| [`plot_training_samples`](#ultralytics.models.yolo.detect.train.DetectionTrainer.plot_training_samples) | Plot training samples with their annotations. |
| [`preprocess_batch`](#ultralytics.models.yolo.detect.train.DetectionTrainer.preprocess_batch) | Preprocess a batch of images by scaling and converting to float. |
| [`progress_string`](#ultralytics.models.yolo.detect.train.DetectionTrainer.progress_string) | Return a formatted string of training progress with epoch, GPU memory, loss, instances and size. |
| [`set_model_attributes`](#ultralytics.models.yolo.detect.train.DetectionTrainer.set_model_attributes) | Set model attributes based on dataset information. |

**Examples**

```python
>>> from ultralytics.models.yolo.detect import DetectionTrainer
>>> args = dict(model="yolo26n.pt", data="coco8.yaml", epochs=3)
>>> trainer = DetectionTrainer(overrides=args)
>>> trainer.train()
```

<details>
<summary>Source code in <code>ultralytics/models/yolo/detect/train.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/models/yolo/detect/train.py#L24-L232"><i class="fa-brands fa-github" aria-hidden="true" style="margin-right:6px;"></i>View on GitHub</a>
```python
class DetectionTrainer(BaseTrainer):
    """A class extending the BaseTrainer class for training based on a detection model.

    This trainer specializes in object detection tasks, handling the specific requirements for training YOLO models for
    object detection including dataset building, data loading, preprocessing, and model configuration.

    Attributes:
        model (DetectionModel): The YOLO detection model being trained.
        data (dict): Dictionary containing dataset information including class names and number of classes.
        loss_names (tuple): Names of the loss components used in training (box_loss, cls_loss, dfl_loss).

    Methods:
        build_dataset: Build YOLO dataset for training or validation.
        get_dataloader: Construct and return dataloader for the specified mode.
        preprocess_batch: Preprocess a batch of images by scaling and converting to float.
        set_model_attributes: Set model attributes based on dataset information.
        get_model: Return a YOLO detection model.
        get_validator: Return a validator for model evaluation.
        label_loss_items: Return a loss dictionary with labeled training loss items.
        progress_string: Return a formatted string of training progress.
        plot_training_samples: Plot training samples with their annotations.
        plot_training_labels: Create a labeled training plot of the YOLO model.
        auto_batch: Calculate optimal batch size based on model memory requirements.

    Examples:
        >>> from ultralytics.models.yolo.detect import DetectionTrainer
        >>> args = dict(model="yolo26n.pt", data="coco8.yaml", epochs=3)
        >>> trainer = DetectionTrainer(overrides=args)
        >>> trainer.train()
    """

    def __init__(self, cfg=DEFAULT_CFG, overrides: dict[str, Any] | None = None, _callbacks=None):
        """Initialize a DetectionTrainer object for training YOLO object detection models.

        Args:
            cfg (dict, optional): Default configuration dictionary containing training parameters.
            overrides (dict, optional): Dictionary of parameter overrides for the default configuration.
            _callbacks (list, optional): List of callback functions to be executed during training.
        """
        super().__init__(cfg, overrides, _callbacks)
```
</details>

<br>

### Method `ultralytics.models.yolo.detect.train.DetectionTrainer.auto_batch` {#ultralytics.models.yolo.detect.train.DetectionTrainer.auto\_batch}

```python
def auto_batch(self)
```

Get optimal batch size by calculating memory occupation of model.

**Returns**

| Type | Description |
| --- | --- |
| `int` | Optimal batch size. |

<details>
<summary>Source code in <code>ultralytics/models/yolo/detect/train.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/models/yolo/detect/train.py#L222-L232"><i class="fa-brands fa-github" aria-hidden="true" style="margin-right:6px;"></i>View on GitHub</a>
```python
def auto_batch(self):
    """Get optimal batch size by calculating memory occupation of model.

    Returns:
        (int): Optimal batch size.
    """
    with override_configs(self.args, overrides={"cache": False}) as self.args:
        train_dataset = self.build_dataset(self.data["train"], mode="train", batch=16)
    max_num_obj = max(len(label["cls"]) for label in train_dataset.labels) * 4  # 4 for mosaic augmentation
    del train_dataset  # free memory
    return super().auto_batch(max_num_obj)
```
</details>

<br>

### Method `ultralytics.models.yolo.detect.train.DetectionTrainer.build_dataset` {#ultralytics.models.yolo.detect.train.DetectionTrainer.build\_dataset}

```python
def build_dataset(self, img_path: str, mode: str = "train", batch: int | None = None)
```

Build YOLO Dataset for training or validation.

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `img_path` | `str` | Path to the folder containing images. | *required* |
| `mode` | `str` | 'train' mode or 'val' mode, users are able to customize different augmentations for each mode. | `"train"` |
| `batch` | `int, optional` | Size of batches, this is for 'rect' mode. | `None` |

**Returns**

| Type | Description |
| --- | --- |
| `Dataset` | YOLO dataset object configured for the specified mode. |

<details>
<summary>Source code in <code>ultralytics/models/yolo/detect/train.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/models/yolo/detect/train.py#L65-L77"><i class="fa-brands fa-github" aria-hidden="true" style="margin-right:6px;"></i>View on GitHub</a>
```python
def build_dataset(self, img_path: str, mode: str = "train", batch: int | None = None):
    """Build YOLO Dataset for training or validation.

    Args:
        img_path (str): Path to the folder containing images.
        mode (str): 'train' mode or 'val' mode, users are able to customize different augmentations for each mode.
        batch (int, optional): Size of batches, this is for 'rect' mode.

    Returns:
        (Dataset): YOLO dataset object configured for the specified mode.
    """
    gs = max(int(unwrap_model(self.model).stride.max()), 32)
    return build_yolo_dataset(self.args, img_path, batch, self.data, mode=mode, rect=mode == "val", stride=gs)
```
</details>

<br>

### Method `ultralytics.models.yolo.detect.train.DetectionTrainer.get_dataloader` {#ultralytics.models.yolo.detect.train.DetectionTrainer.get\_dataloader}

```python
def get_dataloader(self, dataset_path: str, batch_size: int = 16, rank: int = 0, mode: str = "train")
```

Construct and return dataloader for the specified mode.

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `dataset_path` | `str` | Path to the dataset. | *required* |
| `batch_size` | `int` | Number of images per batch. | `16` |
| `rank` | `int` | Process rank for distributed training. | `0` |
| `mode` | `str` | 'train' for training dataloader, 'val' for validation dataloader. | `"train"` |

**Returns**

| Type | Description |
| --- | --- |
| `DataLoader` | PyTorch dataloader object. |

<details>
<summary>Source code in <code>ultralytics/models/yolo/detect/train.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/models/yolo/detect/train.py#L79-L105"><i class="fa-brands fa-github" aria-hidden="true" style="margin-right:6px;"></i>View on GitHub</a>
```python
def get_dataloader(self, dataset_path: str, batch_size: int = 16, rank: int = 0, mode: str = "train"):
    """Construct and return dataloader for the specified mode.

    Args:
        dataset_path (str): Path to the dataset.
        batch_size (int): Number of images per batch.
        rank (int): Process rank for distributed training.
        mode (str): 'train' for training dataloader, 'val' for validation dataloader.

    Returns:
        (DataLoader): PyTorch dataloader object.
    """
    assert mode in {"train", "val"}, f"Mode must be 'train' or 'val', not {mode}."
    with torch_distributed_zero_first(rank):  # init dataset *.cache only once if DDP
        dataset = self.build_dataset(dataset_path, mode, batch_size)
    shuffle = mode == "train"
    if getattr(dataset, "rect", False) and shuffle and not np.all(dataset.batch_shapes == dataset.batch_shapes[0]):
        LOGGER.warning("'rect=True' is incompatible with DataLoader shuffle, setting shuffle=False")
        shuffle = False
    return build_dataloader(
        dataset,
        batch=batch_size,
        workers=self.args.workers if mode == "train" else self.args.workers * 2,
        shuffle=shuffle,
        rank=rank,
        drop_last=self.args.compile and mode == "train",
    )
```
</details>

<br>

### Method `ultralytics.models.yolo.detect.train.DetectionTrainer.get_model` {#ultralytics.models.yolo.detect.train.DetectionTrainer.get\_model}

```python
def get_model(self, cfg: str | None = None, weights: str | None = None, verbose: bool = True)
```

Return a YOLO detection model.

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `cfg` | `str, optional` | Path to model configuration file. | `None` |
| `weights` | `str, optional` | Path to model weights. | `None` |
| `verbose` | `bool` | Whether to display model information. | `True` |

**Returns**

| Type | Description |
| --- | --- |
| `DetectionModel` | YOLO detection model. |

<details>
<summary>Source code in <code>ultralytics/models/yolo/detect/train.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/models/yolo/detect/train.py#L152-L166"><i class="fa-brands fa-github" aria-hidden="true" style="margin-right:6px;"></i>View on GitHub</a>
```python
def get_model(self, cfg: str | None = None, weights: str | None = None, verbose: bool = True):
    """Return a YOLO detection model.

    Args:
        cfg (str, optional): Path to model configuration file.
        weights (str, optional): Path to model weights.
        verbose (bool): Whether to display model information.

    Returns:
        (DetectionModel): YOLO detection model.
    """
    model = DetectionModel(cfg, nc=self.data["nc"], ch=self.data["channels"], verbose=verbose and RANK == -1)
    if weights:
        model.load(weights)
    return model
```
</details>

<br>

### Method `ultralytics.models.yolo.detect.train.DetectionTrainer.get_validator` {#ultralytics.models.yolo.detect.train.DetectionTrainer.get\_validator}

```python
def get_validator(self)
```

Return a DetectionValidator for YOLO model validation.

<details>
<summary>Source code in <code>ultralytics/models/yolo/detect/train.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/models/yolo/detect/train.py#L168-L173"><i class="fa-brands fa-github" aria-hidden="true" style="margin-right:6px;"></i>View on GitHub</a>
```python
def get_validator(self):
    """Return a DetectionValidator for YOLO model validation."""
    self.loss_names = "box_loss", "cls_loss", "dfl_loss"
    return yolo.detect.DetectionValidator(
        self.test_loader, save_dir=self.save_dir, args=copy(self.args), _callbacks=self.callbacks
    )
```
</details>

<br>

### Method `ultralytics.models.yolo.detect.train.DetectionTrainer.label_loss_items` {#ultralytics.models.yolo.detect.train.DetectionTrainer.label\_loss\_items}

```python
def label_loss_items(self, loss_items: list[float] | None = None, prefix: str = "train")
```

Return a loss dict with labeled training loss items tensor.

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `loss_items` | `list[float], optional` | List of loss values. | `None` |
| `prefix` | `str` | Prefix for keys in the returned dictionary. | `"train"` |

**Returns**

| Type | Description |
| --- | --- |
| `dict | list` | Dictionary of labeled loss items if loss_items is provided, otherwise list of keys. |

<details>
<summary>Source code in <code>ultralytics/models/yolo/detect/train.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/models/yolo/detect/train.py#L175-L190"><i class="fa-brands fa-github" aria-hidden="true" style="margin-right:6px;"></i>View on GitHub</a>
```python
def label_loss_items(self, loss_items: list[float] | None = None, prefix: str = "train"):
    """Return a loss dict with labeled training loss items tensor.

    Args:
        loss_items (list[float], optional): List of loss values.
        prefix (str): Prefix for keys in the returned dictionary.

    Returns:
        (dict | list): Dictionary of labeled loss items if loss_items is provided, otherwise list of keys.
    """
    keys = [f"{prefix}/{x}" for x in self.loss_names]
    if loss_items is not None:
        loss_items = [round(float(x), 5) for x in loss_items]  # convert tensors to 5 decimal place floats
        return dict(zip(keys, loss_items))
    else:
        return keys
```
</details>

<br>

### Method `ultralytics.models.yolo.detect.train.DetectionTrainer.plot_training_labels` {#ultralytics.models.yolo.detect.train.DetectionTrainer.plot\_training\_labels}

```python
def plot_training_labels(self)
```

Create a labeled training plot of the YOLO model.

<details>
<summary>Source code in <code>ultralytics/models/yolo/detect/train.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/models/yolo/detect/train.py#L216-L220"><i class="fa-brands fa-github" aria-hidden="true" style="margin-right:6px;"></i>View on GitHub</a>
```python
def plot_training_labels(self):
    """Create a labeled training plot of the YOLO model."""
    boxes = np.concatenate([lb["bboxes"] for lb in self.train_loader.dataset.labels], 0)
    cls = np.concatenate([lb["cls"] for lb in self.train_loader.dataset.labels], 0)
    plot_labels(boxes, cls.squeeze(), names=self.data["names"], save_dir=self.save_dir, on_plot=self.on_plot)
```
</details>

<br>

### Method `ultralytics.models.yolo.detect.train.DetectionTrainer.plot_training_samples` {#ultralytics.models.yolo.detect.train.DetectionTrainer.plot\_training\_samples}

```python
def plot_training_samples(self, batch: dict[str, Any], ni: int) -> None
```

Plot training samples with their annotations.

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `batch` | `dict[str, Any]` | Dictionary containing batch data. | *required* |
| `ni` | `int` | Batch index used for naming the output file. | *required* |

<details>
<summary>Source code in <code>ultralytics/models/yolo/detect/train.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/models/yolo/detect/train.py#L202-L214"><i class="fa-brands fa-github" aria-hidden="true" style="margin-right:6px;"></i>View on GitHub</a>
```python
def plot_training_samples(self, batch: dict[str, Any], ni: int) -> None:
    """Plot training samples with their annotations.

    Args:
        batch (dict[str, Any]): Dictionary containing batch data.
        ni (int): Batch index used for naming the output file.
    """
    plot_images(
        labels=batch,
        paths=batch["im_file"],
        fname=self.save_dir / f"train_batch{ni}.jpg",
        on_plot=self.on_plot,
    )
```
</details>

<br>

### Method `ultralytics.models.yolo.detect.train.DetectionTrainer.preprocess_batch` {#ultralytics.models.yolo.detect.train.DetectionTrainer.preprocess\_batch}

```python
def preprocess_batch(self, batch: dict) -> dict
```

Preprocess a batch of images by scaling and converting to float.

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `batch` | `dict` | Dictionary containing batch data with 'img' tensor. | *required* |

**Returns**

| Type | Description |
| --- | --- |
| `dict` | Preprocessed batch with normalized images. |

<details>
<summary>Source code in <code>ultralytics/models/yolo/detect/train.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/models/yolo/detect/train.py#L107-L137"><i class="fa-brands fa-github" aria-hidden="true" style="margin-right:6px;"></i>View on GitHub</a>
```python
def preprocess_batch(self, batch: dict) -> dict:
    """Preprocess a batch of images by scaling and converting to float.

    Args:
        batch (dict): Dictionary containing batch data with 'img' tensor.

    Returns:
        (dict): Preprocessed batch with normalized images.
    """
    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            batch[k] = v.to(self.device, non_blocking=self.device.type == "cuda")
    batch["img"] = batch["img"].float() / 255
    if self.args.multi_scale > 0.0:
        imgs = batch["img"]
        sz = (
            random.randrange(
                int(self.args.imgsz * (1.0 - self.args.multi_scale)),
                int(self.args.imgsz * (1.0 + self.args.multi_scale) + self.stride),
            )
            // self.stride
            * self.stride
        )  # size
        sf = sz / max(imgs.shape[2:])  # scale factor
        if sf != 1:
            ns = [
                math.ceil(x * sf / self.stride) * self.stride for x in imgs.shape[2:]
            ]  # new shape (stretched to gs-multiple)
            imgs = nn.functional.interpolate(imgs, size=ns, mode="bilinear", align_corners=False)
        batch["img"] = imgs
    return batch
```
</details>

<br>

### Method `ultralytics.models.yolo.detect.train.DetectionTrainer.progress_string` {#ultralytics.models.yolo.detect.train.DetectionTrainer.progress\_string}

```python
def progress_string(self)
```

Return a formatted string of training progress with epoch, GPU memory, loss, instances and size.

<details>
<summary>Source code in <code>ultralytics/models/yolo/detect/train.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/models/yolo/detect/train.py#L192-L200"><i class="fa-brands fa-github" aria-hidden="true" style="margin-right:6px;"></i>View on GitHub</a>
```python
def progress_string(self):
    """Return a formatted string of training progress with epoch, GPU memory, loss, instances and size."""
    return ("\n" + "%11s" * (4 + len(self.loss_names))) % (
        "Epoch",
        "GPU_mem",
        *self.loss_names,
        "Instances",
        "Size",
    )
```
</details>

<br>

### Method `ultralytics.models.yolo.detect.train.DetectionTrainer.set_model_attributes` {#ultralytics.models.yolo.detect.train.DetectionTrainer.set\_model\_attributes}

```python
def set_model_attributes(self)
```

Set model attributes based on dataset information.

<details>
<summary>Source code in <code>ultralytics/models/yolo/detect/train.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/models/yolo/detect/train.py#L139-L149"><i class="fa-brands fa-github" aria-hidden="true" style="margin-right:6px;"></i>View on GitHub</a>
```python
def set_model_attributes(self):
    """Set model attributes based on dataset information."""
    # Nl = de_parallel(self.model).model[-1].nl  # number of detection layers (to scale hyps)
    # self.args.box *= 3 / nl  # scale to layers
    # self.args.cls *= self.data["nc"] / 80 * 3 / nl  # scale to classes and layers
    # self.args.cls *= (self.args.imgsz / 640) ** 2 * 3 / nl  # scale to image size and layers
    self.model.nc = self.data["nc"]  # attach number of classes to model
    self.model.names = self.data["names"]  # attach class names to model
    self.model.args = self.args  # attach hyperparameters to model
    if getattr(self.model, "end2end"):
        self.model.set_head_attr(max_det=self.args.max_det)
```
</details>

<br><br>
