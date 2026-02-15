---
description: Explore the train.py module in Ultralytics YOLO for efficient classification model training. Learn more with examples and detailed code documentation.
keywords: YOLO, Ultralytics, classification, training, machine learning, deep learning, PyTorch, train.py
---

# Reference for `ultralytics/models/yolo/classify/train.py`

!!! success "Improvements"

    This page is sourced from [https://github.com/ultralytics/ultralytics/blob/main/ultralytics/models/yolo/classify/train.py](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/models/yolo/classify/train.py). Have an improvement or example to add? Open a [Pull Request](https://docs.ultralytics.com/help/contributing/) — thank you! 🙏

<br>

!!! abstract "Summary"

    === "<span class="doc-kind doc-kind-class">Classes</span>"

        - [`ClassificationTrainer`](#ultralytics.models.yolo.classify.train.ClassificationTrainer)

    === "<span class="doc-kind doc-kind-method">Methods</span>"

        - [`ClassificationTrainer.set_model_attributes`](#ultralytics.models.yolo.classify.train.ClassificationTrainer.set_model_attributes)
        - [`ClassificationTrainer.get_model`](#ultralytics.models.yolo.classify.train.ClassificationTrainer.get_model)
        - [`ClassificationTrainer.setup_model`](#ultralytics.models.yolo.classify.train.ClassificationTrainer.setup_model)
        - [`ClassificationTrainer.build_dataset`](#ultralytics.models.yolo.classify.train.ClassificationTrainer.build_dataset)
        - [`ClassificationTrainer.get_dataloader`](#ultralytics.models.yolo.classify.train.ClassificationTrainer.get_dataloader)
        - [`ClassificationTrainer.preprocess_batch`](#ultralytics.models.yolo.classify.train.ClassificationTrainer.preprocess_batch)
        - [`ClassificationTrainer.progress_string`](#ultralytics.models.yolo.classify.train.ClassificationTrainer.progress_string)
        - [`ClassificationTrainer.get_validator`](#ultralytics.models.yolo.classify.train.ClassificationTrainer.get_validator)
        - [`ClassificationTrainer.label_loss_items`](#ultralytics.models.yolo.classify.train.ClassificationTrainer.label_loss_items)
        - [`ClassificationTrainer.plot_training_samples`](#ultralytics.models.yolo.classify.train.ClassificationTrainer.plot_training_samples)


## Class `ultralytics.models.yolo.classify.train.ClassificationTrainer` {#ultralytics.models.yolo.classify.train.ClassificationTrainer}

```python
ClassificationTrainer(self, cfg = DEFAULT_CFG, overrides: dict[str, Any] | None = None, _callbacks = None)
```

**Bases:** `BaseTrainer`

A trainer class extending BaseTrainer for training image classification models.

This trainer handles the training process for image classification tasks, supporting both YOLO classification models and torchvision models with comprehensive dataset handling and validation.

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `cfg` | `dict[str, Any], optional` | Default configuration dictionary containing training parameters. | `DEFAULT_CFG` |
| `overrides` | `dict[str, Any], optional` | Dictionary of parameter overrides for the default configuration. | `None` |
| `_callbacks` | `list[Any], optional` | List of callback functions to be executed during training. | `None` |

**Attributes**

| Name | Type | Description |
| --- | --- | --- |
| `model` | `ClassificationModel` | The classification model to be trained. |
| `data` | `dict[str, Any]` | Dictionary containing dataset information including class names and number of classes. |
| `loss_names` | `list[str]` | Names of the loss functions used during training. |
| `validator` | `ClassificationValidator` | Validator instance for model evaluation. |

**Methods**

| Name | Description |
| --- | --- |
| [`build_dataset`](#ultralytics.models.yolo.classify.train.ClassificationTrainer.build_dataset) | Create a ClassificationDataset instance given an image path and mode. |
| [`get_dataloader`](#ultralytics.models.yolo.classify.train.ClassificationTrainer.get_dataloader) | Return PyTorch DataLoader with transforms to preprocess images. |
| [`get_model`](#ultralytics.models.yolo.classify.train.ClassificationTrainer.get_model) | Return a modified PyTorch model configured for training YOLO classification. |
| [`get_validator`](#ultralytics.models.yolo.classify.train.ClassificationTrainer.get_validator) | Return an instance of ClassificationValidator for validation. |
| [`label_loss_items`](#ultralytics.models.yolo.classify.train.ClassificationTrainer.label_loss_items) | Return a loss dict with labeled training loss items tensor. |
| [`plot_training_samples`](#ultralytics.models.yolo.classify.train.ClassificationTrainer.plot_training_samples) | Plot training samples with their annotations. |
| [`preprocess_batch`](#ultralytics.models.yolo.classify.train.ClassificationTrainer.preprocess_batch) | Preprocess a batch of images and classes. |
| [`progress_string`](#ultralytics.models.yolo.classify.train.ClassificationTrainer.progress_string) | Return a formatted string showing training progress. |
| [`set_model_attributes`](#ultralytics.models.yolo.classify.train.ClassificationTrainer.set_model_attributes) | Set the YOLO model's class names from the loaded dataset. |
| [`setup_model`](#ultralytics.models.yolo.classify.train.ClassificationTrainer.setup_model) | Load, create or download model for classification tasks. |

**Examples**

```python
Initialize and train a classification model
>>> from ultralytics.models.yolo.classify import ClassificationTrainer
>>> args = dict(model="yolo26n-cls.pt", data="imagenet10", epochs=3)
>>> trainer = ClassificationTrainer(overrides=args)
>>> trainer.train()
```

<details>
<summary>Source code in <code>ultralytics/models/yolo/classify/train.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/models/yolo/classify/train.py#L19-L214"><i class="fa-brands fa-github" aria-hidden="true" style="margin-right:6px;"></i>View on GitHub</a>
```python
class ClassificationTrainer(BaseTrainer):
    """A trainer class extending BaseTrainer for training image classification models.

    This trainer handles the training process for image classification tasks, supporting both YOLO classification models
    and torchvision models with comprehensive dataset handling and validation.

    Attributes:
        model (ClassificationModel): The classification model to be trained.
        data (dict[str, Any]): Dictionary containing dataset information including class names and number of classes.
        loss_names (list[str]): Names of the loss functions used during training.
        validator (ClassificationValidator): Validator instance for model evaluation.

    Methods:
        set_model_attributes: Set the model's class names from the loaded dataset.
        get_model: Return a modified PyTorch model configured for training.
        setup_model: Load, create or download model for classification.
        build_dataset: Create a ClassificationDataset instance.
        get_dataloader: Return PyTorch DataLoader with transforms for image preprocessing.
        preprocess_batch: Preprocess a batch of images and classes.
        progress_string: Return a formatted string showing training progress.
        get_validator: Return an instance of ClassificationValidator.
        label_loss_items: Return a loss dict with labeled training loss items.
        final_eval: Evaluate trained model and save validation results.
        plot_training_samples: Plot training samples with their annotations.

    Examples:
        Initialize and train a classification model
        >>> from ultralytics.models.yolo.classify import ClassificationTrainer
        >>> args = dict(model="yolo26n-cls.pt", data="imagenet10", epochs=3)
        >>> trainer = ClassificationTrainer(overrides=args)
        >>> trainer.train()
    """

    def __init__(self, cfg=DEFAULT_CFG, overrides: dict[str, Any] | None = None, _callbacks=None):
        """Initialize a ClassificationTrainer object.

        Args:
            cfg (dict[str, Any], optional): Default configuration dictionary containing training parameters.
            overrides (dict[str, Any], optional): Dictionary of parameter overrides for the default configuration.
            _callbacks (list[Any], optional): List of callback functions to be executed during training.
        """
        if overrides is None:
            overrides = {}
        overrides["task"] = "classify"
        if overrides.get("imgsz") is None:
            overrides["imgsz"] = 224
        super().__init__(cfg, overrides, _callbacks)
```
</details>

<br>

### Method `ultralytics.models.yolo.classify.train.ClassificationTrainer.build_dataset` {#ultralytics.models.yolo.classify.train.ClassificationTrainer.build\_dataset}

```python
def build_dataset(self, img_path: str, mode: str = "train", batch = None)
```

Create a ClassificationDataset instance given an image path and mode.

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `img_path` | `str` | Path to the dataset images. | *required* |
| `mode` | `str, optional` | Dataset mode ('train', 'val', or 'test'). | `"train"` |
| `batch` | `Any, optional` | Batch information (unused in this implementation). | `None` |

**Returns**

| Type | Description |
| --- | --- |
| `ClassificationDataset` | Dataset for the specified mode. |

<details>
<summary>Source code in <code>ultralytics/models/yolo/classify/train.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/models/yolo/classify/train.py#L113-L124"><i class="fa-brands fa-github" aria-hidden="true" style="margin-right:6px;"></i>View on GitHub</a>
```python
def build_dataset(self, img_path: str, mode: str = "train", batch=None):
    """Create a ClassificationDataset instance given an image path and mode.

    Args:
        img_path (str): Path to the dataset images.
        mode (str, optional): Dataset mode ('train', 'val', or 'test').
        batch (Any, optional): Batch information (unused in this implementation).

    Returns:
        (ClassificationDataset): Dataset for the specified mode.
    """
    return ClassificationDataset(root=img_path, args=self.args, augment=mode == "train", prefix=mode)
```
</details>

<br>

### Method `ultralytics.models.yolo.classify.train.ClassificationTrainer.get_dataloader` {#ultralytics.models.yolo.classify.train.ClassificationTrainer.get\_dataloader}

```python
def get_dataloader(self, dataset_path: str, batch_size: int = 16, rank: int = 0, mode: str = "train")
```

Return PyTorch DataLoader with transforms to preprocess images.

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `dataset_path` | `str` | Path to the dataset. | *required* |
| `batch_size` | `int, optional` | Number of images per batch. | `16` |
| `rank` | `int, optional` | Process rank for distributed training. | `0` |
| `mode` | `str, optional` | 'train', 'val', or 'test' mode. | `"train"` |

**Returns**

| Type | Description |
| --- | --- |
| `torch.utils.data.DataLoader` | DataLoader for the specified dataset and mode. |

<details>
<summary>Source code in <code>ultralytics/models/yolo/classify/train.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/models/yolo/classify/train.py#L126-L161"><i class="fa-brands fa-github" aria-hidden="true" style="margin-right:6px;"></i>View on GitHub</a>
```python
def get_dataloader(self, dataset_path: str, batch_size: int = 16, rank: int = 0, mode: str = "train"):
    """Return PyTorch DataLoader with transforms to preprocess images.

    Args:
        dataset_path (str): Path to the dataset.
        batch_size (int, optional): Number of images per batch.
        rank (int, optional): Process rank for distributed training.
        mode (str, optional): 'train', 'val', or 'test' mode.

    Returns:
        (torch.utils.data.DataLoader): DataLoader for the specified dataset and mode.
    """
    with torch_distributed_zero_first(rank):  # init dataset *.cache only once if DDP
        dataset = self.build_dataset(dataset_path, mode)

    # Filter out samples with class indices >= nc (prevents CUDA assertion errors)
    nc = self.data.get("nc", 0)
    dataset_nc = len(dataset.base.classes)
    if nc and dataset_nc > nc:
        extra_classes = dataset.base.classes[nc:]
        original_count = len(dataset.samples)
        dataset.samples = [s for s in dataset.samples if s[1] < nc]
        skipped = original_count - len(dataset.samples)
        LOGGER.warning(
            f"{mode} split has {dataset_nc} classes but model expects {nc}. "
            f"Skipping {skipped} samples from extra classes: {extra_classes}"
        )

    loader = build_dataloader(dataset, batch_size, self.args.workers, rank=rank, drop_last=self.args.compile)
    # Attach inference transforms
    if mode != "train":
        if is_parallel(self.model):
            self.model.module.transforms = loader.dataset.torch_transforms
        else:
            self.model.transforms = loader.dataset.torch_transforms
    return loader
```
</details>

<br>

### Method `ultralytics.models.yolo.classify.train.ClassificationTrainer.get_model` {#ultralytics.models.yolo.classify.train.ClassificationTrainer.get\_model}

```python
def get_model(self, cfg = None, weights = None, verbose: bool = True)
```

Return a modified PyTorch model configured for training YOLO classification.

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `cfg` | `Any, optional` | Model configuration. | `None` |
| `weights` | `Any, optional` | Pre-trained model weights. | `None` |
| `verbose` | `bool, optional` | Whether to display model information. | `True` |

**Returns**

| Type | Description |
| --- | --- |
| `ClassificationModel` | Configured PyTorch model for classification. |

<details>
<summary>Source code in <code>ultralytics/models/yolo/classify/train.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/models/yolo/classify/train.py#L71-L93"><i class="fa-brands fa-github" aria-hidden="true" style="margin-right:6px;"></i>View on GitHub</a>
```python
def get_model(self, cfg=None, weights=None, verbose: bool = True):
    """Return a modified PyTorch model configured for training YOLO classification.

    Args:
        cfg (Any, optional): Model configuration.
        weights (Any, optional): Pre-trained model weights.
        verbose (bool, optional): Whether to display model information.

    Returns:
        (ClassificationModel): Configured PyTorch model for classification.
    """
    model = ClassificationModel(cfg, nc=self.data["nc"], ch=self.data["channels"], verbose=verbose and RANK == -1)
    if weights:
        model.load(weights)

    for m in model.modules():
        if not self.args.pretrained and hasattr(m, "reset_parameters"):
            m.reset_parameters()
        if isinstance(m, torch.nn.Dropout) and self.args.dropout:
            m.p = self.args.dropout  # set dropout
    for p in model.parameters():
        p.requires_grad = True  # for training
    return model
```
</details>

<br>

### Method `ultralytics.models.yolo.classify.train.ClassificationTrainer.get_validator` {#ultralytics.models.yolo.classify.train.ClassificationTrainer.get\_validator}

```python
def get_validator(self)
```

Return an instance of ClassificationValidator for validation.

<details>
<summary>Source code in <code>ultralytics/models/yolo/classify/train.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/models/yolo/classify/train.py#L179-L184"><i class="fa-brands fa-github" aria-hidden="true" style="margin-right:6px;"></i>View on GitHub</a>
```python
def get_validator(self):
    """Return an instance of ClassificationValidator for validation."""
    self.loss_names = ["loss"]
    return yolo.classify.ClassificationValidator(
        self.test_loader, self.save_dir, args=copy(self.args), _callbacks=self.callbacks
    )
```
</details>

<br>

### Method `ultralytics.models.yolo.classify.train.ClassificationTrainer.label_loss_items` {#ultralytics.models.yolo.classify.train.ClassificationTrainer.label\_loss\_items}

```python
def label_loss_items(self, loss_items: torch.Tensor | None = None, prefix: str = "train")
```

Return a loss dict with labeled training loss items tensor.

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `loss_items` | `torch.Tensor, optional` | Loss tensor items. | `None` |
| `prefix` | `str, optional` | Prefix to prepend to loss names. | `"train"` |

**Returns**

| Type | Description |
| --- | --- |
| `dict | list` | Dictionary of labeled loss items if loss_items is provided, otherwise list of keys. |

<details>
<summary>Source code in <code>ultralytics/models/yolo/classify/train.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/models/yolo/classify/train.py#L186-L200"><i class="fa-brands fa-github" aria-hidden="true" style="margin-right:6px;"></i>View on GitHub</a>
```python
def label_loss_items(self, loss_items: torch.Tensor | None = None, prefix: str = "train"):
    """Return a loss dict with labeled training loss items tensor.

    Args:
        loss_items (torch.Tensor, optional): Loss tensor items.
        prefix (str, optional): Prefix to prepend to loss names.

    Returns:
        (dict | list): Dictionary of labeled loss items if loss_items is provided, otherwise list of keys.
    """
    keys = [f"{prefix}/{x}" for x in self.loss_names]
    if loss_items is None:
        return keys
    loss_items = [round(float(loss_items), 5)]
    return dict(zip(keys, loss_items))
```
</details>

<br>

### Method `ultralytics.models.yolo.classify.train.ClassificationTrainer.plot_training_samples` {#ultralytics.models.yolo.classify.train.ClassificationTrainer.plot\_training\_samples}

```python
def plot_training_samples(self, batch: dict[str, torch.Tensor], ni: int)
```

Plot training samples with their annotations.

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `batch` | `dict[str, torch.Tensor]` | Batch containing images and class labels. | *required* |
| `ni` | `int` | Batch index used for naming the output file. | *required* |

<details>
<summary>Source code in <code>ultralytics/models/yolo/classify/train.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/models/yolo/classify/train.py#L202-L214"><i class="fa-brands fa-github" aria-hidden="true" style="margin-right:6px;"></i>View on GitHub</a>
```python
def plot_training_samples(self, batch: dict[str, torch.Tensor], ni: int):
    """Plot training samples with their annotations.

    Args:
        batch (dict[str, torch.Tensor]): Batch containing images and class labels.
        ni (int): Batch index used for naming the output file.
    """
    batch["batch_idx"] = torch.arange(batch["img"].shape[0])  # add batch index for plotting
    plot_images(
        labels=batch,
        fname=self.save_dir / f"train_batch{ni}.jpg",
        on_plot=self.on_plot,
    )
```
</details>

<br>

### Method `ultralytics.models.yolo.classify.train.ClassificationTrainer.preprocess_batch` {#ultralytics.models.yolo.classify.train.ClassificationTrainer.preprocess\_batch}

```python
def preprocess_batch(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]
```

Preprocess a batch of images and classes.

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `batch` | `dict[str, torch.Tensor]` |  | *required* |

<details>
<summary>Source code in <code>ultralytics/models/yolo/classify/train.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/models/yolo/classify/train.py#L163-L167"><i class="fa-brands fa-github" aria-hidden="true" style="margin-right:6px;"></i>View on GitHub</a>
```python
def preprocess_batch(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Preprocess a batch of images and classes."""
    batch["img"] = batch["img"].to(self.device, non_blocking=self.device.type == "cuda")
    batch["cls"] = batch["cls"].to(self.device, non_blocking=self.device.type == "cuda")
    return batch
```
</details>

<br>

### Method `ultralytics.models.yolo.classify.train.ClassificationTrainer.progress_string` {#ultralytics.models.yolo.classify.train.ClassificationTrainer.progress\_string}

```python
def progress_string(self) -> str
```

Return a formatted string showing training progress.

<details>
<summary>Source code in <code>ultralytics/models/yolo/classify/train.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/models/yolo/classify/train.py#L169-L177"><i class="fa-brands fa-github" aria-hidden="true" style="margin-right:6px;"></i>View on GitHub</a>
```python
def progress_string(self) -> str:
    """Return a formatted string showing training progress."""
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

### Method `ultralytics.models.yolo.classify.train.ClassificationTrainer.set_model_attributes` {#ultralytics.models.yolo.classify.train.ClassificationTrainer.set\_model\_attributes}

```python
def set_model_attributes(self)
```

Set the YOLO model's class names from the loaded dataset.

<details>
<summary>Source code in <code>ultralytics/models/yolo/classify/train.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/models/yolo/classify/train.py#L67-L69"><i class="fa-brands fa-github" aria-hidden="true" style="margin-right:6px;"></i>View on GitHub</a>
```python
def set_model_attributes(self):
    """Set the YOLO model's class names from the loaded dataset."""
    self.model.names = self.data["names"]
```
</details>

<br>

### Method `ultralytics.models.yolo.classify.train.ClassificationTrainer.setup_model` {#ultralytics.models.yolo.classify.train.ClassificationTrainer.setup\_model}

```python
def setup_model(self)
```

Load, create or download model for classification tasks.

**Returns**

| Type | Description |
| --- | --- |
| `Any` | Model checkpoint if applicable, otherwise None. |

<details>
<summary>Source code in <code>ultralytics/models/yolo/classify/train.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/models/yolo/classify/train.py#L95-L111"><i class="fa-brands fa-github" aria-hidden="true" style="margin-right:6px;"></i>View on GitHub</a>
```python
def setup_model(self):
    """Load, create or download model for classification tasks.

    Returns:
        (Any): Model checkpoint if applicable, otherwise None.
    """
    import torchvision  # scope for faster 'import ultralytics'

    if str(self.model) in torchvision.models.__dict__:
        self.model = torchvision.models.__dict__[self.model](
            weights="IMAGENET1K_V1" if self.args.pretrained else None
        )
        ckpt = None
    else:
        ckpt = super().setup_model()
    ClassificationModel.reshape_outputs(self.model, self.data["nc"])
    return ckpt
```
</details>

<br><br>
