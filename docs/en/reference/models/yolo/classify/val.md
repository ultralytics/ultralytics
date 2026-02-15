---
description: Explore the source code and functionalities of the YOLO Classification Validator in Ultralytics for evaluating classification models effectively.
keywords: Ultralytics, YOLO, classification, validation, ClassifyMetrics, ConfusionMatrix, PyTorch, deep learning, model evaluation, AI, machine learning
---

# Reference for `ultralytics/models/yolo/classify/val.py`

!!! success "Improvements"

    This page is sourced from [https://github.com/ultralytics/ultralytics/blob/main/ultralytics/models/yolo/classify/val.py](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/models/yolo/classify/val.py). Have an improvement or example to add? Open a [Pull Request](https://docs.ultralytics.com/help/contributing/) — thank you! 🙏

<br>

!!! abstract "Summary"

    === "<span class="doc-kind doc-kind-class">Classes</span>"

        - [`ClassificationValidator`](#ultralytics.models.yolo.classify.val.ClassificationValidator)

    === "<span class="doc-kind doc-kind-method">Methods</span>"

        - [`ClassificationValidator.get_desc`](#ultralytics.models.yolo.classify.val.ClassificationValidator.get_desc)
        - [`ClassificationValidator.init_metrics`](#ultralytics.models.yolo.classify.val.ClassificationValidator.init_metrics)
        - [`ClassificationValidator.preprocess`](#ultralytics.models.yolo.classify.val.ClassificationValidator.preprocess)
        - [`ClassificationValidator.update_metrics`](#ultralytics.models.yolo.classify.val.ClassificationValidator.update_metrics)
        - [`ClassificationValidator.finalize_metrics`](#ultralytics.models.yolo.classify.val.ClassificationValidator.finalize_metrics)
        - [`ClassificationValidator.postprocess`](#ultralytics.models.yolo.classify.val.ClassificationValidator.postprocess)
        - [`ClassificationValidator.get_stats`](#ultralytics.models.yolo.classify.val.ClassificationValidator.get_stats)
        - [`ClassificationValidator.gather_stats`](#ultralytics.models.yolo.classify.val.ClassificationValidator.gather_stats)
        - [`ClassificationValidator.build_dataset`](#ultralytics.models.yolo.classify.val.ClassificationValidator.build_dataset)
        - [`ClassificationValidator.get_dataloader`](#ultralytics.models.yolo.classify.val.ClassificationValidator.get_dataloader)
        - [`ClassificationValidator.print_results`](#ultralytics.models.yolo.classify.val.ClassificationValidator.print_results)
        - [`ClassificationValidator.plot_val_samples`](#ultralytics.models.yolo.classify.val.ClassificationValidator.plot_val_samples)
        - [`ClassificationValidator.plot_predictions`](#ultralytics.models.yolo.classify.val.ClassificationValidator.plot_predictions)


## Class `ultralytics.models.yolo.classify.val.ClassificationValidator` {#ultralytics.models.yolo.classify.val.ClassificationValidator}

```python
ClassificationValidator(self, dataloader = None, save_dir = None, args = None, _callbacks = None) -> None
```

**Bases:** `BaseValidator`

A class extending the BaseValidator class for validation based on a classification model.

This validator handles the validation process for classification models, including metrics calculation, confusion matrix generation, and visualization of results.

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `dataloader` | `torch.utils.data.DataLoader, optional` | DataLoader to use for validation. | `None` |
| `save_dir` | `str | Path, optional` | Directory to save results. | `None` |
| `args` | `dict, optional` | Arguments containing model and validation configuration. | `None` |
| `_callbacks` | `list, optional` | List of callback functions to be called during validation. | `None` |

**Attributes**

| Name | Type | Description |
| --- | --- | --- |
| `targets` | `list[torch.Tensor]` | Ground truth class labels. |
| `pred` | `list[torch.Tensor]` | Model predictions. |
| `metrics` | `ClassifyMetrics` | Object to calculate and store classification metrics. |
| `names` | `dict` | Mapping of class indices to class names. |
| `nc` | `int` | Number of classes. |
| `confusion_matrix` | `ConfusionMatrix` | Matrix to evaluate model performance across classes. |

**Methods**

| Name | Description |
| --- | --- |
| [`build_dataset`](#ultralytics.models.yolo.classify.val.ClassificationValidator.build_dataset) | Create a ClassificationDataset instance for validation. |
| [`finalize_metrics`](#ultralytics.models.yolo.classify.val.ClassificationValidator.finalize_metrics) | Finalize metrics including confusion matrix and processing speed. |
| [`gather_stats`](#ultralytics.models.yolo.classify.val.ClassificationValidator.gather_stats) | Gather stats from all GPUs. |
| [`get_dataloader`](#ultralytics.models.yolo.classify.val.ClassificationValidator.get_dataloader) | Build and return a data loader for classification validation. |
| [`get_desc`](#ultralytics.models.yolo.classify.val.ClassificationValidator.get_desc) | Return a formatted string summarizing classification metrics. |
| [`get_stats`](#ultralytics.models.yolo.classify.val.ClassificationValidator.get_stats) | Calculate and return a dictionary of metrics by processing targets and predictions. |
| [`init_metrics`](#ultralytics.models.yolo.classify.val.ClassificationValidator.init_metrics) | Initialize confusion matrix, class names, and tracking containers for predictions and targets. |
| [`plot_predictions`](#ultralytics.models.yolo.classify.val.ClassificationValidator.plot_predictions) | Plot images with their predicted class labels and save the visualization. |
| [`plot_val_samples`](#ultralytics.models.yolo.classify.val.ClassificationValidator.plot_val_samples) | Plot validation image samples with their ground truth labels. |
| [`postprocess`](#ultralytics.models.yolo.classify.val.ClassificationValidator.postprocess) | Extract the primary prediction from model output if it's in a list or tuple format. |
| [`preprocess`](#ultralytics.models.yolo.classify.val.ClassificationValidator.preprocess) | Preprocess input batch by moving data to device and converting to appropriate dtype. |
| [`print_results`](#ultralytics.models.yolo.classify.val.ClassificationValidator.print_results) | Print evaluation metrics for the classification model. |
| [`update_metrics`](#ultralytics.models.yolo.classify.val.ClassificationValidator.update_metrics) | Update running metrics with model predictions and batch targets. |

**Examples**

```python
>>> from ultralytics.models.yolo.classify import ClassificationValidator
>>> args = dict(model="yolo26n-cls.pt", data="imagenet10")
>>> validator = ClassificationValidator(args=args)
>>> validator()
```

!!! note "Notes"

    Torchvision classification models can also be passed to the 'model' argument, i.e. model='resnet18'.

<details>
<summary>Source code in <code>ultralytics/models/yolo/classify/val.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/models/yolo/classify/val.py#L18-L216"><i class="fa-brands fa-github" aria-hidden="true" style="margin-right:6px;"></i>View on GitHub</a>
```python
class ClassificationValidator(BaseValidator):
    """A class extending the BaseValidator class for validation based on a classification model.

    This validator handles the validation process for classification models, including metrics calculation, confusion
    matrix generation, and visualization of results.

    Attributes:
        targets (list[torch.Tensor]): Ground truth class labels.
        pred (list[torch.Tensor]): Model predictions.
        metrics (ClassifyMetrics): Object to calculate and store classification metrics.
        names (dict): Mapping of class indices to class names.
        nc (int): Number of classes.
        confusion_matrix (ConfusionMatrix): Matrix to evaluate model performance across classes.

    Methods:
        get_desc: Return a formatted string summarizing classification metrics.
        init_metrics: Initialize confusion matrix, class names, and tracking containers.
        preprocess: Preprocess input batch by moving data to device.
        update_metrics: Update running metrics with model predictions and batch targets.
        finalize_metrics: Finalize metrics including confusion matrix and processing speed.
        postprocess: Extract the primary prediction from model output.
        get_stats: Calculate and return a dictionary of metrics.
        build_dataset: Create a ClassificationDataset instance for validation.
        get_dataloader: Build and return a data loader for classification validation.
        print_results: Print evaluation metrics for the classification model.
        plot_val_samples: Plot validation image samples with their ground truth labels.
        plot_predictions: Plot images with their predicted class labels.

    Examples:
        >>> from ultralytics.models.yolo.classify import ClassificationValidator
        >>> args = dict(model="yolo26n-cls.pt", data="imagenet10")
        >>> validator = ClassificationValidator(args=args)
        >>> validator()

    Notes:
        Torchvision classification models can also be passed to the 'model' argument, i.e. model='resnet18'.
    """

    def __init__(self, dataloader=None, save_dir=None, args=None, _callbacks=None) -> None:
        """Initialize ClassificationValidator with dataloader, save directory, and other parameters.

        Args:
            dataloader (torch.utils.data.DataLoader, optional): DataLoader to use for validation.
            save_dir (str | Path, optional): Directory to save results.
            args (dict, optional): Arguments containing model and validation configuration.
            _callbacks (list, optional): List of callback functions to be called during validation.
        """
        super().__init__(dataloader, save_dir, args, _callbacks)
        self.targets = None
        self.pred = None
        self.args.task = "classify"
        self.metrics = ClassifyMetrics()
```
</details>

<br>

### Method `ultralytics.models.yolo.classify.val.ClassificationValidator.build_dataset` {#ultralytics.models.yolo.classify.val.ClassificationValidator.build\_dataset}

```python
def build_dataset(self, img_path: str) -> ClassificationDataset
```

Create a ClassificationDataset instance for validation.

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `img_path` | `str` |  | *required* |

<details>
<summary>Source code in <code>ultralytics/models/yolo/classify/val.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/models/yolo/classify/val.py#L149-L151"><i class="fa-brands fa-github" aria-hidden="true" style="margin-right:6px;"></i>View on GitHub</a>
```python
def build_dataset(self, img_path: str) -> ClassificationDataset:
    """Create a ClassificationDataset instance for validation."""
    return ClassificationDataset(root=img_path, args=self.args, augment=False, prefix=self.args.split)
```
</details>

<br>

### Method `ultralytics.models.yolo.classify.val.ClassificationValidator.finalize_metrics` {#ultralytics.models.yolo.classify.val.ClassificationValidator.finalize\_metrics}

```python
def finalize_metrics(self) -> None
```

Finalize metrics including confusion matrix and processing speed.

**Examples**

```python
>>> validator = ClassificationValidator()
>>> validator.pred = [torch.tensor([[0, 1, 2]])]  # Top-3 predictions for one sample
>>> validator.targets = [torch.tensor([0])]  # Ground truth class
>>> validator.finalize_metrics()
>>> print(validator.metrics.confusion_matrix)  # Access the confusion matrix
```

!!! note "Notes"

    This method processes the accumulated predictions and targets to generate the confusion matrix,
    optionally plots it, and updates the metrics object with speed information.

<details>
<summary>Source code in <code>ultralytics/models/yolo/classify/val.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/models/yolo/classify/val.py#L105-L125"><i class="fa-brands fa-github" aria-hidden="true" style="margin-right:6px;"></i>View on GitHub</a>
```python
def finalize_metrics(self) -> None:
    """Finalize metrics including confusion matrix and processing speed.

    Examples:
        >>> validator = ClassificationValidator()
        >>> validator.pred = [torch.tensor([[0, 1, 2]])]  # Top-3 predictions for one sample
        >>> validator.targets = [torch.tensor([0])]  # Ground truth class
        >>> validator.finalize_metrics()
        >>> print(validator.metrics.confusion_matrix)  # Access the confusion matrix

    Notes:
        This method processes the accumulated predictions and targets to generate the confusion matrix,
        optionally plots it, and updates the metrics object with speed information.
    """
    self.confusion_matrix.process_cls_preds(self.pred, self.targets)
    if self.args.plots:
        for normalize in True, False:
            self.confusion_matrix.plot(save_dir=self.save_dir, normalize=normalize, on_plot=self.on_plot)
    self.metrics.speed = self.speed
    self.metrics.save_dir = self.save_dir
    self.metrics.confusion_matrix = self.confusion_matrix
```
</details>

<br>

### Method `ultralytics.models.yolo.classify.val.ClassificationValidator.gather_stats` {#ultralytics.models.yolo.classify.val.ClassificationValidator.gather\_stats}

```python
def gather_stats(self) -> None
```

Gather stats from all GPUs.

<details>
<summary>Source code in <code>ultralytics/models/yolo/classify/val.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/models/yolo/classify/val.py#L136-L147"><i class="fa-brands fa-github" aria-hidden="true" style="margin-right:6px;"></i>View on GitHub</a>
```python
def gather_stats(self) -> None:
    """Gather stats from all GPUs."""
    if RANK == 0:
        gathered_preds = [None] * dist.get_world_size()
        gathered_targets = [None] * dist.get_world_size()
        dist.gather_object(self.pred, gathered_preds, dst=0)
        dist.gather_object(self.targets, gathered_targets, dst=0)
        self.pred = [pred for rank in gathered_preds for pred in rank]
        self.targets = [targets for rank in gathered_targets for targets in rank]
    elif RANK > 0:
        dist.gather_object(self.pred, None, dst=0)
        dist.gather_object(self.targets, None, dst=0)
```
</details>

<br>

### Method `ultralytics.models.yolo.classify.val.ClassificationValidator.get_dataloader` {#ultralytics.models.yolo.classify.val.ClassificationValidator.get\_dataloader}

```python
def get_dataloader(self, dataset_path: Path | str, batch_size: int) -> torch.utils.data.DataLoader
```

Build and return a data loader for classification validation.

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `dataset_path` | `str | Path` | Path to the dataset directory. | *required* |
| `batch_size` | `int` | Number of samples per batch. | *required* |

**Returns**

| Type | Description |
| --- | --- |
| `torch.utils.data.DataLoader` | DataLoader object for the classification validation dataset. |

<details>
<summary>Source code in <code>ultralytics/models/yolo/classify/val.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/models/yolo/classify/val.py#L153-L164"><i class="fa-brands fa-github" aria-hidden="true" style="margin-right:6px;"></i>View on GitHub</a>
```python
def get_dataloader(self, dataset_path: Path | str, batch_size: int) -> torch.utils.data.DataLoader:
    """Build and return a data loader for classification validation.

    Args:
        dataset_path (str | Path): Path to the dataset directory.
        batch_size (int): Number of samples per batch.

    Returns:
        (torch.utils.data.DataLoader): DataLoader object for the classification validation dataset.
    """
    dataset = self.build_dataset(dataset_path)
    return build_dataloader(dataset, batch_size, self.args.workers, rank=-1)
```
</details>

<br>

### Method `ultralytics.models.yolo.classify.val.ClassificationValidator.get_desc` {#ultralytics.models.yolo.classify.val.ClassificationValidator.get\_desc}

```python
def get_desc(self) -> str
```

Return a formatted string summarizing classification metrics.

<details>
<summary>Source code in <code>ultralytics/models/yolo/classify/val.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/models/yolo/classify/val.py#L71-L73"><i class="fa-brands fa-github" aria-hidden="true" style="margin-right:6px;"></i>View on GitHub</a>
```python
def get_desc(self) -> str:
    """Return a formatted string summarizing classification metrics."""
    return ("%22s" + "%11s" * 2) % ("classes", "top1_acc", "top5_acc")
```
</details>

<br>

### Method `ultralytics.models.yolo.classify.val.ClassificationValidator.get_stats` {#ultralytics.models.yolo.classify.val.ClassificationValidator.get\_stats}

```python
def get_stats(self) -> dict[str, float]
```

Calculate and return a dictionary of metrics by processing targets and predictions.

<details>
<summary>Source code in <code>ultralytics/models/yolo/classify/val.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/models/yolo/classify/val.py#L131-L134"><i class="fa-brands fa-github" aria-hidden="true" style="margin-right:6px;"></i>View on GitHub</a>
```python
def get_stats(self) -> dict[str, float]:
    """Calculate and return a dictionary of metrics by processing targets and predictions."""
    self.metrics.process(self.targets, self.pred)
    return self.metrics.results_dict
```
</details>

<br>

### Method `ultralytics.models.yolo.classify.val.ClassificationValidator.init_metrics` {#ultralytics.models.yolo.classify.val.ClassificationValidator.init\_metrics}

```python
def init_metrics(self, model: torch.nn.Module) -> None
```

Initialize confusion matrix, class names, and tracking containers for predictions and targets.

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `model` | `torch.nn.Module` |  | *required* |

<details>
<summary>Source code in <code>ultralytics/models/yolo/classify/val.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/models/yolo/classify/val.py#L75-L81"><i class="fa-brands fa-github" aria-hidden="true" style="margin-right:6px;"></i>View on GitHub</a>
```python
def init_metrics(self, model: torch.nn.Module) -> None:
    """Initialize confusion matrix, class names, and tracking containers for predictions and targets."""
    self.names = model.names
    self.nc = len(model.names)
    self.pred = []
    self.targets = []
    self.confusion_matrix = ConfusionMatrix(names=model.names)
```
</details>

<br>

### Method `ultralytics.models.yolo.classify.val.ClassificationValidator.plot_predictions` {#ultralytics.models.yolo.classify.val.ClassificationValidator.plot\_predictions}

```python
def plot_predictions(self, batch: dict[str, Any], preds: torch.Tensor, ni: int) -> None
```

Plot images with their predicted class labels and save the visualization.

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `batch` | `dict[str, Any]` | Batch data containing images and other information. | *required* |
| `preds` | `torch.Tensor` | Model predictions with shape (batch_size, num_classes). | *required* |
| `ni` | `int` | Batch index used for naming the output file. | *required* |

**Examples**

```python
>>> validator = ClassificationValidator()
>>> batch = {"img": torch.rand(16, 3, 224, 224)}
>>> preds = torch.rand(16, 10)  # 16 images, 10 classes
>>> validator.plot_predictions(batch, preds, 0)
```

<details>
<summary>Source code in <code>ultralytics/models/yolo/classify/val.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/models/yolo/classify/val.py#L191-L216"><i class="fa-brands fa-github" aria-hidden="true" style="margin-right:6px;"></i>View on GitHub</a>
```python
def plot_predictions(self, batch: dict[str, Any], preds: torch.Tensor, ni: int) -> None:
    """Plot images with their predicted class labels and save the visualization.

    Args:
        batch (dict[str, Any]): Batch data containing images and other information.
        preds (torch.Tensor): Model predictions with shape (batch_size, num_classes).
        ni (int): Batch index used for naming the output file.

    Examples:
        >>> validator = ClassificationValidator()
        >>> batch = {"img": torch.rand(16, 3, 224, 224)}
        >>> preds = torch.rand(16, 10)  # 16 images, 10 classes
        >>> validator.plot_predictions(batch, preds, 0)
    """
    batched_preds = dict(
        img=batch["img"],
        batch_idx=torch.arange(batch["img"].shape[0]),
        cls=torch.argmax(preds, dim=1),
        conf=torch.amax(preds, dim=1),
    )
    plot_images(
        batched_preds,
        fname=self.save_dir / f"val_batch{ni}_pred.jpg",
        names=self.names,
        on_plot=self.on_plot,
    )  # pred
```
</details>

<br>

### Method `ultralytics.models.yolo.classify.val.ClassificationValidator.plot_val_samples` {#ultralytics.models.yolo.classify.val.ClassificationValidator.plot\_val\_samples}

```python
def plot_val_samples(self, batch: dict[str, Any], ni: int) -> None
```

Plot validation image samples with their ground truth labels.

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `batch` | `dict[str, Any]` | Dictionary containing batch data with 'img' (images) and 'cls' (class labels). | *required* |
| `ni` | `int` | Batch index used for naming the output file. | *required* |

**Examples**

```python
>>> validator = ClassificationValidator()
>>> batch = {"img": torch.rand(16, 3, 224, 224), "cls": torch.randint(0, 10, (16,))}
>>> validator.plot_val_samples(batch, 0)
```

<details>
<summary>Source code in <code>ultralytics/models/yolo/classify/val.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/models/yolo/classify/val.py#L171-L189"><i class="fa-brands fa-github" aria-hidden="true" style="margin-right:6px;"></i>View on GitHub</a>
```python
def plot_val_samples(self, batch: dict[str, Any], ni: int) -> None:
    """Plot validation image samples with their ground truth labels.

    Args:
        batch (dict[str, Any]): Dictionary containing batch data with 'img' (images) and 'cls' (class labels).
        ni (int): Batch index used for naming the output file.

    Examples:
        >>> validator = ClassificationValidator()
        >>> batch = {"img": torch.rand(16, 3, 224, 224), "cls": torch.randint(0, 10, (16,))}
        >>> validator.plot_val_samples(batch, 0)
    """
    batch["batch_idx"] = torch.arange(batch["img"].shape[0])  # add batch index for plotting
    plot_images(
        labels=batch,
        fname=self.save_dir / f"val_batch{ni}_labels.jpg",
        names=self.names,
        on_plot=self.on_plot,
    )
```
</details>

<br>

### Method `ultralytics.models.yolo.classify.val.ClassificationValidator.postprocess` {#ultralytics.models.yolo.classify.val.ClassificationValidator.postprocess}

```python
def postprocess(self, preds: torch.Tensor | list[torch.Tensor] | tuple[torch.Tensor]) -> torch.Tensor
```

Extract the primary prediction from model output if it's in a list or tuple format.

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `preds` | `torch.Tensor | list[torch.Tensor] | tuple[torch.Tensor]` |  | *required* |

<details>
<summary>Source code in <code>ultralytics/models/yolo/classify/val.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/models/yolo/classify/val.py#L127-L129"><i class="fa-brands fa-github" aria-hidden="true" style="margin-right:6px;"></i>View on GitHub</a>
```python
def postprocess(self, preds: torch.Tensor | list[torch.Tensor] | tuple[torch.Tensor]) -> torch.Tensor:
    """Extract the primary prediction from model output if it's in a list or tuple format."""
    return preds[0] if isinstance(preds, (list, tuple)) else preds
```
</details>

<br>

### Method `ultralytics.models.yolo.classify.val.ClassificationValidator.preprocess` {#ultralytics.models.yolo.classify.val.ClassificationValidator.preprocess}

```python
def preprocess(self, batch: dict[str, Any]) -> dict[str, Any]
```

Preprocess input batch by moving data to device and converting to appropriate dtype.

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `batch` | `dict[str, Any]` |  | *required* |

<details>
<summary>Source code in <code>ultralytics/models/yolo/classify/val.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/models/yolo/classify/val.py#L83-L88"><i class="fa-brands fa-github" aria-hidden="true" style="margin-right:6px;"></i>View on GitHub</a>
```python
def preprocess(self, batch: dict[str, Any]) -> dict[str, Any]:
    """Preprocess input batch by moving data to device and converting to appropriate dtype."""
    batch["img"] = batch["img"].to(self.device, non_blocking=self.device.type == "cuda")
    batch["img"] = batch["img"].half() if self.args.half else batch["img"].float()
    batch["cls"] = batch["cls"].to(self.device, non_blocking=self.device.type == "cuda")
    return batch
```
</details>

<br>

### Method `ultralytics.models.yolo.classify.val.ClassificationValidator.print_results` {#ultralytics.models.yolo.classify.val.ClassificationValidator.print\_results}

```python
def print_results(self) -> None
```

Print evaluation metrics for the classification model.

<details>
<summary>Source code in <code>ultralytics/models/yolo/classify/val.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/models/yolo/classify/val.py#L166-L169"><i class="fa-brands fa-github" aria-hidden="true" style="margin-right:6px;"></i>View on GitHub</a>
```python
def print_results(self) -> None:
    """Print evaluation metrics for the classification model."""
    pf = "%22s" + "%11.3g" * len(self.metrics.keys)  # print format
    LOGGER.info(pf % ("all", self.metrics.top1, self.metrics.top5))
```
</details>

<br>

### Method `ultralytics.models.yolo.classify.val.ClassificationValidator.update_metrics` {#ultralytics.models.yolo.classify.val.ClassificationValidator.update\_metrics}

```python
def update_metrics(self, preds: torch.Tensor, batch: dict[str, Any]) -> None
```

Update running metrics with model predictions and batch targets.

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `preds` | `torch.Tensor` | Model predictions, typically logits or probabilities for each class. | *required* |
| `batch` | `dict` | Batch data containing images and class labels. | *required* |

!!! note "Notes"

    This method appends the top-N predictions (sorted by confidence in descending order) to the
    prediction list for later evaluation. N is limited to the minimum of 5 and the number of classes.

<details>
<summary>Source code in <code>ultralytics/models/yolo/classify/val.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/models/yolo/classify/val.py#L90-L103"><i class="fa-brands fa-github" aria-hidden="true" style="margin-right:6px;"></i>View on GitHub</a>
```python
def update_metrics(self, preds: torch.Tensor, batch: dict[str, Any]) -> None:
    """Update running metrics with model predictions and batch targets.

    Args:
        preds (torch.Tensor): Model predictions, typically logits or probabilities for each class.
        batch (dict): Batch data containing images and class labels.

    Notes:
        This method appends the top-N predictions (sorted by confidence in descending order) to the
        prediction list for later evaluation. N is limited to the minimum of 5 and the number of classes.
    """
    n5 = min(len(self.names), 5)
    self.pred.append(preds.argsort(1, descending=True)[:, :n5].type(torch.int32).cpu())
    self.targets.append(batch["cls"].type(torch.int32).cpu())
```
</details>

<br><br>
