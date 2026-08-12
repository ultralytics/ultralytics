---
title: models.yolo.semantic.val API Reference
description: Reference for YOLO semantic segmentation validator. Learn about the SemanticSegmentationValidator class and its validation metrics.
keywords: Ultralytics, semantic segmentation, validator, validation, mIoU, YOLO, semantic
---

# Reference for `ultralytics/models/yolo/semantic/val.py`

!!! success "Improvements"

    This page is sourced from [https://github.com/ultralytics/ultralytics/blob/main/ultralytics/models/yolo/semantic/val.py](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/models/yolo/semantic/val.py). Have an improvement or example to add? Open a [Pull Request](https://docs.ultralytics.com/help/contributing) — thank you! 🙏

<br>

!!! abstract "Summary"

    === "<span class="doc-kind doc-kind-class">Classes</span>"

        - [`SemanticSegmentationValidator`](#ultralytics.models.yolo.semantic.val.SemanticSegmentationValidator)

    === "<span class="doc-kind doc-kind-method">Methods</span>"

        - [`SemanticSegmentationValidator.init_metrics`](#ultralytics.models.yolo.semantic.val.SemanticSegmentationValidator.init_metrics)
        - [`SemanticSegmentationValidator.preprocess`](#ultralytics.models.yolo.semantic.val.SemanticSegmentationValidator.preprocess)
        - [`SemanticSegmentationValidator.postprocess`](#ultralytics.models.yolo.semantic.val.SemanticSegmentationValidator.postprocess)
        - [`SemanticSegmentationValidator.update_metrics`](#ultralytics.models.yolo.semantic.val.SemanticSegmentationValidator.update_metrics)
        - [`SemanticSegmentationValidator.gather_stats`](#ultralytics.models.yolo.semantic.val.SemanticSegmentationValidator.gather_stats)
        - [`SemanticSegmentationValidator.save_pred_masks`](#ultralytics.models.yolo.semantic.val.SemanticSegmentationValidator.save_pred_masks)
        - [`SemanticSegmentationValidator.get_stats`](#ultralytics.models.yolo.semantic.val.SemanticSegmentationValidator.get_stats)
        - [`SemanticSegmentationValidator.get_desc`](#ultralytics.models.yolo.semantic.val.SemanticSegmentationValidator.get_desc)
        - [`SemanticSegmentationValidator.print_results`](#ultralytics.models.yolo.semantic.val.SemanticSegmentationValidator.print_results)
        - [`SemanticSegmentationValidator.get_dataset`](#ultralytics.models.yolo.semantic.val.SemanticSegmentationValidator.get_dataset)
        - [`SemanticSegmentationValidator.plot_predictions`](#ultralytics.models.yolo.semantic.val.SemanticSegmentationValidator.plot_predictions)


## Class `ultralytics.models.yolo.semantic.val.SemanticSegmentationValidator` {#ultralytics.models.yolo.semantic.val.SemanticSegmentationValidator}

```python
SemanticSegmentationValidator(dataloader=None, save_dir=None, args=None, _callbacks=None)
```

**Bases:** `DetectionValidator`

Validator for semantic segmentation models.

This validator evaluates semantic segmentation models using mIoU and pixel accuracy metrics.

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `dataloader` | `DataLoader, optional` | DataLoader for validation. | `None` |
| `save_dir` | `Path, optional` | Directory to save results. | `None` |
| `args` | `dict, optional` | Arguments for the validator. | `None` |
| `_callbacks` | `dict, optional` | Callback functions. | `None` |

**Attributes**

| Name | Type | Description |
| --- | --- | --- |
| `metrics` | `SemanticMetrics` | Metrics calculator for semantic segmentation. |

**Methods**

| Name | Description |
| --- | --- |
| [`gather_stats`](#ultralytics.models.yolo.semantic.val.SemanticSegmentationValidator.gather_stats) | Reduce semantic confusion matrix to rank 0 during DDP validation. |
| [`get_dataset`](#ultralytics.models.yolo.semantic.val.SemanticSegmentationValidator.get_dataset) | Parse the dataset YAML and add background metadata for polygon labels when required. |
| [`get_desc`](#ultralytics.models.yolo.semantic.val.SemanticSegmentationValidator.get_desc) | Return a formatted description of evaluation metrics. |
| [`get_stats`](#ultralytics.models.yolo.semantic.val.SemanticSegmentationValidator.get_stats) | Return validation statistics. |
| [`init_metrics`](#ultralytics.models.yolo.semantic.val.SemanticSegmentationValidator.init_metrics) | Initialize metrics with model class names. |
| [`plot_predictions`](#ultralytics.models.yolo.semantic.val.SemanticSegmentationValidator.plot_predictions) | Plot predicted semantic masks on input images. |
| [`postprocess`](#ultralytics.models.yolo.semantic.val.SemanticSegmentationValidator.postprocess) | Convert logits or baked class maps to class predictions. |
| [`preprocess`](#ultralytics.models.yolo.semantic.val.SemanticSegmentationValidator.preprocess) | Preprocess a batch of images and masks. |
| [`print_results`](#ultralytics.models.yolo.semantic.val.SemanticSegmentationValidator.print_results) | Print training/validation set metrics per class. |
| [`save_pred_masks`](#ultralytics.models.yolo.semantic.val.SemanticSegmentationValidator.save_pred_masks) | Save semantic predictions as single-channel PNG masks. |
| [`update_metrics`](#ultralytics.models.yolo.semantic.val.SemanticSegmentationValidator.update_metrics) | Update metrics with predictions and ground truth. |

**Examples**

```python
>>> from ultralytics.models.yolo.semantic import SemanticSegmentationValidator
>>> args = dict(model="yolo26n-sem.pt", data="cityscapes8.yaml")
>>> validator = SemanticSegmentationValidator(args=args)
>>> validator()
```

<details>
<summary>Source code in <code>ultralytics/models/yolo/semantic/val.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/models/yolo/semantic/val.py#L23-L200">View on GitHub</a>
```python
class SemanticSegmentationValidator(DetectionValidator):
    """Validator for semantic segmentation models.

    This validator evaluates semantic segmentation models using mIoU and pixel accuracy metrics.

    Attributes:
        metrics (SemanticMetrics): Metrics calculator for semantic segmentation.

    Examples:
        >>> from ultralytics.models.yolo.semantic import SemanticSegmentationValidator
        >>> args = dict(model="yolo26n-sem.pt", data="cityscapes8.yaml")
        >>> validator = SemanticSegmentationValidator(args=args)
        >>> validator()
    """

    def __init__(self, dataloader=None, save_dir=None, args=None, _callbacks=None):
        """Initialize SemanticSegmentationValidator.

        Args:
            dataloader (DataLoader, optional): DataLoader for validation.
            save_dir (Path, optional): Directory to save results.
            args (dict, optional): Arguments for the validator.
            _callbacks (dict, optional): Callback functions.
        """
        super().__init__(dataloader, save_dir, args, _callbacks)
        self.args.task = "semantic"
        self.dataset = None
        self.results_dir = None
        self.metrics = SemanticMetrics()
        self.image_shapes = {}
        self._semantic_target_shape = None
```
</details>

<br>

### Method `ultralytics.models.yolo.semantic.val.SemanticSegmentationValidator.gather_stats` {#ultralytics.models.yolo.semantic.val.SemanticSegmentationValidator.gather\_stats}

```python
def gather_stats(self)
```

Reduce semantic confusion matrix to rank 0 during DDP validation.

<details>
<summary>Source code in <code>ultralytics/models/yolo/semantic/val.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/models/yolo/semantic/val.py#L127-L141">View on GitHub</a>
```python
def gather_stats(self):
    """Reduce semantic confusion matrix to rank 0 during DDP validation."""
    if RANK == -1 or not dist.is_available() or not dist.is_initialized():
        return
    if self.metrics.matrix is None:
        cm_nc = self.metrics.cm_nc
        self.metrics.matrix = torch.zeros((cm_nc, cm_nc), device=self.device, dtype=torch.float32)
    dist.reduce(self.metrics.matrix, dst=0, op=dist.ReduceOp.SUM)
    # Gather nt_per_image across ranks
    if RANK == 0:
        gathered_nt = [None] * dist.get_world_size()
        dist.gather_object(self.metrics.nt_per_image, gathered_nt, dst=0)
        self.metrics.nt_per_image = np.sum(gathered_nt, axis=0)
    elif RANK > 0:
        dist.gather_object(self.metrics.nt_per_image, None, dst=0)
```
</details>

<br>

### Method `ultralytics.models.yolo.semantic.val.SemanticSegmentationValidator.get_dataset` {#ultralytics.models.yolo.semantic.val.SemanticSegmentationValidator.get\_dataset}

```python
def get_dataset(self)
```

Parse the dataset YAML and add background metadata for polygon labels when required.

<details>
<summary>Source code in <code>ultralytics/models/yolo/semantic/val.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/models/yolo/semantic/val.py#L187-L189">View on GitHub</a>
```python
def get_dataset(self):
    """Parse the dataset YAML and add background metadata for polygon labels when required."""
    return add_polygon_background(super().get_dataset())
```
</details>

<br>

### Method `ultralytics.models.yolo.semantic.val.SemanticSegmentationValidator.get_desc` {#ultralytics.models.yolo.semantic.val.SemanticSegmentationValidator.get\_desc}

```python
def get_desc(self)
```

Return a formatted description of evaluation metrics.

**Returns**

| Type | Description |
| --- | --- |
| `str` | Formatted string with metric names. |

<details>
<summary>Source code in <code>ultralytics/models/yolo/semantic/val.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/models/yolo/semantic/val.py#L173-L179">View on GitHub</a>
```python
def get_desc(self):
    """Return a formatted description of evaluation metrics.

    Returns:
        (str): Formatted string with metric names.
    """
    return ("%22s" + "%11s" * 4) % ("Class", "Images", "Pixels", "mIoU", "PixAcc")
```
</details>

<br>

### Method `ultralytics.models.yolo.semantic.val.SemanticSegmentationValidator.get_stats` {#ultralytics.models.yolo.semantic.val.SemanticSegmentationValidator.get\_stats}

```python
def get_stats(self)
```

Return validation statistics.

**Returns**

| Type | Description |
| --- | --- |
| `dict` | Dictionary of validation metrics. |

<details>
<summary>Source code in <code>ultralytics/models/yolo/semantic/val.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/models/yolo/semantic/val.py#L161-L171">View on GitHub</a>
```python
def get_stats(self):
    """Return validation statistics.

    Returns:
        (dict): Dictionary of validation metrics.
    """
    self.metrics.process(save_dir=self.save_dir, plot=self.args.plots, on_plot=self.on_plot)
    if self.metrics.matrix is not None:
        # Internal layout is [gt, pred]; transpose to [pred, gt] for ConfusionMatrix export format.
        self.confusion_matrix.matrix = self.metrics.matrix.detach().cpu().numpy().T.astype(float)
    return self.metrics.results_dict
```
</details>

<br>

### Method `ultralytics.models.yolo.semantic.val.SemanticSegmentationValidator.init_metrics` {#ultralytics.models.yolo.semantic.val.SemanticSegmentationValidator.init\_metrics}

```python
def init_metrics(self, model)
```

Initialize metrics with model class names.

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `model` | `nn.Module` | Model to validate. | *required* |

<details>
<summary>Source code in <code>ultralytics/models/yolo/semantic/val.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/models/yolo/semantic/val.py#L55-L78">View on GitHub</a>
```python
def init_metrics(self, model):
    """Initialize metrics with model class names.

    Args:
        model (nn.Module): Model to validate.
    """
    self.names = model.names
    self.nc = len(self.names)
    self.metrics = SemanticMetrics(names=self.names)
    self.seen = 0
    self.dataset = getattr(self.dataloader, "dataset", None)
    labels = getattr(self.dataset, "labels", []) if self.dataset is not None else []
    self.image_shapes = {lb["im_file"]: tuple(lb["shape"]) for lb in labels if "im_file" in lb and "shape" in lb}
    self.results_dir = None
    if self.args.save_json:
        self.results_dir = self.save_dir / "results"
        self.results_dir.mkdir(parents=True, exist_ok=True)
    cm_nc = self.metrics.cm_nc
    if cm_nc == 2 and len(self.names) == 1:  # binary segmentation, expand to include background
        cm_names = {0: "background", 1: next(iter(self.names.values()))}
    else:
        base = list(self.names.values()) + [str(i) for i in range(len(self.names), cm_nc)]
        cm_names = {i: base[i] for i in range(cm_nc)}
    self.confusion_matrix = ConfusionMatrix(names=cm_names, task="semantic")
```
</details>

<br>

### Method `ultralytics.models.yolo.semantic.val.SemanticSegmentationValidator.plot_predictions` {#ultralytics.models.yolo.semantic.val.SemanticSegmentationValidator.plot\_predictions}

```python
def plot_predictions(self, batch, preds, ni)
```

Plot predicted semantic masks on input images.

<details>
<summary>Source code in <code>ultralytics/models/yolo/semantic/val.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/models/yolo/semantic/val.py#L191-L200">View on GitHub</a>
```python
def plot_predictions(self, batch, preds, ni):
    """Plot predicted semantic masks on input images."""
    plot_images(
        images=batch["img"],
        labels={"semantic_mask": preds},
        paths=batch["im_file"],
        fname=self.save_dir / f"val_batch{ni}_pred.jpg",
        names=self.names,
        on_plot=self.on_plot,
    )
```
</details>

<br>

### Method `ultralytics.models.yolo.semantic.val.SemanticSegmentationValidator.postprocess` {#ultralytics.models.yolo.semantic.val.SemanticSegmentationValidator.postprocess}

```python
def postprocess(self, preds)
```

Convert logits or baked class maps to class predictions.

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `preds` | `torch.Tensor` | Raw model output logits [B, nc, H, W] or baked class map [B, H, W]. | *required* |

**Returns**

| Type | Description |
| --- | --- |
| `torch.Tensor` | Predicted class IDs [B, H, W]. |

<details>
<summary>Source code in <code>ultralytics/models/yolo/semantic/val.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/models/yolo/semantic/val.py#L94-L113">View on GitHub</a>
```python
def postprocess(self, preds):
    """Convert logits or baked class maps to class predictions.

    Args:
        preds (torch.Tensor): Raw model output logits [B, nc, H, W] or baked class map [B, H, W].

    Returns:
        (torch.Tensor): Predicted class IDs [B, H, W].
    """
    if isinstance(preds, (tuple, list)):
        preds = preds[0]
    if preds.ndim == 3:
        # [B, H, W] class map with argmax already baked into the graph. Nearest-resize only.
        if tuple(preds.shape[-2:]) != self._semantic_target_shape:
            preds = F.interpolate(preds[:, None].float(), size=self._semantic_target_shape, mode="nearest")[:, 0]
        return preds.to(torch.int32)
    pred_hw = preds.shape[2:]
    if pred_hw[0] != self._semantic_target_shape[0] or pred_hw[1] != self._semantic_target_shape[1]:
        preds = F.interpolate(preds, size=self._semantic_target_shape, mode="bilinear", align_corners=False)
    return preds.argmax(dim=1).to(torch.int32) if self.nc > 1 else preds.gt(0).squeeze(1).to(torch.int32)
```
</details>

<br>

### Method `ultralytics.models.yolo.semantic.val.SemanticSegmentationValidator.preprocess` {#ultralytics.models.yolo.semantic.val.SemanticSegmentationValidator.preprocess}

```python
def preprocess(self, batch)
```

Preprocess a batch of images and masks.

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `batch` | `dict` | Batch data containing images and masks. | *required* |

**Returns**

| Type | Description |
| --- | --- |
| `dict` | Preprocessed batch. |

<details>
<summary>Source code in <code>ultralytics/models/yolo/semantic/val.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/models/yolo/semantic/val.py#L80-L92">View on GitHub</a>
```python
def preprocess(self, batch):
    """Preprocess a batch of images and masks.

    Args:
        batch (dict): Batch data containing images and masks.

    Returns:
        (dict): Preprocessed batch.
    """
    batch = super().preprocess(batch)
    batch["semantic_mask"] = batch["semantic_mask"].to(self.device, dtype=torch.int32)
    self._semantic_target_shape = tuple(batch["semantic_mask"].shape[-2:])
    return batch
```
</details>

<br>

### Method `ultralytics.models.yolo.semantic.val.SemanticSegmentationValidator.print_results` {#ultralytics.models.yolo.semantic.val.SemanticSegmentationValidator.print\_results}

```python
def print_results(self) -> None
```

Print training/validation set metrics per class.

<details>
<summary>Source code in <code>ultralytics/models/yolo/semantic/val.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/models/yolo/semantic/val.py#L181-L185">View on GitHub</a>
```python
def print_results(self) -> None:
    """Print training/validation set metrics per class."""
    super().print_results()
    if self.args.save_json and self.results_dir is not None:
        LOGGER.info(f"Semantic prediction masks saved to {self.results_dir}")
```
</details>

<br>

### Method `ultralytics.models.yolo.semantic.val.SemanticSegmentationValidator.save_pred_masks` {#ultralytics.models.yolo.semantic.val.SemanticSegmentationValidator.save\_pred\_masks}

```python
def save_pred_masks(self, preds: torch.Tensor, batch: dict[str, Any]) -> None
```

Save semantic predictions as single-channel PNG masks.

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `preds` | `torch.Tensor` |  | *required* |
| `batch` | `dict[str, Any]` |  | *required* |

<details>
<summary>Source code in <code>ultralytics/models/yolo/semantic/val.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/models/yolo/semantic/val.py#L143-L159">View on GitHub</a>
```python
def save_pred_masks(self, preds: torch.Tensor, batch: dict[str, Any]) -> None:
    """Save semantic predictions as single-channel PNG masks."""
    if self.results_dir is None:
        return
    im_files = batch.get("im_file", [])
    if not im_files:
        return
    preds = preds.cpu().numpy()
    if isinstance(self.dataset, SemanticDataset) and self.dataset.label_mapping:
        preds = self.dataset.convert_label(preds, inverse=True)
    preds = preds.astype(np.uint8, copy=False)
    for pred, im_file in zip(preds, im_files):
        orig_shape = self.image_shapes.get(im_file)
        if orig_shape and pred.shape != orig_shape:
            pred = cv2.resize(pred, (orig_shape[1], orig_shape[0]), interpolation=cv2.INTER_NEAREST)
        save_path = self.results_dir / Path(im_file).with_suffix(".png").name
        Image.fromarray(pred).save(save_path)
```
</details>

<br>

### Method `ultralytics.models.yolo.semantic.val.SemanticSegmentationValidator.update_metrics` {#ultralytics.models.yolo.semantic.val.SemanticSegmentationValidator.update\_metrics}

```python
def update_metrics(self, preds, batch)
```

Update metrics with predictions and ground truth.

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `preds` | `torch.Tensor` | Predicted class IDs [B, H, W]. | *required* |
| `batch` | `dict` | Batch containing 'semantic_mask'. | *required* |

<details>
<summary>Source code in <code>ultralytics/models/yolo/semantic/val.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/models/yolo/semantic/val.py#L115-L125">View on GitHub</a>
```python
def update_metrics(self, preds, batch):
    """Update metrics with predictions and ground truth.

    Args:
        preds (torch.Tensor): Predicted class IDs [B, H, W].
        batch (dict): Batch containing 'semantic_mask'.
    """
    if self.args.save_json:
        self.save_pred_masks(preds, batch)
    self.metrics.update_stats(preds, batch["semantic_mask"])
    self.seen += preds.shape[0]
```
</details>

<br><br>
