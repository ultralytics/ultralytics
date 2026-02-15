---
description: Explore the SAM (Segment Anything Model) and SAM 2 (Segment Anything Model 2) interface for real-time image segmentation. Learn about promptable segmentation and zero-shot capabilities.
keywords: Ultralytics, SAM, Segment Anything Model, SAM 2, Segment Anything Model 2, image segmentation, real-time segmentation, zero-shot performance, promptable segmentation, SA-1B dataset
---

# Reference for `ultralytics/models/sam/model.py`

!!! success "Improvements"

    This page is sourced from [https://github.com/ultralytics/ultralytics/blob/main/ultralytics/models/sam/model.py](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/models/sam/model.py). Have an improvement or example to add? Open a [Pull Request](https://docs.ultralytics.com/help/contributing/) — thank you! 🙏

<br>

!!! abstract "Summary"

    === "<span class="doc-kind doc-kind-class">Classes</span>"

        - [`SAM`](#ultralytics.models.sam.model.SAM)

    === "<span class="doc-kind doc-kind-property">Properties</span>"

        - [`SAM.task_map`](#ultralytics.models.sam.model.SAM.task_map)

    === "<span class="doc-kind doc-kind-method">Methods</span>"

        - [`SAM._load`](#ultralytics.models.sam.model.SAM._load)
        - [`SAM.predict`](#ultralytics.models.sam.model.SAM.predict)
        - [`SAM.__call__`](#ultralytics.models.sam.model.SAM.__call__)
        - [`SAM.info`](#ultralytics.models.sam.model.SAM.info)


## Class `ultralytics.models.sam.model.SAM` {#ultralytics.models.sam.model.SAM}

```python
SAM(self, model: str = "sam_b.pt") -> None
```

**Bases:** `Model`

SAM (Segment Anything Model) interface class for real-time image segmentation tasks.

This class provides an interface to the Segment Anything Model (SAM) from Ultralytics, designed for promptable segmentation with versatility in image analysis. It supports various prompts such as bounding boxes, points, or labels, and features zero-shot performance capabilities.

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `model` | `str` | Path to the pre-trained SAM model file. File should have a .pt or .pth extension. | `"sam_b.pt"` |

**Attributes**

| Name | Type | Description |
| --- | --- | --- |
| `model` | `torch.nn.Module` | The loaded SAM model. |
| `is_sam2` | `bool` | Indicates whether the model is SAM2 variant. |
| `task` | `str` | The task type, set to "segment" for SAM models. |

**Methods**

| Name | Description |
| --- | --- |
| [`task_map`](#ultralytics.models.sam.model.SAM.task_map) | Provide a mapping from the 'segment' task to its corresponding 'Predictor'. |
| [`__call__`](#ultralytics.models.sam.model.SAM.__call__) | Perform segmentation prediction on the given image or video source. |
| [`_load`](#ultralytics.models.sam.model.SAM._load) | Load the specified weights into the SAM model. |
| [`info`](#ultralytics.models.sam.model.SAM.info) | Log information about the SAM model. |
| [`predict`](#ultralytics.models.sam.model.SAM.predict) | Perform segmentation prediction on the given image or video source. |

**Examples**

```python
>>> sam = SAM("sam_b.pt")
>>> results = sam.predict("image.jpg", points=[[500, 375]])
>>> for r in results:
...     print(f"Detected {len(r.masks)} masks")
```

**Raises**

| Type | Description |
| --- | --- |
| `NotImplementedError` | If the model file extension is not .pt or .pth. |

<details>
<summary>Source code in <code>ultralytics/models/sam/model.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/models/sam/model.py#L27-L169"><i class="fa-brands fa-github" aria-hidden="true" style="margin-right:6px;"></i>View on GitHub</a>
```python
class SAM(Model):
    """SAM (Segment Anything Model) interface class for real-time image segmentation tasks.

    This class provides an interface to the Segment Anything Model (SAM) from Ultralytics, designed for promptable
    segmentation with versatility in image analysis. It supports various prompts such as bounding boxes, points, or
    labels, and features zero-shot performance capabilities.

    Attributes:
        model (torch.nn.Module): The loaded SAM model.
        is_sam2 (bool): Indicates whether the model is SAM2 variant.
        task (str): The task type, set to "segment" for SAM models.

    Methods:
        predict: Perform segmentation prediction on the given image or video source.
        info: Log information about the SAM model.

    Examples:
        >>> sam = SAM("sam_b.pt")
        >>> results = sam.predict("image.jpg", points=[[500, 375]])
        >>> for r in results:
        ...     print(f"Detected {len(r.masks)} masks")
    """

    def __init__(self, model: str = "sam_b.pt") -> None:
        """Initialize the SAM (Segment Anything Model) instance.

        Args:
            model (str): Path to the pre-trained SAM model file. File should have a .pt or .pth extension.

        Raises:
            NotImplementedError: If the model file extension is not .pt or .pth.
        """
        if model and Path(model).suffix not in {".pt", ".pth"}:
            raise NotImplementedError("SAM prediction requires pre-trained *.pt or *.pth model.")
        self.is_sam2 = "sam2" in Path(model).stem
        self.is_sam3 = "sam3" in Path(model).stem
        super().__init__(model=model, task="segment")
```
</details>

<br>

### Property `ultralytics.models.sam.model.SAM.task_map` {#ultralytics.models.sam.model.SAM.task\_map}

```python
def task_map(self) -> dict[str, dict[str, type[Predictor]]]
```

Provide a mapping from the 'segment' task to its corresponding 'Predictor'.

**Returns**

| Type | Description |
| --- | --- |
| `dict[str, dict[str, type[Predictor]]]` | A dictionary mapping the 'segment' task to its corresponding |

**Examples**

```python
>>> sam = SAM("sam_b.pt")
>>> task_map = sam.task_map
>>> print(task_map)
{'segment': {'predictor': <class 'ultralytics.models.sam.predict.Predictor'>}}
```

<details>
<summary>Source code in <code>ultralytics/models/sam/model.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/models/sam/model.py#L154-L169"><i class="fa-brands fa-github" aria-hidden="true" style="margin-right:6px;"></i>View on GitHub</a>
```python
@property
def task_map(self) -> dict[str, dict[str, type[Predictor]]]:
    """Provide a mapping from the 'segment' task to its corresponding 'Predictor'.

    Returns:
        (dict[str, dict[str, type[Predictor]]]): A dictionary mapping the 'segment' task to its corresponding
            Predictor class. For SAM2 models, it maps to SAM2Predictor, otherwise to the standard Predictor.

    Examples:
        >>> sam = SAM("sam_b.pt")
        >>> task_map = sam.task_map
        >>> print(task_map)
        {'segment': {'predictor': <class 'ultralytics.models.sam.predict.Predictor'>}}
    """
    return {
        "segment": {"predictor": SAM2Predictor if self.is_sam2 else SAM3Predictor if self.is_sam3 else Predictor}
    }
```
</details>

<br>

### Method `ultralytics.models.sam.model.SAM.__call__` {#ultralytics.models.sam.model.SAM.\_\_call\_\_}

```python
def __call__(self, source = None, stream: bool = False, bboxes = None, points = None, labels = None, **kwargs)
```

Perform segmentation prediction on the given image or video source.

This method is an alias for the 'predict' method, providing a convenient way to call the SAM model for segmentation tasks.

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `source` | `str | PIL.Image | np.ndarray | None` | Path to the image or video file, or a PIL.Image object, or a<br>    np.ndarray object. | `None` |
| `stream` | `bool` | If True, enables real-time streaming. | `False` |
| `bboxes` | `list[list[float]] | None` | List of bounding box coordinates for prompted segmentation. | `None` |
| `points` | `list[list[float]] | None` | List of points for prompted segmentation. | `None` |
| `labels` | `list[int] | None` | List of labels for prompted segmentation. | `None` |
| `**kwargs` | `Any` | Additional keyword arguments to be passed to the predict method. | *required* |

**Returns**

| Type | Description |
| --- | --- |
| `list` | The model predictions, typically containing segmentation masks and other relevant information. |

**Examples**

```python
>>> sam = SAM("sam_b.pt")
>>> results = sam("image.jpg", points=[[500, 375]])
>>> print(f"Detected {len(results[0].masks)} masks")
```

<details>
<summary>Source code in <code>ultralytics/models/sam/model.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/models/sam/model.py#L111-L134"><i class="fa-brands fa-github" aria-hidden="true" style="margin-right:6px;"></i>View on GitHub</a>
```python
def __call__(self, source=None, stream: bool = False, bboxes=None, points=None, labels=None, **kwargs):
    """Perform segmentation prediction on the given image or video source.

    This method is an alias for the 'predict' method, providing a convenient way to call the SAM model for
    segmentation tasks.

    Args:
        source (str | PIL.Image | np.ndarray | None): Path to the image or video file, or a PIL.Image object, or a
            np.ndarray object.
        stream (bool): If True, enables real-time streaming.
        bboxes (list[list[float]] | None): List of bounding box coordinates for prompted segmentation.
        points (list[list[float]] | None): List of points for prompted segmentation.
        labels (list[int] | None): List of labels for prompted segmentation.
        **kwargs (Any): Additional keyword arguments to be passed to the predict method.

    Returns:
        (list): The model predictions, typically containing segmentation masks and other relevant information.

    Examples:
        >>> sam = SAM("sam_b.pt")
        >>> results = sam("image.jpg", points=[[500, 375]])
        >>> print(f"Detected {len(results[0].masks)} masks")
    """
    return self.predict(source, stream, bboxes, points, labels, **kwargs)
```
</details>

<br>

### Method `ultralytics.models.sam.model.SAM._load` {#ultralytics.models.sam.model.SAM.\_load}

```python
def _load(self, weights: str, task = None)
```

Load the specified weights into the SAM model.

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `weights` | `str` | Path to the weights file. Should be a .pt or .pth file containing the model parameters. | *required* |
| `task` | `str | None` | Task name. If provided, it specifies the particular task the model is being loaded for. | `None` |

**Examples**

```python
>>> sam = SAM("sam_b.pt")
>>> sam._load("path/to/custom_weights.pt")
```

<details>
<summary>Source code in <code>ultralytics/models/sam/model.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/models/sam/model.py#L65-L83"><i class="fa-brands fa-github" aria-hidden="true" style="margin-right:6px;"></i>View on GitHub</a>
```python
def _load(self, weights: str, task=None):
    """Load the specified weights into the SAM model.

    Args:
        weights (str): Path to the weights file. Should be a .pt or .pth file containing the model parameters.
        task (str | None): Task name. If provided, it specifies the particular task the model is being loaded for.

    Examples:
        >>> sam = SAM("sam_b.pt")
        >>> sam._load("path/to/custom_weights.pt")
    """
    if self.is_sam3:
        from .build_sam3 import build_interactive_sam3

        self.model = build_interactive_sam3(weights)
    else:
        from .build import build_sam  # slow import

        self.model = build_sam(weights)
```
</details>

<br>

### Method `ultralytics.models.sam.model.SAM.info` {#ultralytics.models.sam.model.SAM.info}

```python
def info(self, detailed: bool = False, verbose: bool = True)
```

Log information about the SAM model.

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `detailed` | `bool` | If True, displays detailed information about the model layers and operations. | `False` |
| `verbose` | `bool` | If True, prints the information to the console. | `True` |

**Returns**

| Type | Description |
| --- | --- |
| `tuple` | A tuple containing the model's information (string representations of the model). |

**Examples**

```python
>>> sam = SAM("sam_b.pt")
>>> info = sam.info()
>>> print(info[0])  # Print summary information
```

<details>
<summary>Source code in <code>ultralytics/models/sam/model.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/models/sam/model.py#L136-L151"><i class="fa-brands fa-github" aria-hidden="true" style="margin-right:6px;"></i>View on GitHub</a>
```python
def info(self, detailed: bool = False, verbose: bool = True):
    """Log information about the SAM model.

    Args:
        detailed (bool): If True, displays detailed information about the model layers and operations.
        verbose (bool): If True, prints the information to the console.

    Returns:
        (tuple): A tuple containing the model's information (string representations of the model).

    Examples:
        >>> sam = SAM("sam_b.pt")
        >>> info = sam.info()
        >>> print(info[0])  # Print summary information
    """
    return model_info(self.model, detailed=detailed, verbose=verbose)
```
</details>

<br>

### Method `ultralytics.models.sam.model.SAM.predict` {#ultralytics.models.sam.model.SAM.predict}

```python
def predict(self, source, stream: bool = False, bboxes = None, points = None, labels = None, **kwargs)
```

Perform segmentation prediction on the given image or video source.

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `source` | `str | PIL.Image | np.ndarray` | Path to the image or video file, or a PIL.Image object, or a<br>    np.ndarray object. | *required* |
| `stream` | `bool` | If True, enables real-time streaming. | `False` |
| `bboxes` | `list[list[float]] | None` | List of bounding box coordinates for prompted segmentation. | `None` |
| `points` | `list[list[float]] | None` | List of points for prompted segmentation. | `None` |
| `labels` | `list[int] | None` | List of labels for prompted segmentation. | `None` |
| `**kwargs` | `Any` | Additional keyword arguments for prediction. | *required* |

**Returns**

| Type | Description |
| --- | --- |
| `list` | The model predictions. |

**Examples**

```python
>>> sam = SAM("sam_b.pt")
>>> results = sam.predict("image.jpg", points=[[500, 375]])
>>> for r in results:
...     print(f"Detected {len(r.masks)} masks")
```

<details>
<summary>Source code in <code>ultralytics/models/sam/model.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/models/sam/model.py#L85-L109"><i class="fa-brands fa-github" aria-hidden="true" style="margin-right:6px;"></i>View on GitHub</a>
```python
def predict(self, source, stream: bool = False, bboxes=None, points=None, labels=None, **kwargs):
    """Perform segmentation prediction on the given image or video source.

    Args:
        source (str | PIL.Image | np.ndarray): Path to the image or video file, or a PIL.Image object, or a
            np.ndarray object.
        stream (bool): If True, enables real-time streaming.
        bboxes (list[list[float]] | None): List of bounding box coordinates for prompted segmentation.
        points (list[list[float]] | None): List of points for prompted segmentation.
        labels (list[int] | None): List of labels for prompted segmentation.
        **kwargs (Any): Additional keyword arguments for prediction.

    Returns:
        (list): The model predictions.

    Examples:
        >>> sam = SAM("sam_b.pt")
        >>> results = sam.predict("image.jpg", points=[[500, 375]])
        >>> for r in results:
        ...     print(f"Detected {len(r.masks)} masks")
    """
    overrides = dict(conf=0.25, task="segment", mode="predict", imgsz=1024)
    kwargs = {**overrides, **kwargs}
    prompts = dict(bboxes=bboxes, points=points, labels=labels)
    return super().predict(source, stream, prompts=prompts, **kwargs)
```
</details>

<br><br>
