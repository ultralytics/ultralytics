---
description: Discover how to use the FastSAM model with Ultralytics. Learn about its interface and implementation details with practical examples.
keywords: FastSAM, Ultralytics, model interface, YOLO, deep learning, machine learning, segmentation, predictor, validator, Python
---

# Reference for `ultralytics/models/fastsam/model.py`

!!! success "Improvements"

    This page is sourced from [https://github.com/ultralytics/ultralytics/blob/main/ultralytics/models/fastsam/model.py](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/models/fastsam/model.py). Have an improvement or example to add? Open a [Pull Request](https://docs.ultralytics.com/help/contributing/) — thank you! 🙏

<br>

!!! abstract "Summary"

    === "<span class="doc-kind doc-kind-class">Classes</span>"

        - [`FastSAM`](#ultralytics.models.fastsam.model.FastSAM)

    === "<span class="doc-kind doc-kind-property">Properties</span>"

        - [`FastSAM.task_map`](#ultralytics.models.fastsam.model.FastSAM.task_map)

    === "<span class="doc-kind doc-kind-method">Methods</span>"

        - [`FastSAM.predict`](#ultralytics.models.fastsam.model.FastSAM.predict)


## Class `ultralytics.models.fastsam.model.FastSAM` {#ultralytics.models.fastsam.model.FastSAM}

```python
FastSAM(self, model: str | Path = "FastSAM-x.pt")
```

**Bases:** `Model`

FastSAM model interface for Segment Anything tasks.

This class extends the base Model class to provide specific functionality for the FastSAM (Fast Segment Anything Model) implementation, allowing for efficient and accurate image segmentation with optional prompting support.

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `model` | `str | Path` |  | `"FastSAM-x.pt"` |

**Attributes**

| Name | Type | Description |
| --- | --- | --- |
| `model` | `str` | Path to the pre-trained FastSAM model file. |
| `task` | `str` | The task type, set to "segment" for FastSAM models. |

**Methods**

| Name | Description |
| --- | --- |
| [`task_map`](#ultralytics.models.fastsam.model.FastSAM.task_map) | Returns a dictionary mapping segment task to corresponding predictor and validator classes. |
| [`predict`](#ultralytics.models.fastsam.model.FastSAM.predict) | Perform segmentation prediction on image or video source. |

**Examples**

```python
Initialize FastSAM model and run prediction
>>> from ultralytics import FastSAM
>>> model = FastSAM("FastSAM-x.pt")
>>> results = model.predict("ultralytics/assets/bus.jpg")

Run prediction with bounding box prompts
>>> results = model.predict("image.jpg", bboxes=[[100, 100, 200, 200]])
```

<details>
<summary>Source code in <code>ultralytics/models/fastsam/model.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/models/fastsam/model.py#L14-L79"><i class="fa-brands fa-github" aria-hidden="true" style="margin-right:6px;"></i>View on GitHub</a>
```python
class FastSAM(Model):
    """FastSAM model interface for Segment Anything tasks.

    This class extends the base Model class to provide specific functionality for the FastSAM (Fast Segment Anything
    Model) implementation, allowing for efficient and accurate image segmentation with optional prompting support.

    Attributes:
        model (str): Path to the pre-trained FastSAM model file.
        task (str): The task type, set to "segment" for FastSAM models.

    Methods:
        predict: Perform segmentation prediction on image or video source with optional prompts.
        task_map: Returns mapping of segment task to predictor and validator classes.

    Examples:
        Initialize FastSAM model and run prediction
        >>> from ultralytics import FastSAM
        >>> model = FastSAM("FastSAM-x.pt")
        >>> results = model.predict("ultralytics/assets/bus.jpg")

        Run prediction with bounding box prompts
        >>> results = model.predict("image.jpg", bboxes=[[100, 100, 200, 200]])
    """

    def __init__(self, model: str | Path = "FastSAM-x.pt"):
        """Initialize the FastSAM model with the specified pre-trained weights."""
        if str(model) == "FastSAM.pt":
            model = "FastSAM-x.pt"
        assert Path(model).suffix not in {".yaml", ".yml"}, "FastSAM only supports pre-trained weights."
        super().__init__(model=model, task="segment")
```
</details>

<br>

### Property `ultralytics.models.fastsam.model.FastSAM.task_map` {#ultralytics.models.fastsam.model.FastSAM.task\_map}

```python
def task_map(self) -> dict[str, dict[str, Any]]
```

Returns a dictionary mapping segment task to corresponding predictor and validator classes.

<details>
<summary>Source code in <code>ultralytics/models/fastsam/model.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/models/fastsam/model.py#L77-L79"><i class="fa-brands fa-github" aria-hidden="true" style="margin-right:6px;"></i>View on GitHub</a>
```python
@property
def task_map(self) -> dict[str, dict[str, Any]]:
    """Returns a dictionary mapping segment task to corresponding predictor and validator classes."""
    return {"segment": {"predictor": FastSAMPredictor, "validator": FastSAMValidator}}
```
</details>

<br>

### Method `ultralytics.models.fastsam.model.FastSAM.predict` {#ultralytics.models.fastsam.model.FastSAM.predict}

```python
def predict(
    self,
    source,
    stream: bool = False,
    bboxes: list | None = None,
    points: list | None = None,
    labels: list | None = None,
    texts: list | None = None,
    **kwargs: Any,
)
```

Perform segmentation prediction on image or video source.

Supports prompted segmentation with bounding boxes, points, labels, and texts. The method packages these prompts and passes them to the parent class predict method for processing.

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `source` | `str | PIL.Image | np.ndarray` | Input source for prediction, can be a file path, URL, PIL image, or<br>    numpy array. | *required* |
| `stream` | `bool` | Whether to enable real-time streaming mode for video inputs. | `False` |
| `bboxes` | `list, optional` | Bounding box coordinates for prompted segmentation in format [[x1, y1, x2, y2]]. | `None` |
| `points` | `list, optional` | Point coordinates for prompted segmentation in format [[x, y]]. | `None` |
| `labels` | `list, optional` | Class labels for prompted segmentation. | `None` |
| `texts` | `list, optional` | Text prompts for segmentation guidance. | `None` |
| `**kwargs` | `Any` | Additional keyword arguments passed to the predictor. | *required* |

**Returns**

| Type | Description |
| --- | --- |
| `list` | List of Results objects containing the prediction results. |

<details>
<summary>Source code in <code>ultralytics/models/fastsam/model.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/models/fastsam/model.py#L45-L74"><i class="fa-brands fa-github" aria-hidden="true" style="margin-right:6px;"></i>View on GitHub</a>
```python
def predict(
    self,
    source,
    stream: bool = False,
    bboxes: list | None = None,
    points: list | None = None,
    labels: list | None = None,
    texts: list | None = None,
    **kwargs: Any,
):
    """Perform segmentation prediction on image or video source.

    Supports prompted segmentation with bounding boxes, points, labels, and texts. The method packages these prompts
    and passes them to the parent class predict method for processing.

    Args:
        source (str | PIL.Image | np.ndarray): Input source for prediction, can be a file path, URL, PIL image, or
            numpy array.
        stream (bool): Whether to enable real-time streaming mode for video inputs.
        bboxes (list, optional): Bounding box coordinates for prompted segmentation in format [[x1, y1, x2, y2]].
        points (list, optional): Point coordinates for prompted segmentation in format [[x, y]].
        labels (list, optional): Class labels for prompted segmentation.
        texts (list, optional): Text prompts for segmentation guidance.
        **kwargs (Any): Additional keyword arguments passed to the predictor.

    Returns:
        (list): List of Results objects containing the prediction results.
    """
    prompts = dict(bboxes=bboxes, points=points, labels=labels, texts=texts)
    return super().predict(source, stream, prompts=prompts, **kwargs)
```
</details>

<br><br>
