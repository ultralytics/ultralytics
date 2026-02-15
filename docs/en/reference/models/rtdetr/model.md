---
description: Explore the interface for Baidu's RT-DETR, a Vision Transformer-based real-time object detector in the Ultralytics Docs. Learn more about its efficient hybrid encoding and IoU-aware query selection.
keywords: RT-DETR, real-time object detection, Vision Transformer, Ultralytics, model interface, Baidu, hybrid encoding, IoU-aware query selection, machine learning, AI
---

# Reference for `ultralytics/models/rtdetr/model.py`

!!! success "Improvements"

    This page is sourced from [https://github.com/ultralytics/ultralytics/blob/main/ultralytics/models/rtdetr/model.py](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/models/rtdetr/model.py). Have an improvement or example to add? Open a [Pull Request](https://docs.ultralytics.com/help/contributing/) — thank you! 🙏

<br>

!!! abstract "Summary"

    === "<span class="doc-kind doc-kind-class">Classes</span>"

        - [`RTDETR`](#ultralytics.models.rtdetr.model.RTDETR)

    === "<span class="doc-kind doc-kind-property">Properties</span>"

        - [`RTDETR.task_map`](#ultralytics.models.rtdetr.model.RTDETR.task_map)


## Class `ultralytics.models.rtdetr.model.RTDETR` {#ultralytics.models.rtdetr.model.RTDETR}

```python
RTDETR(self, model: str = "rtdetr-l.pt") -> None
```

**Bases:** `Model`

Interface for Baidu's RT-DETR model, a Vision Transformer-based real-time object detector.

This model provides real-time performance with high accuracy. It supports efficient hybrid encoding, IoU-aware query selection, and adaptable inference speed.

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `model` | `str` | Path to the pre-trained model. Supports .pt, .yaml, and .yml formats. | `"rtdetr-l.pt"` |

**Attributes**

| Name | Type | Description |
| --- | --- | --- |
| `model` | `str` | Path to the pre-trained model. |

**Methods**

| Name | Description |
| --- | --- |
| [`task_map`](#ultralytics.models.rtdetr.model.RTDETR.task_map) | Return a task map for RT-DETR, associating tasks with corresponding Ultralytics classes. |

**Examples**

```python
Initialize RT-DETR with a pre-trained model
>>> from ultralytics import RTDETR
>>> model = RTDETR("rtdetr-l.pt")
>>> results = model("image.jpg")
```

<details>
<summary>Source code in <code>ultralytics/models/rtdetr/model.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/models/rtdetr/model.py#L21-L63"><i class="fa-brands fa-github" aria-hidden="true" style="margin-right:6px;"></i>View on GitHub</a>
```python
class RTDETR(Model):
    """Interface for Baidu's RT-DETR model, a Vision Transformer-based real-time object detector.

    This model provides real-time performance with high accuracy. It supports efficient hybrid encoding, IoU-aware query
    selection, and adaptable inference speed.

    Attributes:
        model (str): Path to the pre-trained model.

    Methods:
        task_map: Return a task map for RT-DETR, associating tasks with corresponding Ultralytics classes.

    Examples:
        Initialize RT-DETR with a pre-trained model
        >>> from ultralytics import RTDETR
        >>> model = RTDETR("rtdetr-l.pt")
        >>> results = model("image.jpg")
    """

    def __init__(self, model: str = "rtdetr-l.pt") -> None:
        """Initialize the RT-DETR model with the given pre-trained model file.

        Args:
            model (str): Path to the pre-trained model. Supports .pt, .yaml, and .yml formats.
        """
        assert TORCH_1_11, "RTDETR requires torch>=1.11"
        super().__init__(model=model, task="detect")
```
</details>

<br>

### Property `ultralytics.models.rtdetr.model.RTDETR.task_map` {#ultralytics.models.rtdetr.model.RTDETR.task\_map}

```python
def task_map(self) -> dict
```

Return a task map for RT-DETR, associating tasks with corresponding Ultralytics classes.

**Returns**

| Type | Description |
| --- | --- |
| `dict` | A dictionary mapping task names to Ultralytics task classes for the RT-DETR model. |

<details>
<summary>Source code in <code>ultralytics/models/rtdetr/model.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/models/rtdetr/model.py#L50-L63"><i class="fa-brands fa-github" aria-hidden="true" style="margin-right:6px;"></i>View on GitHub</a>
```python
@property
def task_map(self) -> dict:
    """Return a task map for RT-DETR, associating tasks with corresponding Ultralytics classes.

    Returns:
        (dict): A dictionary mapping task names to Ultralytics task classes for the RT-DETR model.
    """
    return {
        "detect": {
            "predictor": RTDETRPredictor,
            "validator": RTDETRValidator,
            "trainer": RTDETRTrainer,
            "model": RTDETRDetectionModel,
        }
    }
```
</details>

<br><br>
