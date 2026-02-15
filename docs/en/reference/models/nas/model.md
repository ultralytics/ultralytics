---
description: Explore the YOLO-NAS model interface and learn how to utilize pretrained YOLO-NAS models for object detection with Ultralytics.
keywords: Ultralytics, YOLO, YOLO-NAS, object detection, pretrained models, machine learning, deep learning, NAS model
---

# Reference for `ultralytics/models/nas/model.py`

!!! success "Improvements"

    This page is sourced from [https://github.com/ultralytics/ultralytics/blob/main/ultralytics/models/nas/model.py](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/models/nas/model.py). Have an improvement or example to add? Open a [Pull Request](https://docs.ultralytics.com/help/contributing/) — thank you! 🙏

<br>

!!! abstract "Summary"

    === "<span class="doc-kind doc-kind-class">Classes</span>"

        - [`NAS`](#ultralytics.models.nas.model.NAS)

    === "<span class="doc-kind doc-kind-property">Properties</span>"

        - [`NAS.task_map`](#ultralytics.models.nas.model.NAS.task_map)

    === "<span class="doc-kind doc-kind-method">Methods</span>"

        - [`NAS._load`](#ultralytics.models.nas.model.NAS._load)
        - [`NAS.info`](#ultralytics.models.nas.model.NAS.info)


## Class `ultralytics.models.nas.model.NAS` {#ultralytics.models.nas.model.NAS}

```python
NAS(self, model: str = "yolo_nas_s.pt") -> None
```

**Bases:** `Model`

YOLO-NAS model for object detection.

This class provides an interface for the YOLO-NAS models and extends the `Model` class from Ultralytics engine. It is designed to facilitate the task of object detection using pre-trained or custom-trained YOLO-NAS models.

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `model` | `str` |  | `"yolo_nas_s.pt"` |

**Attributes**

| Name | Type | Description |
| --- | --- | --- |
| `model` | `torch.nn.Module` | The loaded YOLO-NAS model. |
| `task` | `str` | The task type for the model, defaults to 'detect'. |
| `predictor` | `NASPredictor` | The predictor instance for making predictions. |
| `validator` | `NASValidator` | The validator instance for model validation. |

**Methods**

| Name | Description |
| --- | --- |
| [`task_map`](#ultralytics.models.nas.model.NAS.task_map) | Return a dictionary mapping tasks to respective predictor and validator classes. |
| [`_load`](#ultralytics.models.nas.model.NAS._load) | Load an existing NAS model weights or create a new NAS model with pretrained weights. |
| [`info`](#ultralytics.models.nas.model.NAS.info) | Log model information. |

**Examples**

```python
>>> from ultralytics import NAS
>>> model = NAS("yolo_nas_s")
>>> results = model.predict("ultralytics/assets/bus.jpg")
```

!!! note "Notes"

    YOLO-NAS models only support pre-trained models. Do not provide YAML configuration files.

<details>
<summary>Source code in <code>ultralytics/models/nas/model.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/models/nas/model.py#L20-L98"><i class="fa-brands fa-github" aria-hidden="true" style="margin-right:6px;"></i>View on GitHub</a>
```python
class NAS(Model):
    """YOLO-NAS model for object detection.

    This class provides an interface for the YOLO-NAS models and extends the `Model` class from Ultralytics engine. It
    is designed to facilitate the task of object detection using pre-trained or custom-trained YOLO-NAS models.

    Attributes:
        model (torch.nn.Module): The loaded YOLO-NAS model.
        task (str): The task type for the model, defaults to 'detect'.
        predictor (NASPredictor): The predictor instance for making predictions.
        validator (NASValidator): The validator instance for model validation.

    Methods:
        info: Log model information and return model details.

    Examples:
        >>> from ultralytics import NAS
        >>> model = NAS("yolo_nas_s")
        >>> results = model.predict("ultralytics/assets/bus.jpg")

    Notes:
        YOLO-NAS models only support pre-trained models. Do not provide YAML configuration files.
    """

    def __init__(self, model: str = "yolo_nas_s.pt") -> None:
        """Initialize the NAS model with the provided or default model."""
        assert Path(model).suffix not in {".yaml", ".yml"}, "YOLO-NAS models only support pre-trained models."
        super().__init__(model, task="detect")
```
</details>

<br>

### Property `ultralytics.models.nas.model.NAS.task_map` {#ultralytics.models.nas.model.NAS.task\_map}

```python
def task_map(self) -> dict[str, dict[str, Any]]
```

Return a dictionary mapping tasks to respective predictor and validator classes.

<details>
<summary>Source code in <code>ultralytics/models/nas/model.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/models/nas/model.py#L96-L98"><i class="fa-brands fa-github" aria-hidden="true" style="margin-right:6px;"></i>View on GitHub</a>
```python
@property
def task_map(self) -> dict[str, dict[str, Any]]:
    """Return a dictionary mapping tasks to respective predictor and validator classes."""
    return {"detect": {"predictor": NASPredictor, "validator": NASValidator}}
```
</details>

<br>

### Method `ultralytics.models.nas.model.NAS._load` {#ultralytics.models.nas.model.NAS.\_load}

```python
def _load(self, weights: str, task = None) -> None
```

Load an existing NAS model weights or create a new NAS model with pretrained weights.

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `weights` | `str` | Path to the model weights file or model name. | *required* |
| `task` | `str, optional` | Task type for the model. | `None` |

<details>
<summary>Source code in <code>ultralytics/models/nas/model.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/models/nas/model.py#L49-L81"><i class="fa-brands fa-github" aria-hidden="true" style="margin-right:6px;"></i>View on GitHub</a>
```python
def _load(self, weights: str, task=None) -> None:
    """Load an existing NAS model weights or create a new NAS model with pretrained weights.

    Args:
        weights (str): Path to the model weights file or model name.
        task (str, optional): Task type for the model.
    """
    import super_gradients

    suffix = Path(weights).suffix
    if suffix == ".pt":
        self.model = torch_load(attempt_download_asset(weights))
    elif suffix == "":
        self.model = super_gradients.training.models.get(weights, pretrained_weights="coco")

    # Override the forward method to ignore additional arguments
    def new_forward(x, *args, **kwargs):
        """Ignore additional __call__ arguments."""
        return self.model._original_forward(x)

    self.model._original_forward = self.model.forward
    self.model.forward = new_forward

    # Standardize model attributes for compatibility
    self.model.fuse = lambda verbose=True: self.model
    self.model.stride = torch.tensor([32])
    self.model.names = dict(enumerate(self.model._class_names))
    self.model.is_fused = lambda: False  # for info()
    self.model.yaml = {}  # for info()
    self.model.pt_path = weights  # for export()
    self.model.task = "detect"  # for export()
    self.model.args = {**DEFAULT_CFG_DICT, **self.overrides}  # for export()
    self.model.eval()
```
</details>

<br>

### Method `ultralytics.models.nas.model.NAS.info` {#ultralytics.models.nas.model.NAS.info}

```python
def info(self, detailed: bool = False, verbose: bool = True) -> dict[str, Any]
```

Log model information.

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `detailed` | `bool` | Show detailed information about model. | `False` |
| `verbose` | `bool` | Controls verbosity. | `True` |

**Returns**

| Type | Description |
| --- | --- |
| `tuple` | Model information as a tuple of (layers, parameters, gradients, GFLOPs). |

<details>
<summary>Source code in <code>ultralytics/models/nas/model.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/models/nas/model.py#L83-L93"><i class="fa-brands fa-github" aria-hidden="true" style="margin-right:6px;"></i>View on GitHub</a>
```python
def info(self, detailed: bool = False, verbose: bool = True) -> dict[str, Any]:
    """Log model information.

    Args:
        detailed (bool): Show detailed information about model.
        verbose (bool): Controls verbosity.

    Returns:
        (tuple): Model information as a tuple of (layers, parameters, gradients, GFLOPs).
    """
    return model_info(self.model, detailed=detailed, verbose=verbose, imgsz=640)
```
</details>

<br><br>
