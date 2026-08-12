---
title: nn.backends.base API Reference
description: Explore the BaseBackend class, the abstract foundation for all inference backends in Ultralytics, defining the interface for model loading and inference.
keywords: Ultralytics, BaseBackend, inference backend, abstract class, model loading, deep learning, neural network inference
---

# Reference for `ultralytics/nn/backends/base.py`

!!! success "Improvements"

    This page is sourced from [https://github.com/ultralytics/ultralytics/blob/main/ultralytics/nn/backends/base.py](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/nn/backends/base.py). Have an improvement or example to add? Open a [Pull Request](https://docs.ultralytics.com/help/contributing) — thank you! 🙏

<br>

!!! abstract "Summary"

    === "<span class="doc-kind doc-kind-class">Classes</span>"

        - [`BaseBackend`](#ultralytics.nn.backends.base.BaseBackend)

    === "<span class="doc-kind doc-kind-method">Methods</span>"

        - [`BaseBackend.load_model`](#ultralytics.nn.backends.base.BaseBackend.load_model)
        - [`BaseBackend.forward`](#ultralytics.nn.backends.base.BaseBackend.forward)
        - [`BaseBackend.__call__`](#ultralytics.nn.backends.base.BaseBackend.__call__)
        - [`BaseBackend.apply_metadata`](#ultralytics.nn.backends.base.BaseBackend.apply_metadata)

    === "<span class="doc-kind doc-kind-function">Functions</span>"

        - [`read_tflite_metadata`](#ultralytics.nn.backends.base.read_tflite_metadata)


## Class `ultralytics.nn.backends.base.BaseBackend` {#ultralytics.nn.backends.base.BaseBackend}

```python
BaseBackend(weight: str | torch.nn.Module, device: torch.device | str, fp16: bool = False)
```

**Bases:** `ABC`

Base class for all inference backends.

This abstract class defines the interface that all inference backends must implement. It provides common functionality for model loading, metadata processing, and device management.

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `weight` | `str \| torch.nn.Module` | Path to the model weights file or a PyTorch module instance. | *required* |
| `device` | `torch.device \| str` | Device to run inference on (e.g., 'cpu', 'cuda:0'). | *required* |
| `fp16` | `bool` | Whether to use FP16 half-precision inference. | `False` |

**Attributes**

| Name | Type | Description |
| --- | --- | --- |
| `model` |  | The underlying inference model or runtime session. |
| `device` | `torch.device` | The device to run inference on. |
| `fp16` | `bool` | Whether to use FP16 (half-precision) inference. |
| `nhwc` | `bool` | Whether the model expects NHWC input format instead of NCHW. |
| `stride` | `int` | Model stride, typically 32 for YOLO models. |
| `names` | `dict` | Dictionary mapping class indices to class names. |
| `task` | `str \| None` | The task type (detect, segment, semantic, classify, pose, obb). |
| `batch` | `int` | Batch size for inference. |
| `imgsz` | `tuple` | Input image size as (height, width). |
| `channels` | `int` | Number of input channels, typically 3 for RGB. |
| `end2end` | `bool` | Whether the model includes end-to-end NMS post-processing. |
| `dynamic` | `bool` | Whether the model supports dynamic input shapes. |
| `base_model` | `bool` | Whether the loaded model is an Ultralytics `BaseModel`, and so implements the `augment` and `embed` forward arguments. |
| `metadata` | `dict` | Model metadata dictionary containing export configuration. |

**Methods**

| Name | Description |
| --- | --- |
| [`__call__`](#ultralytics.nn.backends.base.BaseBackend.__call__) | Allow the backend instance to be called directly to perform inference, forwarding arguments to the `forward` |
| [`apply_metadata`](#ultralytics.nn.backends.base.BaseBackend.apply_metadata) | Process and apply model metadata to backend attributes. |
| [`forward`](#ultralytics.nn.backends.base.BaseBackend.forward) | Run inference on the input image tensor. |
| [`load_model`](#ultralytics.nn.backends.base.BaseBackend.load_model) | Load the model from a weights file or module instance. |

<details>
<summary>Source code in <code>ultralytics/nn/backends/base.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/nn/backends/base.py#L41-L143">View on GitHub</a>
```python
class BaseBackend(ABC):
    """Base class for all inference backends.

    This abstract class defines the interface that all inference backends must implement. It provides common
    functionality for model loading, metadata processing, and device management.

    Attributes:
        model: The underlying inference model or runtime session.
        device (torch.device): The device to run inference on.
        fp16 (bool): Whether to use FP16 (half-precision) inference.
        nhwc (bool): Whether the model expects NHWC input format instead of NCHW.
        stride (int): Model stride, typically 32 for YOLO models.
        names (dict): Dictionary mapping class indices to class names.
        task (str | None): The task type (detect, segment, semantic, classify, pose, obb).
        batch (int): Batch size for inference.
        imgsz (tuple): Input image size as (height, width).
        channels (int): Number of input channels, typically 3 for RGB.
        end2end (bool): Whether the model includes end-to-end NMS post-processing.
        dynamic (bool): Whether the model supports dynamic input shapes.
        base_model (bool): Whether the loaded model is an Ultralytics `BaseModel`, and so implements the `augment` and
            `embed` forward arguments.
        metadata (dict): Model metadata dictionary containing export configuration.
    """

    def __init__(self, weight: str | torch.nn.Module, device: torch.device | str, fp16: bool = False):
        """Initialize the base backend with common attributes and load the model.

        Args:
            weight (str | torch.nn.Module): Path to the model weights file or a PyTorch module instance.
            device (torch.device | str): Device to run inference on (e.g., 'cpu', 'cuda:0').
            fp16 (bool): Whether to use FP16 half-precision inference.
        """
        self.device = device
        self.fp16 = fp16
        self.nhwc = False
        self.stride = 32
        self.names = {}
        self.task = None
        self.batch = 1
        self.channels = 3
        self.end2end = False
        self.dynamic = False
        self.base_model = False
        self.metadata = {}
        self.model = None
        self.load_model(weight)
```
</details>

<br>

### Method `ultralytics.nn.backends.base.BaseBackend.__call__` {#ultralytics.nn.backends.base.BaseBackend.\_\_call\_\_}

```python
def __call__(self, *args, **kwargs) -> Any
```

Allow the backend instance to be called directly to perform inference, forwarding arguments to the `forward`

method.

<details>
<summary>Source code in <code>ultralytics/nn/backends/base.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/nn/backends/base.py#L109-L113">View on GitHub</a>
```python
def __call__(self, *args, **kwargs) -> Any:
    """Allow the backend instance to be called directly to perform inference, forwarding arguments to the `forward`
    method.
    """
    return self.forward(*args, **kwargs)
```
</details>

<br>

### Method `ultralytics.nn.backends.base.BaseBackend.apply_metadata` {#ultralytics.nn.backends.base.BaseBackend.apply\_metadata}

```python
def apply_metadata(self, metadata: dict | None) -> None
```

Process and apply model metadata to backend attributes.

Handles type conversions for common metadata fields (e.g., stride, batch, names) and sets them as instance attributes. Also resolves end-to-end NMS and dynamic shape settings from export args.

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `metadata` | `dict \| None` | Dictionary containing metadata key-value pairs from model export. | *required* |

<details>
<summary>Source code in <code>ultralytics/nn/backends/base.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/nn/backends/base.py#L115-L143">View on GitHub</a>
```python
def apply_metadata(self, metadata: dict | None) -> None:
    """Process and apply model metadata to backend attributes.

    Handles type conversions for common metadata fields (e.g., stride, batch, names) and sets them as
    instance attributes. Also resolves end-to-end NMS and dynamic shape settings from export args.

    Args:
        metadata (dict | None): Dictionary containing metadata key-value pairs from model export.
    """
    if not metadata:
        return

    # Store raw metadata
    self.metadata = metadata

    # Process type conversions for known fields
    for k, v in metadata.items():
        if k in {"stride", "batch", "channels"}:
            metadata[k] = int(v)
        elif k in {"imgsz", "names", "kpt_shape", "kpt_names", "args", "end2end"} and isinstance(v, str):
            metadata[k] = ast.literal_eval(v)

    # Handle models exported with end-to-end NMS
    metadata["end2end"] = metadata.get("end2end", False) or metadata.get("args", {}).get("nms", False)
    metadata["dynamic"] = metadata.get("args", {}).get("dynamic", self.dynamic)

    # Apply all metadata fields as backend attributes
    for k, v in metadata.items():
        setattr(self, k, v)
```
</details>

<br>

### Method `ultralytics.nn.backends.base.BaseBackend.forward` {#ultralytics.nn.backends.base.BaseBackend.forward}

```python
def forward(self, im: torch.Tensor) -> Any
```

Run inference on the input image tensor.

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `im` | `torch.Tensor` | Input image tensor in BCHW format, normalized to [0, 1]. | *required* |

**Returns**

| Type | Description |
| --- | --- |
| `Any` | The raw output from the model's forward pass, which may require post-processing. |

<details>
<summary>Source code in <code>ultralytics/nn/backends/base.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/nn/backends/base.py#L98-L107">View on GitHub</a>
```python
@abstractmethod
def forward(self, im: torch.Tensor) -> Any:
    """Run inference on the input image tensor.

    Args:
        im (torch.Tensor): Input image tensor in BCHW format, normalized to [0, 1].

    Returns:
        (Any): The raw output from the model's forward pass, which may require post-processing.
    """
    raise NotImplementedError
```
</details>

<br>

### Method `ultralytics.nn.backends.base.BaseBackend.load_model` {#ultralytics.nn.backends.base.BaseBackend.load\_model}

```python
def load_model(self, weight: str | torch.nn.Module) -> None
```

Load the model from a weights file or module instance.

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `weight` | `str \| torch.nn.Module` | Path to model weights or a PyTorch module. | *required* |

<details>
<summary>Source code in <code>ultralytics/nn/backends/base.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/nn/backends/base.py#L89-L95">View on GitHub</a>
```python
@abstractmethod
def load_model(self, weight: str | torch.nn.Module) -> None:
    """Load the model from a weights file or module instance.

    Args:
        weight (str | torch.nn.Module): Path to model weights or a PyTorch module.
    """
    raise NotImplementedError
```
</details>


<br><br><hr><br>

## Function `ultralytics.nn.backends.base.read_tflite_metadata` {#ultralytics.nn.backends.base.read\_tflite\_metadata}

```python
def read_tflite_metadata(file: str | Path) -> dict | None
```

Read Ultralytics metadata embedded in a ``.tflite`` file.

Ultralytics appends metadata to the end of ``.tflite`` flatbuffers as a zip entry (``metadata.json`` for litert-torch/single-file exports, or a single literal-dict entry for legacy onnx2tf exports). Returns the parsed metadata dict, or ``None`` if the file has no readable embedded metadata.

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `file` | `str \| Path` | Path to the ``.tflite`` model file. | *required* |

**Returns**

| Type | Description |
| --- | --- |
| `dict \| None` | Parsed metadata dictionary, or ``None`` if absent or unreadable. |

<details>
<summary>Source code in <code>ultralytics/nn/backends/base.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/nn/backends/base.py#L13-L38">View on GitHub</a>
```python
def read_tflite_metadata(file: str | Path) -> dict | None:
    """Read Ultralytics metadata embedded in a ``.tflite`` file.

    Ultralytics appends metadata to the end of ``.tflite`` flatbuffers as a zip entry (``metadata.json`` for
    litert-torch/single-file exports, or a single literal-dict entry for legacy onnx2tf exports). Returns the parsed
    metadata dict, or ``None`` if the file has no readable embedded metadata.

    Args:
        file (str | Path): Path to the ``.tflite`` model file.

    Returns:
        (dict | None): Parsed metadata dictionary, or ``None`` if absent or unreadable.
    """
    import json
    import zipfile

    try:
        with zipfile.ZipFile(file, "r") as zf:
            names = zf.namelist()
            if "metadata.json" in names:
                return json.loads(zf.read("metadata.json"))
            if names:  # legacy onnx2tf exports store a single Python-literal dict entry
                return ast.literal_eval(zf.read(names[0]).decode("utf-8"))
    except (zipfile.BadZipFile, SyntaxError, ValueError, KeyError, json.JSONDecodeError):
        return None
    return None
```
</details>

<br><br>
