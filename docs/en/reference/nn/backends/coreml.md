---
title: nn.backends.coreml API Reference
description: Explore CoreMLBackend for Apple CoreML inference, enabling efficient YOLO model deployment on iOS, macOS, and Apple Silicon devices.
keywords: Ultralytics, CoreMLBackend, CoreML inference, Apple CoreML, iOS deployment, macOS inference, Apple Silicon, mobile AI
---

# Reference for `ultralytics/nn/backends/coreml.py`

!!! success "Improvements"

    This page is sourced from [https://github.com/ultralytics/ultralytics/blob/main/ultralytics/nn/backends/coreml.py](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/nn/backends/coreml.py). Have an improvement or example to add? Open a [Pull Request](https://docs.ultralytics.com/help/contributing) — thank you! 🙏

<br>

!!! abstract "Summary"

    === "<span class="doc-kind doc-kind-class">Classes</span>"

        - [`CoreMLBackend`](#ultralytics.nn.backends.coreml.CoreMLBackend)

    === "<span class="doc-kind doc-kind-method">Methods</span>"

        - [`CoreMLBackend.load_model`](#ultralytics.nn.backends.coreml.CoreMLBackend.load_model)
        - [`CoreMLBackend.forward`](#ultralytics.nn.backends.coreml.CoreMLBackend.forward)


## Class `ultralytics.nn.backends.coreml.CoreMLBackend` {#ultralytics.nn.backends.coreml.CoreMLBackend}

```python
CoreMLBackend()
```

**Bases:** `BaseBackend`

CoreML inference backend for Apple hardware.

Loads and runs inference with CoreML models (.mlpackage files) using the coremltools library. Supports both static and dynamic input shapes and handles NMS-included model outputs.

**Methods**

| Name | Description |
| --- | --- |
| [`forward`](#ultralytics.nn.backends.coreml.CoreMLBackend.forward) | Run CoreML inference with automatic input format handling. |
| [`load_model`](#ultralytics.nn.backends.coreml.CoreMLBackend.load_model) | Load a CoreML model from a .mlpackage file. |

<details>
<summary>Source code in <code>ultralytics/nn/backends/coreml.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/nn/backends/coreml.py#L17-L75">View on GitHub</a>
```python
class CoreMLBackend(BaseBackend):
    """CoreML inference backend for Apple hardware.

    Loads and runs inference with CoreML models (.mlpackage files) using the coremltools library. Supports both static
    and dynamic input shapes and handles NMS-included model outputs.
    """
```
</details>

<br>

### Method `ultralytics.nn.backends.coreml.CoreMLBackend.forward` {#ultralytics.nn.backends.coreml.CoreMLBackend.forward}

```python
def forward(self, im: torch.Tensor) -> np.ndarray | list[np.ndarray]
```

Run CoreML inference with automatic input format handling.

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `im` | `torch.Tensor` | Input image tensor in BHWC format (converted from BCHW by AutoBackend). | *required* |

**Returns**

| Type | Description |
| --- | --- |
| `np.ndarray \| list[np.ndarray]` | Model predictions as numpy array(s). |

<details>
<summary>Source code in <code>ultralytics/nn/backends/coreml.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/nn/backends/coreml.py#L51-L75">View on GitHub</a>
```python
def forward(self, im: torch.Tensor) -> np.ndarray | list[np.ndarray]:
    """Run CoreML inference with automatic input format handling.

    Args:
        im (torch.Tensor): Input image tensor in BHWC format (converted from BCHW by AutoBackend).

    Returns:
        (np.ndarray | list[np.ndarray]): Model predictions as numpy array(s).
    """
    im = im.cpu().numpy()
    h, w = im.shape[1:3]

    im = im.transpose(0, 3, 1, 2) if self.dynamic else Image.fromarray((im[0] * 255).astype("uint8"))
    y = self.model.predict({self.input_name: im})
    if "confidence" in y:  # NMS included
        from ultralytics.utils.ops import xywh2xyxy

        box = xywh2xyxy(y["coordinates"] * [[w, h, w, h]])
        cls = y["confidence"].argmax(1, keepdims=True)
        y = np.concatenate((box, np.take_along_axis(y["confidence"], cls, axis=1), cls), 1)[None]
    else:
        y = list(y.values())
    if len(y) == 2 and len(y[1].shape) != 4:  # segmentation model
        y = list(reversed(y))
    return y
```
</details>

<br>

### Method `ultralytics.nn.backends.coreml.CoreMLBackend.load_model` {#ultralytics.nn.backends.coreml.CoreMLBackend.load\_model}

```python
def load_model(self, weight: str | Path) -> None
```

Load a CoreML model from a .mlpackage file.

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `weight` | `str \| Path` | Path to the .mlpackage model file. | *required* |

<details>
<summary>Source code in <code>ultralytics/nn/backends/coreml.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/nn/backends/coreml.py#L24-L49">View on GitHub</a>
```python
def load_model(self, weight: str | Path) -> None:
    """Load a CoreML model from a .mlpackage file.

    Args:
        weight (str | Path): Path to the .mlpackage model file.
    """
    check_requirements(["coremltools>=9.0", "numpy>=1.14.5,<=2.3.5"])
    import coremltools as ct

    LOGGER.info(f"Loading {weight} for CoreML inference...")
    # Run on the Neural Engine (CPU_AND_NE): ~3x faster than CPU, and the default ComputeUnit.ALL / CPU_AND_GPU
    # abort the process via an MPSGraph compiler bug on macOS hosts (coremltools 9.x). CPU_AND_NE needs macOS >= 13,
    # so fall back to CPU_ONLY below that. CoreML inference is macOS-only, so this applies wherever the backend runs.
    # Exception: RT-DETR loses FP16 accuracy and runs slower on the Neural Engine alone, so route it through ALL.
    meta = dict(ct.utils.load_spec(str(weight)).description.metadata.userDefined)
    default_unit = ct.ComputeUnit.ALL if meta.get("head") == "RTDETRDecoder" else ct.ComputeUnit.CPU_AND_NE
    try:
        self.model = ct.models.MLModel(weight, compute_units=default_unit)
    except Exception:
        self.model = ct.models.MLModel(weight, compute_units=ct.ComputeUnit.CPU_ONLY)
    spec = self.model.get_spec()
    self.input_name = spec.description.input[0].name
    self.dynamic = spec.description.input[0].type.HasField("multiArrayType")

    # Load metadata
    self.apply_metadata(dict(self.model.user_defined_metadata))
```
</details>

<br><br>
