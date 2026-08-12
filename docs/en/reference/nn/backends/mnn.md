---
title: nn.backends.mnn API Reference
description: Explore MNNBackend for Alibaba MNN inference, enabling lightweight and efficient model deployment on mobile and edge devices.
keywords: Ultralytics, MNNBackend, MNN inference, Alibaba MNN, mobile inference, edge AI, .mnn models, deep learning
---

# Reference for `ultralytics/nn/backends/mnn.py`

!!! success "Improvements"

    This page is sourced from [https://github.com/ultralytics/ultralytics/blob/main/ultralytics/nn/backends/mnn.py](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/nn/backends/mnn.py). Have an improvement or example to add? Open a [Pull Request](https://docs.ultralytics.com/help/contributing) — thank you! 🙏

<br>

!!! abstract "Summary"

    === "<span class="doc-kind doc-kind-class">Classes</span>"

        - [`MNNBackend`](#ultralytics.nn.backends.mnn.MNNBackend)

    === "<span class="doc-kind doc-kind-method">Methods</span>"

        - [`MNNBackend.load_model`](#ultralytics.nn.backends.mnn.MNNBackend.load_model)
        - [`MNNBackend.forward`](#ultralytics.nn.backends.mnn.MNNBackend.forward)


## Class `ultralytics.nn.backends.mnn.MNNBackend` {#ultralytics.nn.backends.mnn.MNNBackend}

```python
MNNBackend()
```

**Bases:** `BaseBackend`

MNN (Mobile Neural Network) inference backend.

Loads and runs inference with MNN models (.mnn files) using the Alibaba MNN framework. Optimized for mobile and edge deployment with configurable thread count and precision.

**Methods**

| Name | Description |
| --- | --- |
| [`forward`](#ultralytics.nn.backends.mnn.MNNBackend.forward) | Run inference using the MNN runtime. |
| [`load_model`](#ultralytics.nn.backends.mnn.MNNBackend.load_model) | Load an Alibaba MNN model from a .mnn file. |

<details>
<summary>Source code in <code>ultralytics/nn/backends/mnn.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/nn/backends/mnn.py#L18-L64">View on GitHub</a>
```python
class MNNBackend(BaseBackend):
    """MNN (Mobile Neural Network) inference backend.

    Loads and runs inference with MNN models (.mnn files) using the Alibaba MNN framework. Optimized for mobile and edge
    deployment with configurable thread count and precision.
    """
```
</details>

<br>

### Method `ultralytics.nn.backends.mnn.MNNBackend.forward` {#ultralytics.nn.backends.mnn.MNNBackend.forward}

```python
def forward(self, im: torch.Tensor) -> list
```

Run inference using the MNN runtime.

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `im` | `torch.Tensor` | Input image tensor in BCHW format, normalized to [0, 1]. | *required* |

**Returns**

| Type | Description |
| --- | --- |
| `list` | Model predictions as a list of numpy arrays. |

<details>
<summary>Source code in <code>ultralytics/nn/backends/mnn.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/nn/backends/mnn.py#L48-L64">View on GitHub</a>
```python
def forward(self, im: torch.Tensor) -> list:
    """Run inference using the MNN runtime.

    Args:
        im (torch.Tensor): Input image tensor in BCHW format, normalized to [0, 1].

    Returns:
        (list): Model predictions as a list of numpy arrays.
    """
    input_var = self.expr.const(im.data_ptr(), im.shape)
    output_var = self.net.onForward([input_var])
    # NOTE: need this copy(), or it'd get incorrect results on ARM devices
    if output_var:
        return [x.read().copy() for x in output_var]
    if self.metadata.get("args", {}).get("nms") and self.task in {"detect", "pose"}:
        return [np.empty((im.shape[0], 0, 6))]
    raise RuntimeError("Alibaba MNN inference returned no output tensors.")
```
</details>

<br>

### Method `ultralytics.nn.backends.mnn.MNNBackend.load_model` {#ultralytics.nn.backends.mnn.MNNBackend.load\_model}

```python
def load_model(self, weight: str | Path) -> None
```

Load an Alibaba MNN model from a .mnn file.

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `weight` | `str \| Path` | Path to the .mnn model file. | *required* |

<details>
<summary>Source code in <code>ultralytics/nn/backends/mnn.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/nn/backends/mnn.py#L25-L46">View on GitHub</a>
```python
def load_model(self, weight: str | Path) -> None:
    """Load an Alibaba MNN model from a .mnn file.

    Args:
        weight (str | Path): Path to the .mnn model file.
    """
    LOGGER.info(f"Loading {weight} for MNN inference...")
    check_requirements("MNN")
    import MNN

    config = {"precision": "low", "backend": "CPU", "numThread": (os.cpu_count() + 1) // 2}
    rt = MNN.nn.create_runtime_manager((config,))
    self.net = MNN.nn.load_module_from_file(weight, [], [], runtime_manager=rt, rearrange=True)
    self.expr = MNN.expr

    # Load metadata from bizCode
    info = self.net.get_info()
    if "bizCode" in info:
        try:
            self.apply_metadata(json.loads(info["bizCode"]))
        except json.JSONDecodeError:
            pass
```
</details>

<br><br>
