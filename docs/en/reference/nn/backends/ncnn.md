---
title: nn.backends.ncnn API Reference
description: Explore NCNNBackend for Tencent NCNN inference, optimized for mobile and embedded platforms with Vulkan acceleration support.
keywords: Ultralytics, NCNNBackend, NCNN inference, Tencent NCNN, mobile inference, Vulkan acceleration, embedded AI, deep learning
---

# Reference for `ultralytics/nn/backends/ncnn.py`

!!! success "Improvements"

    This page is sourced from [https://github.com/ultralytics/ultralytics/blob/main/ultralytics/nn/backends/ncnn.py](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/nn/backends/ncnn.py). Have an improvement or example to add? Open a [Pull Request](https://docs.ultralytics.com/help/contributing) — thank you! 🙏

<br>

!!! abstract "Summary"

    === "<span class="doc-kind doc-kind-class">Classes</span>"

        - [`NCNNBackend`](#ultralytics.nn.backends.ncnn.NCNNBackend)

    === "<span class="doc-kind doc-kind-method">Methods</span>"

        - [`NCNNBackend.load_model`](#ultralytics.nn.backends.ncnn.NCNNBackend.load_model)
        - [`NCNNBackend.forward`](#ultralytics.nn.backends.ncnn.NCNNBackend.forward)


## Class `ultralytics.nn.backends.ncnn.NCNNBackend` {#ultralytics.nn.backends.ncnn.NCNNBackend}

```python
NCNNBackend()
```

**Bases:** `BaseBackend`

Tencent NCNN inference backend for mobile and embedded deployment.

Loads and runs inference with Tencent NCNN models (*_ncnn_model/ directories). Optimized for mobile platforms with optional Vulkan GPU acceleration when available.

**Methods**

| Name | Description |
| --- | --- |
| [`forward`](#ultralytics.nn.backends.ncnn.NCNNBackend.forward) | Run inference using the NCNN runtime. |
| [`load_model`](#ultralytics.nn.backends.ncnn.NCNNBackend.load_model) | Load an NCNN model from a .param/.bin file pair or model directory. |

<details>
<summary>Source code in <code>ultralytics/nn/backends/ncnn.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/nn/backends/ncnn.py#L16-L73">View on GitHub</a>
```python
class NCNNBackend(BaseBackend):
    """Tencent NCNN inference backend for mobile and embedded deployment.

    Loads and runs inference with Tencent NCNN models (*_ncnn_model/ directories). Optimized for mobile platforms with
    optional Vulkan GPU acceleration when available.
    """
```
</details>

<br>

### Method `ultralytics.nn.backends.ncnn.NCNNBackend.forward` {#ultralytics.nn.backends.ncnn.NCNNBackend.forward}

```python
def forward(self, im: torch.Tensor) -> list[np.ndarray]
```

Run inference using the NCNN runtime.

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `im` | `torch.Tensor` | Input image tensor in BCHW format, normalized to [0, 1]. | *required* |

**Returns**

| Type | Description |
| --- | --- |
| `list[np.ndarray]` | Model predictions as a list of numpy arrays, one per output layer. |

<details>
<summary>Source code in <code>ultralytics/nn/backends/ncnn.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/nn/backends/ncnn.py#L58-L73">View on GitHub</a>
```python
def forward(self, im: torch.Tensor) -> list[np.ndarray]:
    """Run inference using the NCNN runtime.

    Args:
        im (torch.Tensor): Input image tensor in BCHW format, normalized to [0, 1].

    Returns:
        (list[np.ndarray]): Model predictions as a list of numpy arrays, one per output layer.
    """
    outputs = []
    for sample in im.cpu().numpy():
        with self.net.create_extractor() as ex:
            ex.input(self.net.input_names()[0], self.pyncnn.Mat(sample))
            # Sort output names as temporary fix for pnnx issue
            outputs.append([np.array(ex.extract(x)[1]) for x in sorted(self.net.output_names())])
    return [np.stack(y) for y in zip(*outputs)]
```
</details>

<br>

### Method `ultralytics.nn.backends.ncnn.NCNNBackend.load_model` {#ultralytics.nn.backends.ncnn.NCNNBackend.load\_model}

```python
def load_model(self, weight: str | Path) -> None
```

Load an NCNN model from a .param/.bin file pair or model directory.

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `weight` | `str \| Path` | Path to the .param file or directory containing NCNN model files. | *required* |

<details>
<summary>Source code in <code>ultralytics/nn/backends/ncnn.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/nn/backends/ncnn.py#L23-L56">View on GitHub</a>
```python
def load_model(self, weight: str | Path) -> None:
    """Load an NCNN model from a .param/.bin file pair or model directory.

    Args:
        weight (str | Path): Path to the .param file or directory containing NCNN model files.
    """
    LOGGER.info(f"Loading {weight} for NCNN inference...")
    check_requirements("ncnn", cmds="--no-deps")
    import ncnn as pyncnn

    self.pyncnn = pyncnn
    self.net = pyncnn.Net()

    # Setup Vulkan if available
    if isinstance(self.device, str) and self.device.startswith("vulkan"):
        self.net.opt.use_vulkan_compute = True
        self.net.set_vulkan_device(int(self.device.split(":")[1]))
        self.device = torch.device("cpu")
    else:
        self.net.opt.use_vulkan_compute = False

    w = Path(weight)
    if not w.is_file():
        w = next(w.glob("*.param"))

    self.net.load_param(str(w))
    self.net.load_model(str(w.with_suffix(".bin")))

    # Load metadata
    metadata_file = w.parent / "metadata.yaml"
    if metadata_file.exists():
        from ultralytics.utils import YAML

        self.apply_metadata(YAML.load(metadata_file))
```
</details>

<br><br>
