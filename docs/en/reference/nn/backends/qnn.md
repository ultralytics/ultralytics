---
title: nn.backends.qnn API Reference
description: Reference for the Ultralytics QNNBackend. Learn how to run inference with Qualcomm QNN context-binary models using ONNX Runtime's QNN Execution Provider.
keywords: QNNBackend, Qualcomm QNN, onnxruntime-qnn, QNN Execution Provider, Snapdragon, Hexagon NPU, inference backend, Ultralytics, YOLO, edge AI
---

# Reference for `ultralytics/nn/backends/qnn.py`

!!! success "Improvements"

    This page is sourced from [https://github.com/ultralytics/ultralytics/blob/main/ultralytics/nn/backends/qnn.py](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/nn/backends/qnn.py). Have an improvement or example to add? Open a [Pull Request](https://docs.ultralytics.com/help/contributing) — thank you! 🙏

<br>

!!! abstract "Summary"

    === "<span class="doc-kind doc-kind-class">Classes</span>"

        - [`QNNBackend`](#ultralytics.nn.backends.qnn.QNNBackend)

    === "<span class="doc-kind doc-kind-method">Methods</span>"

        - [`QNNBackend.load_model`](#ultralytics.nn.backends.qnn.QNNBackend.load_model)
        - [`QNNBackend.forward`](#ultralytics.nn.backends.qnn.QNNBackend.forward)


## Class `ultralytics.nn.backends.qnn.QNNBackend` {#ultralytics.nn.backends.qnn.QNNBackend}

```python
QNNBackend()
```

**Bases:** `BaseBackend`

Qualcomm QNN inference backend for Snapdragon hardware.

Loads and runs the QNN context binary produced by the Ultralytics QNN export (`*_qnn.onnx`) using ONNX Runtime with the QNN Execution Provider plugin (`onnxruntime-qnn`). Inference runs on Qualcomm Snapdragon devices (Android, Windows on Snapdragon, or Qualcomm Linux boards) via the HTP (NPU) backend.

**Methods**

| Name | Description |
| --- | --- |
| [`forward`](#ultralytics.nn.backends.qnn.QNNBackend.forward) | Run inference on the Qualcomm QNN runtime. |
| [`load_model`](#ultralytics.nn.backends.qnn.QNNBackend.load_model) | Load a QNN context-binary model with ONNX Runtime's QNN Execution Provider plugin. |

<details>
<summary>Source code in <code>ultralytics/nn/backends/qnn.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/nn/backends/qnn.py#L15-L79">View on GitHub</a>
```python
class QNNBackend(BaseBackend):
    """Qualcomm QNN inference backend for Snapdragon hardware.

    Loads and runs the QNN context binary produced by the Ultralytics QNN export (`*_qnn.onnx`) using ONNX Runtime with
    the QNN Execution Provider plugin (`onnxruntime-qnn`). Inference runs on Qualcomm Snapdragon devices (Android,
    Windows on Snapdragon, or Qualcomm Linux boards) via the HTP (NPU) backend.
    """
```
</details>

<br>

### Method `ultralytics.nn.backends.qnn.QNNBackend.forward` {#ultralytics.nn.backends.qnn.QNNBackend.forward}

```python
def forward(self, im: torch.Tensor) -> list
```

Run inference on the Qualcomm QNN runtime.

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `im` | `torch.Tensor` | Input image tensor in BCHW format, normalized to [0, 1]. | *required* |

**Returns**

| Type | Description |
| --- | --- |
| `list` | Model predictions as a list of output arrays. |

<details>
<summary>Source code in <code>ultralytics/nn/backends/qnn.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/nn/backends/qnn.py#L68-L79">View on GitHub</a>
```python
def forward(self, im: torch.Tensor) -> list:
    """Run inference on the Qualcomm QNN runtime.

    Args:
        im (torch.Tensor): Input image tensor in BCHW format, normalized to [0, 1].

    Returns:
        (list): Model predictions as a list of output arrays.
    """
    if self.nhwc:
        im = im.permute(0, 2, 3, 1)
    return self.session.run(self.output_names, {self.session.get_inputs()[0].name: im.cpu().numpy()})
```
</details>

<br>

### Method `ultralytics.nn.backends.qnn.QNNBackend.load_model` {#ultralytics.nn.backends.qnn.QNNBackend.load\_model}

```python
def load_model(self, weight: str | Path) -> None
```

Load a QNN context-binary model with ONNX Runtime's QNN Execution Provider plugin.

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `weight` | `str \| Path` | Path to the `*_qnn.onnx` file. | *required* |

**Raises**

| Type | Description |
| --- | --- |
| `OSError` | If the QNN Execution Provider cannot be registered (e.g. not on Snapdragon hardware). |

<details>
<summary>Source code in <code>ultralytics/nn/backends/qnn.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/nn/backends/qnn.py#L23-L66">View on GitHub</a>
```python
def load_model(self, weight: str | Path) -> None:
    """Load a QNN context-binary model with ONNX Runtime's QNN Execution Provider plugin.

    Args:
        weight (str | Path): Path to the `*_qnn.onnx` file.

    Raises:
        OSError: If the QNN Execution Provider cannot be registered (e.g. not on Snapdragon hardware).
    """
    check_requirements("onnxruntime-qnn")
    import onnxruntime

    from ultralytics.utils.export.qnn import qnn_library_paths

    onnx_file = Path(weight)
    LOGGER.info(f"Loading {onnx_file} for Qualcomm QNN inference...")

    # Register the QNN EP (libraries resolved from the plugin helper or the onnxruntime/capi bundle) and select
    # it; ep_library is None when QNN is already built into ONNX Runtime and needs no plugin registration
    ep_name = "QNNExecutionProvider"
    ep_library, htp_backend = qnn_library_paths()
    ep_options = {"backend_path": htp_backend}
    options = onnxruntime.SessionOptions()
    if ep_library:
        onnxruntime.register_execution_provider_library(ep_name, ep_library)
        devices = [d for d in onnxruntime.get_ep_devices() if d.ep_name == ep_name]
        if not devices:
            raise OSError(
                "QNN Execution Provider registered but no QNN devices were found. Run on a Qualcomm Snapdragon "
                "device with 'onnxruntime-qnn' installed."
            )
        options.add_provider_for_devices(devices, ep_options)
        self.session = onnxruntime.InferenceSession(str(onnx_file), sess_options=options)
    else:
        self.session = onnxruntime.InferenceSession(
            str(onnx_file), sess_options=options, providers=[ep_name], provider_options=[ep_options]
        )
    self.output_names = [x.name for x in self.session.get_outputs()]
    shape = self.session.get_inputs()[0].shape  # channel-last exports take [N, H, W, C] input
    self.nhwc = len(shape) == 4 and shape[3] in {1, 3} and shape[1] not in {1, 3}

    metadata_map = self.session.get_modelmeta().custom_metadata_map
    if metadata_map:
        self.apply_metadata(dict(metadata_map))
```
</details>

<br><br>
