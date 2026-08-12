---
title: nn.backends.deepx API Reference
description: Reference for the Ultralytics DeepXBackend. Learn how to run inference with DEEPX NPU compiled models using the DX-Runtime.
keywords: DeepXBackend, DEEPX, NPU, inference backend, dx_engine, Ultralytics, YOLO, edge AI
---

# Reference for `ultralytics/nn/backends/deepx.py`

!!! success "Improvements"

    This page is sourced from [https://github.com/ultralytics/ultralytics/blob/main/ultralytics/nn/backends/deepx.py](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/nn/backends/deepx.py). Have an improvement or example to add? Open a [Pull Request](https://docs.ultralytics.com/help/contributing) — thank you! 🙏

<br>

!!! abstract "Summary"

    === "<span class="doc-kind doc-kind-class">Classes</span>"

        - [`DeepXBackend`](#ultralytics.nn.backends.deepx.DeepXBackend)

    === "<span class="doc-kind doc-kind-method">Methods</span>"

        - [`DeepXBackend.load_model`](#ultralytics.nn.backends.deepx.DeepXBackend.load_model)
        - [`DeepXBackend.forward`](#ultralytics.nn.backends.deepx.DeepXBackend.forward)


## Class `ultralytics.nn.backends.deepx.DeepXBackend` {#ultralytics.nn.backends.deepx.DeepXBackend}

```python
DeepXBackend()
```

**Bases:** `BaseBackend`

DEEPX NPU inference backend for DEEPX hardware accelerators.

Loads compiled DEEPX models (.dxnn files) and runs inference using the DEEPX DX-Runtime.

**Methods**

| Name | Description |
| --- | --- |
| [`forward`](#ultralytics.nn.backends.deepx.DeepXBackend.forward) | Run inference on the DEEPX NPU. |
| [`load_model`](#ultralytics.nn.backends.deepx.DeepXBackend.load_model) | Load a DEEPX model from a directory containing a .dxnn file. |

<details>
<summary>Source code in <code>ultralytics/nn/backends/deepx.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/nn/backends/deepx.py#L15-L75">View on GitHub</a>
```python
class DeepXBackend(BaseBackend):
    """DEEPX NPU inference backend for DEEPX hardware accelerators.

    Loads compiled DEEPX models (.dxnn files) and runs inference using the DEEPX DX-Runtime.
    """
```
</details>

<br>

### Method `ultralytics.nn.backends.deepx.DeepXBackend.forward` {#ultralytics.nn.backends.deepx.DeepXBackend.forward}

```python
def forward(self, im: torch.Tensor) -> np.ndarray | list[np.ndarray]
```

Run inference on the DEEPX NPU.

Converts each image from BCHW float [0, 1] to HWC uint8 [0, 255] per the DEEPX runtime contract, runs the engine per image, then stacks outputs along the batch dimension.

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `im` | `torch.Tensor` | Input image tensor in BCHW format, normalized to [0, 1]. | *required* |

**Returns**

| Type | Description |
| --- | --- |
| `np.ndarray \| list[np.ndarray]` | Model predictions as a single array or list of arrays. |

<details>
<summary>Source code in <code>ultralytics/nn/backends/deepx.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/nn/backends/deepx.py#L55-L75">View on GitHub</a>
```python
def forward(self, im: torch.Tensor) -> np.ndarray | list[np.ndarray]:
    """Run inference on the DEEPX NPU.

    Converts each image from BCHW float [0, 1] to HWC uint8 [0, 255] per the DEEPX runtime contract,
    runs the engine per image, then stacks outputs along the batch dimension.

    Args:
        im (torch.Tensor): Input image tensor in BCHW format, normalized to [0, 1].

    Returns:
        (np.ndarray | list[np.ndarray]): Model predictions as a single array or list of arrays.
    """
    outputs = []
    for sample in im.cpu().numpy():
        sample = np.ascontiguousarray(np.clip(np.transpose(sample, (1, 2, 0)) * 255, 0, 255).astype(np.uint8))
        for i, out in enumerate(map(np.asarray, self.model.run([sample]))):
            if i == len(outputs):
                outputs.append([])
            outputs[i].append(out if out.ndim and out.shape[0] == 1 else out[None])
    y = [np.concatenate(x, axis=0) for x in outputs]
    return y[0] if len(y) == 1 else y
```
</details>

<br>

### Method `ultralytics.nn.backends.deepx.DeepXBackend.load_model` {#ultralytics.nn.backends.deepx.DeepXBackend.load\_model}

```python
def load_model(self, weight: str | Path) -> None
```

Load a DEEPX model from a directory containing a .dxnn file.

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `weight` | `str \| Path` | Path to the DEEPX model directory containing the .dxnn binary. | *required* |

**Raises**

| Type | Description |
| --- | --- |
| `ImportError` | If the ``dx_engine`` Python package is not installed. |
| `FileNotFoundError` | If no .dxnn file is found in the given directory. |

<details>
<summary>Source code in <code>ultralytics/nn/backends/deepx.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/nn/backends/deepx.py#L21-L53">View on GitHub</a>
```python
def load_model(self, weight: str | Path) -> None:
    """Load a DEEPX model from a directory containing a .dxnn file.

    Args:
        weight (str | Path): Path to the DEEPX model directory containing the .dxnn binary.

    Raises:
        ImportError: If the ``dx_engine`` Python package is not installed.
        FileNotFoundError: If no .dxnn file is found in the given directory.
    """
    try:
        from dx_engine import InferenceEngine
    except ImportError as e:
        raise ImportError(
            "DEEPX inference requires the DEEPX DX-Runtime and `dx_engine` Python package. "
            "See https://docs.ultralytics.com/integrations/deepx/#runtime-installation for installation instructions."
        ) from e

    LOGGER.info(f"Loading {weight} for DEEPX inference...")

    w = Path(weight)
    found = next(w.rglob("*.dxnn"), None)
    if found is None:
        raise FileNotFoundError(f"No .dxnn file found in: {w}")

    self.model = InferenceEngine(str(found))

    # Load metadata
    metadata_file = found.parent / "metadata.yaml"
    if metadata_file.exists():
        from ultralytics.utils import YAML

        self.apply_metadata(YAML.load(metadata_file))
```
</details>

<br><br>
