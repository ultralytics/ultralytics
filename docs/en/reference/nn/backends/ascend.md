---
title: nn.backends.ascend API Reference
description: Reference for `ultralytics.nn.backends.ascend` in the Ultralytics package.
keywords: Ultralytics, ultralytics.nn.backends.ascend, API reference, YOLO, Python
---

# Reference for `ultralytics/nn/backends/ascend.py`

!!! success "Improvements"

    This page is sourced from [https://github.com/ultralytics/ultralytics/blob/main/ultralytics/nn/backends/ascend.py](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/nn/backends/ascend.py). Have an improvement or example to add? Open a [Pull Request](https://docs.ultralytics.com/help/contributing) — thank you! 🙏

<br>

!!! abstract "Summary"

    === "<span class="doc-kind doc-kind-class">Classes</span>"

        - [`AscendBackend`](#ultralytics.nn.backends.ascend.AscendBackend)

    === "<span class="doc-kind doc-kind-method">Methods</span>"

        - [`AscendBackend.load_model`](#ultralytics.nn.backends.ascend.AscendBackend.load_model)
        - [`AscendBackend.__del__`](#ultralytics.nn.backends.ascend.AscendBackend.__del__)
        - [`AscendBackend.forward`](#ultralytics.nn.backends.ascend.AscendBackend.forward)


## Class `ultralytics.nn.backends.ascend.AscendBackend` {#ultralytics.nn.backends.ascend.AscendBackend}

```python
AscendBackend()
```

**Bases:** `BaseBackend`

Huawei Ascend NPU inference backend for CANN offline models.

Loads a compiled .om offline model and runs inference on the Ascend AI Processor through the ais_bench runtime, which wraps CANN's pyACL bindings.

**Methods**

| Name | Description |
| --- | --- |
| [`__del__`](#ultralytics.nn.backends.ascend.AscendBackend.__del__) | Release the Ascend device-side resources held by the inference session. |
| [`forward`](#ultralytics.nn.backends.ascend.AscendBackend.forward) | Run inference on the Ascend NPU. |
| [`load_model`](#ultralytics.nn.backends.ascend.AscendBackend.load_model) | Load an Ascend model from a directory containing a .om file. |

<details>
<summary>Source code in <code>ultralytics/nn/backends/ascend.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/nn/backends/ascend.py#L15-L69">View on GitHub</a>
```python
class AscendBackend(BaseBackend):
    """Huawei Ascend NPU inference backend for CANN offline models.

    Loads a compiled .om offline model and runs inference on the Ascend AI Processor through the ais_bench runtime,
    which wraps CANN's pyACL bindings.
    """
```
</details>

<br>

### Method `ultralytics.nn.backends.ascend.AscendBackend.__del__` {#ultralytics.nn.backends.ascend.AscendBackend.\_\_del\_\_}

```python
def __del__(self)
```

Release the Ascend device-side resources held by the inference session.

<details>
<summary>Source code in <code>ultralytics/nn/backends/ascend.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/nn/backends/ascend.py#L54-L57">View on GitHub</a>
```python
def __del__(self):
    """Release the Ascend device-side resources held by the inference session."""
    if model := getattr(self, "model", None):
        model.free_resource()
```
</details>

<br>

### Method `ultralytics.nn.backends.ascend.AscendBackend.forward` {#ultralytics.nn.backends.ascend.AscendBackend.forward}

```python
def forward(self, im: torch.Tensor) -> np.ndarray | list[np.ndarray]
```

Run inference on the Ascend NPU.

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `im` | `torch.Tensor` | Input image tensor in BCHW format, normalized to [0, 1]. | *required* |

**Returns**

| Type | Description |
| --- | --- |
| `np.ndarray \| list[np.ndarray]` | Model predictions as a single array or list of arrays. |

<details>
<summary>Source code in <code>ultralytics/nn/backends/ascend.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/nn/backends/ascend.py#L59-L69">View on GitHub</a>
```python
def forward(self, im: torch.Tensor) -> np.ndarray | list[np.ndarray]:
    """Run inference on the Ascend NPU.

    Args:
        im (torch.Tensor): Input image tensor in BCHW format, normalized to [0, 1].

    Returns:
        (np.ndarray | list[np.ndarray]): Model predictions as a single array or list of arrays.
    """
    y = self.model.infer([im.cpu().numpy()])
    return y[0] if len(y) == 1 else y
```
</details>

<br>

### Method `ultralytics.nn.backends.ascend.AscendBackend.load_model` {#ultralytics.nn.backends.ascend.AscendBackend.load\_model}

```python
def load_model(self, weight: str | Path) -> None
```

Load an Ascend model from a directory containing a .om file.

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `weight` | `str \| Path` | Path to the Ascend model directory containing the .om offline model. | *required* |

**Raises**

| Type | Description |
| --- | --- |
| `ImportError` | If the ``ais_bench`` Python package is not installed. |
| `FileNotFoundError` | If no .om file is found in the given directory. |

<details>
<summary>Source code in <code>ultralytics/nn/backends/ascend.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/nn/backends/ascend.py#L22-L52">View on GitHub</a>
```python
def load_model(self, weight: str | Path) -> None:
    """Load an Ascend model from a directory containing a .om file.

    Args:
        weight (str | Path): Path to the Ascend model directory containing the .om offline model.

    Raises:
        ImportError: If the ``ais_bench`` Python package is not installed.
        FileNotFoundError: If no .om file is found in the given directory.
    """
    try:
        from ais_bench.infer.interface import InferSession
    except ImportError as e:
        raise ImportError(
            "Ascend inference requires the CANN runtime and `ais_bench` Python package. "
            "See https://docs.ultralytics.com/integrations/ascend/#runtime-installation for instructions."
        ) from e

    LOGGER.info(f"Loading {weight} for Huawei Ascend inference...")

    w = Path(weight)
    found = next(w.rglob("*.om"), None)
    if found is None:
        raise FileNotFoundError(f"No .om file found in: {w}")

    self.model = InferSession(getattr(self.device, "index", None) or 0, str(found))

    # Load metadata
    metadata_file = found.parent / "metadata.yaml"
    if metadata_file.exists():
        self.apply_metadata(YAML.load(metadata_file))
```
</details>

<br><br>
