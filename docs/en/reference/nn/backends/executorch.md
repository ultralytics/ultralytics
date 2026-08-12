---
title: nn.backends.executorch API Reference
description: Explore ExecuTorchBackend for Meta ExecuTorch inference, enabling efficient PyTorch model deployment on mobile and edge devices.
keywords: Ultralytics, ExecuTorchBackend, ExecuTorch inference, Meta ExecuTorch, mobile inference, edge deployment, PyTorch Mobile
---

# Reference for `ultralytics/nn/backends/executorch.py`

!!! success "Improvements"

    This page is sourced from [https://github.com/ultralytics/ultralytics/blob/main/ultralytics/nn/backends/executorch.py](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/nn/backends/executorch.py). Have an improvement or example to add? Open a [Pull Request](https://docs.ultralytics.com/help/contributing) — thank you! 🙏

<br>

!!! abstract "Summary"

    === "<span class="doc-kind doc-kind-class">Classes</span>"

        - [`ExecuTorchBackend`](#ultralytics.nn.backends.executorch.ExecuTorchBackend)

    === "<span class="doc-kind doc-kind-method">Methods</span>"

        - [`ExecuTorchBackend.load_model`](#ultralytics.nn.backends.executorch.ExecuTorchBackend.load_model)
        - [`ExecuTorchBackend.forward`](#ultralytics.nn.backends.executorch.ExecuTorchBackend.forward)


## Class `ultralytics.nn.backends.executorch.ExecuTorchBackend` {#ultralytics.nn.backends.executorch.ExecuTorchBackend}

```python
ExecuTorchBackend()
```

**Bases:** `BaseBackend`

Meta ExecuTorch inference backend for on-device deployment.

Loads and runs inference with Meta ExecuTorch models (.pte files) using the ExecuTorch runtime. Supports both standalone .pte files and directory-based model packages with metadata.

**Methods**

| Name | Description |
| --- | --- |
| [`forward`](#ultralytics.nn.backends.executorch.ExecuTorchBackend.forward) | Run inference using the ExecuTorch runtime. |
| [`load_model`](#ultralytics.nn.backends.executorch.ExecuTorchBackend.load_model) | Load an ExecuTorch model from a .pte file or directory. |

<details>
<summary>Source code in <code>ultralytics/nn/backends/executorch.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/nn/backends/executorch.py#L15-L59">View on GitHub</a>
```python
class ExecuTorchBackend(BaseBackend):
    """Meta ExecuTorch inference backend for on-device deployment.

    Loads and runs inference with Meta ExecuTorch models (.pte files) using the ExecuTorch runtime. Supports both
    standalone .pte files and directory-based model packages with metadata.
    """
```
</details>

<br>

### Method `ultralytics.nn.backends.executorch.ExecuTorchBackend.forward` {#ultralytics.nn.backends.executorch.ExecuTorchBackend.forward}

```python
def forward(self, im: torch.Tensor) -> list
```

Run inference using the ExecuTorch runtime.

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `im` | `torch.Tensor` | Input image tensor in BCHW format, normalized to [0, 1]. | *required* |

**Returns**

| Type | Description |
| --- | --- |
| `list` | Model predictions as a list of ExecuTorch output values. |

<details>
<summary>Source code in <code>ultralytics/nn/backends/executorch.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/nn/backends/executorch.py#L50-L59">View on GitHub</a>
```python
def forward(self, im: torch.Tensor) -> list:
    """Run inference using the ExecuTorch runtime.

    Args:
        im (torch.Tensor): Input image tensor in BCHW format, normalized to [0, 1].

    Returns:
        (list): Model predictions as a list of ExecuTorch output values.
    """
    return self.model.execute([im])
```
</details>

<br>

### Method `ultralytics.nn.backends.executorch.ExecuTorchBackend.load_model` {#ultralytics.nn.backends.executorch.ExecuTorchBackend.load\_model}

```python
def load_model(self, weight: str | Path) -> None
```

Load an ExecuTorch model from a .pte file or directory.

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `weight` | `str \| Path` | Path to the .pte model file or directory containing the model. | *required* |

<details>
<summary>Source code in <code>ultralytics/nn/backends/executorch.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/nn/backends/executorch.py#L22-L48">View on GitHub</a>
```python
def load_model(self, weight: str | Path) -> None:
    """Load an ExecuTorch model from a .pte file or directory.

    Args:
        weight (str | Path): Path to the .pte model file or directory containing the model.
    """
    LOGGER.info(f"Loading {weight} for ExecuTorch inference...")
    check_executorch_requirements()

    from executorch.runtime import Runtime

    w = Path(weight)
    if w.is_dir():
        model_file = next(w.rglob("*.pte"))
        metadata_file = w / "metadata.yaml"
    else:
        model_file = w
        metadata_file = w.parent / "metadata.yaml"

    program = Runtime.get().load_program(str(model_file))
    self.model = program.load_method("forward")

    # Load metadata
    if metadata_file.exists():
        from ultralytics.utils import YAML

        self.apply_metadata(YAML.load(metadata_file))
```
</details>

<br><br>
