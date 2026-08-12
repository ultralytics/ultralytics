---
title: nn.backends.axelera API Reference
description: Explore AxeleraBackend for Axelera hardware inference, deploying YOLO models on Axelera AI accelerators with optimized performance.
keywords: Ultralytics, AxeleraBackend, Axelera inference, AI accelerator, hardware inference, edge AI, deep learning acceleration
---

# Reference for `ultralytics/nn/backends/axelera.py`

!!! success "Improvements"

    This page is sourced from [https://github.com/ultralytics/ultralytics/blob/main/ultralytics/nn/backends/axelera.py](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/nn/backends/axelera.py). Have an improvement or example to add? Open a [Pull Request](https://docs.ultralytics.com/help/contributing) — thank you! 🙏

<br>

!!! abstract "Summary"

    === "<span class="doc-kind doc-kind-class">Classes</span>"

        - [`AxeleraBackend`](#ultralytics.nn.backends.axelera.AxeleraBackend)

    === "<span class="doc-kind doc-kind-method">Methods</span>"

        - [`AxeleraBackend.load_model`](#ultralytics.nn.backends.axelera.AxeleraBackend.load_model)
        - [`AxeleraBackend.forward`](#ultralytics.nn.backends.axelera.AxeleraBackend.forward)


## Class `ultralytics.nn.backends.axelera.AxeleraBackend` {#ultralytics.nn.backends.axelera.AxeleraBackend}

```python
AxeleraBackend()
```

**Bases:** `BaseBackend`

Axelera AI inference backend for Axelera Metis AI accelerators.

Loads compiled Axelera models (.axm files) and runs inference using the Axelera AI runtime SDK.

**Methods**

| Name | Description |
| --- | --- |
| [`forward`](#ultralytics.nn.backends.axelera.AxeleraBackend.forward) | Run inference on the Axelera hardware accelerator. |
| [`load_model`](#ultralytics.nn.backends.axelera.AxeleraBackend.load_model) | Load an Axelera model from a directory containing a .axm file. |

<details>
<summary>Source code in <code>ultralytics/nn/backends/axelera.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/nn/backends/axelera.py#L14-L59">View on GitHub</a>
```python
class AxeleraBackend(BaseBackend):
    """Axelera AI inference backend for Axelera Metis AI accelerators.

    Loads compiled Axelera models (.axm files) and runs inference using the Axelera AI runtime SDK.
    """
```
</details>

<br>

### Method `ultralytics.nn.backends.axelera.AxeleraBackend.forward` {#ultralytics.nn.backends.axelera.AxeleraBackend.forward}

```python
def forward(self, im: torch.Tensor) -> list
```

Run inference on the Axelera hardware accelerator.

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `im` | `torch.Tensor` | Input image tensor in BCHW format, normalized to [0, 1]. | *required* |

**Returns**

| Type | Description |
| --- | --- |
| `list` | Model predictions as a list of output arrays. |

<details>
<summary>Source code in <code>ultralytics/nn/backends/axelera.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/nn/backends/axelera.py#L50-L59">View on GitHub</a>
```python
def forward(self, im: torch.Tensor) -> list:
    """Run inference on the Axelera hardware accelerator.

    Args:
        im (torch.Tensor): Input image tensor in BCHW format, normalized to [0, 1].

    Returns:
        (list): Model predictions as a list of output arrays.
    """
    return self.model(im.cpu())
```
</details>

<br>

### Method `ultralytics.nn.backends.axelera.AxeleraBackend.load_model` {#ultralytics.nn.backends.axelera.AxeleraBackend.load\_model}

```python
def load_model(self, weight: str | Path) -> None
```

Load an Axelera model from a directory containing a .axm file.

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `weight` | `str \| Path` | Path to the Axelera model directory containing the .axm binary. | *required* |

<details>
<summary>Source code in <code>ultralytics/nn/backends/axelera.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/nn/backends/axelera.py#L20-L48">View on GitHub</a>
```python
def load_model(self, weight: str | Path) -> None:
    """Load an Axelera model from a directory containing a .axm file.

    Args:
        weight (str | Path): Path to the Axelera model directory containing the .axm binary.
    """
    try:
        from axelera.runtime import op
    except ImportError:
        check_requirements(
            "axelera-rt==1.7.0",
            cmds="--extra-index-url https://software.axelera.ai/artifactory/api/pypi/axelera-pypi/simple",
        )

    from axelera.runtime import op

    w = Path(weight)
    found = next(w.rglob("*.axm"), None)
    if found is None:
        raise FileNotFoundError(f"No .axm file found in: {w}")

    self.model = op.load(str(found)).optimized()

    # Load metadata
    metadata_file = found.parent / "metadata.yaml"
    if metadata_file.exists():
        from ultralytics.utils import YAML

        self.apply_metadata(YAML.load(metadata_file))
```
</details>

<br><br>
