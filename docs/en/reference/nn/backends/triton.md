---
title: nn.backends.triton API Reference
description: Explore TritonBackend for NVIDIA Triton Inference Server, enabling scalable cloud and edge deployment of YOLO models.
keywords: Ultralytics, TritonBackend, Triton Inference Server, NVIDIA Triton, cloud inference, model serving, scalable deployment
---

# Reference for `ultralytics/nn/backends/triton.py`

!!! success "Improvements"

    This page is sourced from [https://github.com/ultralytics/ultralytics/blob/main/ultralytics/nn/backends/triton.py](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/nn/backends/triton.py). Have an improvement or example to add? Open a [Pull Request](https://docs.ultralytics.com/help/contributing) — thank you! 🙏

<br>

!!! abstract "Summary"

    === "<span class="doc-kind doc-kind-class">Classes</span>"

        - [`TritonBackend`](#ultralytics.nn.backends.triton.TritonBackend)

    === "<span class="doc-kind doc-kind-method">Methods</span>"

        - [`TritonBackend.load_model`](#ultralytics.nn.backends.triton.TritonBackend.load_model)
        - [`TritonBackend.forward`](#ultralytics.nn.backends.triton.TritonBackend.forward)


## Class `ultralytics.nn.backends.triton.TritonBackend` {#ultralytics.nn.backends.triton.TritonBackend}

```python
TritonBackend()
```

**Bases:** `BaseBackend`

NVIDIA Triton Inference Server backend for remote model serving.

Connects to and runs inference with models hosted on an NVIDIA Triton Inference Server instance via HTTP or gRPC protocols. The model is specified using a triton:// URL scheme.

**Methods**

| Name | Description |
| --- | --- |
| [`forward`](#ultralytics.nn.backends.triton.TritonBackend.forward) | Run inference via the NVIDIA Triton Inference Server. |
| [`load_model`](#ultralytics.nn.backends.triton.TritonBackend.load_model) | Connect to a remote model on an NVIDIA Triton Inference Server. |

<details>
<summary>Source code in <code>ultralytics/nn/backends/triton.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/nn/backends/triton.py#L14-L45">View on GitHub</a>
```python
class TritonBackend(BaseBackend):
    """NVIDIA Triton Inference Server backend for remote model serving.

    Connects to and runs inference with models hosted on an NVIDIA Triton Inference Server instance via HTTP or gRPC
    protocols. The model is specified using a triton:// URL scheme.
    """
```
</details>

<br>

### Method `ultralytics.nn.backends.triton.TritonBackend.forward` {#ultralytics.nn.backends.triton.TritonBackend.forward}

```python
def forward(self, im: torch.Tensor) -> list
```

Run inference via the NVIDIA Triton Inference Server.

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `im` | `torch.Tensor` | Input image tensor in BCHW format, normalized to [0, 1]. | *required* |

**Returns**

| Type | Description |
| --- | --- |
| `list` | Model predictions as a list of numpy arrays from the Triton server. |

<details>
<summary>Source code in <code>ultralytics/nn/backends/triton.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/nn/backends/triton.py#L36-L45">View on GitHub</a>
```python
def forward(self, im: torch.Tensor) -> list:
    """Run inference via the NVIDIA Triton Inference Server.

    Args:
        im (torch.Tensor): Input image tensor in BCHW format, normalized to [0, 1].

    Returns:
        (list): Model predictions as a list of numpy arrays from the Triton server.
    """
    return self.model(im.cpu().numpy())
```
</details>

<br>

### Method `ultralytics.nn.backends.triton.TritonBackend.load_model` {#ultralytics.nn.backends.triton.TritonBackend.load\_model}

```python
def load_model(self, weight: str | Path) -> None
```

Connect to a remote model on an NVIDIA Triton Inference Server.

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `weight` | `str \| Path` | Triton model URL (e.g., 'triton://host:8000/model_name'). | *required* |

<details>
<summary>Source code in <code>ultralytics/nn/backends/triton.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/nn/backends/triton.py#L21-L34">View on GitHub</a>
```python
def load_model(self, weight: str | Path) -> None:
    """Connect to a remote model on an NVIDIA Triton Inference Server.

    Args:
        weight (str | Path): Triton model URL (e.g., 'triton://host:8000/model_name').
    """
    check_requirements("tritonclient[all]")
    from ultralytics.utils.triton import TritonRemoteModel

    self.model = TritonRemoteModel(weight)

    # Copy metadata from Triton model
    if hasattr(self.model, "metadata"):
        self.apply_metadata(self.model.metadata)
```
</details>

<br><br>
