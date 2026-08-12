---
title: nn.backends.pytorch API Reference
description: Explore PyTorchBackend and TorchScriptBackend for native PyTorch and TorchScript model inference in Ultralytics YOLO models.
keywords: Ultralytics, PyTorchBackend, TorchScriptBackend, PyTorch inference, TorchScript inference, .pt models, deep learning, YOLO
---

# Reference for `ultralytics/nn/backends/pytorch.py`

!!! success "Improvements"

    This page is sourced from [https://github.com/ultralytics/ultralytics/blob/main/ultralytics/nn/backends/pytorch.py](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/nn/backends/pytorch.py). Have an improvement or example to add? Open a [Pull Request](https://docs.ultralytics.com/help/contributing) — thank you! 🙏

<br>

!!! abstract "Summary"

    === "<span class="doc-kind doc-kind-class">Classes</span>"

        - [`PyTorchBackend`](#ultralytics.nn.backends.pytorch.PyTorchBackend)
        - [`TorchScriptBackend`](#ultralytics.nn.backends.pytorch.TorchScriptBackend)

    === "<span class="doc-kind doc-kind-method">Methods</span>"

        - [`PyTorchBackend.load_model`](#ultralytics.nn.backends.pytorch.PyTorchBackend.load_model)
        - [`PyTorchBackend.forward`](#ultralytics.nn.backends.pytorch.PyTorchBackend.forward)
        - [`TorchScriptBackend.load_model`](#ultralytics.nn.backends.pytorch.TorchScriptBackend.load_model)
        - [`TorchScriptBackend.forward`](#ultralytics.nn.backends.pytorch.TorchScriptBackend.forward)


## Class `ultralytics.nn.backends.pytorch.PyTorchBackend` {#ultralytics.nn.backends.pytorch.PyTorchBackend}

```python
PyTorchBackend(
    weight: str | Path | nn.Module,
    device: torch.device,
    fp16: bool = False,
    fuse: bool = True,
    verbose: bool = True,
)
```

**Bases:** `BaseBackend`

PyTorch inference backend for native model execution.

Loads and runs inference with native PyTorch models (.pt checkpoint files) or pre-loaded nn.Module instances. Supports model layer fusion, FP16 precision, and NVIDIA Jetson compatibility.

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `weight` | `str \| Path \| nn.Module` | Path to the .pt model file or a pre-loaded nn.Module instance. | *required* |
| `device` | `torch.device` | Device to run inference on (e.g., 'cpu', 'cuda:0'). | *required* |
| `fp16` | `bool` | Whether to use FP16 half-precision inference. | `False` |
| `fuse` | `bool` | Whether to fuse Conv2D + BatchNorm layers for optimization. | `True` |
| `verbose` | `bool` | Whether to print verbose model loading messages. | `True` |

**Methods**

| Name | Description |
| --- | --- |
| [`forward`](#ultralytics.nn.backends.pytorch.PyTorchBackend.forward) | Run native PyTorch inference with support for augmentation and embeddings. |
| [`load_model`](#ultralytics.nn.backends.pytorch.PyTorchBackend.load_model) | Load a PyTorch model from a checkpoint file or nn.Module instance. |

<details>
<summary>Source code in <code>ultralytics/nn/backends/pytorch.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/nn/backends/pytorch.py#L17-L93">View on GitHub</a>
```python
class PyTorchBackend(BaseBackend):
    """PyTorch inference backend for native model execution.

    Loads and runs inference with native PyTorch models (.pt checkpoint files) or pre-loaded nn.Module
    instances. Supports model layer fusion, FP16 precision, and NVIDIA Jetson compatibility.
    """

    def __init__(
        self,
        weight: str | Path | nn.Module,
        device: torch.device,
        fp16: bool = False,
        fuse: bool = True,
        verbose: bool = True,
    ):
        """Initialize the PyTorch backend.

        Args:
            weight (str | Path | nn.Module): Path to the .pt model file or a pre-loaded nn.Module instance.
            device (torch.device): Device to run inference on (e.g., 'cpu', 'cuda:0').
            fp16 (bool): Whether to use FP16 half-precision inference.
            fuse (bool): Whether to fuse Conv2D + BatchNorm layers for optimization.
            verbose (bool): Whether to print verbose model loading messages.
        """
        self.fuse = fuse
        self.verbose = verbose
        super().__init__(weight, device, fp16)
```
</details>

<br>

### Method `ultralytics.nn.backends.pytorch.PyTorchBackend.forward` {#ultralytics.nn.backends.pytorch.PyTorchBackend.forward}

```python
def forward(
    self, im: torch.Tensor, augment: bool = False, embed: list | None = None, **kwargs: Any
) -> torch.Tensor | list[torch.Tensor]
```

Run native PyTorch inference with support for augmentation and embeddings.

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `im` | `torch.Tensor` | Input image tensor in BCHW format, normalized to [0, 1]. | *required* |
| `augment` | `bool` | Whether to apply test-time augmentation. | `False` |
| `embed` | `list \| None` | List of layer indices to extract embeddings from, or None. | `None` |
| `**kwargs` | `Any` | Additional keyword arguments passed to the model forward method. | *required* |

**Returns**

| Type | Description |
| --- | --- |
| `torch.Tensor \| list[torch.Tensor]` | Model predictions as tensor(s). |

<details>
<summary>Source code in <code>ultralytics/nn/backends/pytorch.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/nn/backends/pytorch.py#L77-L93">View on GitHub</a>
```python
def forward(
    self, im: torch.Tensor, augment: bool = False, embed: list | None = None, **kwargs: Any
) -> torch.Tensor | list[torch.Tensor]:
    """Run native PyTorch inference with support for augmentation and embeddings.

    Args:
        im (torch.Tensor): Input image tensor in BCHW format, normalized to [0, 1].
        augment (bool): Whether to apply test-time augmentation.
        embed (list | None): List of layer indices to extract embeddings from, or None.
        **kwargs (Any): Additional keyword arguments passed to the model forward method.

    Returns:
        (torch.Tensor | list[torch.Tensor]): Model predictions as tensor(s).
    """
    if not self.base_model:  # a foreign nn.Module defines no `augment`/`embed` contract to honor
        return self.model(im, **kwargs)
    return self.model(im, augment=augment, embed=embed, **kwargs)
```
</details>

<br>

### Method `ultralytics.nn.backends.pytorch.PyTorchBackend.load_model` {#ultralytics.nn.backends.pytorch.PyTorchBackend.load\_model}

```python
def load_model(self, weight: str | torch.nn.Module) -> None
```

Load a PyTorch model from a checkpoint file or nn.Module instance.

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `weight` | `str \| torch.nn.Module` | Path to the .pt checkpoint or a pre-loaded module. | *required* |

<details>
<summary>Source code in <code>ultralytics/nn/backends/pytorch.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/nn/backends/pytorch.py#L45-L75">View on GitHub</a>
```python
def load_model(self, weight: str | torch.nn.Module) -> None:
    """Load a PyTorch model from a checkpoint file or nn.Module instance.

    Args:
        weight (str | torch.nn.Module): Path to the .pt checkpoint or a pre-loaded module.
    """
    from ultralytics.nn.tasks import BaseModel, load_checkpoint

    if isinstance(weight, torch.nn.Module):
        if self.fuse and hasattr(weight, "fuse"):
            if IS_JETSON and is_jetson(jetpack=5):
                weight = weight.to(self.device)
            weight = weight.fuse(verbose=self.verbose) if isinstance(weight, BaseModel) else weight.fuse()
        model = weight.to(self.device)
    else:
        model, _ = load_checkpoint(weight, device=self.device, fuse=self.fuse)

    # Extract model attributes
    if hasattr(model, "kpt_shape"):
        self.kpt_shape = model.kpt_shape
    self.stride = max(int(model.stride.max()), 32) if hasattr(model, "stride") else 32
    self.names = model.module.names if hasattr(model, "module") else getattr(model, "names", {})
    self.channels = model.yaml.get("channels", 3) if hasattr(model, "yaml") else 3
    model.half() if self.fp16 else model.float()

    for p in model.parameters():
        p.requires_grad = False

    self.model = model
    self.end2end = getattr(model, "end2end", False)
    self.base_model = isinstance(unwrap_model(model), BaseModel)
```
</details>


<br><br><hr><br>

## Class `ultralytics.nn.backends.pytorch.TorchScriptBackend` {#ultralytics.nn.backends.pytorch.TorchScriptBackend}

```python
TorchScriptBackend(weight: str | Path, device: torch.device, fp16: bool = False)
```

**Bases:** `BaseBackend`

PyTorch TorchScript inference backend for serialized model execution.

Loads and runs inference with TorchScript models (.torchscript files) created via torch.jit.trace or torch.jit.script. Supports FP16 precision and embedded metadata extraction.

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `weight` | `str \| Path` | Path to the .torchscript model file. | *required* |
| `device` | `torch.device` | Device to run inference on (e.g., 'cpu', 'cuda:0'). | *required* |
| `fp16` | `bool` | Whether to use FP16 half-precision inference. | `False` |

**Methods**

| Name | Description |
| --- | --- |
| [`forward`](#ultralytics.nn.backends.pytorch.TorchScriptBackend.forward) | Run TorchScript inference. |
| [`load_model`](#ultralytics.nn.backends.pytorch.TorchScriptBackend.load_model) | Load a TorchScript model from a .torchscript file with optional embedded metadata. |

<details>
<summary>Source code in <code>ultralytics/nn/backends/pytorch.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/nn/backends/pytorch.py#L96-L140">View on GitHub</a>
```python
class TorchScriptBackend(BaseBackend):
    """PyTorch TorchScript inference backend for serialized model execution.

    Loads and runs inference with TorchScript models (.torchscript files) created via torch.jit.trace or
    torch.jit.script. Supports FP16 precision and embedded metadata extraction.
    """

    def __init__(self, weight: str | Path, device: torch.device, fp16: bool = False):
        """Initialize the TorchScript backend.

        Args:
            weight (str | Path): Path to the .torchscript model file.
            device (torch.device): Device to run inference on (e.g., 'cpu', 'cuda:0').
            fp16 (bool): Whether to use FP16 half-precision inference.
        """
        super().__init__(weight, device, fp16)
```
</details>

<br>

### Method `ultralytics.nn.backends.pytorch.TorchScriptBackend.forward` {#ultralytics.nn.backends.pytorch.TorchScriptBackend.forward}

```python
def forward(self, im: torch.Tensor) -> torch.Tensor | list[torch.Tensor]
```

Run TorchScript inference.

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `im` | `torch.Tensor` | Input image tensor in BCHW format, normalized to [0, 1]. | *required* |

**Returns**

| Type | Description |
| --- | --- |
| `torch.Tensor \| list[torch.Tensor]` | Model predictions as tensor(s). |

<details>
<summary>Source code in <code>ultralytics/nn/backends/pytorch.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/nn/backends/pytorch.py#L131-L140">View on GitHub</a>
```python
def forward(self, im: torch.Tensor) -> torch.Tensor | list[torch.Tensor]:
    """Run TorchScript inference.

    Args:
        im (torch.Tensor): Input image tensor in BCHW format, normalized to [0, 1].

    Returns:
        (torch.Tensor | list[torch.Tensor]): Model predictions as tensor(s).
    """
    return self.model(im)
```
</details>

<br>

### Method `ultralytics.nn.backends.pytorch.TorchScriptBackend.load_model` {#ultralytics.nn.backends.pytorch.TorchScriptBackend.load\_model}

```python
def load_model(self, weight: str) -> None
```

Load a TorchScript model from a .torchscript file with optional embedded metadata.

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `weight` | `str` | Path to the .torchscript model file. | *required* |

<details>
<summary>Source code in <code>ultralytics/nn/backends/pytorch.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/nn/backends/pytorch.py#L113-L129">View on GitHub</a>
```python
def load_model(self, weight: str) -> None:
    """Load a TorchScript model from a .torchscript file with optional embedded metadata.

    Args:
        weight (str): Path to the .torchscript model file.
    """
    import json

    import torchvision  # noqa - required for TorchScript model deserialization

    LOGGER.info(f"Loading {weight} for TorchScript inference...")
    extra_files = {"config.txt": ""}
    self.model = torch.jit.load(weight, _extra_files=extra_files, map_location=self.device)
    self.model.half() if self.fp16 else self.model.float()

    if extra_files["config.txt"]:
        self.apply_metadata(json.loads(extra_files["config.txt"], object_hook=lambda x: dict(x.items())))
```
</details>

<br><br>
