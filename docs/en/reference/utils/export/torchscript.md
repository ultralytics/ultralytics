---
title: utils.export.torchscript API Reference
description: TorchScript export utilities for converting PyTorch YOLO models to TorchScript format with metadata for production deployment and C++ inference.
keywords: Ultralytics, TorchScript, model export, PyTorch, JIT trace, production deployment, C++ inference, libtorch, model serialization
---

# Reference for `ultralytics/utils/export/torchscript.py`

!!! success "Improvements"

    This page is sourced from [https://github.com/ultralytics/ultralytics/blob/main/ultralytics/utils/export/torchscript.py](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/utils/export/torchscript.py). Have an improvement or example to add? Open a [Pull Request](https://docs.ultralytics.com/help/contributing) — thank you! 🙏

<br>

!!! abstract "Summary"

    === "<span class="doc-kind doc-kind-function">Functions</span>"

        - [`torch2torchscript`](#ultralytics.utils.export.torchscript.torch2torchscript)


## Function `ultralytics.utils.export.torchscript.torch2torchscript` {#ultralytics.utils.export.torchscript.torch2torchscript}

```python
def torch2torchscript(
    model: torch.nn.Module,
    im: torch.Tensor,
    output_file: Path | str,
    metadata: dict | None = None,
    prefix: str = "",
) -> str
```

Export a PyTorch model to TorchScript format.

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `model` | `torch.nn.Module` | The PyTorch model to export (may be NMS-wrapped). | *required* |
| `im` | `torch.Tensor` | Example input tensor for tracing. | *required* |
| `output_file` | `Path \| str` | Path to save the exported TorchScript model. | *required* |
| `metadata` | `dict \| None` | Optional metadata to embed in the TorchScript archive. | `None` |
| `prefix` | `str` | Prefix for log messages. | `""` |

**Returns**

| Type | Description |
| --- | --- |
| `str` | Path to the exported ``.torchscript`` file. |

<details>
<summary>Source code in <code>ultralytics/utils/export/torchscript.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/utils/export/torchscript.py#L13-L38">View on GitHub</a>
```python
def torch2torchscript(
    model: torch.nn.Module,
    im: torch.Tensor,
    output_file: Path | str,
    metadata: dict | None = None,
    prefix: str = "",
) -> str:
    """Export a PyTorch model to TorchScript format.

    Args:
        model (torch.nn.Module): The PyTorch model to export (may be NMS-wrapped).
        im (torch.Tensor): Example input tensor for tracing.
        output_file (Path | str): Path to save the exported TorchScript model.
        metadata (dict | None): Optional metadata to embed in the TorchScript archive.
        prefix (str): Prefix for log messages.

    Returns:
        (str): Path to the exported ``.torchscript`` file.
    """
    LOGGER.info(f"\n{prefix} starting export with torch {TORCH_VERSION}...")

    output_file = str(output_file)
    ts = torch.jit.trace(model, im, strict=False, check_trace=False)
    extra_files = {"config.txt": json.dumps(metadata or {})}  # torch._C.ExtraFilesMap()
    ts.save(output_file, _extra_files=extra_files)
    return output_file
```
</details>

<br><br>
