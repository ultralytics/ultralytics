---
title: utils.export.ncnn API Reference
description: NCNN export utilities for converting PyTorch YOLO models to NCNN format using PNNX. Optimized for mobile and embedded platforms with support for FP16 inference on ARM architectures.
keywords: Ultralytics, NCNN, model export, PyTorch to NCNN, PNNX, mobile deployment, ARM, embedded systems, FP16, lightweight inference, Tencent NCNN, edge AI
---

# Reference for `ultralytics/utils/export/ncnn.py`

!!! success "Improvements"

    This page is sourced from [https://github.com/ultralytics/ultralytics/blob/main/ultralytics/utils/export/ncnn.py](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/utils/export/ncnn.py). Have an improvement or example to add? Open a [Pull Request](https://docs.ultralytics.com/help/contributing) — thank you! 🙏

<br>

!!! abstract "Summary"

    === "<span class="doc-kind doc-kind-function">Functions</span>"

        - [`torch2ncnn`](#ultralytics.utils.export.ncnn.torch2ncnn)


## Function `ultralytics.utils.export.ncnn.torch2ncnn` {#ultralytics.utils.export.ncnn.torch2ncnn}

```python
def torch2ncnn(
    model: torch.nn.Module,
    im: torch.Tensor,
    output_dir: Path | str,
    quantize: int | str | None = None,
    metadata: dict | None = None,
    device: torch.device | None = None,
    prefix: str = "",
) -> str
```

Export a PyTorch model to NCNN format using PNNX.

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `model` | `torch.nn.Module` | The PyTorch model to export. | *required* |
| `im` | `torch.Tensor` | Example input tensor for tracing. | *required* |
| `output_dir` | `Path \| str` | Directory to save the exported NCNN model. | *required* |
| `quantize` | `int \| str \| None` | Precision scheme, e.g. 16 for FP16. | `None` |
| `metadata` | `dict \| None` | Optional metadata saved as ``metadata.yaml``. | `None` |
| `device` | `torch.device \| None` | Device the model lives on. | `None` |
| `prefix` | `str` | Prefix for log messages. | `""` |

**Returns**

| Type | Description |
| --- | --- |
| `str` | Path to the exported ``_ncnn_model`` directory. |

<details>
<summary>Source code in <code>ultralytics/utils/export/ncnn.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/utils/export/ncnn.py#L12-L68">View on GitHub</a>
```python
def torch2ncnn(
    model: torch.nn.Module,
    im: torch.Tensor,
    output_dir: Path | str,
    quantize: int | str | None = None,
    metadata: dict | None = None,
    device: torch.device | None = None,
    prefix: str = "",
) -> str:
    """Export a PyTorch model to NCNN format using PNNX.

    Args:
        model (torch.nn.Module): The PyTorch model to export.
        im (torch.Tensor): Example input tensor for tracing.
        output_dir (Path | str): Directory to save the exported NCNN model.
        quantize (int | str | None): Precision scheme, e.g. 16 for FP16.
        metadata (dict | None): Optional metadata saved as ``metadata.yaml``.
        device (torch.device | None): Device the model lives on.
        prefix (str): Prefix for log messages.

    Returns:
        (str): Path to the exported ``_ncnn_model`` directory.
    """
    from ultralytics.utils.checks import check_requirements

    check_requirements("ncnn", cmds="--no-deps")  # no deps to avoid installing opencv-python
    # Pin until PNNX 20260704 NCNN inference segfault is fixed: https://github.com/pnnx/pnnx/issues/293
    check_requirements("pnnx==20260526")
    import ncnn
    import pnnx

    LOGGER.info(f"\n{prefix} starting export with NCNN {ncnn.__version__} and PNNX {pnnx.__version__}...")
    output_dir = Path(output_dir)

    ncnn_args = {
        "ncnnparam": (output_dir / "model.ncnn.param").as_posix(),
        "ncnnbin": (output_dir / "model.ncnn.bin").as_posix(),
        "ncnnpy": (output_dir / "model_ncnn.py").as_posix(),
    }
    pnnx_args = {
        "ptpath": (output_dir / "model.pt").as_posix(),
        "pnnxparam": (output_dir / "model.pnnx.param").as_posix(),
        "pnnxbin": (output_dir / "model.pnnx.bin").as_posix(),
        "pnnxpy": (output_dir / "model_pnnx.py").as_posix(),
        "pnnxonnx": (output_dir / "model.pnnx.onnx").as_posix(),
    }

    output_dir.mkdir(parents=True, exist_ok=True)  # make ncnn_model directory
    device_type = device.type if device is not None else "cpu"
    pnnx.export(model, inputs=im, **ncnn_args, **pnnx_args, fp16=quantize == 16, device=device_type)

    for f_debug in ("debug.bin", "debug.param", "debug2.bin", "debug2.param", *pnnx_args.values()):
        Path(f_debug).unlink(missing_ok=True)

    if metadata:
        YAML.save(output_dir / "metadata.yaml", metadata)  # add metadata.yaml
    return str(output_dir)
```
</details>

<br><br>
