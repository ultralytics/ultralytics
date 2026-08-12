---
title: utils.export.openvino API Reference
description: OpenVINO export utilities for converting PyTorch YOLO models to OpenVINO format with support for FP16 compression and INT8 quantization via NNCF. Provides optimized inference for Intel hardware including CPUs, GPUs, and VPUs.
keywords: Ultralytics, OpenVINO, model export, PyTorch to OpenVINO, INT8 quantization, FP16, NNCF, Intel, model optimization, inference acceleration, CPU inference, GPU inference, VPU, edge deployment
---

# Reference for `ultralytics/utils/export/openvino.py`

!!! success "Improvements"

    This page is sourced from [https://github.com/ultralytics/ultralytics/blob/main/ultralytics/utils/export/openvino.py](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/utils/export/openvino.py). Have an improvement or example to add? Open a [Pull Request](https://docs.ultralytics.com/help/contributing) — thank you! 🙏

<br>

!!! abstract "Summary"

    === "<span class="doc-kind doc-kind-function">Functions</span>"

        - [`torch2openvino`](#ultralytics.utils.export.openvino.torch2openvino)


## Function `ultralytics.utils.export.openvino.torch2openvino` {#ultralytics.utils.export.openvino.torch2openvino}

```python
def torch2openvino(
    model: torch.nn.Module,
    im: torch.Tensor | list[torch.Tensor] | tuple[torch.Tensor, ...],
    output_dir: Path | str | None = None,
    dynamic: bool = False,
    quantize: int | str | None = None,
    calibration_dataset: Any | None = None,
    ignored_scope: dict | None = None,
    prefix: str = "",
) -> Any
```

Export a PyTorch model to OpenVINO format with optional INT8 quantization.

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `model` | `torch.nn.Module` | The model to export (may be NMS-wrapped). | *required* |
| `im` | `torch.Tensor \| list[torch.Tensor] \| tuple[torch.Tensor, ...]` | Example input tensor(s) for tracing. | *required* |
| `output_dir` | `Path \| str \| None` | Directory to save the exported OpenVINO model. | `None` |
| `dynamic` | `bool` | Whether to use dynamic input shapes. | `False` |
| `quantize` | `int \| str \| None` | Precision scheme, e.g. 16 for FP16 or 8 for INT8. | `None` |
| `calibration_dataset` | `nncf.Dataset \| None` | Dataset for INT8 calibration (required when ``quantize=8``). | `None` |
| `ignored_scope` | `dict \| None` | Kwargs passed to ``nncf.IgnoredScope`` for head patterns. | `None` |
| `prefix` | `str` | Prefix for log messages. | `""` |

**Returns**

| Type | Description |
| --- | --- |
| `ov.Model` | The converted OpenVINO model. |

<details>
<summary>Source code in <code>ultralytics/utils/export/openvino.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/utils/export/openvino.py#L13-L66">View on GitHub</a>
```python
def torch2openvino(
    model: torch.nn.Module,
    im: torch.Tensor | list[torch.Tensor] | tuple[torch.Tensor, ...],
    output_dir: Path | str | None = None,
    dynamic: bool = False,
    quantize: int | str | None = None,
    calibration_dataset: Any | None = None,
    ignored_scope: dict | None = None,
    prefix: str = "",
) -> Any:
    """Export a PyTorch model to OpenVINO format with optional INT8 quantization.

    Args:
        model (torch.nn.Module): The model to export (may be NMS-wrapped).
        im (torch.Tensor | list[torch.Tensor] | tuple[torch.Tensor, ...]): Example input tensor(s) for tracing.
        output_dir (Path | str | None): Directory to save the exported OpenVINO model.
        dynamic (bool): Whether to use dynamic input shapes.
        quantize (int | str | None): Precision scheme, e.g. 16 for FP16 or 8 for INT8.
        calibration_dataset (nncf.Dataset | None): Dataset for INT8 calibration (required when ``quantize=8``).
        ignored_scope (dict | None): Kwargs passed to ``nncf.IgnoredScope`` for head patterns.
        prefix (str): Prefix for log messages.

    Returns:
        (ov.Model): The converted OpenVINO model.
    """
    import openvino as ov

    LOGGER.info(f"\n{prefix} starting export with openvino {ov.__version__}...")

    input_shape = [i.shape for i in im] if isinstance(im, (list, tuple)) else im.shape
    # Hand OpenVINO an already-traced ScriptModule (torchscript/coreml exports trace the same way), not a raw
    # nn.Module, so it doesn't re-trace internally with check_trace=True - that re-trace-and-diff sanity check is
    # non-deterministic on NMS models and fails with "Graphs differed across invocations!". check_trace=False skips
    # the same check on our own trace.
    ts = torch.jit.trace(model, im, strict=False, check_trace=False)
    ov_model = ov.convert_model(ts, input=None if dynamic else input_shape, example_input=im)
    if quantize == 8:
        import nncf

        ov_model = nncf.quantize(
            model=ov_model,
            calibration_dataset=calibration_dataset,
            preset=nncf.QuantizationPreset.MIXED,
            # Calibrate on the full dataset like other INT8 backends, not nncf's 300-batch default
            subset_size=calibration_dataset.get_length() or 300,
            ignored_scope=ignored_scope,
        )

    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        output_file = output_dir / "model.xml"
        ov.save_model(ov_model, output_file, compress_to_fp16=quantize == 16)
    return ov_model
```
</details>

<br><br>
