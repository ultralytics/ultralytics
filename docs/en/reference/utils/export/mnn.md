---
title: utils.export.mnn API Reference
description: MNN export utilities for converting ONNX models to MNN format for efficient inference on mobile and embedded devices. Supports FP16 and INT8 weight quantization for optimized deployment using Alibaba's MNN framework.
keywords: Ultralytics, MNN, model export, ONNX to MNN, Alibaba MNN, mobile deployment, embedded systems, FP16, INT8 quantization, lightweight inference, edge deployment
---

# Reference for `ultralytics/utils/export/mnn.py`

!!! success "Improvements"

    This page is sourced from [https://github.com/ultralytics/ultralytics/blob/main/ultralytics/utils/export/mnn.py](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/utils/export/mnn.py). Have an improvement or example to add? Open a [Pull Request](https://docs.ultralytics.com/help/contributing) — thank you! 🙏

<br>

!!! abstract "Summary"

    === "<span class="doc-kind doc-kind-function">Functions</span>"

        - [`onnx2mnn`](#ultralytics.utils.export.mnn.onnx2mnn)


## Function `ultralytics.utils.export.mnn.onnx2mnn` {#ultralytics.utils.export.mnn.onnx2mnn}

```python
def onnx2mnn(
    onnx_file: str,
    output_file: Path | str,
    quantize: int | str | None = None,
    metadata: dict | None = None,
    prefix: str = "",
) -> str
```

Convert an ONNX model to MNN format.

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `onnx_file` | `str` | Path to the source ONNX file. | *required* |
| `output_file` | `Path \| str` | Path to save the exported MNN model. | *required* |
| `quantize` | `int \| str \| None` | Precision scheme, e.g. 16 for FP16 or 8 for INT8 weights. | `None` |
| `metadata` | `dict \| None` | Optional metadata embedded via ``--bizCode``. | `None` |
| `prefix` | `str` | Prefix for log messages. | `""` |

**Returns**

| Type | Description |
| --- | --- |
| `str` | Path to the exported ``.mnn`` file. |

<details>
<summary>Source code in <code>ultralytics/utils/export/mnn.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/utils/export/mnn.py#L11-L61">View on GitHub</a>
```python
def onnx2mnn(
    onnx_file: str,
    output_file: Path | str,
    quantize: int | str | None = None,
    metadata: dict | None = None,
    prefix: str = "",
) -> str:
    """Convert an ONNX model to MNN format.

    Args:
        onnx_file (str): Path to the source ONNX file.
        output_file (Path | str): Path to save the exported MNN model.
        quantize (int | str | None): Precision scheme, e.g. 16 for FP16 or 8 for INT8 weights.
        metadata (dict | None): Optional metadata embedded via ``--bizCode``.
        prefix (str): Prefix for log messages.

    Returns:
        (str): Path to the exported ``.mnn`` file.
    """
    from ultralytics.utils.checks import check_requirements
    from ultralytics.utils.torch_utils import TORCH_1_10

    assert TORCH_1_10, "MNN export requires torch>=1.10.0 to avoid segmentation faults"
    assert Path(onnx_file).exists(), f"failed to export ONNX file: {onnx_file}"

    check_requirements("MNN>=2.9.6")
    import MNN
    from MNN.tools import mnnconvert

    LOGGER.info(f"\n{prefix} starting export with MNN {MNN.version()}...")
    mnn_args = [
        "",
        "-f",
        "ONNX",
        "--modelFile",
        onnx_file,
        "--MNNModel",
        str(output_file),
        "--bizCode",
        json.dumps(metadata or {}),
    ]
    if quantize == 8:
        mnn_args.extend(("--weightQuantBits", "8"))
    if quantize == 16:
        mnn_args.append("--fp16")
    mnnconvert.convert(mnn_args)
    # Remove scratch file created during model convert optimize
    convert_scratch = Path(output_file).parent / ".__convert_external_data.bin"
    if convert_scratch.exists():
        convert_scratch.unlink()
    return str(output_file)
```
</details>

<br><br>
