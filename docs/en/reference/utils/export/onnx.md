---
title: utils.export.onnx API Reference
description: Reference for the Ultralytics ONNX export utilities. Learn how ONNX INT8 quantization uses ONNX Runtime calibration data readers.
keywords: ONNX export, ONNX INT8, onnxruntime quantization, CalibrationDataReader, Ultralytics, model export, quantization
---

# Reference for `ultralytics/utils/export/onnx.py`

!!! success "Improvements"

    This page is sourced from [https://github.com/ultralytics/ultralytics/blob/main/ultralytics/utils/export/onnx.py](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/utils/export/onnx.py). Have an improvement or example to add? Open a [Pull Request](https://docs.ultralytics.com/help/contributing) — thank you! 🙏

<br>

!!! abstract "Summary"

    === "<span class="doc-kind doc-kind-function">Functions</span>"

        - [`onnx_calibration_reader`](#ultralytics.utils.export.onnx.onnx_calibration_reader)
        - [`onnx_int8_quantize`](#ultralytics.utils.export.onnx.onnx_int8_quantize)


## Function `ultralytics.utils.export.onnx.onnx_calibration_reader` {#ultralytics.utils.export.onnx.onnx\_calibration\_reader}

```python
def onnx_calibration_reader(dataset, transform_fn, input_name: str = "images", batch: int = 0)
```

Create an ONNX Runtime calibration data reader from an Ultralytics calibration dataloader.

`batch` is the graph's static batch dimension (0 for dynamic-batch models): calibration datasets smaller than the export batch yield undersized batches that static graphs reject, so samples are tiled up to exactly `batch`.

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `dataset` |  |  | *required* |
| `transform_fn` |  |  | *required* |
| `input_name` | `str` |  | `"images"` |
| `batch` | `int` |  | `0` |

<details>
<summary>Source code in <code>ultralytics/utils/export/onnx.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/utils/export/onnx.py#L8-L34">View on GitHub</a>
```python
def onnx_calibration_reader(dataset, transform_fn, input_name: str = "images", batch: int = 0):
    """Create an ONNX Runtime calibration data reader from an Ultralytics calibration dataloader.

    `batch` is the graph's static batch dimension (0 for dynamic-batch models): calibration datasets smaller than the
    export batch yield undersized batches that static graphs reject, so samples are tiled up to exactly `batch`.
    """
    from onnxruntime.quantization import CalibrationDataReader

    class _CalibrationReader(CalibrationDataReader):
        def __init__(self):
            """Initialize calibration dataset iteration."""
            self.iterator = iter(dataset)

        def get_next(self):
            """Return the next calibration sample, or None when exhausted."""
            if (b := next(self.iterator, None)) is None:
                return None
            im = transform_fn(b)
            if batch and im.shape[0] != batch:  # tile up to the static batch dimension
                im = np.tile(im, (-(-batch // im.shape[0]), 1, 1, 1))[:batch]
            return {input_name: im}

        def rewind(self):
            """Reset the iterator for an additional calibration pass."""
            self.iterator = iter(dataset)

    return _CalibrationReader()
```
</details>


<br><br><hr><br>

## Function `ultralytics.utils.export.onnx.onnx_int8_quantize` {#ultralytics.utils.export.onnx.onnx\_int8\_quantize}

```python
def onnx_int8_quantize(
    onnx_file,
    output_file,
    dataset,
    transform_fn,
    input_name: str = "images",
    batch: int = 0,
    prefix: str = "",
) -> str
```

Quantize an ONNX model to INT8 using ONNX Runtime static quantization.

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `onnx_file` |  |  | *required* |
| `output_file` |  |  | *required* |
| `dataset` |  |  | *required* |
| `transform_fn` |  |  | *required* |
| `input_name` | `str` |  | `"images"` |
| `batch` | `int` |  | `0` |
| `prefix` | `str` |  | `""` |

<details>
<summary>Source code in <code>ultralytics/utils/export/onnx.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/utils/export/onnx.py#L37-L63">View on GitHub</a>
```python
def onnx_int8_quantize(
    onnx_file,
    output_file,
    dataset,
    transform_fn,
    input_name: str = "images",
    batch: int = 0,
    prefix: str = "",
) -> str:
    """Quantize an ONNX model to INT8 using ONNX Runtime static quantization."""
    import onnx
    from onnxruntime.quantization import quantize_static

    # Quantize only weighted ops so the head decode stays float: one INT8 scale spanning box pixels (~0-640) and class
    # probs (0-1) rounds every score to 0. Excluding by node (not op_types) still calibrates all tensors, avoiding an
    # ONNX Runtime crash on the uncalibrated attention Softmax.
    graph = onnx.load(onnx_file).graph
    exclude = [n.name for n in graph.node if n.op_type not in {"Conv", "Gemm", "MatMul"}]

    LOGGER.info(f"{prefix} quantizing INT8 with ONNX Runtime...")
    quantize_static(
        onnx_file,
        output_file,
        onnx_calibration_reader(dataset, transform_fn, input_name, batch),
        nodes_to_exclude=exclude,
    )
    return str(output_file)
```
</details>

<br><br>
