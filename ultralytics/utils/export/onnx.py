# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from ultralytics.utils import LOGGER


def onnx_calibration_reader(dataset, transform_fn, input_name: str = "images"):
    """Create an ONNX Runtime calibration data reader from an Ultralytics calibration dataloader.

    Args:
        dataset (Iterable): Calibration dataloader yielding batch dicts.
        transform_fn (Callable): Function converting a batch dict to a float32 NCHW numpy array.
        input_name (str): Name of the ONNX graph input to feed.

    Returns:
        (onnxruntime.quantization.CalibrationDataReader): Calibration data reader over `dataset`.
    """
    from onnxruntime.quantization import CalibrationDataReader

    class _CalibrationReader(CalibrationDataReader):
        def __init__(self):
            """Initialize calibration dataset iteration."""
            self.iterator = iter(dataset)

        def get_next(self):
            """Return the next calibration sample, or None when exhausted."""
            b = next(self.iterator, None)
            return None if b is None else {input_name: transform_fn(b)}

        def rewind(self):
            """Reset the iterator for an additional calibration pass."""
            self.iterator = iter(dataset)

    return _CalibrationReader()


def onnx_int8_quantize(
    onnx_file,
    output_file,
    dataset,
    transform_fn,
    input_name: str = "images",
    prefix: str = "",
) -> str:
    """Quantize an ONNX model to INT8 using ONNX Runtime static quantization.

    Args:
        onnx_file (str | Path): Path to the FP32 ONNX model.
        output_file (str | Path): Path to save the INT8 ONNX model.
        dataset (Iterable): Calibration dataloader yielding batch dicts.
        transform_fn (Callable): Function converting a batch dict to a float32 NCHW numpy array.
        input_name (str): Name of the ONNX graph input to feed.
        prefix (str): Prefix for log messages.

    Returns:
        (str): Path to the quantized ONNX file.
    """
    import onnx
    from onnxruntime.quantization import quantize_static

    # Quantize only weighted ops so the head decode stays float: one INT8 scale spanning box pixels (~0-640) and class
    # probs (0-1) rounds every score to 0. Excluding by node (not op_types) still calibrates all tensors, avoiding an
    # ONNX Runtime crash on the uncalibrated attention Softmax.
    graph = onnx.load(onnx_file).graph
    exclude = [n.name for n in graph.node if n.op_type not in {"Conv", "Gemm", "MatMul"}]
    del graph

    LOGGER.info(f"{prefix} quantizing INT8 with ONNX Runtime...")
    quantize_static(
        onnx_file,
        output_file,
        onnx_calibration_reader(dataset, transform_fn, input_name),
        nodes_to_exclude=exclude,
    )
    return str(output_file)
