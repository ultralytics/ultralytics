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
    model = onnx.load(onnx_file)
    exclude = [n.name for n in model.graph.node if n.op_type not in {"Conv", "Gemm", "MatMul"}]
    # Per-channel weight scales keep small-magnitude output channels from rounding away under one per-tensor scale
    # (INT8 depth maps go near-constant). Per-axis DequantizeLinear needs opset>=13, so opset 12 stays per-tensor.
    per_channel = next(o.version for o in model.opset_import if o.domain in {"", "ai.onnx"}) >= 13
    del model

    LOGGER.info(f"{prefix} quantizing INT8 with ONNX Runtime...")
    quantize_static(
        onnx_file,
        output_file,
        onnx_calibration_reader(dataset, transform_fn, input_name),
        nodes_to_exclude=exclude,
        per_channel=per_channel,
    )
    return str(output_file)


def head_outputs(graph, head):
    """Return the tensors where an INT8 NPU graph ends: the head's per-level output convs plus extra graph outputs.

    Per-tensor INT8 cannot hold the decode's pixel-range DFL, anchor and stride math, so NPU exports stop at the head
    convs and run the remaining decode on the host in floating point.

    Args:
        graph (onnx.GraphProto): Graph of the exported ONNX model.
        head (str): ONNX node-name scope of the model head, e.g. ``"/model.23/"``.

    Returns:
        (list[str]): Output tensor names, empty when the graph has no per-level head convs (e.g. classification).
    """
    ops = {"Reshape", "Concat"}  # head convs feed only these per-level ops, e.g. Concat in WorldDetect
    ends = {x for n in graph.node if n.op_type in ops for x in n.input}
    ends -= {x for n in graph.node if n.op_type not in ops for x in n.input}
    outputs = [
        n.output[0]
        for n in graph.node
        if n.op_type == "Conv" and head in n.name and "/dfl/" not in n.name and n.output[0] in ends
    ]
    return outputs + [x.name for x in graph.output if x.name != "output0"] if outputs else []  # e.g. segment protos
