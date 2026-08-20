# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

import numpy as np

from ultralytics.utils import LOGGER


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
    del graph

    LOGGER.info(f"{prefix} quantizing INT8 with ONNX Runtime...")
    quantize_static(
        onnx_file,
        output_file,
        onnx_calibration_reader(dataset, transform_fn, input_name, batch),
        nodes_to_exclude=exclude,
    )
    return str(output_file)
