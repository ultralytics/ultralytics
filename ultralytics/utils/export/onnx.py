# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from pathlib import Path

from ultralytics.utils import LOGGER


def onnx_calibration_reader(dataset, transform_fn, input_name: str = "images"):
    """Create an ONNX Runtime calibration data reader from an Ultralytics calibration dataloader."""
    from onnxruntime.quantization import CalibrationDataReader

    class _CalibrationReader(CalibrationDataReader):
        def __init__(self):
            """Materialize calibration inputs as `{input_name: float32_NCHW}` dicts."""
            self.samples = [{input_name: transform_fn(batch)} for batch in dataset]
            self.iterator = iter(self.samples)

        def get_next(self):
            """Return the next calibration sample, or None when exhausted."""
            return next(self.iterator, None)

        def rewind(self):
            """Reset the iterator for an additional calibration pass."""
            self.iterator = iter(self.samples)

    return _CalibrationReader()


def onnx_int8_quantize(
    onnx_file,
    output_file,
    dataset,
    transform_fn,
    input_name: str = "images",
    prefix: str = "",
) -> str:
    """Quantize an ONNX model to INT8 using ONNX Runtime static quantization."""
    from onnxruntime.quantization import quantize_static

    onnx_file, output_file = Path(onnx_file), Path(output_file)
    LOGGER.info(f"{prefix} quantizing INT8 with ONNX Runtime...")
    quantize_static(str(onnx_file), str(output_file), onnx_calibration_reader(dataset, transform_fn, input_name))
    return str(output_file)
