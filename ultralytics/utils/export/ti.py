# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from __future__ import annotations

from pathlib import Path

from ultralytics.utils import LOGGER, YAML
from ultralytics.utils.checks import check_requirements


def onnx2tidl(
    onnx_file: str | Path,
    output_dir: Path | str,
    target_device: str,
    dataset,
    transform_fn,
    imgsz: tuple[int, int],
    batch: int = 1,
    tensor_bits: int = 8,
    metadata: dict | None = None,
    prefix: str = "",
) -> str:
    """Compile an ONNX model to TI Deep Learning (TIDL) artifacts using ``edgeai-tidl-runtime``.

    TIDL requires a static input shape, so the source ONNX graph is first rewritten to a fixed shape before compilation;
    the resulting artifacts are only valid for that exact shape. Compilation runs entirely on the host with no TI
    account or device connection required (``edgeai-tidl-runtime`` bundles the native TIDL tools and the TIDL-enabled
    ONNX Runtime build), producing a self-contained artifacts directory that can be copied directly to a TI Edge AI
    device.

    Args:
        onnx_file (str | Path): Path to the source ONNX file (already exported).
        output_dir (Path | str): Directory to save the compiled TIDL model artifacts.
        target_device (str): Target TI SoC or device family, e.g. ``"tda4vh"``, ``"am62a"``.
        dataset (DataLoader): Calibration dataloader (from `Exporter.get_int8_calibration_dataloader`) used for
            quantization during compilation.
        transform_fn (Callable): Preprocessing transform (`Exporter._transform_fn`) converting a calibration item to a
            normalized `float32` NCHW array.
        imgsz (tuple[int, int]): Static export image size as `(height, width)`.
        batch (int): Static batch dimension of the fixed ONNX graph.
        tensor_bits (int): TIDL tensor/weight precision, 8 or 16.
        metadata (dict | None): Ultralytics model metadata saved as `metadata.yaml` alongside the artifacts.
        prefix (str): Prefix for log messages.

    Returns:
        (str): Path to the directory containing the compiled TIDL artifacts.
    """
    check_requirements("edgeai-tidl-runtime")
    from edgeai_tidl_runtime import compile_onnx

    LOGGER.info(f"\n{prefix} starting TIDL compilation for target device '{target_device}'...")
    output_dir = compile_onnx(
        onnx_file=onnx_file,
        output_dir=output_dir,
        target_device=target_device,
        calibration_dataset=dataset,
        transform_fn=transform_fn,
        imgsz=imgsz,
        batch=batch,
        tensor_bits=tensor_bits,
    )
    if metadata:
        YAML.save(Path(output_dir) / "metadata.yaml", metadata)
    return str(output_dir)
