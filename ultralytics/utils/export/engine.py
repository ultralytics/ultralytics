# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from __future__ import annotations

import json
import types
from pathlib import Path

import torch

from ultralytics.utils import IS_JETSON, LOGGER, TORCH_VERSION, ThreadingLocked, is_dgx, is_jetson
from ultralytics.utils.checks import check_tensorrt, check_version
from ultralytics.utils.torch_utils import TORCH_2_4, TORCH_2_9


def best_onnx_opset(onnx: types.ModuleType, cuda: bool = False) -> int:
    """Return max ONNX opset for this torch version with ONNX fallback."""
    if TORCH_2_4:  # _constants.ONNX_MAX_OPSET first defined in torch 1.13
        opset = torch.onnx.utils._constants.ONNX_MAX_OPSET - 1  # use second-latest version for safety
        if TORCH_2_9:
            opset = min(opset, 20)  # legacy TorchScript exporter caps at opset 20 in torch 2.9+
        if cuda:
            opset -= 2  # fix CUDA ONNXRuntime NMS squeeze op errors
    else:
        version = ".".join(TORCH_VERSION.split(".")[:2])
        opset = {
            "1.8": 12,
            "1.9": 12,
            "1.10": 13,
            "1.11": 14,
            "1.12": 15,
            "1.13": 17,
            "2.0": 17,  # reduced from 18 to fix ONNX errors
            "2.1": 17,  # reduced from 19
            "2.2": 17,  # reduced from 19
            "2.3": 17,  # reduced from 19
            "2.4": 20,
            "2.5": 20,
            "2.6": 20,
            "2.7": 20,
            "2.8": 23,
        }.get(version, 12)
    return min(opset, onnx.defs.onnx_opset_version())


@ThreadingLocked()
def torch2onnx(
    model: torch.nn.Module,
    im: torch.Tensor | tuple[torch.Tensor, ...],
    output_file: Path | str,
    opset: int = 14,
    input_names: list[str] | None = None,
    output_names: list[str] | None = None,
    dynamic: dict | None = None,
) -> str:
    """Export a PyTorch model to ONNX format.

    Args:
        model (torch.nn.Module): The PyTorch model to export.
        im (torch.Tensor | tuple[torch.Tensor, ...]): Example input tensor(s) for tracing.
        output_file (Path | str): Path to save the exported ONNX file.
        opset (int): ONNX opset version to use for export.
        input_names (list[str] | None): List of input tensor names. Defaults to ``["images"]``.
        output_names (list[str] | None): List of output tensor names. Defaults to ``["output0"]``.
        dynamic (dict | None): Dictionary specifying dynamic axes for inputs and outputs.

    Returns:
        (str): Path to the exported ONNX file.

    Notes:
        Setting `do_constant_folding=True` may cause issues with DNN inference for torch>=1.12.
    """
    if input_names is None:
        input_names = ["images"]
    if output_names is None:
        output_names = ["output0"]
    kwargs = {"dynamo": False} if TORCH_2_4 else {}
    torch.onnx.export(
        model,
        im,
        output_file,
        verbose=False,
        opset_version=opset,
        do_constant_folding=True,  # WARNING: DNN inference with torch>=1.12 may require do_constant_folding=False
        input_names=input_names,
        output_names=output_names,
        dynamic_axes=dynamic,
        **kwargs,
    )
    return str(output_file)


def onnx2engine(
    onnx_file: str,
    output_file: Path | str | None = None,
    workspace: int | None = None,
    half: bool = False,
    int8: bool = False,
    dynamic: bool = False,
    shape: tuple[int, int, int, int] = (1, 3, 640, 640),
    dla: int | None = None,
    dataset=None,
    metadata: dict | None = None,
    verbose: bool = False,
    prefix: str = "",
) -> str:
    """Export a YOLO model to TensorRT engine format.

    Args:
        onnx_file (str): Path to the ONNX file to be converted.
        output_file (Path | str | None): Path to save the generated TensorRT engine file.
        workspace (int | None): Workspace size in GB for TensorRT.
        half (bool, optional): Enable FP16 precision.
        int8 (bool, optional): Enable INT8 precision.
        dynamic (bool, optional): Enable dynamic input shapes.
        shape (tuple[int, int, int, int], optional): Input shape (batch, channels, height, width).
        dla (int | None): DLA core to use (Jetson devices only).
        dataset (ultralytics.data.build.InfiniteDataLoader, optional): Dataset for INT8 calibration.
        metadata (dict | None): Metadata to include in the engine file.
        verbose (bool, optional): Enable verbose logging.
        prefix (str, optional): Prefix for log messages.

    Returns:
        (str): Path to the exported engine file.

    Raises:
        ValueError: If DLA is enabled on non-Jetson devices or required precision is not set.
        RuntimeError: If the ONNX file cannot be parsed.

    Notes:
        TensorRT version compatibility is handled for workspace size and engine building.
        INT8 calibration requires a dataset and generates a calibration cache.
        Metadata is serialized and written to the engine file if provided.
    """
    # Force re-install TensorRT on CUDA 13 ARM devices to 10.15.x versions for RT-DETR exports
    # https://github.com/ultralytics/ultralytics/issues/22873
    if is_jetson(jetpack=7) or is_dgx():
        check_tensorrt("10.15")

    try:
        import tensorrt as trt
    except ImportError:
        check_tensorrt()
        import tensorrt as trt
    check_version(trt.__version__, ">=7.0.0", hard=True)
    check_version(trt.__version__, "!=10.1.0", msg="https://github.com/ultralytics/ultralytics/pull/14239")

    LOGGER.info(f"\n{prefix} starting export with TensorRT {trt.__version__}...")
    output_file = output_file or Path(onnx_file).with_suffix(".engine")

    logger = trt.Logger(trt.Logger.INFO)
    if verbose:
        logger.min_severity = trt.Logger.Severity.VERBOSE

    # Engine builder
    builder = trt.Builder(logger)
    config = builder.create_builder_config()
    workspace_bytes = int((workspace or 0) * (1 << 30))
    is_trt10 = int(trt.__version__.split(".", 1)[0]) >= 10  # is TensorRT >= 10
    if is_trt10 and workspace_bytes > 0:
        config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, workspace_bytes)
    elif workspace_bytes > 0:  # TensorRT versions 7, 8
        config.max_workspace_size = workspace_bytes
    flag = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    network = builder.create_network(flag)
    half = builder.platform_has_fast_fp16 and half
    int8 = builder.platform_has_fast_int8 and int8

    # Optionally switch to DLA if enabled
    if dla is not None:
        if not IS_JETSON:
            raise ValueError("DLA is only available on NVIDIA Jetson devices")
        LOGGER.info(f"{prefix} enabling DLA on core {dla}...")
        if not half and not int8:
            raise ValueError(
                "DLA requires either 'half=True' (FP16) or 'int8=True' (INT8) to be enabled. Please enable one of them and try again."
            )
        config.default_device_type = trt.DeviceType.DLA
        config.DLA_core = int(dla)
        config.set_flag(trt.BuilderFlag.GPU_FALLBACK)

    # Read ONNX file
    parser = trt.OnnxParser(network, logger)
    if not parser.parse_from_file(onnx_file):
        raise RuntimeError(f"failed to load ONNX file: {onnx_file}")

    # Network inputs
    inputs = [network.get_input(i) for i in range(network.num_inputs)]
    outputs = [network.get_output(i) for i in range(network.num_outputs)]
    for inp in inputs:
        LOGGER.info(f'{prefix} input "{inp.name}" with shape{inp.shape} {inp.dtype}')
    for out in outputs:
        LOGGER.info(f'{prefix} output "{out.name}" with shape{out.shape} {out.dtype}')

    if dynamic:
        profile = builder.create_optimization_profile()
        min_shape = (1, shape[1], 32, 32)  # minimum input shape
        max_shape = (*shape[:2], *(int(max(2, workspace or 2) * d) for d in shape[2:]))  # max input shape
        for inp in inputs:
            profile.set_shape(inp.name, min=min_shape, opt=shape, max=max_shape)
        config.add_optimization_profile(profile)
        if int8 and not is_trt10:  # deprecated in TensorRT 10, causes internal errors
            config.set_calibration_profile(profile)

    LOGGER.info(f"{prefix} building {'INT8' if int8 else 'FP' + ('16' if half else '32')} engine as {output_file}")
    if int8:
        config.set_flag(trt.BuilderFlag.INT8)
        config.profiling_verbosity = trt.ProfilingVerbosity.DETAILED

        class EngineCalibrator(trt.IInt8Calibrator):
            """Custom INT8 calibrator for TensorRT engine optimization.

            This calibrator provides the necessary interface for TensorRT to perform INT8 quantization calibration using
            a dataset. It handles batch generation, caching, and calibration algorithm selection.

            Attributes:
                dataset: Dataset for calibration.
                data_iter: Iterator over the calibration dataset.
                algo (trt.CalibrationAlgoType): Calibration algorithm type.
                batch (int): Batch size for calibration.
                cache (Path): Path to save the calibration cache.

            Methods:
                get_algorithm: Get the calibration algorithm to use.
                get_batch_size: Get the batch size to use for calibration.
                get_batch: Get the next batch to use for calibration.
                read_calibration_cache: Use existing cache instead of calibrating again.
                write_calibration_cache: Write calibration cache to disk.
            """

            def __init__(
                self,
                dataset,  # ultralytics.data.build.InfiniteDataLoader
                cache: str = "",
            ) -> None:
                """Initialize the INT8 calibrator with dataset and cache path."""
                trt.IInt8Calibrator.__init__(self)
                self.dataset = dataset
                self.data_iter = iter(dataset)
                self.algo = (
                    trt.CalibrationAlgoType.ENTROPY_CALIBRATION_2  # DLA quantization needs ENTROPY_CALIBRATION_2
                    if dla is not None
                    else trt.CalibrationAlgoType.MINMAX_CALIBRATION
                )
                self.batch = dataset.batch_size
                self.cache = Path(cache)

            def get_algorithm(self) -> trt.CalibrationAlgoType:
                """Get the calibration algorithm to use."""
                return self.algo

            def get_batch_size(self) -> int:
                """Get the batch size to use for calibration."""
                return self.batch or 1

            def get_batch(self, names) -> list[int] | None:
                """Get the next batch to use for calibration, as a list of device memory pointers."""
                try:
                    im0s = next(self.data_iter)["img"] / 255.0
                    im0s = im0s.to("cuda") if im0s.device.type == "cpu" else im0s
                    return [int(im0s.data_ptr())]
                except StopIteration:
                    # Return None to signal to TensorRT there is no calibration data remaining
                    return None

            def read_calibration_cache(self) -> bytes | None:
                """Use existing cache instead of calibrating again, otherwise, implicitly return None."""
                if self.cache.exists() and self.cache.suffix == ".cache":
                    return self.cache.read_bytes()

            def write_calibration_cache(self, cache: bytes) -> None:
                """Write calibration cache to disk."""
                _ = self.cache.write_bytes(cache)

        # Load dataset w/ builder (for batching) and calibrate
        config.int8_calibrator = EngineCalibrator(
            dataset=dataset,
            cache=str(Path(onnx_file).with_suffix(".cache")),
        )

    elif half:
        config.set_flag(trt.BuilderFlag.FP16)

    # Write file
    if is_trt10:
        # TensorRT 10+ returns bytes directly, not a context manager
        engine = builder.build_serialized_network(network, config)
        if engine is None:
            raise RuntimeError("TensorRT engine build failed, check logs for errors")
        with open(output_file, "wb") as t:
            if metadata is not None:
                meta = json.dumps(metadata)
                t.write(len(meta).to_bytes(4, byteorder="little", signed=True))
                t.write(meta.encode())
            t.write(engine)
    else:
        with builder.build_engine(network, config) as engine, open(output_file, "wb") as t:
            if metadata is not None:
                meta = json.dumps(metadata)
                t.write(len(meta).to_bytes(4, byteorder="little", signed=True))
                t.write(meta.encode())
            t.write(engine.serialize())
    return str(output_file)
