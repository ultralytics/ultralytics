# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

import json
from pathlib import Path

import torch

from ultralytics.utils import IS_JETSON, LOGGER


def export_onnx(
    torch_model,
    im,
    onnx_file,
    opset=14,
    input_names=["images"],
    output_names=["output0"],
    dynamic=False,
):
    """
    Exports a PyTorch model to ONNX format.

    Args:
        torch_model (torch.nn.Module): The PyTorch model to export.
        im (torch.Tensor): Example input tensor for the model.
        onnx_file (str): Path to save the exported ONNX file.
        opset (int): ONNX opset version to use for export.
        input_names (list): List of input tensor names.
        output_names (list): List of output tensor names.
        dynamic (bool | dict, optional): Whether to enable dynamic axes. Defaults to False.

    Notes:
        - Setting `do_constant_folding=True` may cause issues with DNN inference for torch>=1.12.
    """
    torch.onnx.export(
        torch_model,
        im,
        onnx_file,
        verbose=False,
        opset_version=opset,
        do_constant_folding=True,  # WARNING: DNN inference with torch>=1.12 may require do_constant_folding=False
        input_names=input_names,
        output_names=output_names,
        dynamic_axes=dynamic or None,
    )


def export_engine(
    onnx_file,
    engine_file=None,
    workspace=None,
    half=False,
    int8=False,
    dynamic=False,
    shape=(1, 3, 640, 640),
    dla=None,
    dataset=None,
    metadata=None,
    verbose=False,
    prefix="",
):
    """
    Exports a YOLO model to TensorRT engine format.

    Args:
        onnx_file (str): Path to the ONNX file to be converted.
        engine_file (str, optional): Path to save the generated TensorRT engine file.
        workspace (int, optional): Workspace size in GB for TensorRT. Defaults to None.
        half (bool, optional): Enable FP16 precision. Defaults to False.
        int8 (bool, optional): Enable INT8 precision. Defaults to False.
        dynamic (bool, optional): Enable dynamic input shapes. Defaults to False.
        shape (tuple, optional): Input shape (batch, channels, height, width). Defaults to (1, 3, 640, 640).
        dla (int, optional): DLA core to use (Jetson devices only). Defaults to None.
        dataset (ultralytics.data.build.InfiniteDataLoader, optional): Dataset for INT8 calibration. Defaults to None.
        metadata (dict, optional): Metadata to include in the engine file. Defaults to None.
        verbose (bool, optional): Enable verbose logging. Defaults to False.
        prefix (str, optional): Prefix for log messages. Defaults to "".

    Raises:
        ValueError: If DLA is enabled on non-Jetson devices or required precision is not set.
        RuntimeError: If the ONNX file cannot be parsed.

    Notes:
        - TensorRT version compatibility is handled for workspace size and engine building.
        - INT8 calibration requires a dataset and generates a calibration cache.
        - Metadata is serialized and written to the engine file if provided.
    """
    import tensorrt as trt  # noqa

    engine_file = engine_file or Path(onnx_file).with_suffix(".engine")

    logger = trt.Logger(trt.Logger.INFO)
    if verbose:
        logger.min_severity = trt.Logger.Severity.VERBOSE

    # Engine builder
    builder = trt.Builder(logger)
    config = builder.create_builder_config()
    workspace = int((workspace or 0) * (1 << 30))
    is_trt10 = int(trt.__version__.split(".", 1)[0]) >= 10  # is TensorRT >= 10
    if is_trt10 and workspace > 0:
        config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, workspace)
    elif workspace > 0:  # TensorRT versions 7, 8
        config.max_workspace_size = workspace
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
        if shape[0] <= 1:
            LOGGER.warning(f"{prefix} 'dynamic=True' model requires max batch size, i.e. 'batch=16'")
        profile = builder.create_optimization_profile()
        min_shape = (1, shape[1], 32, 32)  # minimum input shape
        max_shape = (*shape[:2], *(int(max(2, workspace or 2) * d) for d in shape[2:]))  # max input shape
        for inp in inputs:
            profile.set_shape(inp.name, min=min_shape, opt=shape, max=max_shape)
        config.add_optimization_profile(profile)

    LOGGER.info(f"{prefix} building {'INT8' if int8 else 'FP' + ('16' if half else '32')} engine as {engine_file}")
    if int8:
        config.set_flag(trt.BuilderFlag.INT8)
        config.set_calibration_profile(profile)
        config.profiling_verbosity = trt.ProfilingVerbosity.DETAILED

        class EngineCalibrator(trt.IInt8Calibrator):
            """
            Custom INT8 calibrator for TensorRT.

            Args:
                dataset (object): Dataset for calibration.
                batch (int): Batch size for calibration.
                cache (str, optional): Path to save the calibration cache. Defaults to "".
            """

            def __init__(
                self,
                dataset,  # ultralytics.data.build.InfiniteDataLoader
                cache: str = "",
            ) -> None:
                trt.IInt8Calibrator.__init__(self)
                self.dataset = dataset
                self.data_iter = iter(dataset)
                self.algo = trt.CalibrationAlgoType.MINMAX_CALIBRATION
                self.batch = dataset.batch_size
                self.cache = Path(cache)

            def get_algorithm(self) -> trt.CalibrationAlgoType:
                """Get the calibration algorithm to use."""
                return self.algo

            def get_batch_size(self) -> int:
                """Get the batch size to use for calibration."""
                return self.batch or 1

            def get_batch(self, names) -> list:
                """Get the next batch to use for calibration, as a list of device memory pointers."""
                try:
                    im0s = next(self.data_iter)["img"] / 255.0
                    im0s = im0s.to("cuda") if im0s.device.type == "cpu" else im0s
                    return [int(im0s.data_ptr())]
                except StopIteration:
                    # Return [] or None, signal to TensorRT there is no calibration data remaining
                    return None

            def read_calibration_cache(self) -> bytes:
                """Use existing cache instead of calibrating again, otherwise, implicitly return None."""
                if self.cache.exists() and self.cache.suffix == ".cache":
                    return self.cache.read_bytes()

            def write_calibration_cache(self, cache) -> None:
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
    build = builder.build_serialized_network if is_trt10 else builder.build_engine
    with build(network, config) as engine, open(engine_file, "wb") as t:
        # Metadata
        if metadata is not None:
            meta = json.dumps(metadata)
            t.write(len(meta).to_bytes(4, byteorder="little", signed=True))
            t.write(meta.encode())
        # Model
        t.write(engine if is_trt10 else engine.serialize())
