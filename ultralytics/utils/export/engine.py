# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from __future__ import annotations

import json
import re
import types
from pathlib import Path

import numpy as np
import torch

from ultralytics.utils import IS_JETSON, LOGGER, TORCH_VERSION, ThreadingLocked, is_dgx, is_jetson
from ultralytics.utils.checks import check_requirements, check_tensorrt, check_version
from ultralytics.utils.torch_utils import TORCH_2_4


class _NormalizeCoords(torch.nn.Module):
    """Wrap a model with input-relative box and pose coordinates for per-tensor quantization."""

    def __init__(self, model: torch.nn.Module, h: int, w: int, task: str, nc: int, kpt_shape: tuple | None):
        """Initialize with the wrapped model and prediction metadata."""
        super().__init__()
        self.model = model
        self.h = h
        self.w = w
        self.task = task
        self.nc = nc
        self.kpt_shape = kpt_shape

    def forward(self, x: torch.Tensor):
        """Run the wrapped model and normalize its coordinate channels by input size."""
        y = self.model(x)
        det = y[0] if isinstance(y, (tuple, list)) else y
        box_wh = torch.tensor([self.w, self.h, self.w, self.h], dtype=det.dtype, device=det.device).view(1, 4, 1)
        parts = [det[:, :4] / box_wh]
        if self.task == "pose" and self.kpt_shape:
            parts.append(det[:, 4 : 4 + self.nc])
            b, _, a = det.shape
            kpts = det[:, 4 + self.nc :].view(b, self.kpt_shape[0], self.kpt_shape[1], a)
            kpt_wh = torch.tensor([self.w, self.h], dtype=det.dtype, device=det.device).view(1, 1, 2, 1)
            kpts = torch.cat([kpts[:, :, :2] / kpt_wh, kpts[:, :, 2:]], dim=2)
            parts.append(kpts.reshape(b, -1, a))
        else:
            parts.append(det[:, 4 : 4 + self.nc])
            if det.shape[1] > 4 + self.nc:
                parts.append(det[:, 4 + self.nc :])
        det = torch.cat(parts, dim=1)
        return (det, *y[1:]) if isinstance(y, (tuple, list)) else det


def best_onnx_opset(onnx: types.ModuleType) -> int:
    """Return max ONNX opset for this torch version with ONNX fallback."""
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
    }.get(version, 18)
    # torch>=2.4 supports opset>=19, but ONNX Runtime CUDA has no Resize-19 or ReduceMax-20 kernel, so opset>=19 runs
    # those nodes on the CPU and copies their tensors back and forth. Its static INT8 quantization also rejects opset>=21.
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
        opset_version=opset,
        input_names=input_names,
        output_names=output_names,
        dynamic_axes=dynamic,
        **kwargs,
    )
    return str(output_file)


def modelopt_quantize_onnx(
    onnx_file: str,
    quantize: int | str | None = None,
    dataset=None,
    shape: tuple[int, int, int, int] = (1, 3, 640, 640),
    dynamic: bool = False,
    dynamic_dim: int = 1,
    calib_shapes: dict[str, tuple] | None = None,
    prefix: str = "",
) -> str:
    """Bake reduced precision into an ONNX model for TensorRT 11 strongly-typed builds using NVIDIA ModelOpt.

    TensorRT 11 is strongly-typed only: it removed the FP16/INT8 builder flags and the ``IInt8Calibrator`` interface, so
    reduced precision must be expressed in the ONNX graph itself before building. FP16 is applied via ModelOpt AutoCast
    mixed-precision conversion and INT8 via explicit Q/DQ quantization with calibration.

    Args:
        onnx_file (str): Path to the FP32 ONNX file to convert.
        quantize (int | str | None): Precision scheme, 8 for INT8 Q/DQ nodes or 16 for FP16 precision.
        dataset (ultralytics.data.build.InfiniteDataLoader | None): Dataloader providing INT8 calibration images.
            Required when ``quantize=8``.
        shape (tuple[int, int, int, int]): Input shape (batch, channels, height, width) used for dynamic calibration.
        dynamic (bool): Whether the ONNX model uses dynamic input shapes.
        dynamic_dim (int): Size to substitute for symbolic dimensions when synthesizing calibration data.
        calib_shapes (dict[str, tuple] | None): Exact calibration shape per input name, overriding the synthesized
            one. Needed when several inputs carry related symbolic dimensions, such as feature pyramid levels that
            must stay at their fixed ratio for the graph to run at all.
        prefix (str): Prefix for log messages.

    Returns:
        (str): Path to the precision-converted ONNX file.
    """
    if quantize == 8 and dataset is None:
        raise ValueError("INT8 ModelOpt quantization requires a calibration dataset.")

    # Require modelopt >= 0.44: older releases import onnx.mapping which was removed in onnx >= 1.18 and crash
    check_requirements("nvidia-modelopt[onnx]>=0.44")
    import onnx

    graph_inputs = onnx.load(onnx_file, load_external_data=False).graph.input
    input_name = graph_inputs[0].name
    if quantize == 8:
        from modelopt.onnx.quantization import quantize as modelopt_quantize

        out_file = str(Path(onnx_file).with_suffix(".int8.onnx"))
        # Collect up to ~500 calibration images (TensorRT recommendation); ModelOpt holds them in memory at once,
        # so cap the count to bound memory instead of materializing the entire (possibly thousands-image) dataset.
        images, n = [], 0
        for batch in dataset:
            images.append(batch["img"])
            n += images[-1].shape[0]
            if n >= 512:
                break
        calib = torch.cat(images).to(torch.float32) / 255.0
        LOGGER.info(f"{prefix} quantizing ONNX to INT8 with ModelOpt using {calib.shape[0]} calibration images...")
        kwargs = {"calibration_shapes": f"{input_name}:{'x'.join(str(d) for d in shape)}"} if dynamic else {}
        modelopt_quantize(
            onnx_file,
            quantize_mode="int8",
            calibration_data={input_name: calib.cpu().numpy()},
            calibration_method="max",
            # Calibrate on CPU. ModelOpt's CUDA EP session can hit an uncatchable cuDNN-ABI segfault (its pinned
            # onnxruntime-gpu's cuDNN vs the installed torch's) and the TensorRT EP aborts on RTX cards (NvTensorRTRTX);
            # scales are EP-independent, so the INT8 engine is equivalent and only this one-time step is slower.
            calibration_eps=["cpu"],
            output_path=out_file,
            **kwargs,
        )
        return out_file

    from modelopt.onnx import autocast

    # AutoCast only needs representative shapes and ranges, so synthesize one tensor per graph input
    # from its own declared rank and dtype. A single 4D float image reproduces the original behavior,
    # while multi input graphs also get valid token ids, masks and symbolic dims.
    calib = {}
    for inp in graph_inputs:
        tt = inp.type.tensor_type
        dims = [d.dim_value if d.dim_value > 0 else dynamic_dim for d in tt.shape.dim]
        if not dims:  # a scalar input still needs an array
            dims = [1]
        if calib_shapes and inp.name in calib_shapes:
            dims = list(calib_shapes[inp.name])
        # Only a dynamic image input needs the caller's size; a declared shape is already correct and
        # must be kept, otherwise a non image first input would be calibrated at the wrong shape.
        elif inp.name == input_name and len(dims) == len(shape) and any(d.dim_value <= 0 for d in tt.shape.dim):
            dims = list(shape)
        np_dtype = onnx.helper.tensor_dtype_to_np_dtype(tt.elem_type)
        if np_dtype == np.bool_:
            calib[inp.name] = np.zeros(dims, dtype=np.bool_)
        elif np.issubdtype(np_dtype, np.integer):
            # Integer inputs are usually indices into an embedding, so keep them small and in range.
            calib[inp.name] = np.ones(dims, dtype=np_dtype)
        else:
            calib[inp.name] = np.random.randn(*dims).astype(np_dtype)

    out_file = str(Path(onnx_file).with_suffix(".fp16.onnx"))
    LOGGER.info(f"{prefix} converting ONNX to FP16 mixed precision with ModelOpt AutoCast...")
    onnx.save(
        autocast.convert_to_mixed_precision(
            onnx_file, low_precision_type="fp16", keep_io_types=True, calibration_data=calib
        ),
        out_file,
    )
    return out_file


def onnx2engine(
    onnx_file: str,
    output_file: Path | str | None = None,
    workspace: int | None = None,
    quantize: int | str | None = None,
    dynamic: bool = False,
    shape: tuple[int, int, int, int] = (1, 3, 640, 640),
    dla: int | None = None,
    dataset=None,
    metadata: dict | None = None,
    verbose: bool = False,
    prefix: str = "",
    profile_shapes: dict[str, tuple[tuple, ...]] | None = None,
    strongly_typed: bool = False,
    onnx_bytes: bytes | None = None,
) -> str:
    """Export a YOLO model to TensorRT engine format.

    Args:
        onnx_file (str): Path to the ONNX file to be converted.
        output_file (Path | str | None): Path to save the generated TensorRT engine file.
        workspace (int | None): Workspace size in GB for TensorRT.
        quantize (int | str | None): Precision scheme, 16 for FP16 or 8 for INT8.
        dynamic (bool, optional): Enable dynamic input shapes.
        shape (tuple[int, int, int, int], optional): Input shape (batch, channels, height, width).
        dla (int | None): DLA core to use (Jetson devices only).
        dataset (ultralytics.data.build.InfiniteDataLoader, optional): Dataset for INT8 calibration.
        metadata (dict | None): Metadata to include in the engine file.
        verbose (bool, optional): Enable verbose logging.
        prefix (str, optional): Prefix for log messages.
        profile_shapes (dict[str, tuple[tuple, ...]] | None): Per-input ``(min, opt, max)`` shapes for the optimization
            profile, covering every network input. For multi-input graphs where one uniform ``dynamic`` profile is
            wrong. Implies a profile is always added, and each input must be present.
        strongly_typed (bool): Create a strongly-typed network (TensorRT >= 10) that honors per-node precision baked
            into the ONNX, instead of the legacy FP16 builder flag. Pair with a ModelOpt-quantized ``onnx_file``.
        onnx_bytes (bytes | None): Parse these bytes instead of reading ``onnx_file``, for graphs that need a
            TensorRT-specific adjustment the exported file should not carry.

    Returns:
        (str): Path to the exported engine file.

    Raises:
        ValueError: If DLA is enabled on non-Jetson devices or required precision is not set.
        RuntimeError: If the ONNX file cannot be parsed.

    Notes:
        TensorRT version compatibility is handled for workspace size and engine building. On TensorRT 7-10, INT8
        calibration uses an ``IInt8Calibrator`` over ``dataset`` and writes a calibration cache, while FP16/INT8 are
        enabled with builder flags. On TensorRT 11 these were removed in favor of strongly-typed networks, so reduced
        precision is baked into the ONNX with NVIDIA ModelOpt before building (FP16 AutoCast, INT8 explicit Q/DQ) by
        `modelopt_quantize_onnx`. The TensorRT 7-10 path keeps the Sigmoid layers at higher precision to preserve
        confidence-score calibration (see #24668). Metadata is serialized and written to the engine file if provided.
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
    check_version(trt.__version__, "!=10.2.0", msg="https://github.com/ultralytics/ultralytics/pull/24367")

    LOGGER.info(f"\n{prefix} starting export with TensorRT {trt.__version__}...")
    output_file = output_file or Path(onnx_file).with_suffix(".engine")

    logger = trt.Logger(trt.Logger.INFO)
    if verbose:
        logger.min_severity = trt.Logger.Severity.VERBOSE

    # Engine builder
    builder = trt.Builder(logger)
    config = builder.create_builder_config()
    workspace_bytes = int((workspace or 0) * (1 << 30))
    trt_major = int(trt.__version__.split(".", 1)[0])
    is_trt10 = trt_major >= 10
    # TensorRT >= 11 is strongly-typed only: precision builder flags and IInt8Calibrator removed
    is_trt11 = trt_major >= 11
    if workspace_bytes > 0:
        if hasattr(config, "set_memory_pool_limit"):
            config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, workspace_bytes)
        else:  # TensorRT 7 fallback
            config.max_workspace_size = workspace_bytes
    # EXPLICIT_BATCH flag is removed in TensorRT 10 (explicit batch is the only/default mode); keep it for TRT 7/8
    flag = 0 if is_trt10 else (1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    if strongly_typed:  # a strongly-typed network honors per-node precision baked into the ONNX graph
        check_version(trt.__version__, ">=10.0.0", name="TensorRT strongly-typed build", hard=True)
        flag = 1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED)
    network = builder.create_network(flag)
    # platform_has_fast_fp16/int8 were removed from the Builder in TensorRT 10; default to True when absent
    use_fp16 = getattr(builder, "platform_has_fast_fp16", True) and quantize == 16
    use_int8 = getattr(builder, "platform_has_fast_int8", True) and quantize == 8
    if use_int8 and dataset is None:
        raise ValueError("INT8 TensorRT export requires a calibration dataset.")

    # Optionally switch to DLA if enabled
    if dla is not None:
        if not IS_JETSON:
            raise ValueError("DLA is only available on NVIDIA Jetson devices")
        if check_version(trt.__version__, ">=11.0.0,<11.1.0"):
            # DLA is unsupported in TensorRT 11.0 and is planned to return in a later release
            # https://docs.nvidia.com/deeplearning/tensorrt/latest/api/migration/tensorrt-10x-to-11x-jetson.html
            raise ValueError("DLA is not supported in TensorRT 11.0; export with TensorRT 10.x to use DLA.")
        LOGGER.info(f"{prefix} enabling DLA on core {dla}...")
        if not use_fp16 and not use_int8:
            raise ValueError(
                "DLA requires either quantize=16 (FP16) or quantize=8 (INT8). Please enable one of them and try again."
            )
        config.default_device_type = trt.DeviceType.DLA
        config.DLA_core = int(dla)
        config.set_flag(trt.BuilderFlag.GPU_FALLBACK)

    # TensorRT 11 is strongly-typed and removed the FP16/INT8 builder flags and INT8 calibrator, so reduced
    # precision must be baked into the ONNX graph with NVIDIA ModelOpt before parsing (FP16 AutoCast, INT8 Q/DQ).
    # A strongly-typed build on TensorRT 10 follows the same path so its per-node precision is honored.
    if (is_trt11 or strongly_typed) and (use_fp16 or use_int8):
        onnx_file = modelopt_quantize_onnx(onnx_file, quantize, dataset, shape, dynamic, prefix=prefix)

    # Read ONNX file
    parser = trt.OnnxParser(network, logger)
    if not (parser.parse(onnx_bytes) if onnx_bytes is not None else parser.parse_from_file(onnx_file)):
        raise RuntimeError(f"failed to load ONNX file: {onnx_file}")

    # Network inputs
    inputs = [network.get_input(i) for i in range(network.num_inputs)]
    outputs = [network.get_output(i) for i in range(network.num_outputs)]
    for inp in inputs:
        LOGGER.info(f'{prefix} input "{inp.name}" with shape{inp.shape} {inp.dtype}')
    for out in outputs:
        LOGGER.info(f'{prefix} output "{out.name}" with shape{out.shape} {out.dtype}')

    if profile_shapes:
        profile = builder.create_optimization_profile()
        for inp in inputs:
            assert inp.name in profile_shapes, f"{prefix} no profile shape for input '{inp.name}'"
            profile.set_shape(inp.name, *profile_shapes[inp.name])
        config.add_optimization_profile(profile)
    elif dynamic:
        profile = builder.create_optimization_profile()
        min_shape = (1, shape[1], 32, 32)  # minimum input shape
        max_shape = (*shape[:2], *(int(max(2, workspace or 2) * d) for d in shape[2:]))  # max input shape
        for inp in inputs:
            inp_min = tuple(d if d != -1 else lo for d, lo in zip(inp.shape, min_shape))
            inp_max = tuple(d if d != -1 else hi for d, hi in zip(inp.shape, max_shape))
            profile.set_shape(inp.name, min=inp_min, opt=shape, max=inp_max)
        config.add_optimization_profile(profile)
        if use_int8 and not is_trt10:  # deprecated in TensorRT 10, causes internal errors
            config.set_calibration_profile(profile)

    precision = "INT8" if use_int8 else "mixed FP16" if strongly_typed else f"FP{'16' if use_fp16 else '32'}"
    LOGGER.info(f"{prefix} building {precision} engine as {output_file}")
    if use_int8 and not (is_trt11 or strongly_typed):
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

        # Implicit quantization cannot exclude op types like ModelOpt on TRT 11, so keep the head Sigmoid (an
        # ACTIVATION layer named after its ONNX node) in FP32 via per-layer precision constraints to preserve
        # confidence-score calibration, mirroring the OpenVINO IgnoredScope
        # https://github.com/ultralytics/ultralytics/issues/24668. Scope this to the head: every SiLU activation is
        # also a Sigmoid, and constraining all of them costs INT8 speed across backbone and neck.
        names = [network.get_layer(i).name for i in range(network.num_layers)]
        indices = [int(m.group(1)) for n in names if (m := re.match(r"/model\.(\d+)/", n))]
        head = f"/model.{max(indices)}/" if indices else "/"
        count = 0
        for i in range(network.num_layers):
            layer = network.get_layer(i)
            if (
                layer.type == trt.LayerType.ACTIVATION
                and "sigmoid" in layer.name.lower()
                and layer.name.startswith(head)
            ):
                layer.precision = trt.float32
                for j in range(layer.num_outputs):
                    layer.set_output_type(j, trt.float32)
                count += 1
        if count:
            flag = (
                trt.BuilderFlag.OBEY_PRECISION_CONSTRAINTS
                if hasattr(trt.BuilderFlag, "OBEY_PRECISION_CONSTRAINTS")
                else trt.BuilderFlag.STRICT_TYPES
            )
            config.set_flag(flag)  # OBEY_PRECISION_CONSTRAINTS replaced STRICT_TYPES in TensorRT 8.2
            LOGGER.info(f"{prefix} keeping {count} head Sigmoid layers in FP32 for INT8 accuracy")

    elif use_fp16 and not (is_trt11 or strongly_typed):
        config.set_flag(trt.BuilderFlag.FP16)

    # Write file
    if hasattr(builder, "build_serialized_network"):
        engine = builder.build_serialized_network(network, config)
    else:
        engine = builder.build_engine(network, config)
        engine = None if engine is None else engine.serialize()
    if engine is None:
        raise RuntimeError("TensorRT engine build failed, check logs for errors")
    with open(output_file, "wb") as t:
        if metadata is not None:
            meta = json.dumps(metadata)
            t.write(len(meta).to_bytes(4, byteorder="little", signed=True))
            t.write(meta.encode())
        t.write(engine)
    return str(output_file)
