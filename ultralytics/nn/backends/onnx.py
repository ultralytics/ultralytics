# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from __future__ import annotations

import hashlib
import os
import shutil
from functools import lru_cache
from pathlib import Path

import numpy as np
import torch

from ultralytics.utils import ARM64, LOGGER, USER_CONFIG_DIR, ThreadingLocked
from ultralytics.utils.checks import IS_PYTHON_MINIMUM_3_11, check_requirements, rocm_is_available

from .base import BaseBackend

# AMD wheel indexes for the MIGraphX EP plugin and its C library (ROCm 10 / MIGraphX 2.17 / onnxruntime 1.29)
ROCM_EXTRA_INDEX = (
    "--extra-index-url https://stable.repo.amd.com/rocm/onnxruntime/whl-next/ "
    "--extra-index-url https://stable.repo.amd.com/rocm/migraphx/whl-next/"
)
# migraphx-libs is not a declared plugin dependency (ROCm/AMDMIGraphX#5235). Exact local-version pins keep the rolling
# whl-next indexes on the validated stack and can only resolve from AMD's index (PyPI rejects local versions).
ROCM_EP_PACKAGES = ["onnxruntime-ep-migraphx==1.0.0+rocm10.0.0", "migraphx-libs==2.17.0+rocm10.0.0"]
MIGRAPHX_CACHE_MODELS = 8  # compiled programs kept in the MIGraphX cache, least recently used evicted first


def _register_migraphx_ep(onnxruntime) -> str:
    """Register the MIGraphX plugin EP once per process and return its name.

    The plugin (`onnxruntime-ep-migraphx`) is not auto-registered. Its libs are preloaded with RTLD_GLOBAL to work
    around a missing `libonnxruntime.so.1` soname link (ROCm/AMDMIGraphX#5235).

    Args:
        onnxruntime (module): The imported onnxruntime module.

    Returns:
        (str): The registered execution provider name.
    """
    import ctypes
    import glob

    import migraphx_libs
    import onnxruntime_ep_migraphx as ep

    name = ep.get_ep_name()
    if name not in onnxruntime.get_available_providers():  # not yet registered in this process
        for pattern in (
            Path(onnxruntime.__file__).parent / "capi" / "libonnxruntime.so.1*",
            Path(migraphx_libs.__file__).parent / "libmigraphx_c.so.3*",
        ):
            if hits := sorted(glob.glob(str(pattern))):
                ctypes.CDLL(hits[0], mode=ctypes.RTLD_GLOBAL)
        onnxruntime.register_execution_provider_library(name, ep.get_library_path())
    return name


@lru_cache(maxsize=1)
def _migraphx_cache_root() -> Path:
    """Resolve the MIGraphX compiled-program cache root once per process.

    Cached so per-model ORT_MIGRAPHX_CACHE_DIR overwrites don't nest each cache under the last.

    Returns:
        (Path): Cache root from ORT_MIGRAPHX_CACHE_DIR if set, else under USER_CONFIG_DIR.
    """
    return Path(os.environ.get("ORT_MIGRAPHX_CACHE_DIR") or USER_CONFIG_DIR / "migraphx_cache")


def _migraphx_cache_dir(weight: str | Path) -> Path:
    """Return a per-model cache subdirectory for the MIGraphX compiled program, evicting least recently used models.

    The EP keys its cache by graph and input shapes (not weights), so hashing the model bytes isolates each model. Only
    the MIGRAPHX_CACHE_MODELS most recently used models are kept, bounding the cache on disk.

    Args:
        weight (str | Path): Path to the .onnx model file, hashed to key the cache.

    Returns:
        (Path): Per-model cache subdirectory under the resolved cache root.
    """
    cache_dir = _migraphx_cache_root() / hashlib.sha256(Path(weight).read_bytes()).hexdigest()[:16]
    cache_dir.mkdir(parents=True, exist_ok=True)
    os.utime(cache_dir)  # mark as most recently used
    dirs = sorted(cache_dir.parent.glob("[0-9a-f]" * 16), key=lambda d: d.stat().st_mtime, reverse=True)  # ours only
    for d in dirs[MIGRAPHX_CACHE_MODELS:]:
        shutil.rmtree(d, ignore_errors=True)
    return cache_dir


@ThreadingLocked()  # ORT_MIGRAPHX_CACHE_DIR is process-global until the session is built
def _load_migraphx_session(onnxruntime, weight: str | Path, index: int):
    """Create an InferenceSession on the MIGraphX plugin EP for GPU `index`.

    Raises if the plugin is missing, the GPU is not enumerated, or compilation fails, so the caller can fall back to
    CPU.

    Args:
        onnxruntime (module): The imported onnxruntime module.
        weight (str | Path): Path to the .onnx model file.
        index (int): GPU device index.

    Returns:
        (onnxruntime.InferenceSession): The session on the MIGraphX EP.
    """
    ep = _register_migraphx_ep(onnxruntime)
    devices = [d for d in onnxruntime.get_ep_devices() if d.ep_name == ep]
    if index >= len(devices):
        raise ValueError(f"MIGraphX device {index} not found ({len(devices)} available), set HIP_VISIBLE_DEVICES")

    # Disabling Winograd cuts cold-compile time with no accuracy change (ROCm/AMDMIGraphX#5234).
    os.environ.setdefault("MIGRAPHX_DISABLE_WINOGRAD", "1")

    # Cache the JIT-compiled program so repeat loads skip compilation.
    options = {}
    try:
        cache_dir = _migraphx_cache_dir(weight)
        if not any(cache_dir.iterdir()):
            LOGGER.info("MIGraphX is compiling the model; this runs once per model and the result is cached...")
        # The EP reads ORT_MIGRAPHX_CACHE_DIR ahead of the cache_dir option, so set both.
        os.environ["ORT_MIGRAPHX_CACHE_DIR"] = options["cache_dir"] = str(cache_dir)
    except OSError as e:
        os.environ.pop("ORT_MIGRAPHX_CACHE_DIR", None)  # never leave the EP on another model's cached programs
        LOGGER.warning(f"MIGraphX cache disabled ({e}); recompiling each session init.")

    LOGGER.info(f"Using ONNX Runtime {onnxruntime.__version__} with {ep}")
    session_options = onnxruntime.SessionOptions()
    session_options.add_provider_for_devices(devices[index : index + 1], options)
    # Disable ORT's silent CPU retry so a MIGraphX failure raises to ONNXBackend's own CPU fallback
    return onnxruntime.InferenceSession(weight, session_options, enable_fallback=0)


# ONNX Runtime output type string -> (torch dtype, numpy dtype) for IO binding.
_ORT_DTYPES = {
    "tensor(float16)": (torch.float16, np.float16),
    "tensor(float)": (torch.float32, np.float32),
    "tensor(double)": (torch.float64, np.float64),
    "tensor(uint8)": (torch.uint8, np.uint8),
    "tensor(int8)": (torch.int8, np.int8),
    "tensor(int32)": (torch.int32, np.int32),
    "tensor(int64)": (torch.int64, np.int64),
}


class ONNXBackend(BaseBackend):
    """Microsoft ONNX Runtime inference backend with optional OpenCV DNN support.

    Loads and runs inference with ONNX models (.onnx files) using either Microsoft ONNX Runtime with CUDA,
    ROCm/MIGraphX, or CoreML execution providers, or OpenCV DNN for lightweight CPU inference. Supports IO binding for
    optimized GPU inference with static input shapes.
    """

    def __init__(
        self,
        weight: str | Path,
        device: torch.device,
        fp16: bool = False,
        format: str = "onnx",
        session_options: object | None = None,
    ):
        """Initialize the ONNX backend.

        Args:
            weight (str | Path): Path to the .onnx model file.
            device (torch.device): Device to run inference on.
            fp16 (bool): Whether to use FP16 half-precision inference.
            format (str): Inference engine, either "onnx" for ONNX Runtime or "dnn" for OpenCV DNN.
            session_options (object | None): Optional ONNX Runtime session options.
        """
        assert format in {"onnx", "dnn"}, f"Unsupported ONNX format: {format}."
        self.format = format
        self.session_options = session_options
        super().__init__(weight, device, fp16)

    def load_model(self, weight: str | Path) -> None:
        """Load an ONNX model using ONNX Runtime or OpenCV DNN.

        Args:
            weight (str | Path): Path to the .onnx model file.
        """
        if not isinstance(self.device, torch.device):  # 'intel', 'tpu' or 'vulkan' device strings run on CPU
            self.device = torch.device("cpu")
        cuda = torch.cuda.is_available() and self.device.type != "cpu"

        self.apply_metadata(self.read_metadata(weight))

        if self.format == "dnn":
            # OpenCV DNN
            LOGGER.info(f"Loading {weight} for ONNX OpenCV DNN inference...")
            import cv2

            self.net = cv2.dnn.readNetFromONNX(weight)
        else:
            # ONNX Runtime
            LOGGER.info(f"Loading {weight} for ONNX Runtime inference...")
            rocm = cuda and rocm_is_available()  # AMD GPUs run on the MIGraphX plugin EP
            check_requirements("onnx")
            if rocm and IS_PYTHON_MINIMUM_3_11 and not ARM64:  # MIGraphX EP wheels are Python>=3.11 x86_64 only
                check_requirements(ROCM_EP_PACKAGES, cmds=ROCM_EXTRA_INDEX)
            # Stock onnxruntime underlies the MIGraphX plugin and is the CPU fallback if the plugin is unavailable
            check_requirements(
                [("onnxruntime", "onnxruntime-gpu")] if rocm else "onnxruntime-gpu" if cuda else "onnxruntime"
            )
            import onnxruntime

            if rocm:
                try:
                    self.session = _load_migraphx_session(onnxruntime, weight, self.device.index or 0)
                except Exception as e:  # plugin missing, GPU not enumerated, or unsupported-op compile failure
                    LOGGER.warning(f"MIGraphX execution provider unavailable ({e}). Using CPU...")
                    self.device, cuda, rocm = torch.device("cpu"), False, False
            if not rocm:
                # Select execution provider
                available = onnxruntime.get_available_providers()
                if cuda and "CUDAExecutionProvider" in available:
                    providers = [("CUDAExecutionProvider", {"device_id": self.device.index}), "CPUExecutionProvider"]
                elif self.device.type == "mps" and "CoreMLExecutionProvider" in available:
                    providers = ["CoreMLExecutionProvider", "CPUExecutionProvider"]
                else:
                    providers = ["CPUExecutionProvider"]

                LOGGER.info(
                    f"Using ONNX Runtime {onnxruntime.__version__} with "
                    f"{providers[0] if isinstance(providers[0], str) else providers[0][0]}"
                )

                try:
                    self.session = onnxruntime.InferenceSession(weight, self.session_options, providers=providers)
                except onnxruntime.capi.onnxruntime_pybind11_state.InvalidProtobuf as e:
                    # ONNX Runtime reports an unparsable graph as a raw protobuf error naming neither the problem
                    # nor a remedy. Only this one type is caught: other load failures are execution-provider or
                    # model-support issues, where the runtime's own message is the useful one.
                    raise TypeError(
                        f"ERROR ❌️ {weight} is not a loadable ONNX model — the file is empty, truncated or corrupted "
                        f"({type(e).__name__}: {e}).\nRecommended fixes are to re-export it with "
                        f"'yolo export model=yolo26n.pt format=onnx', or to re-download the file."
                    ) from e
                if cuda and "CUDAExecutionProvider" not in self.session.get_providers():
                    LOGGER.warning("CUDA requested but CUDAExecutionProvider not available. Using CPU...")
                    self.device = torch.device("cpu")
                    cuda = False
            self.output_names = [x.name for x in self.session.get_outputs()]

            # Check if dynamic shapes
            self.dynamic = isinstance(self.session.get_outputs()[0].shape[0], str)
            self.fp16 = "float16" in self.session.get_inputs()[0].type

            # Setup IO binding for CUDA and MIGraphX
            self.use_io_binding = not self.dynamic and cuda
            if self.use_io_binding:
                self.io = self.session.io_binding()
                self.bindings = []
                for output in self.session.get_outputs():
                    torch_dtype, np_dtype = _ORT_DTYPES.get(output.type, (torch.float32, np.float32))
                    y_tensor = torch.empty(output.shape, dtype=torch_dtype).to(self.device)
                    if rocm:  # the MIGraphX plugin EP binds GPU tensors through DLPack
                        self.io.bind_ortvalue_output(output.name, onnxruntime.OrtValue.from_dlpack(y_tensor))
                    else:
                        self.io.bind_output(
                            name=output.name,
                            device_type=self.device.type,
                            device_id=self.device.index if cuda else 0,
                            element_type=np_dtype,
                            shape=tuple(y_tensor.shape),
                            buffer_ptr=y_tensor.data_ptr(),
                        )
                    self.bindings.append(y_tensor)

    def forward(
        self, im: torch.Tensor | dict[str, torch.Tensor | np.ndarray]
    ) -> np.ndarray | list[np.ndarray] | list[torch.Tensor]:
        """Run ONNX inference using IO binding (CUDA and ROCm/MIGraphX) or standard session execution.

        Args:
            im (torch.Tensor | dict): Input image tensor in BCHW format, normalized to [0, 1], or a dictionary mapping
                input names to tensors/arrays for multi-input ONNX Runtime models.

        Returns:
            (np.ndarray | list[np.ndarray] | list[torch.Tensor]): Model predictions as a numpy array (OpenCV DNN), a
                list of numpy arrays (ONNX Runtime), or a list of bound output tensors (GPU IO binding).
        """
        if self.format == "dnn":
            # OpenCV DNN
            self.net.setInput(im.cpu().numpy())
            return self.net.forward()

        # ONNX Runtime
        if isinstance(im, dict):  # multi-input model
            im = {k: v.cpu().numpy() if isinstance(v, torch.Tensor) else v for k, v in im.items()}
            return self.session.run(self.output_names, im)

        if self.use_io_binding:
            if self.device.type == "cpu":
                im = im.cpu()
            if torch.version.hip:  # the MIGraphX plugin EP binds GPU tensors through DLPack
                from onnxruntime import OrtValue

                self.io.bind_ortvalue_input("images", OrtValue.from_dlpack(im))
            else:
                self.io.bind_input(
                    name="images",
                    device_type=im.device.type,
                    device_id=im.device.index if im.device.type == "cuda" else 0,
                    element_type=np.float16 if self.fp16 else np.float32,
                    shape=tuple(im.shape),
                    buffer_ptr=im.data_ptr(),
                )
            self.session.run_with_iobinding(self.io)
            return self.bindings
        else:
            return self.session.run(self.output_names, {self.session.get_inputs()[0].name: im.cpu().numpy()})


class ONNXIMXBackend(ONNXBackend):
    """ONNX IMX inference backend for NXP i.MX processors.

    Extends `ONNXBackend` with support for quantized models targeting NXP i.MX edge devices. Uses MCT (Model Compression
    Toolkit) quantizers and custom NMS operations for optimized inference.
    """

    def load_model(self, weight: str | Path) -> None:
        """Load a quantized ONNX model from an IMX model directory.

        Args:
            weight (str | Path): Path to the IMX model directory containing the .onnx file.
        """
        check_requirements(("model-compression-toolkit>=2.4.1", "edge-mdt-cl<1.1.0", "onnxruntime-extensions"))
        check_requirements(("onnx", "onnxruntime"))
        import mct_quantizers as mctq
        import onnxruntime
        from edgemdt_cl.pytorch.nms import nms_ort  # noqa - register custom NMS ops

        w = Path(weight)
        onnx_file = next(w.glob("*.onnx"))
        LOGGER.info(f"Loading {onnx_file} for ONNX IMX inference...")

        session_options = mctq.get_ort_session_options()
        session_options.enable_mem_reuse = False

        self.session = onnxruntime.InferenceSession(onnx_file, session_options, providers=["CPUExecutionProvider"])
        self.output_names = [x.name for x in self.session.get_outputs()]
        self.dynamic = isinstance(self.session.get_outputs()[0].shape[0], str)
        self.fp16 = "float16" in self.session.get_inputs()[0].type
        self.apply_metadata(self.read_metadata(w))

    def forward(self, im: torch.Tensor) -> np.ndarray | list[np.ndarray] | tuple[np.ndarray, ...]:
        """Run IMX inference with task-specific output concatenation for detect, pose, and segment tasks.

        Args:
            im (torch.Tensor): Input image tensor in BCHW format, normalized to [0, 1].

        Returns:
            (np.ndarray | list[np.ndarray] | tuple[np.ndarray, ...]): Task-formatted model predictions.
        """
        y = self.session.run(self.output_names, {self.session.get_inputs()[0].name: im.cpu().numpy()})

        if self.task == "detect":
            # boxes, conf, cls
            return np.concatenate([y[0], y[1][:, :, None], y[2][:, :, None]], axis=-1)
        elif self.task == "pose":
            # boxes, conf, cls, kpts
            return np.concatenate([y[0], y[1][:, :, None], y[2][:, :, None], y[3]], axis=-1, dtype=y[0].dtype)
        elif self.task == "segment":
            return (
                np.concatenate([y[0], y[1][:, :, None], y[2][:, :, None], y[3]], axis=-1, dtype=y[0].dtype),
                y[4],
            )
        return y
