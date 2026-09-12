# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from __future__ import annotations

import hashlib
import os
from functools import lru_cache
from pathlib import Path

import numpy as np
import torch

from ultralytics.utils import LOGGER, ROCM_EP_PACKAGES, ROCM_EXTRA_INDEX, USER_CONFIG_DIR
from ultralytics.utils.checks import check_requirements, rocm_is_available

from .base import BaseBackend


def _register_migraphx_ep(onnxruntime) -> str | None:
    """Register the MIGraphX plugin EP and return its name, or None if unavailable.

    The plugin (`onnxruntime-ep-migraphx`) is not auto-registered. Its libs are preloaded with RTLD_GLOBAL to work
    around a missing `libonnxruntime.so.1` soname link (ROCm/AMDMIGraphX#5235). Registration is idempotent.

    Args:
        onnxruntime (module): The imported onnxruntime module.

    Returns:
        (str | None): The registered EP name, or None if the plugin or its registration is unavailable.
    """
    try:
        import onnxruntime_ep_migraphx as ep
    except ImportError:
        return None

    name = ep.get_ep_name()
    if name in onnxruntime.get_available_providers():
        return name  # already registered in this process

    import ctypes
    import glob

    # Preload libs the plugin links but cannot locate itself (see docstring).
    search = [str(Path(onnxruntime.__file__).parent / "capi" / "libonnxruntime.so.1*")]
    try:
        import migraphx_libs

        search.append(str(Path(migraphx_libs.__file__).parent / "libmigraphx_c.so.3*"))
    except ImportError:
        pass
    for pattern in search:
        hits = sorted(glob.glob(pattern))
        if hits:
            try:
                ctypes.CDLL(hits[0], mode=ctypes.RTLD_GLOBAL)
            except OSError as e:
                LOGGER.debug(f"MIGraphX EP preload of {hits[0]} failed: {e}")

    try:
        onnxruntime.register_execution_provider_library(name, ep.get_library_path())
    except Exception as e:
        LOGGER.warning(f"Failed to register the MIGraphX execution provider ({e}). Using CPU...")
        return None
    return name


@lru_cache(maxsize=1)
def _migraphx_cache_root() -> Path:
    """Resolve the MIGraphX compiled-program cache root once per process.

    Cached so per-model ORT_MIGRAPHX_CACHE_DIR overwrites don't nest each cache under the last, keeping the dirs
    siblings.

    Returns:
        (Path): Cache root from ORT_MIGRAPHX_CACHE_DIR if set, else under USER_CONFIG_DIR.
    """
    return Path(os.environ.get("ORT_MIGRAPHX_CACHE_DIR") or USER_CONFIG_DIR / "migraphx_cache")


def _migraphx_cache_dir(weight: str | Path) -> Path:
    """Return a per-model cache subdirectory for the MIGraphX compiled program.

    The EP keys its cache by graph and input shapes (not weights), so hashing the model bytes isolates each model.

    Args:
        weight (str | Path): Path to the .onnx model file, hashed to key the cache.

    Returns:
        (Path): Per-model cache subdirectory under the resolved cache root.
    """
    return _migraphx_cache_root() / hashlib.sha256(Path(weight).read_bytes()).hexdigest()[:16]


def _create_session(onnxruntime, weight: str | Path, session_options, providers=None):
    """Create an ONNX Runtime InferenceSession, raising a clear error on an unparsable model.

    `providers=None` uses the execution providers already configured on `session_options` (e.g. the MIGraphX plugin).

    Args:
        onnxruntime (module): The imported onnxruntime module.
        weight (str | Path): Path to the .onnx model file.
        session_options (onnxruntime.SessionOptions): Session options, optionally carrying pre-configured providers.
        providers (list | None): Explicit execution providers, or None to use those set on `session_options`.

    Returns:
        (onnxruntime.InferenceSession): The loaded inference session.
    """
    try:
        if providers is None:
            return onnxruntime.InferenceSession(weight, session_options)
        return onnxruntime.InferenceSession(weight, session_options, providers=providers)
    except onnxruntime.capi.onnxruntime_pybind11_state.InvalidProtobuf as e:
        # ONNX Runtime reports an unparsable graph as a raw protobuf error naming neither the problem nor a remedy.
        # Only this type is caught: other load failures are EP or model-support issues where ORT's message is useful.
        raise TypeError(
            f"ERROR ❌️ {weight} is not a loadable ONNX model — the file is empty, truncated or corrupted "
            f"({type(e).__name__}: {e}).\nRecommend fixes are to re-export it with "
            f"'yolo export model=yolo26n.pt format=onnx', or to re-download the file."
        ) from e


def _load_migraphx_session(onnxruntime, session_options, weight: str | Path, index: int):
    """Build an InferenceSession on the MIGraphX plugin EP, or return None if MIGraphX is unavailable.

    Registers the plugin, selects the requested GPU, disables Winograd to speed cold compiles, enables the per-model
    compiled-program cache, and creates the session. The EP is added via `add_provider_for_devices`, not `providers=`.

    Args:
        onnxruntime (module): The imported onnxruntime module.
        session_options (onnxruntime.SessionOptions): Session options the MIGraphX provider is added to.
        weight (str | Path): Path to the .onnx model file.
        index (int): Requested GPU device index, clamped to device 0 if not enumerated by the EP.

    Returns:
        (onnxruntime.InferenceSession | None): The session on the MIGraphX EP, or None if MIGraphX is unavailable.
    """
    ep = _register_migraphx_ep(onnxruntime)
    devices = [d for d in onnxruntime.get_ep_devices() if d.ep_name == ep] if ep else []
    if not devices:
        return None
    if index >= len(devices):  # requested GPU not enumerated by the EP
        LOGGER.warning(f"MIGraphX device {index} unavailable ({len(devices)} found); using device 0.")
        index = 0

    # Disabling Winograd cuts cold-compile time with no accuracy change (ROCm/AMDMIGraphX#5234).
    os.environ.setdefault("MIGRAPHX_DISABLE_WINOGRAD", "1")

    # Cache the JIT-compiled program so repeat loads skip compilation.
    options = {}
    compiling = True  # unless a populated cache is found below
    try:
        cache_dir = _migraphx_cache_dir(weight)
        compiling = not (cache_dir.is_dir() and any(cache_dir.iterdir()))
        cache_dir.mkdir(parents=True, exist_ok=True)
        # The EP reads ORT_MIGRAPHX_CACHE_DIR ahead of the cache_dir option, so set both.
        os.environ["ORT_MIGRAPHX_CACHE_DIR"] = str(cache_dir)
        options["cache_dir"] = str(cache_dir)
        LOGGER.info(f"MIGraphX compiled-program cache at {cache_dir}")
    except OSError as e:
        LOGGER.warning(f"MIGraphX cache disabled ({e}); recompiling each session init.")
    if compiling:  # note the one-time JIT compile so the wait is expected
        LOGGER.info("MIGraphX is compiling the model; this runs once per model and the result is cached...")

    session_options.add_provider_for_devices(devices[index : index + 1], options)
    return _create_session(onnxruntime, weight, session_options)


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
        cuda = isinstance(self.device, torch.device) and torch.cuda.is_available() and self.device.type != "cpu"

        self.apply_metadata(self.read_metadata(weight))

        if self.format == "dnn":
            # OpenCV DNN
            LOGGER.info(f"Loading {weight} for ONNX OpenCV DNN inference...")
            import cv2

            self.net = cv2.dnn.readNetFromONNX(weight)
        else:
            # ONNX Runtime
            LOGGER.info(f"Loading {weight} for ONNX Runtime inference...")
            rocm = cuda and rocm_is_available()  # the MIGraphX plugin EP targets AMD GPUs; CPU/CUDA use stock wheels
            check_requirements("onnx")
            if rocm:
                check_requirements(ROCM_EP_PACKAGES, cmds=ROCM_EXTRA_INDEX)
                # Ensure stock ONNX Runtime as a fallback so a missing plugin wheel degrades to CPU instead of crashing.
                check_requirements([("onnxruntime", "onnxruntime-gpu")])
            else:
                ort = "onnxruntime-gpu" if cuda else "onnxruntime"
                check_requirements([(ort, "onnxruntime", "onnxruntime-gpu")])
            import onnxruntime

            session_options = self.session_options or onnxruntime.SessionOptions()

            # On ROCm, try the MIGraphX plugin EP first; it returns a configured session, or None if unavailable.
            self.session = None
            if rocm:
                try:
                    self.session = _load_migraphx_session(onnxruntime, session_options, weight, self.device.index or 0)
                except Exception as e:
                    # Plugin init can raise (e.g. unsupported-op compile); fall back to CPU with fresh options.
                    LOGGER.warning(f"MIGraphX EP init failed ({e}). Using CPU...")
                    session_options = onnxruntime.SessionOptions()
            plugin_ep = self.session is not None
            if not plugin_ep:
                available = onnxruntime.get_available_providers()
                if cuda and not rocm and "CUDAExecutionProvider" in available:
                    providers = [("CUDAExecutionProvider", {"device_id": self.device.index}), "CPUExecutionProvider"]
                elif self.device.type == "mps" and "CoreMLExecutionProvider" in available:
                    providers = ["CoreMLExecutionProvider", "CPUExecutionProvider"]
                else:
                    providers = ["CPUExecutionProvider"]
                    if cuda:
                        ep_name = "MIGraphXExecutionProvider" if rocm else "CUDAExecutionProvider"
                        pkg = "onnxruntime-ep-migraphx" if rocm else "onnxruntime-gpu"
                        fix = f"pip install {pkg}" + (f" {ROCM_EXTRA_INDEX}" if rocm else "")
                        LOGGER.warning(f"GPU requested but {ep_name} not available. Using CPU... Fix with '{fix}'")
                        self.device = torch.device("cpu")
                        cuda = False
                self.session = _create_session(onnxruntime, weight, session_options, providers)

            if cuda and not rocm and "CUDAExecutionProvider" not in self.session.get_providers():
                LOGGER.warning("CUDA requested but CUDAExecutionProvider not available. Using CPU...")
                self.device = torch.device("cpu")
                cuda = False
            LOGGER.info(f"Using ONNX Runtime {onnxruntime.__version__} with {self.session.get_providers()[0]}")
            self.output_names = [x.name for x in self.session.get_outputs()]

            # Check if dynamic shapes
            self.dynamic = isinstance(self.session.get_outputs()[0].shape[0], str)
            self.fp16 = "float16" in self.session.get_inputs()[0].type

            # Zero-copy GPU IO binding (CUDA buffer pointers or MIGraphX DLPack OrtValues).
            self.use_io_binding = not self.dynamic and cuda
            self._from_dlpack = onnxruntime.OrtValue.from_dlpack if plugin_ep else None  # onnxruntime is a local import
            if self.use_io_binding:
                self._setup_io_binding(plugin_ep)

    def _setup_io_binding(self, plugin_ep: bool) -> None:
        """Bind persistent output buffers for zero-copy GPU inference.

        Args:
            plugin_ep (bool): True for the MIGraphX plugin EP (DLPack OrtValues), False for CUDA (buffer pointers).
        """
        self.io = self.session.io_binding()
        self.bindings = []
        for output in self.session.get_outputs():
            torch_dtype, np_dtype = _ORT_DTYPES.get(output.type, (torch.float32, np.float32))
            y_tensor = torch.empty(output.shape, dtype=torch_dtype).to(self.device)
            if plugin_ep:
                self.io.bind_ortvalue_output(output.name, self._from_dlpack(y_tensor))
            else:
                self.io.bind_output(
                    name=output.name,
                    device_type=self.device.type,
                    device_id=self.device.index if self.device.type == "cuda" else 0,
                    element_type=np_dtype,
                    shape=tuple(y_tensor.shape),
                    buffer_ptr=y_tensor.data_ptr(),
                )
            self.bindings.append(y_tensor)

    def forward(
        self, im: torch.Tensor | dict[str, torch.Tensor | np.ndarray]
    ) -> torch.Tensor | list[torch.Tensor] | np.ndarray:
        """Run ONNX inference using IO binding (CUDA and ROCm/MIGraphX) or standard session execution.

        Args:
            im (torch.Tensor | dict): Input image tensor in BCHW format, normalized to [0, 1], or a dictionary mapping
                input names to tensors/arrays for multi-input ONNX Runtime models.

        Returns:
            (torch.Tensor | list[torch.Tensor] | np.ndarray): Model predictions as tensor(s) or numpy array(s).
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
            if self._from_dlpack is not None:
                self.io.bind_ortvalue_input("images", self._from_dlpack(im))
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
        check_requirements(("onnx", ("onnxruntime", "onnxruntime-gpu")))
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
            # boxes, conf, kpts
            return np.concatenate([y[0], y[1][:, :, None], y[2][:, :, None], y[3]], axis=-1, dtype=y[0].dtype)
        elif self.task == "segment":
            return (
                np.concatenate([y[0], y[1][:, :, None], y[2][:, :, None], y[3]], axis=-1, dtype=y[0].dtype),
                y[4],
            )
        return y
