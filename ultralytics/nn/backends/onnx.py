# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import torch

from ultralytics.utils import LOGGER
from ultralytics.utils.checks import check_requirements

from .base import BaseBackend


class ONNXBackend(BaseBackend):
    """Microsoft ONNX Runtime inference backend with optional OpenCV DNN support.

    Loads and runs inference with ONNX models (.onnx files) using Microsoft ONNX Runtime with MIGraphX/CUDA/CoreML
    execution providers, or OpenCV DNN for lightweight CPU inference. Supports IO binding for optimized CUDA inference
    with static input shapes.
    """

    def __init__(self, weight: str | Path, device: torch.device, fp16: bool = False, format: str = "onnx"):
        """Initialize the ONNX backend.

        Args:
            weight (str | Path): Path to the .onnx model file.
            device (torch.device): Device to run inference on.
            fp16 (bool): Whether to use FP16 half-precision inference.
            format (str): Inference engine, either "onnx" for ONNX Runtime or "dnn" for OpenCV DNN.
        """
        assert format in {"onnx", "dnn"}, f"Unsupported ONNX format: {format}."
        self.format = format
        super().__init__(weight, device, fp16)

    def load_model(self, weight: str | Path) -> None:
        """Load an ONNX model using ONNX Runtime or OpenCV DNN.

        Args:
            weight (str | Path): Path to the .onnx model file.
        """
        gpu = isinstance(self.device, torch.device) and torch.cuda.is_available() and self.device.type != "cpu"
        rocm = bool(getattr(torch.version, "hip", None))
        requested_fp16 = self.fp16

        if self.format == "dnn":
            # OpenCV DNN
            LOGGER.info(f"Loading {weight} for ONNX OpenCV DNN inference...")
            check_requirements("opencv-python>=4.5.4")
            import cv2

            self.net = cv2.dnn.readNetFromONNX(weight)
        else:
            # ONNX Runtime
            LOGGER.info(f"Loading {weight} for ONNX Runtime inference...")
            runtime_candidates = (
                ("onnxruntime-migraphx", "onnxruntime", "onnxruntime-gpu")
                if gpu and rocm
                else ("onnxruntime-gpu", "onnxruntime-migraphx", "onnxruntime")
                if gpu
                else ("onnxruntime", "onnxruntime-migraphx", "onnxruntime-gpu")
            )
            check_requirements(["onnx", runtime_candidates])
            import onnxruntime

            # Select execution provider
            available = onnxruntime.get_available_providers()
            device_id = self.device.index if isinstance(self.device, torch.device) and self.device.index else 0
            use_cuda_io_binding = False
            if gpu and "MIGraphXExecutionProvider" in available:
                migraphx_options = {"device_id": str(device_id)}
                if requested_fp16:
                    migraphx_options["migraphx_fp16_enable"] = "1"
                if cache_dir := os.environ.get("ULTRALYTICS_MIGRAPHX_CACHE_DIR"):
                    migraphx_options["migraphx_model_cache_dir"] = cache_dir
                providers = [("MIGraphXExecutionProvider", migraphx_options), "CPUExecutionProvider"]
            elif gpu and "CUDAExecutionProvider" in available:
                providers = [("CUDAExecutionProvider", {"device_id": device_id}), "CPUExecutionProvider"]
                use_cuda_io_binding = True
            elif self.device.type == "mps" and "CoreMLExecutionProvider" in available:
                providers = ["CoreMLExecutionProvider", "CPUExecutionProvider"]
            else:
                providers = ["CPUExecutionProvider"]
                if gpu:
                    LOGGER.warning("GPU requested but no compatible ONNX Runtime GPU provider is available. Using CPU...")
                    self.device = torch.device("cpu")

            LOGGER.info(
                f"Using ONNX Runtime {onnxruntime.__version__} with "
                f"{providers[0] if isinstance(providers[0], str) else providers[0][0]}"
            )

            self.session = onnxruntime.InferenceSession(weight, providers=providers)
            self.providers = self.session.get_providers()
            self.provider = self.providers[0] if self.providers else None
            self.provider_options = self.session.get_provider_options()
            self.output_names = [x.name for x in self.session.get_outputs()]

            # Get metadata
            metadata_map = self.session.get_modelmeta().custom_metadata_map
            if metadata_map:
                self.apply_metadata(dict(metadata_map))

            # Check if dynamic shapes
            self.dynamic = isinstance(self.session.get_outputs()[0].shape[0], str)
            self.fp16 = "float16" in self.session.get_inputs()[0].type
            self.migraphx_fp16 = self.provider == "MIGraphXExecutionProvider" and requested_fp16
            if self.migraphx_fp16 and not self.fp16:
                LOGGER.info("MIGraphX FP16 provider option is enabled; model input remains FP32.")

            # Setup IO binding for CUDA
            self.use_io_binding = not self.dynamic and use_cuda_io_binding
            if self.use_io_binding:
                self.io = self.session.io_binding()
                self.bindings = []
                for output in self.session.get_outputs():
                    out_fp16 = "float16" in output.type
                    y_tensor = torch.empty(output.shape, dtype=torch.float16 if out_fp16 else torch.float32).to(
                        self.device
                    )
                    self.io.bind_output(
                        name=output.name,
                        device_type=self.device.type,
                        device_id=device_id,
                        element_type=np.float16 if out_fp16 else np.float32,
                        shape=tuple(y_tensor.shape),
                        buffer_ptr=y_tensor.data_ptr(),
                    )
                    self.bindings.append(y_tensor)

    def forward(self, im: torch.Tensor) -> torch.Tensor | list[torch.Tensor] | np.ndarray:
        """Run ONNX inference using IO binding (CUDA) or standard session execution.

        Args:
            im (torch.Tensor): Input image tensor in BCHW format, normalized to [0, 1].

        Returns:
            (torch.Tensor | list[torch.Tensor] | np.ndarray): Model predictions as tensor(s) or numpy array(s).
        """
        if self.format == "dnn":
            # OpenCV DNN
            self.net.setInput(im.cpu().numpy())
            return self.net.forward()

        # ONNX Runtime
        if self.use_io_binding:
            if self.device.type == "cpu":
                im = im.cpu()
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
        metadata_map = self.session.get_modelmeta().custom_metadata_map
        if metadata_map:
            self.apply_metadata(dict(metadata_map))

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
