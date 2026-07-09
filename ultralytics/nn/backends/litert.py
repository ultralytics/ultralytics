# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from ultralytics.utils import LOGGER
from ultralytics.utils.checks import check_requirements

from .base import BaseBackend, read_tflite_metadata


class LiteRTBackend(BaseBackend):
    """Google LiteRT (formerly TensorFlow Lite) inference backend.

    Loads and runs inference with LiteRT models (.tflite files) exported via ai-edge-litert/litert-torch. Ultralytics
    exports keep float graph I/O (weights/activations may be int8/int16 internally); full-integer .tflite (legacy
    onnx2tf or third-party) with int8/int16 graph I/O are also handled by (de)quantizing at the boundary. Box and
    keypoint coordinates are denormalized by image size.

    On Intel Core Ultra hardware the same .tflite runs on the NPU via the LiteRT OpenVINO dispatch (JIT), selected with
    ``device='intel:npu'`` or auto-detected on Intel CPUs; any failure transparently falls back to the CPU interpreter.
    """

    def load_model(self, weight: str | Path) -> None:
        """Load a LiteRT model from a .tflite file, on the Intel NPU when available else the CPU interpreter.

        Args:
            weight (str | Path): Path to the .tflite model file (metadata embedded as a metadata.json entry).
        """
        tflite_file = Path(weight)
        self.npu = False
        mode = self._want_npu()  # 'explicit' | 'auto' | None
        if mode:
            self._load_npu(tflite_file, mode)  # sets self.npu=True on success

        if not self.npu:
            check_requirements("ai-edge-litert>=2.1.4")
            from ai_edge_litert.interpreter import Interpreter

            LOGGER.info(f"Loading {tflite_file} for LiteRT inference...")
            self.interpreter = Interpreter(str(tflite_file))
            self.interpreter.allocate_tensors()
            self.input_details = self.interpreter.get_input_details()
            self.output_details = self.interpreter.get_output_details()
            # Legacy onnx2tf TFLite exports are NHWC; litert-torch exports are NCHW. Detect from the input tensor so
            # both load through this backend (detection/proto outputs share the same layout for either export path).
            self.nhwc = self.input_details[0]["shape"][-1] == 3

        # Load the metadata.json embedded in the .tflite (single self-contained file); shared by both load paths.
        metadata = read_tflite_metadata(tflite_file)
        if metadata:
            self.apply_metadata(metadata)

    def _want_npu(self) -> str | None:
        """Return NPU mode: ``'explicit'`` (``device='intel:npu'``), ``'auto'`` (Intel CPU/default), else None.

        Mirrors ``OpenVINOBackend``: an explicit ``intel:*`` device string resets ``self.device`` to CPU so downstream
        torch ops stay on CPU. Only NPU (or unqualified ``intel``/``intel:auto``) targets take the OpenVINO path here.
        """
        device = self.device
        if isinstance(device, str) and device.startswith("intel"):
            suffix = device.split(":", 1)[1].lower() if ":" in device else ""
            self.device = torch.device("cpu")
            return "explicit" if suffix in {"", "npu", "auto"} else None
        if isinstance(device, torch.device) and device.type != "cpu":
            return None
        from ultralytics.utils.checks import is_intel

        return "auto" if is_intel() else None

    def _load_npu(self, tflite_file: Path, mode: str) -> None:
        """Load the model for Intel NPU inference via the LiteRT OpenVINO dispatch, else leave ``self.npu`` False.

        The dispatch lives in the nightly ``ai-edge-litert-sdk-intel`` (which replaces stable ai-edge-litert), so it is
        auto-installed only on an ``'explicit'`` ``device='intel:npu'`` request; ``'auto'`` mode uses it only when it is
        already installed and is otherwise silent. Any failure falls through to the CPU interpreter.
        """
        try:
            try:
                import ai_edge_litert_sdk_intel  # noqa: F401  probe the OpenVINO NPU dispatch SDK
            except ImportError:
                if mode != "explicit":
                    return  # auto mode never installs the nightly SDK; run on CPU
                check_requirements(
                    ("ai-edge-litert-nightly", "ai-edge-litert-sdk-intel-nightly"),
                    cmds="--pre --extra-index-url https://storage.openvinotoolkit.org/simple/wheels/nightly",
                )

            from ai_edge_litert.compiled_model import CompiledModel
            from ai_edge_litert.hardware_accelerator import HardwareAccelerator

            LOGGER.info(f"Loading {tflite_file} for LiteRT inference on Intel NPU (OpenVINO dispatch)...")
            self.compiled = CompiledModel.from_file(
                str(tflite_file), hardware_accel=HardwareAccelerator.NPU | HardwareAccelerator.CPU
            )
            self.sig_idx = self.compiled.get_signature_index(next(iter(self.compiled.get_signature_list())))
            self.in_buffers = self.compiled.create_input_buffers(self.sig_idx)
            self.out_buffers = self.compiled.create_output_buffers(self.sig_idx)
            # Cache (num_elements, shape, dtype) so forward() can reshape the flat read() output of each buffer.
            self.out_specs = [
                (int(np.prod(d["shape"])), tuple(d["shape"]), d["dtype"])
                for d in (b.get_tensor_details() for b in self.out_buffers)
            ]
            self.nhwc = False  # litert-torch NPU graphs are NCHW float I/O
            self.npu = True
            if not self.compiled.is_fully_accelerated():
                LOGGER.warning(f"{tflite_file} is not fully NPU-accelerated; unsupported ops run on CPU.")
        except Exception as e:
            if mode == "explicit":
                LOGGER.warning(f"Intel NPU acceleration unavailable ({e}); using CPU.")

    def forward(self, im: torch.Tensor) -> list[np.ndarray]:
        """Run inference on the Intel NPU or the CPU interpreter, then denormalize coordinates.

        Box and pose keypoint coordinates are exported normalized to [0, 1] (so INT8 quantization preserves class
        score resolution) and denormalized here by the input image size, mirroring the TensorFlow Lite backend.

        Args:
            im (torch.Tensor): Input image tensor in BCHW format, normalized to [0, 1].

        Returns:
            (list[np.ndarray]): Model predictions as a list of numpy arrays.
        """
        if self.npu:
            arr = im.cpu().numpy().astype(np.float32)  # BCHW float, matches the litert-torch NCHW graph
            h, w = arr.shape[2:4]
            self.in_buffers[0].write(arr)
            self.compiled.run_by_index(self.sig_idx, self.in_buffers, self.out_buffers)
            y = [buf.read(n, dtype).reshape(shape) for buf, (n, shape, dtype) in zip(self.out_buffers, self.out_specs)]
        else:
            im = im.cpu().numpy()  # BCHW
            if self.nhwc:
                im = im.transpose(0, 2, 3, 1)  # BCHW to BHWC for legacy onnx2tf TFLite
            h, w = im.shape[1:3] if self.nhwc else im.shape[2:4]
            details = self.input_details[0]
            # Ultralytics exports keep float I/O, but quantize here too for full-integer .tflite (legacy onnx2tf or
            # third-party) whose graph input is int8/int16.
            if details["dtype"] in {np.int8, np.int16}:
                scale, zero_point = details["quantization"]
                im = (im / scale + zero_point).astype(details["dtype"])
            self.interpreter.set_tensor(details["index"], im)
            self.interpreter.invoke()
            y = []
            for output in self.output_details:
                x = self.interpreter.get_tensor(output["index"])
                # Dequantize full-integer .tflite outputs; skip semantic argmax maps (integer class IDs, no scale).
                if not (self.task == "semantic" and x.ndim == 3) and output["dtype"] in {np.int8, np.int16}:
                    scale, zero_point = output["quantization"]
                    x = (x.astype(np.float32) - zero_point) * scale
                y.append(x)

        kpt_start = 4 + len(self.names)  # pose keypoints follow the box (4) and class-score (nc) channels
        out = []
        for x in y:
            if self.task == "semantic" and x.ndim == 3:
                # Legacy onnx2tf baked argmax class map [B, H, W] of integer IDs: skip xywh denorm, which would
                # corrupt and overflow the indices.
                out.append(x)
                continue
            # Denormalize xywh (and pose keypoints) by image size. litert-torch end2end output is already post-NMS
            # pixel coordinates (batch, max_det, 6+), so it is left as-is; legacy onnx2tf TFLite normalizes even the
            # end2end output, so it is denormalized on the last axis.
            if x.ndim == 3 and not self.end2end:
                x[:, [0, 2]] *= w
                x[:, [1, 3]] *= h
                if self.task == "pose":
                    x[:, kpt_start::3] *= w
                    x[:, kpt_start + 1 :: 3] *= h
            elif x.ndim == 3 and self.end2end and self.nhwc:
                x[:, :, [0, 2]] *= w
                x[:, :, [1, 3]] *= h
                if self.task == "pose":  # post-NMS [B, N, box(4)+conf+cls+kpts], keypoints start at 6
                    x[:, :, 6::3] *= w
                    x[:, :, 7::3] *= h
            out.append(x)
        y = out

        if self.task == "segment" and y[0].ndim == 4:  # order as (detections, protos)
            y = [y[1], y[0]]
        # litert-torch exports are NCHW; legacy onnx2tf masks/logits are NHWC and need a channels-last→first transpose.
        if self.nhwc:
            if self.task == "segment" and len(y) > 1 and y[1].ndim == 4:
                y[1] = np.transpose(y[1], (0, 3, 1, 2))  # protos NHWC → NCHW
            elif self.task == "semantic" and len(y) == 1 and y[0].ndim == 4:
                y[0] = np.transpose(y[0], (0, 3, 1, 2))  # logits NHWC → NCHW

        return y
