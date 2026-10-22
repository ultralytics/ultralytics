# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from ultralytics.utils import LOGGER, NUM_THREADS
from ultralytics.utils.checks import check_requirements

from .base import BaseBackend


class LiteRTBackend(BaseBackend):
    """Google LiteRT (formerly TensorFlow Lite) inference backend.

    Loads and runs inference with LiteRT models (.tflite files) exported via ai-edge-litert/litert-torch. Ultralytics
    exports keep float graph I/O (weights/activations may be int8/int16 internally); full-integer .tflite (legacy
    onnx2tf or third-party) with int8/int16 graph I/O are also handled by (de)quantizing at the boundary. Box and
    keypoint coordinates are denormalized by image size.
    """

    def load_model(self, weight: str | Path) -> None:
        """Load a LiteRT model from a .tflite file.

        Args:
            weight (str | Path): Path to the .tflite model file (metadata embedded as a metadata.json entry).
        """
        tflite_file = Path(weight)
        self.intel = str(self.device).startswith("intel")  # intel:cpu|npu

        LOGGER.info(f"Loading {tflite_file} for LiteRT inference...")
        if self.intel:  # Intel OpenVINO plugin through CompiledModel
            check_requirements(["ai-edge-litert>=2.3.0", "ai-edge-litert-sdk-intel>=2.3.0"])
            from ai_edge_litert.compiled_model import CompiledModel, HardwareAccelerator, IntelOpenVinoOptions, Options

            backend = getattr(IntelOpenVinoOptions.GRAPH_BACKEND, self.device.partition(":")[2].upper(), None)
            self.device = torch.device("cpu")
            self.model = CompiledModel.from_file(
                str(tflite_file),
                options=Options(
                    hardware_accelerators=HardwareAccelerator.NPU | HardwareAccelerator.CPU,
                    intel_openvino_options=IntelOpenVinoOptions(graph_backend=backend),
                ),
            )
            sig = next(iter(self.model.get_signature_list()))
            self.input_buffers = self.model.create_input_buffers(0)
            self.output_buffers = self.model.create_output_buffers(0)
            self.input_details = list(self.model.get_input_tensor_details(sig).values())
            self.output_details = list(self.model.get_output_tensor_details(sig).values())
            for d in self.input_details + self.output_details:
                d["dtype"] = np.dtype(d["dtype"])
        else:
            check_requirements("ai-edge-litert>=2.1.4")
            from ai_edge_litert.interpreter import Interpreter

            self.interpreter = Interpreter(str(tflite_file), num_threads=NUM_THREADS)  # Enable multi-core inference
            self.interpreter.allocate_tensors()
            self.input_details = self.interpreter.get_input_details()
            self.output_details = self.interpreter.get_output_details()
        # Legacy onnx2tf TFLite exports are NHWC; litert-torch exports are NCHW. Detect from the input tensor so both
        # load through this backend (the detection/proto outputs share the same layout for either export path).
        metadata = self.read_metadata(tflite_file)
        self.nhwc = self.input_details[0]["shape"][-1] == int(metadata.get("channels", self.channels))

        self.apply_metadata(metadata)

    def forward(self, im: torch.Tensor) -> list[np.ndarray]:
        """Run inference using the LiteRT interpreter.

        Box and pose keypoint coordinates are exported normalized to [0, 1] (so INT8 quantization preserves class
        score resolution) and denormalized here by the input image size, mirroring the TensorFlow Lite backend.

        Args:
            im (torch.Tensor): Input image tensor in BCHW format, normalized to [0, 1].

        Returns:
            (list[np.ndarray]): Model predictions as a list of numpy arrays.
        """
        im = im.cpu().numpy()  # BCHW
        if self.nhwc:
            im = im.transpose(0, 2, 3, 1)  # BCHW to BHWC for legacy onnx2tf TFLite
        h, w = im.shape[1:3] if self.nhwc else im.shape[2:4]
        details = self.input_details[0]
        # Ultralytics exports keep float I/O, but quantize here too for full-integer .tflite (legacy onnx2tf or
        # third-party) whose graph input is int8/int16.
        if details["dtype"] in {np.int8, np.int16}:
            scale, zero_point = details["quantization"]
            limits = np.iinfo(details["dtype"])
            im = im / scale + zero_point
            im = im.clip(limits.min, limits.max, out=im).astype(details["dtype"])
        if self.intel:
            self.input_buffers[0].write(np.ascontiguousarray(im))
            self.model.run_by_index(0, self.input_buffers, self.output_buffers)
        else:
            self.interpreter.set_tensor(details["index"], im)
            self.interpreter.invoke()

        kpt_start = 4 + len(self.names)  # pose keypoints follow the box (4) and class-score (nc) channels
        y = []
        for i, output in enumerate(self.output_details):
            if self.intel:
                x = self.output_buffers[i].read(int(np.prod(output["shape"])), output["dtype"]).reshape(output["shape"])
            else:
                x = self.interpreter.get_tensor(output["index"])
            if self.task == "semantic" and x.ndim == 3:
                # Legacy onnx2tf baked argmax class map [B, H, W] of integer IDs: skip xywh denorm, which would
                # corrupt and overflow the indices.
                y.append(x)
                continue
            if output["dtype"] in {np.int8, np.int16}:  # dequantize full-integer .tflite outputs (see input note)
                scale, zero_point = output["quantization"]
                x = (x.astype(np.float32) - zero_point) * scale
            # Denormalize xywh (and pose keypoints) by image size. litert-torch end2end output is already post-NMS
            # pixel coordinates (batch, max_det, 6+), so it is left as-is; legacy onnx2tf TFLite normalizes even the
            # end2end output, so it is denormalized on the last axis.
            if x.ndim == 3 and not self.end2end and getattr(self, "head", None) != "RTDETRDecoder":
                x[:, [0, 2]] *= w
                x[:, [1, 3]] *= h
                if self.task == "pose":
                    nd = self.kpt_shape[1]  # 2 (x, y) or 3 (x, y, visibility) values per keypoint
                    x[:, kpt_start::nd] *= w
                    x[:, kpt_start + 1 :: nd] *= h
            elif x.ndim == 3 and self.end2end and self.nhwc:
                x[:, :, [0, 2]] *= w
                x[:, :, [1, 3]] *= h
                if self.task == "pose":  # post-NMS [B, N, box(4)+conf+cls+kpts], keypoints start at 6
                    nd = self.kpt_shape[1]
                    x[:, :, 6::nd] *= w
                    x[:, :, 7::nd] *= h
            y.append(x)

        if self.task == "segment" and y[0].ndim == 4:  # order as (detections, protos)
            y = [y[1], y[0]]
        # litert-torch exports are NCHW; legacy onnx2tf masks/logits are NHWC and need a channels-last→first transpose.
        if self.nhwc:
            if self.task == "segment" and len(y) > 1 and y[1].ndim == 4:
                y[1] = np.transpose(y[1], (0, 3, 1, 2))  # protos NHWC → NCHW
            elif self.task in {"semantic", "depth"} and len(y) == 1 and y[0].ndim == 4:
                y[0] = np.transpose(y[0], (0, 3, 1, 2))  # logits NHWC → NCHW

        return y
