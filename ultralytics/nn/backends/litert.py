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
    exports keep float graph I/O (weights/activations may be int8/int16 internally), so this backend feeds float input
    and reads float output, denormalizing box/keypoint coordinates by image size.
    """

    def load_model(self, weight: str | Path) -> None:
        """Load a LiteRT model from a .tflite file.

        Args:
            weight (str | Path): Path to the .tflite model file (metadata embedded as a metadata.json entry).
        """
        check_requirements("ai-edge-litert>=2.1.4")
        from ai_edge_litert.interpreter import Interpreter

        tflite_file = Path(weight)

        LOGGER.info(f"Loading {tflite_file} for LiteRT inference...")
        self.interpreter = Interpreter(str(tflite_file))
        self.interpreter.allocate_tensors()
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()
        # Legacy onnx2tf TFLite exports are NHWC; litert-torch exports are NCHW. Detect from the input tensor so both
        # load through this backend (the detection/proto outputs share the same layout for either export path).
        self.nhwc = self.input_details[0]["shape"][-1] == 3

        # Load the metadata.json embedded in the .tflite (single self-contained file)
        metadata = read_tflite_metadata(tflite_file)
        if metadata:
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
        self.interpreter.set_tensor(details["index"], im)
        self.interpreter.invoke()

        kpt_start = 4 + len(self.names)  # pose keypoints follow the box (4) and class-score (nc) channels
        y = []
        for output in self.output_details:
            x = self.interpreter.get_tensor(output["index"])
            if self.task == "semantic" and x.ndim == 3:
                # Legacy onnx2tf baked argmax class map [B, H, W] of integer IDs: skip xywh denorm, which would
                # corrupt and overflow the indices.
                y.append(x)
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
            y.append(x)

        if self.task == "segment" and y[0].ndim == 4:  # order as (detections, protos)
            y = [y[1], y[0]]
        # litert-torch exports are NCHW; legacy onnx2tf masks/logits are NHWC and need a channels-last→first transpose.
        if self.nhwc:
            if self.task == "segment" and len(y) > 1 and y[1].ndim == 4:
                y[1] = np.transpose(y[1], (0, 3, 1, 2))  # protos NHWC → NCHW
            elif self.task == "semantic" and len(y) == 1 and y[0].ndim == 4:
                y[0] = np.transpose(y[0], (0, 3, 1, 2))  # logits NHWC → NCHW

        return y
