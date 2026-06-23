# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from ultralytics.utils import LOGGER
from ultralytics.utils.checks import check_requirements

from .base import BaseBackend


class LiteRTBackend(BaseBackend):
    """Google LiteRT (formerly TensorFlow Lite) inference backend.

    Loads and runs inference with LiteRT models (.tflite files) exported via ai-edge-litert/litert-torch. Supports FP32
    and static INT8 (int8 weights and activations) models, with automatic input/output dequantization and
    denormalization of box/keypoint coordinates by image size.
    """

    def load_model(self, weight: str | Path) -> None:
        """Load a LiteRT model from a .tflite file or directory.

        Args:
            weight (str | Path): Path to the .tflite model file or directory containing the model.
        """
        check_requirements("ai-edge-litert>=2.1.4")
        from ai_edge_litert.interpreter import Interpreter

        w = Path(weight)
        if w.is_dir():
            tflite_file = next(w.glob("*.tflite"))
            metadata_file = w / "metadata.yaml"
        else:
            tflite_file = w
            metadata_file = w.parent / "metadata.yaml"

        LOGGER.info(f"Loading {tflite_file} for LiteRT inference...")
        self.interpreter = Interpreter(str(tflite_file))
        self.interpreter.allocate_tensors()
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()

        # Load metadata
        if metadata_file.exists():
            from ultralytics.utils import YAML

            self.apply_metadata(YAML.load(metadata_file))

    def forward(self, im: torch.Tensor) -> list[np.ndarray]:
        """Run inference using the LiteRT interpreter.

        Box and pose keypoint coordinates are exported normalized to [0, 1] (so INT8 quantization preserves class
        score resolution) and denormalized here by the input image size, mirroring the TensorFlow Lite backend.

        Args:
            im (torch.Tensor): Input image tensor in BCHW format, normalized to [0, 1].

        Returns:
            (list[np.ndarray]): Model predictions as a list of numpy arrays.
        """
        im = im.cpu().numpy()
        h, w = im.shape[2:4]  # BCHW input (LiteRT keeps NCHW, unlike NHWC TFLite)
        details = self.input_details[0]
        is_int = details["dtype"] in {np.int8, np.int16}

        if is_int:
            scale, zero_point = details["quantization"]
            im = (im / scale + zero_point).astype(details["dtype"])

        self.interpreter.set_tensor(details["index"], im)
        self.interpreter.invoke()

        y = []
        for output in self.output_details:
            x = self.interpreter.get_tensor(output["index"])
            if output["dtype"] in {np.int8, np.int16}:
                scale, zero_point = output["quantization"]
                x = (x.astype(np.float32) - zero_point) * scale
            if x.ndim == 3:  # denormalize xywh (and pose keypoints) by image size on raw (batch, channels, anchors)
                x[:, [0, 2]] *= w
                x[:, [1, 3]] *= h
                if self.task == "pose":
                    x[:, 5::3] *= w
                    x[:, 6::3] *= h
            y.append(x)

        if self.task == "segment" and y[0].ndim == 4:  # order as (detections, protos); protos already NCHW
            y = [y[1], y[0]]

        return y
