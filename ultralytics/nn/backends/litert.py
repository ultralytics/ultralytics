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
    and INT8 (dynamic-range, int8 weights / fp32 activations) quantized models with automatic dequantization.
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

        Args:
            im (torch.Tensor): Input image tensor in BCHW format, normalized to [0, 1].

        Returns:
            (list[np.ndarray]): Model predictions as a list of numpy arrays.
        """
        im = im.cpu().numpy()
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
            y.append(x)

        # Rejoin split detection output (boxes, classes) produced by the INT8 export wrapper
        if len(y) == 2 and all(x.ndim == 3 for x in y) and y[0].shape[0] == y[1].shape[0]:
            if y[0].shape[1] == y[1].shape[1]:  # end2end: same N_det → concat on last dim
                y = [np.concatenate(y, axis=-1)]  # (B, N_det, 4) + (B, N_det, 2) → (B, N_det, 6)
            elif y[0].shape[2] == y[1].shape[2]:  # non-end2end: same anchors → concat on axis 1
                y = [np.concatenate(y, axis=1)]  # (B, 4, N) + (B, C, N) → (B, 4+C, N)

        return y
