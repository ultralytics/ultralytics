# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from PIL import Image

from ultralytics.utils import LOGGER
from ultralytics.utils.checks import check_requirements

from .base import BaseBackend


class CoreMLBackend(BaseBackend):
    """CoreML inference backend for Apple hardware.

    Loads and runs inference with CoreML models (.mlpackage files) using the coremltools library. Supports both static
    and dynamic input shapes and handles NMS-included model outputs.
    """

    def load_model(self, weight: str | Path) -> None:
        """Load a CoreML model from a .mlpackage file.

        Args:
            weight (str | Path): Path to the .mlpackage model file.
        """
        check_requirements(["coremltools>=9.0", "numpy>=1.14.5,<=2.3.5"])
        import coremltools as ct

        LOGGER.info(f"Loading {weight} for CoreML inference...")
        # Run on the Neural Engine (CPU_AND_NE): ~3x faster than CPU, and the default ComputeUnit.ALL / CPU_AND_GPU
        # abort the process via an MPSGraph compiler bug on macOS hosts (coremltools 9.x). CPU_AND_NE needs macOS >= 13,
        # so fall back to CPU_ONLY below that. CoreML inference is macOS-only, so this applies wherever the backend runs.
        # Exception: RT-DETR loses FP16 accuracy and runs slower on the Neural Engine alone, so route it through ALL.
        meta = self.read_metadata(weight)
        default_unit = ct.ComputeUnit.ALL if meta.get("head") == "RTDETRDecoder" else ct.ComputeUnit.CPU_AND_NE
        try:
            self.model = ct.models.MLModel(weight, compute_units=default_unit)
        except Exception:
            self.model = ct.models.MLModel(weight, compute_units=ct.ComputeUnit.CPU_ONLY)
        spec = self.model.get_spec()
        self.input_name = spec.description.input[0].name
        self.dynamic = spec.description.input[0].type.HasField("multiArrayType")

        self.apply_metadata(meta)

    def forward(self, im: torch.Tensor) -> np.ndarray | list[np.ndarray]:
        """Run CoreML inference with automatic input format handling.

        Args:
            im (torch.Tensor): Input image tensor in BHWC format (converted from BCHW by AutoBackend).

        Returns:
            (np.ndarray | list[np.ndarray]): Model predictions as numpy array(s).
        """
        im = im.cpu().numpy()
        h, w = im.shape[1:3]

        if self.dynamic:
            y = list(self.model.predict({self.input_name: im.transpose(0, 3, 1, 2)}).values())
        else:  # static exports take one image per request: predict the batch as a list and stack the outputs
            y = self.model.predict([{self.input_name: Image.fromarray((x * 255).astype("uint8"))} for x in im])
            if "confidence" in y[0]:  # NMS included: zero-pad per image, return (B, n, 6) before the swap below
                from ultralytics.utils.ops import xywh2xyxy

                y = [(xywh2xyxy(d["coordinates"] * [[w, h, w, h]]), d["confidence"]) for d in y]
                y = [
                    np.concatenate((box, conf.max(1, keepdims=True), conf.argmax(1, keepdims=True)), 1)
                    for box, conf in y
                ]
                n = max(map(len, y))
                return np.stack([np.pad(d, ((0, n - len(d)), (0, 0))) for d in y])
            arrays = [k for k, v in y[0].items() if isinstance(v, np.ndarray)]
            if arrays:
                y = [np.concatenate([d[k] for d in y]) for k in arrays]
            else:  # neuralnetwork classifiers emit only the label and a name-to-probability dict
                probs = next(k for k, v in y[0].items() if isinstance(v, dict))
                y = [np.array([[d[probs][n] for n in self.names.values()] for d in y])]
        if len(y) == 2 and len(y[1].shape) != 4:  # segmentation model
            y = list(reversed(y))
        return y
