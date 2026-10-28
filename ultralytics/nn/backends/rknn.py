# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from ultralytics.utils import LOGGER
from ultralytics.utils.checks import check_requirements, is_rockchip

from .base import BaseBackend


class RKNNBackend(BaseBackend):
    """Rockchip RKNN inference backend for Rockchip NPU hardware.

    Loads and runs inference with RKNN models (.rknn files) using the RKNN-Toolkit-Lite2 runtime. Only supported on
    Rockchip devices with NPU hardware (e.g., RK3588, RK3566).
    """

    def load_model(self, weight: str | Path) -> None:
        """Load a Rockchip RKNN model from a .rknn file or model directory.

        Args:
            weight (str | Path): Path to the .rknn file or directory containing the model.

        Raises:
            OSError: If not running on a Rockchip device.
            RuntimeError: If model loading or runtime initialization fails.
        """
        if not is_rockchip():
            raise OSError("RKNN inference is only supported on Rockchip devices.")

        LOGGER.info(f"Loading {weight} for RKNN inference...")
        check_requirements("rknn-toolkit-lite2")
        from rknnlite.api import RKNNLite

        w = Path(weight)
        if not w.is_file():
            w = next(w.rglob("*.rknn"))

        self.model = RKNNLite()
        ret = self.model.load_rknn(str(w))
        if ret != 0:
            raise RuntimeError(f"Failed to load RKNN model: {ret}")

        ret = self.model.init_runtime()
        if ret != 0:
            raise RuntimeError(f"Failed to init RKNN runtime: {ret}")

        self.tail = None
        if (w.parent / "tail.onnx").is_file():
            check_requirements("onnxruntime")
            import onnxruntime

            self.tail = onnxruntime.InferenceSession(str(w.parent / "tail.onnx"), providers=["CPUExecutionProvider"])
        self.apply_metadata(self.read_metadata(w))

    def forward(self, im: torch.Tensor) -> list:
        """Run inference on the Rockchip NPU.

        Args:
            im (torch.Tensor): Input image tensor in BHWC format, normalized to [0, 1].

        Returns:
            (list): Model predictions as a list of output arrays.
        """
        im = (im.cpu().numpy() * 255).astype("uint8")
        y = self.model.inference(inputs=[im])
        if self.tail:  # INT8 exports decode head outputs on the host in floating point with a batch-1 tail graph
            feeds = [
                {t.name: v[i : i + 1].astype(np.float32) for t, v in zip(self.tail.get_inputs(), y)}
                for i in range(len(im))
            ]
            y = [np.concatenate(x) for x in zip(*(self.tail.run(None, f) for f in feeds))]
        return y
