# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from __future__ import annotations

from pathlib import Path

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

        # Load metadata
        metadata_file = w.parent / "metadata.yaml"
        if metadata_file.exists():
            from ultralytics.utils import YAML

            self.apply_metadata(YAML.load(metadata_file))

    def forward(self, im: torch.Tensor) -> list:
        """Run inference on the Rockchip NPU.

        Args:
            im (torch.Tensor): Input image tensor in BCHW format, normalized to [0, 1].

        Returns:
            (list): Model predictions as a list of output arrays.
        """
        im = (im.cpu().numpy() * 255).astype("uint8")
        im = im if isinstance(im, (list, tuple)) else [im]
        return self.model.inference(inputs=im)
