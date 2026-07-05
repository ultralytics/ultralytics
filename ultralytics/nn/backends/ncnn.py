# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from ultralytics.utils import LOGGER
from ultralytics.utils.checks import check_requirements

from .base import BaseBackend


class NCNNBackend(BaseBackend):
    """Tencent NCNN inference backend for mobile and embedded deployment.

    Loads and runs inference with Tencent NCNN models (*_ncnn_model/ directories). Optimized for mobile platforms with
    optional Vulkan GPU acceleration when available.
    """

    def load_model(self, weight: str | Path) -> None:
        """Load an NCNN model from a .param/.bin file pair or model directory.

        Args:
            weight (str | Path): Path to the .param file or directory containing NCNN model files.
        """
        LOGGER.info(f"Loading {weight} for NCNN inference...")
        check_requirements("ncnn", cmds="--no-deps")
        import ncnn as pyncnn

        self.pyncnn = pyncnn
        self.net = pyncnn.Net()

        # Setup Vulkan if available
        if isinstance(self.device, str) and self.device.startswith("vulkan"):
            self.net.opt.use_vulkan_compute = True
            self.net.set_vulkan_device(int(self.device.split(":")[1]))
            self.device = torch.device("cpu")
        else:
            self.net.opt.use_vulkan_compute = False

        w = Path(weight)
        if not w.is_file():
            w = next(w.glob("*.param"))

        self.net.load_param(str(w))
        self.net.load_model(str(w.with_suffix(".bin")))

        # Load metadata
        metadata_file = w.parent / "metadata.yaml"
        if metadata_file.exists():
            from ultralytics.utils import YAML

            self.apply_metadata(YAML.load(metadata_file))

    def forward(self, im: torch.Tensor) -> list[np.ndarray]:
        """Run inference using the NCNN runtime.

        Args:
            im (torch.Tensor): Input image tensor in BCHW format, normalized to [0, 1].

        Returns:
            (list[np.ndarray]): Model predictions as a list of numpy arrays, one per output layer.
        """
        mat_in = self.pyncnn.Mat(im[0].cpu().numpy())
        with self.net.create_extractor() as ex:
            ex.input(self.net.input_names()[0], mat_in)
            # Sort output names as temporary fix for pnnx issue
            y = [np.array(ex.extract(x)[1])[None] for x in sorted(self.net.output_names())]
        return y
