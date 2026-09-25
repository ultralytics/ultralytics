# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from __future__ import annotations

from pathlib import Path

import torch

from ultralytics.utils import LOGGER, YAML
from ultralytics.utils.checks import check_requirements

from .base import BaseBackend


class TIDLBackend(BaseBackend):
    """Texas Instruments TIDL inference backend for TI Edge AI (C7x DSP/MMA NPU) hardware.

    Loads and runs the TIDL artifacts produced by the Ultralytics TI export (`*_ti_model/`) using the
    `edgeai-tidl-runtime` package, which bundles the native TIDL tools and TIDL-enabled ONNX Runtime build.
    Inference runs on TI Edge AI MPU devices (e.g. TDA4VH, AM62A) via the C7x DSP/MMA NPU.
    """

    def load_model(self, weight: str | Path) -> None:
        """Load a compiled TIDL model directory with the `edgeai-tidl-runtime` TIDLExecutionProvider.

        Args:
            weight (str | Path): Path to the `*_ti_model/` directory produced by TI export.

        Raises:
            ValueError: If the export metadata (and the target device it records) cannot be found.
        """
        check_requirements("edgeai-tidl-runtime")
        from edgeai_tidl_runtime import create_session

        w = Path(weight)
        metadata_file = w / "metadata.yaml"
        if not metadata_file.exists():
            raise ValueError(f"No metadata.yaml found in {w}; re-export with 'format=ti' to regenerate it.")
        metadata = YAML.load(metadata_file)
        target_device = metadata.get("args", {}).get("name")
        if not target_device:
            raise ValueError(f"No TIDL target device recorded in {metadata_file}; re-export with a valid 'name' arg.")

        LOGGER.info(f"Loading {w} for TI Edge AI (TIDL) inference on '{target_device}'...")
        self.session = create_session(
            w, target_device=target_device, tensor_bits=metadata.get("args", {}).get("quantize", 8)
        )
        self.output_names = [x.name for x in self.session.get_outputs()]
        self.apply_metadata(metadata)

    def forward(self, im: torch.Tensor) -> list:
        """Run inference on the TI Edge AI TIDL runtime.

        Args:
            im (torch.Tensor): Input image tensor in BCHW format, normalized to [0, 1].

        Returns:
            (list): Model predictions as a list of output arrays.
        """
        return self.session.run(self.output_names, {self.session.get_inputs()[0].name: im.cpu().numpy()})
