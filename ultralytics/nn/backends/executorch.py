# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from __future__ import annotations

from pathlib import Path

import torch

from ultralytics.utils import LOGGER
from ultralytics.utils.checks import check_executorch_requirements

from .base import BaseBackend


class ExecuTorchBackend(BaseBackend):
    """Meta ExecuTorch inference backend for on-device deployment.

    Loads and runs inference with Meta ExecuTorch models (.pte files) using the ExecuTorch runtime. Supports both
    standalone .pte files and directory-based model packages with metadata.
    """

    def load_model(self, weight: str | Path) -> None:
        """Load an ExecuTorch model from a .pte file or directory.

        Args:
            weight (str | Path): Path to the .pte model file or directory containing the model.
        """
        LOGGER.info(f"Loading {weight} for ExecuTorch inference...")
        check_executorch_requirements()

        from executorch.runtime import Runtime

        w = Path(weight)
        if w.is_dir():
            model_file = next(w.rglob("*.pte"))
            metadata_file = w / "metadata.yaml"
        else:
            model_file = w
            metadata_file = w.parent / "metadata.yaml"

        program = Runtime.get().load_program(str(model_file))
        self.model = program.load_method("forward")

        # Load metadata
        if metadata_file.exists():
            from ultralytics.utils import YAML

            self.apply_metadata(YAML.load(metadata_file))

    def forward(self, im: torch.Tensor) -> list:
        """Run inference using the ExecuTorch runtime.

        Args:
            im (torch.Tensor): Input image tensor in BCHW format, normalized to [0, 1].

        Returns:
            (list): Model predictions as a list of ExecuTorch output values.
        """
        return self.model.execute([im])
