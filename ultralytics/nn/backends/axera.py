# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from __future__ import annotations

from pathlib import Path

import torch

from ultralytics.utils import LOGGER

from .base import BaseBackend


class AxeraBackend(BaseBackend):
    """AXERA NPU inference backend for Pulsar2 models.

    Loads a compiled .axmodel and runs it on an AXERA board or M.2 card through the ``axengine`` runtime (pyaxengine).
    """

    def load_model(self, weight: str | Path) -> None:
        """Load an AXERA model from a .axmodel file or model directory.

        Args:
            weight (str | Path): Path to the .axmodel file or directory containing the model.

        Raises:
            ImportError: If the ``axengine`` Python package is not installed.
        """
        try:
            import axengine
        except ImportError as e:
            raise ImportError(
                "AXERA inference requires the 'axengine' runtime from https://github.com/AXERA-TECH/pyaxengine. "
                "See https://docs.ultralytics.com/integrations/axera"
            ) from e

        LOGGER.info(f"Loading {weight} for AXERA inference...")
        w = Path(weight)
        if not w.is_file():
            w = next(w.rglob("*.axmodel"))
        self.model = axengine.InferenceSession(str(w))
        self.load_tail(w)
        self.apply_metadata(self.read_metadata(w))

    def forward(self, im: torch.Tensor) -> list:
        """Run inference on the AXERA NPU.

        Args:
            im (torch.Tensor): Input image tensor in BHWC format, normalized to [0, 1].

        Returns:
            (list): Model predictions as a list of output arrays.
        """
        im = (im.cpu().numpy() * 255).astype("uint8")
        return self.run_tail(self.model.run(None, {self.model.get_inputs()[0].name: im}))
