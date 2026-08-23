# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from ultralytics.utils import LOGGER, YAML

from .base import BaseBackend


class AscendBackend(BaseBackend):
    """Huawei Ascend NPU inference backend for CANN offline models.

    Loads a compiled .om offline model and runs inference on the Ascend AI Processor through the ais_bench runtime,
    which wraps CANN's pyACL bindings.
    """

    def load_model(self, weight: str | Path) -> None:
        """Load an Ascend model from a directory containing a .om file.

        Args:
            weight (str | Path): Path to the Ascend model directory containing the .om offline model.

        Raises:
            ImportError: If the ``ais_bench`` Python package is not installed.
            FileNotFoundError: If no .om file is found in the given directory.
        """
        try:
            from ais_bench.infer.interface import InferSession
        except ImportError as e:
            raise ImportError(
                "Ascend inference requires the CANN runtime and `ais_bench` Python package. "
                "See https://docs.ultralytics.com/integrations/ascend#runtime-installation for instructions."
            ) from e

        LOGGER.info(f"Loading {weight} for Huawei Ascend inference...")

        w = Path(weight)
        found = next(w.rglob("*.om"), None)
        if found is None:
            raise FileNotFoundError(f"No .om file found in: {w}")

        self.model = InferSession(getattr(self.device, "index", None) or 0, str(found))

        # Load metadata
        metadata_file = found.parent / "metadata.yaml"
        if metadata_file.exists():
            self.apply_metadata(YAML.load(metadata_file))

    def __del__(self):
        """Release the Ascend device-side resources held by the inference session."""
        if model := getattr(self, "model", None):
            model.free_resource()

    def forward(self, im: torch.Tensor) -> np.ndarray | list[np.ndarray]:
        """Run inference on the Ascend NPU.

        Args:
            im (torch.Tensor): Input image tensor in BCHW format, normalized to [0, 1].

        Returns:
            (np.ndarray | list[np.ndarray]): Model predictions as a single array or list of arrays.
        """
        y = self.model.infer([im.cpu().numpy()])
        return y[0] if len(y) == 1 else y
