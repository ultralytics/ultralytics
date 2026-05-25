# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from ultralytics.utils import LOGGER

from .base import BaseBackend


class DeepXBackend(BaseBackend):
    """DeepX NPU inference backend for DeepX hardware accelerators.

    Loads compiled DeepX models (.dxnn files) and runs inference using the DeepX DX-Runtime.
    """

    def load_model(self, weight: str | Path) -> None:
        """Load a DeepX model from a directory containing a .dxnn file.

        Args:
            weight (str | Path): Path to the DeepX model directory containing the .dxnn binary.

        Raises:
            ImportError: If the ``dx_engine`` Python package is not installed.
            FileNotFoundError: If no .dxnn file is found in the given directory.
        """
        try:
            from dx_engine import InferenceEngine
        except ImportError as e:
            raise ImportError(
                "DeepX inference requires the DeepX DX-Runtime and `dx_engine` Python package. "
                "See https://docs.ultralytics.com/integrations/deepx/#runtime-installation for installation instructions."
            ) from e

        LOGGER.info(f"Loading {weight} for DeepX inference...")

        w = Path(weight)
        found = next(w.rglob("*.dxnn"), None)
        if found is None:
            raise FileNotFoundError(f"No .dxnn file found in: {w}")

        self.model = InferenceEngine(str(found))

        # Load metadata
        metadata_file = found.parent / "metadata.yaml"
        if metadata_file.exists():
            from ultralytics.utils import YAML

            self.apply_metadata(YAML.load(metadata_file))

    def forward(self, im: torch.Tensor) -> np.ndarray | list[np.ndarray]:
        """Run inference on the DeepX NPU.

        Converts each image from BCHW float [0, 1] to HWC uint8 [0, 255] per the DeepX runtime contract,
        runs the engine per image, then stacks outputs along the batch dimension.

        Args:
            im (torch.Tensor): Input image tensor in BCHW format, normalized to [0, 1].

        Returns:
            (np.ndarray | list[np.ndarray]): Model predictions as a single array or list of arrays.
        """
        outputs = []
        for sample in im.cpu().numpy():
            sample = np.ascontiguousarray(np.clip(np.transpose(sample, (1, 2, 0)) * 255, 0, 255).astype(np.uint8))
            for i, out in enumerate(map(np.asarray, self.model.run([sample]))):
                if i == len(outputs):
                    outputs.append([])
                outputs[i].append(out if out.ndim and out.shape[0] == 1 else out[None])
        y = [np.concatenate(x, axis=0) for x in outputs]
        return y[0] if len(y) == 1 else y
