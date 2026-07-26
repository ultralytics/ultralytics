# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from __future__ import annotations

from pathlib import Path

import torch

from ultralytics.utils.checks import check_requirements

from .base import BaseBackend


class TritonBackend(BaseBackend):
    """NVIDIA Triton Inference Server backend for remote model serving.

    Connects to and runs inference with models hosted on an NVIDIA Triton Inference Server instance via HTTP or gRPC
    protocols. The model is specified using a triton:// URL scheme.
    """

    def load_model(self, weight: str | Path) -> None:
        """Connect to a remote model on an NVIDIA Triton Inference Server.

        Args:
            weight (str | Path): Triton model URL (e.g., 'triton://host:8000/model_name').
        """
        check_requirements("tritonclient[all]")
        from ultralytics.utils.triton import TritonRemoteModel

        self.model = TritonRemoteModel(weight)

        # Copy metadata from Triton model
        if hasattr(self.model, "metadata"):
            self.apply_metadata(self.model.metadata)

    def forward(self, im: torch.Tensor) -> list:
        """Run inference via the NVIDIA Triton Inference Server.

        Args:
            im (torch.Tensor): Input image tensor in BCHW format, normalized to [0, 1].

        Returns:
            (list): Model predictions as a list of numpy arrays from the Triton server.
        """
        return self.model(im.cpu().numpy())
