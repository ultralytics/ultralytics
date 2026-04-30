# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

from ultralytics.utils import IS_JETSON, LOGGER, is_jetson

from .base import BaseBackend


class PyTorchBackend(BaseBackend):
    """PyTorch inference backend for native model execution.

    Loads and runs inference with native PyTorch models (.pt checkpoint files) or pre-loaded nn.Module
    instances. Supports model layer fusion, FP16 precision, and NVIDIA Jetson compatibility.
    """

    def __init__(
        self,
        weight: str | Path | nn.Module,
        device: torch.device,
        fp16: bool = False,
        fuse: bool = True,
        verbose: bool = True,
    ):
        """Initialize the PyTorch backend.

        Args:
            weight (str | Path | nn.Module): Path to the .pt model file or a pre-loaded nn.Module instance.
            device (torch.device): Device to run inference on (e.g., 'cpu', 'cuda:0').
            fp16 (bool): Whether to use FP16 half-precision inference.
            fuse (bool): Whether to fuse Conv2D + BatchNorm layers for optimization.
            verbose (bool): Whether to print verbose model loading messages.
        """
        self.fuse = fuse
        self.verbose = verbose
        super().__init__(weight, device, fp16)

    def load_model(self, weight: str | torch.nn.Module) -> None:
        """Load a PyTorch model from a checkpoint file or nn.Module instance.

        Args:
            weight (str | torch.nn.Module): Path to the .pt checkpoint or a pre-loaded module.
        """
        from ultralytics.nn.tasks import load_checkpoint

        if isinstance(weight, torch.nn.Module):
            if self.fuse and hasattr(weight, "fuse"):
                if IS_JETSON and is_jetson(jetpack=5):
                    weight = weight.to(self.device)
                weight = weight.fuse(verbose=self.verbose)
            model = weight.to(self.device)
        else:
            model, _ = load_checkpoint(weight, device=self.device, fuse=self.fuse)

        # Extract model attributes
        if hasattr(model, "kpt_shape"):
            self.kpt_shape = model.kpt_shape
        self.stride = max(int(model.stride.max()), 32) if hasattr(model, "stride") else 32
        self.names = model.module.names if hasattr(model, "module") else getattr(model, "names", {})
        self.channels = model.yaml.get("channels", 3) if hasattr(model, "yaml") else 3
        model.half() if self.fp16 else model.float()

        for p in model.parameters():
            p.requires_grad = False

        self.model = model
        self.end2end = getattr(model, "end2end", False)

    def forward(
        self, im: torch.Tensor, augment: bool = False, visualize: bool = False, embed: list | None = None, **kwargs: Any
    ) -> torch.Tensor | list[torch.Tensor]:
        """Run native PyTorch inference with support for augmentation, visualization, and embeddings.

        Args:
            im (torch.Tensor): Input image tensor in BCHW format, normalized to [0, 1].
            augment (bool): Whether to apply test-time augmentation.
            visualize (bool): Whether to visualize intermediate feature maps.
            embed (list | None): List of layer indices to extract embeddings from, or None.
            **kwargs (Any): Additional keyword arguments passed to the model forward method.

        Returns:
            (torch.Tensor | list[torch.Tensor]): Model predictions as tensor(s).
        """
        return self.model(im, augment=augment, visualize=visualize, embed=embed, **kwargs)


class TorchScriptBackend(BaseBackend):
    """PyTorch TorchScript inference backend for serialized model execution.

    Loads and runs inference with TorchScript models (.torchscript files) created via torch.jit.trace or
    torch.jit.script. Supports FP16 precision and embedded metadata extraction.
    """

    def __init__(self, weight: str | Path, device: torch.device, fp16: bool = False):
        """Initialize the TorchScript backend.

        Args:
            weight (str | Path): Path to the .torchscript model file.
            device (torch.device): Device to run inference on (e.g., 'cpu', 'cuda:0').
            fp16 (bool): Whether to use FP16 half-precision inference.
        """
        super().__init__(weight, device, fp16)

    def load_model(self, weight: str) -> None:
        """Load a TorchScript model from a .torchscript file with optional embedded metadata.

        Args:
            weight (str): Path to the .torchscript model file.
        """
        import json

        import torchvision  # noqa - required for TorchScript model deserialization

        LOGGER.info(f"Loading {weight} for TorchScript inference...")
        extra_files = {"config.txt": ""}
        self.model = torch.jit.load(weight, _extra_files=extra_files, map_location=self.device)
        self.model.half() if self.fp16 else self.model.float()

        if extra_files["config.txt"]:
            self.apply_metadata(json.loads(extra_files["config.txt"], object_hook=lambda x: dict(x.items())))

    def forward(self, im: torch.Tensor) -> torch.Tensor | list[torch.Tensor]:
        """Run TorchScript inference.

        Args:
            im (torch.Tensor): Input image tensor in BCHW format, normalized to [0, 1].

        Returns:
            (torch.Tensor | list[torch.Tensor]): Model predictions as tensor(s).
        """
        return self.model(im)
