# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

"""6-channel input backbone for stereo vision."""

from __future__ import annotations

import torch
import torch.nn as nn

from ultralytics.nn.modules import Conv, C3k2, SPPF, C2PSA
# parse_model imported locally when needed to avoid circular import


class StereoConv(nn.Module):
    """6-channel input convolution layer for stereo image pairs.

    Extends standard Conv to accept 6 channels (RGB left + RGB right).
    """

    def __init__(self, c1: int, c2: int, k: int = 3, s: int = 1, p: int | None = None, act: bool = True):
        """Initialize StereoConv layer.

        Args:
            c1: Input channels (must be 6 for first layer).
            c2: Output channels.
            k: Kernel size.
            s: Stride.
            p: Padding.
            act: Activation function.
        """
        super().__init__()
        from ultralytics.nn.modules.conv import autopad

        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p), bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.SiLU() if act is True else (act if isinstance(act, nn.Module) else nn.Identity())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor [B, 6, H, W] for first layer, or [B, C, H, W] for subsequent layers.

        Returns:
            Output tensor [B, C', H', W'].
        """
        return self.act(self.bn(self.conv(x)))


class StereoBackbone(nn.Module):
    """6-channel input backbone extending YOLOv11 architecture.

    Modifies first convolution layer to accept 6 channels (RGB left + RGB right)
    while maintaining compatibility with YOLOv11 backbone structure.
    """

    def __init__(self, cfg: dict, ch: int = 6):
        """Initialize stereo backbone.

        Args:
            cfg: Backbone configuration from YAML (list of layer definitions).
            ch: Input channels (6 for stereo, 3 for single image).
        """
        super().__init__()
        self.ch = ch

        # Parse backbone layers from config
        # Replace first Conv with StereoConv if input_channels == 6
        layers = []
        for i, layer_def in enumerate(cfg):
            if i == 0 and layer_def[2] == "Conv" and ch == 6:
                # First layer: use StereoConv for 6-channel input
                args = layer_def[3]
                layers.append(("StereoConv", StereoConv(ch, args[0], args[1], args[2])))
            else:
                # Subsequent layers: use standard modules
                layers.append(layer_def)

        # Build backbone using parse_model (Ultralytics model builder)
        # Import locally to avoid circular import
        from ultralytics.nn.tasks import parse_model

        self.model, self.save = parse_model(layers, ch=[ch])

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        """Forward pass through backbone.

        Args:
            x: Input tensor [B, 6, H, W] (stereo image pair concatenated).

        Returns:
            List of feature maps at different scales [P3, P4, P5].
        """
        y = []  # outputs
        for m in self.model:
            if m.f != -1:  # if not from previous layer
                x = y[m.f] if isinstance(m.f, int) else [x if j == -1 else y[j] for j in m.f]  # from earlier layers
            x = m(x)  # run
            y.append(x if m.i in self.save else None)  # save output

        # Return feature maps at P3, P4, P5 scales
        # Assuming save indices correspond to P3, P4, P5
        return [y[i] for i in self.save if y[i] is not None]

