# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

"""10-branch detection head for Stereo CenterNet."""

from __future__ import annotations

import torch
import torch.nn as nn

from ultralytics.nn.modules import Conv


class StereoCenterNetHead(nn.Module):
    """Stereo CenterNet detection head with 10 parallel branches.

    Outputs 10 branches as specified in constitution:
    1. heatmap: [B, C, H/4, W/4] - Center point heatmap
    2. offset: [B, 2, H/4, W/4] - Sub-pixel offset (δx, δy)
    3. bbox_size: [B, 2, H/4, W/4] - 2D box size (w, h)
    4. lr_distance: [B, 1, H/4, W/4] - Left-right center distance d
    5. right_width: [B, 1, H/4, W/4] - Right box width wr
    6. dimensions: [B, 3, H/4, W/4] - 3D dimension offsets (ΔH, ΔW, ΔL)
    7. orientation: [B, 8, H/4, W/4] - Multi-Bin orientation encoding
    8. vertices: [B, 8, H/4, W/4] - Bottom 4 vertex coordinates
    9. vertex_offset: [B, 8, H/4, W/4] - Vertex sub-pixel offsets
    10. vertex_dist: [B, 4, H/4, W/4] - Center to vertex distances
    """

    def __init__(self, in_channels: int = 256, num_classes: int = 3):
        """Initialize Stereo CenterNet head.

        Args:
            in_channels: Number of input feature channels.
            num_classes: Number of object classes (Car, Pedestrian, Cyclist).
        """
        super().__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes

        # Shared feature extractor
        self.shared_head = nn.Sequential(
            Conv(in_channels, 256, 3, 1, 1),  # [B, 256, H, W] -> [B, 256, H, W]
            Conv(256, 256, 3, 1, 1),  # [B, 256, H, W] -> [B, 256, H, W]
        )

        # 10 branches as per constitution
        self.branches = nn.ModuleDict(
            {
                "heatmap": self._build_branch(256, num_classes),  # [B, C, H/4, W/4]
                "offset": self._build_branch(256, 2),  # [B, 2, H/4, W/4]
                "bbox_size": self._build_branch(256, 2),  # [B, 2, H/4, W/4]
                "lr_distance": self._build_branch(256, 1),  # [B, 1, H/4, W/4]
                "right_width": self._build_branch(256, 1),  # [B, 1, H/4, W/4]
                "dimensions": self._build_branch(256, 3),  # [B, 3, H/4, W/4]
                "orientation": self._build_branch(256, 8),  # [B, 8, H/4, W/4]
                "vertices": self._build_branch(256, 8),  # [B, 8, H/4, W/4]
                "vertex_offset": self._build_branch(256, 8),  # [B, 8, H/4, W/4]
                "vertex_dist": self._build_branch(256, 4),  # [B, 4, H/4, W/4]
            }
        )

        # Initialize heatmap branch bias for focal loss (bias=-2.19)
        self._init_heatmap_bias()

    def _build_branch(self, in_channels: int, out_channels: int) -> nn.Sequential:
        """Build a single branch: Conv(3×3) → BN → ReLU → Conv(1×1).

        Args:
            in_channels: Input channels.
            out_channels: Output channels.

        Returns:
            Sequential module for the branch.
        """
        return nn.Sequential(
            Conv(in_channels, 256, 3, 1, 1),  # [B, 256, H, W] -> [B, 256, H, W]
            Conv(256, out_channels, 1, 1, 0),  # [B, 256, H, W] -> [B, out_channels, H, W]
        )

    def _init_heatmap_bias(self):
        """Initialize heatmap branch bias to -2.19 for focal loss prior."""
        heatmap_conv = self.branches["heatmap"][-1].conv
        if heatmap_conv.bias is not None:
            nn.init.constant_(heatmap_conv.bias, -2.19)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        """Forward pass through detection head.

        Args:
            x: Input features [B, in_channels, H/4, W/4] from backbone/neck.

        Returns:
            Dictionary with 10 branch outputs, each [B, C, H/4, W/4].
        """
        # Shared feature extraction
        shared_feat = self.shared_head(x)  # [B, 256, H/4, W/4]

        # Run all 10 branches in parallel
        outputs = {}
        for branch_name, branch_module in self.branches.items():
            outputs[branch_name] = branch_module(shared_feat)

        # Apply sigmoid to heatmap and vertices (for focal loss)
        outputs["heatmap"] = torch.sigmoid(outputs["heatmap"])
        outputs["vertices"] = torch.sigmoid(outputs["vertices"])

        return outputs

