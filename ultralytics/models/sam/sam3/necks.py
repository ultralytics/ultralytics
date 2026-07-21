# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved

"""Necks are the interface between a vision backbone and the rest of the detection model."""

from __future__ import annotations

from copy import deepcopy

import torch
import torch.nn as nn


class Sam3DualViTDetNeck(nn.Module):
    """A neck that implements a simple FPN as in ViTDet, with support for dual necks (for SAM3 and SAM2)."""

    def __init__(
        self,
        trunk: nn.Module,
        position_encoding: nn.Module,
        d_model: int,
        scale_factors=(4.0, 2.0, 1.0, 0.5),
        add_sam2_neck: bool = False,
    ):
        """SimpleFPN neck a la ViTDet, very lightly adapted from detectron2.

        Supports a "dual neck" setting with two identical necks (for SAM3 and SAM2) that have different weights.

        Args:
            trunk (nn.Module): The backbone.
            position_encoding (nn.Module): The positional encoding to use.
            d_model (int): The dimension of the model.
            scale_factors (tuple): Scale factors for each FPN level.
            add_sam2_neck (bool): Whether to add a second neck for SAM2.
        """
        super().__init__()
        self.trunk = trunk
        self.position_encoding = position_encoding
        self.convs = nn.ModuleList()

        self.scale_factors = scale_factors
        use_bias = True
        dim: int = self.trunk.channel_list[-1]

        for _, scale in enumerate(scale_factors):
            current = nn.Sequential()

            if scale == 4.0:
                current.add_module(
                    "dconv_2x2_0",
                    nn.ConvTranspose2d(dim, dim // 2, kernel_size=2, stride=2),
                )
                current.add_module(
                    "gelu",
                    nn.GELU(),
                )
                current.add_module(
                    "dconv_2x2_1",
                    nn.ConvTranspose2d(dim // 2, dim // 4, kernel_size=2, stride=2),
                )
                out_dim = dim // 4
            elif scale == 2.0:
                current.add_module(
                    "dconv_2x2",
                    nn.ConvTranspose2d(dim, dim // 2, kernel_size=2, stride=2),
                )
                out_dim = dim // 2
            elif scale == 1.0:
                out_dim = dim
            elif scale == 0.5:
                current.add_module(
                    "maxpool_2x2",
                    nn.MaxPool2d(kernel_size=2, stride=2),
                )
                out_dim = dim
            else:
                raise NotImplementedError(f"scale_factor={scale} is not supported yet.")

            current.add_module(
                "conv_1x1",
                nn.Conv2d(
                    in_channels=out_dim,
                    out_channels=d_model,
                    kernel_size=1,
                    bias=use_bias,
                ),
            )
            current.add_module(
                "conv_3x3",
                nn.Conv2d(
                    in_channels=d_model,
                    out_channels=d_model,
                    kernel_size=3,
                    padding=1,
                    bias=use_bias,
                ),
            )
            self.convs.append(current)

        self.sam2_convs = None
        if add_sam2_neck:
            # Assumes sam2 neck is just a clone of the original neck
            self.sam2_convs = deepcopy(self.convs)

    def forward(
        self, tensor_list: list[torch.Tensor]
    ) -> tuple[list[torch.Tensor], list[torch.Tensor], list[torch.Tensor] | None, list[torch.Tensor] | None]:
        """Get feature maps and positional encodings from the neck."""
        xs = self.trunk(tensor_list)
        x = xs[-1]  # simpleFPN
        sam3_out, sam3_pos = self.sam_forward_feature_levels(x, self.convs)
        if self.sam2_convs is None:
            return sam3_out, sam3_pos, None, None
        sam2_out, sam2_pos = self.sam_forward_feature_levels(x, self.sam2_convs)
        return sam3_out, sam3_pos, sam2_out, sam2_pos

    def sam_forward_feature_levels(
        self, x: torch.Tensor, convs: nn.ModuleList
    ) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
        """Run neck convolutions and compute positional encodings for each feature level."""
        outs, poss = [], []
        for conv in convs:
            feat = conv(x)
            outs.append(feat)
            poss.append(self.position_encoding(feat).to(feat.dtype))
        return outs, poss

    def set_imgsz(self, imgsz: list[int] | None = None):
        """Set the image size for the trunk backbone."""
        imgsz = imgsz if imgsz is not None else [1008, 1008]
        self.trunk.set_imgsz(imgsz)


class Sam3TriViTDetNeck(Sam3DualViTDetNeck):
    """SimpleFPN neck with three heads (sam3 detection, interactive, propagation) for SAM 3.1 multiplex.

    The three heads are weight-independent clones of the same conv pyramid: ``convs`` feeds the DETR
    detector, ``interactive_convs`` the click-refinement SAM head, and ``propagation_convs`` the
    multiplex tracker (memory encoder/attention features).
    """

    def __init__(
        self,
        trunk: nn.Module,
        position_encoding: nn.Module,
        d_model: int,
        scale_factors=(4.0, 2.0, 1.0),
    ):
        """Initialize the tri-head neck.

        Args:
            trunk (nn.Module): The ViT backbone.
            position_encoding (nn.Module): The positional encoding module shared by all heads.
            d_model (int): The output dimension of each FPN level.
            scale_factors (tuple): Scale factors for each FPN level.
        """
        super().__init__(trunk, position_encoding, d_model, scale_factors, add_sam2_neck=False)
        self.interactive_convs = deepcopy(self.convs)
        self.propagation_convs = deepcopy(self.convs)

    def forward(
        self,
        tensor_list: list[torch.Tensor],
        need_sam3_out: bool = True,
        need_interactive_out: bool = True,
        need_propagation_out: bool = True,
    ):
        """Run the trunk once and the requested neck heads.

        Returns:
            sam3_out (list[torch.Tensor]): Detection features per level (empty if not requested).
            sam3_pos (list[torch.Tensor]): Their positional encodings.
            interactive_out (list[torch.Tensor]): Interactive-head features per level.
            interactive_pos (list[torch.Tensor]): Their positional encodings.
            propagation_out (list[torch.Tensor]): Propagation-head features per level.
            propagation_pos (list[torch.Tensor]): Their positional encodings.
        """
        xs = self.trunk(tensor_list)
        x = xs[-1]  # simpleFPN
        sam3_out, sam3_pos = self.sam_forward_feature_levels(x, self.convs) if need_sam3_out else ([], [])
        interactive_out, interactive_pos = (
            self.sam_forward_feature_levels(x, self.interactive_convs) if need_interactive_out else ([], [])
        )
        propagation_out, propagation_pos = (
            self.sam_forward_feature_levels(x, self.propagation_convs) if need_propagation_out else ([], [])
        )
        return sam3_out, sam3_pos, interactive_out, interactive_pos, propagation_out, propagation_pos
