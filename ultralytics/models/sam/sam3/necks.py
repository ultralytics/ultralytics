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
        """
        SimpleFPN neck a la ViTDet
        (From detectron2, very lightly adapted)
        It supports a "dual neck" setting, where we have two identical necks (for SAM3 and SAM2), with different weights.

        :param trunk: the backbone
        :param position_encoding: the positional encoding to use
        :param d_model: the dimension of the model
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

    def set_imgsz(self, imgsz: list[int] = [1008, 1008]):
        """Set the image size for the trunk backbone."""
        self.trunk.set_imgsz(imgsz)
