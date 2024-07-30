# Ultralytics YOLO 🚀, AGPL-3.0 license

from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from ultralytics.models.sam.modules.encoders import PatchEmbed

from .sam2_blocks import CXBlock, Fuser, MaskDownSampler, MultiScaleBlock, PositionEmbeddingSine


class MemoryEncoder(nn.Module):
    """Encodes pixel features and masks into a memory representation for efficient image segmentation."""

    def __init__(
        self,
        out_dim,
        in_dim=256,  # in_dim of pix_feats
    ):
        """Initializes the MemoryEncoder module for encoding pixel features and masks in SAM-like models."""
        super().__init__()

        self.mask_downsampler = MaskDownSampler(kernel_size=3, stride=2, padding=1)

        self.pix_feat_proj = nn.Conv2d(in_dim, in_dim, kernel_size=1)
        self.fuser = Fuser(CXBlock(dim=256), num_layers=2)
        self.position_encoding = PositionEmbeddingSine(num_pos_feats=64)
        self.out_proj = nn.Identity()
        if out_dim != in_dim:
            self.out_proj = nn.Conv2d(in_dim, out_dim, kernel_size=1)

    def forward(
        self,
        pix_feat: torch.Tensor,
        masks: torch.Tensor,
        skip_mask_sigmoid: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Processes pixel features and masks, fusing them to generate encoded memory representations."""
        if not skip_mask_sigmoid:
            masks = F.sigmoid(masks)
        masks = self.mask_downsampler(masks)

        # Fuse pix_feats and downsampled masks, in case the visual features are on CPU, cast them to CUDA
        pix_feat = pix_feat.to(masks.device)

        x = self.pix_feat_proj(pix_feat)
        x = x + masks
        x = self.fuser(x)
        x = self.out_proj(x)

        pos = self.position_encoding(x).to(x.dtype)

        return {"vision_features": x, "vision_pos_enc": [pos]}


class ImageEncoder(nn.Module):
    """Encodes images using a trunk-neck architecture, producing multiscale features and positional encodings."""

    def __init__(
        self,
        trunk: nn.Module,
        neck: nn.Module,
        scalp: int = 0,
    ):
        """Initializes an image encoder with a trunk, neck, and optional scalp for feature extraction."""
        super().__init__()
        self.trunk = trunk
        self.neck = neck
        self.scalp = scalp
        assert (
            self.trunk.channel_list == self.neck.backbone_channel_list
        ), f"Channel dims of trunk {self.trunk.channel_list} and neck {self.neck.backbone_channel_list} do not match."

    def forward(self, sample: torch.Tensor):
        """Processes image input through trunk and neck, returning features, positional encodings, and FPN outputs."""
        features, pos = self.neck(self.trunk(sample))
        if self.scalp > 0:
            # Discard the lowest resolution features
            features, pos = features[: -self.scalp], pos[: -self.scalp]

        src = features[-1]
        output = {
            "vision_features": src,
            "vision_pos_enc": pos,
            "backbone_fpn": features,
        }
        return output


class FpnNeck(nn.Module):
    """Feature Pyramid Network (FPN) neck variant for multiscale feature fusion in object detection models."""

    def __init__(
        self,
        d_model: int,
        backbone_channel_list: List[int],
        kernel_size: int = 1,
        stride: int = 1,
        padding: int = 0,
        fpn_interp_model: str = "bilinear",
        fuse_type: str = "sum",
        fpn_top_down_levels: Optional[List[int]] = None,
    ):
        """
        Initializes a modified Feature Pyramid Network (FPN) neck.

        This FPN variant removes the output convolution and uses bicubic interpolation for feature resizing,
        similar to ViT positional embedding interpolation.

        Args:
            d_model (int): Dimension of the model.
            backbone_channel_list (List[int]): List of channel dimensions from the backbone.
            kernel_size (int): Kernel size for the convolutional layers.
            stride (int): Stride for the convolutional layers.
            padding (int): Padding for the convolutional layers.
            fpn_interp_model (str): Interpolation mode for FPN feature resizing.
            fuse_type (str): Type of feature fusion, either 'sum' or 'avg'.
            fpn_top_down_levels (Optional[List[int]]): Levels to have top-down features in outputs.

        Attributes:
            position_encoding (PositionEmbeddingSine): Sinusoidal positional encoding.
            convs (nn.ModuleList): List of convolutional layers for each backbone level.
            backbone_channel_list (List[int]): List of channel dimensions from the backbone.
            fpn_interp_model (str): Interpolation mode for FPN feature resizing.
            fuse_type (str): Type of feature fusion.
            fpn_top_down_levels (List[int]): Levels with top-down feature propagation.

        Examples:
            >>> backbone_channels = [64, 128, 256, 512]
            >>> fpn_neck = FpnNeck(256, backbone_channels)
            >>> print(fpn_neck)
        """
        super().__init__()
        self.position_encoding = PositionEmbeddingSine(num_pos_feats=256)
        self.convs = nn.ModuleList()
        self.backbone_channel_list = backbone_channel_list
        for dim in backbone_channel_list:
            current = nn.Sequential()
            current.add_module(
                "conv",
                nn.Conv2d(
                    in_channels=dim,
                    out_channels=d_model,
                    kernel_size=kernel_size,
                    stride=stride,
                    padding=padding,
                ),
            )

            self.convs.append(current)
        self.fpn_interp_model = fpn_interp_model
        assert fuse_type in ["sum", "avg"]
        self.fuse_type = fuse_type

        # levels to have top-down features in its outputs
        # e.g. if fpn_top_down_levels is [2, 3], then only outputs of level 2 and 3
        # have top-down propagation, while outputs of level 0 and level 1 have only
        # lateral features from the same backbone level.
        if fpn_top_down_levels is None:
            # default is to have top-down features on all levels
            fpn_top_down_levels = range(len(self.convs))
        self.fpn_top_down_levels = list(fpn_top_down_levels)

    def forward(self, xs: List[torch.Tensor]):
        """
        Performs forward pass through the Feature Pyramid Network (FPN) neck.

        Args:
            xs (List[torch.Tensor]): List of input tensors from the backbone, with shape (B, C, H, W) for each tensor.

        Returns:
            (Tuple[List[torch.Tensor], List[torch.Tensor]]): A tuple containing two lists:
                - out: List of output feature maps after FPN processing, with shape (B, d_model, H, W) for each tensor.
                - pos: List of positional encodings corresponding to each output feature map.

        Examples:
            >>> fpn_neck = FpnNeck(d_model=256, backbone_channel_list=[64, 128, 256, 512])
            >>> inputs = [torch.rand(1, c, 32, 32) for c in [64, 128, 256, 512]]
            >>> outputs, positions = fpn_neck(inputs)
        """
        out = [None] * len(self.convs)
        pos = [None] * len(self.convs)
        assert len(xs) == len(self.convs)
        # fpn forward pass
        # see https://github.com/facebookresearch/detectron2/blob/main/detectron2/modeling/backbone/fpn.py
        prev_features = None
        # forward in top-down order (from low to high resolution)
        n = len(self.convs) - 1
        for i in range(n, -1, -1):
            x = xs[i]
            lateral_features = self.convs[n - i](x)
            if i in self.fpn_top_down_levels and prev_features is not None:
                top_down_features = F.interpolate(
                    prev_features.to(dtype=torch.float32),
                    scale_factor=2.0,
                    mode=self.fpn_interp_model,
                    align_corners=(None if self.fpn_interp_model == "nearest" else False),
                    antialias=False,
                )
                prev_features = lateral_features + top_down_features
                if self.fuse_type == "avg":
                    prev_features /= 2
            else:
                prev_features = lateral_features
            x_out = prev_features
            out[i] = x_out
            pos[i] = self.position_encoding(x_out).to(x_out.dtype)

        return out, pos


class Hiera(nn.Module):
    """Hierarchical vision transformer for efficient multiscale feature extraction in image processing tasks."""

    def __init__(
        self,
        embed_dim: int = 96,  # initial embed dim
        num_heads: int = 1,  # initial number of heads
        drop_path_rate: float = 0.0,  # stochastic depth
        q_pool: int = 3,  # number of q_pool stages
        q_stride: Tuple[int, int] = (2, 2),  # downsample stride bet. stages
        stages: Tuple[int, ...] = (2, 3, 16, 3),  # blocks per stage
        dim_mul: float = 2.0,  # dim_mul factor at stage shift
        head_mul: float = 2.0,  # head_mul factor at stage shift
        window_pos_embed_bkg_spatial_size: Tuple[int, int] = (14, 14),
        # window size per stage, when not using global att.
        window_spec: Tuple[int, ...] = (
            8,
            4,
            14,
            7,
        ),
        # global attn in these blocks
        global_att_blocks: Tuple[int, ...] = (
            12,
            16,
            20,
        ),
        return_interm_layers=True,  # return feats from every stage
    ):
        """Initializes a Hiera model with configurable architecture for hierarchical vision transformers."""
        super().__init__()

        assert len(stages) == len(window_spec)
        self.window_spec = window_spec

        depth = sum(stages)
        self.q_stride = q_stride
        self.stage_ends = [sum(stages[:i]) - 1 for i in range(1, len(stages) + 1)]
        assert 0 <= q_pool <= len(self.stage_ends[:-1])
        self.q_pool_blocks = [x + 1 for x in self.stage_ends[:-1]][:q_pool]
        self.return_interm_layers = return_interm_layers

        self.patch_embed = PatchEmbed(
            embed_dim=embed_dim,
            kernel_size=(7, 7),
            stride=(4, 4),
            padding=(3, 3),
        )
        # Which blocks have global att?
        self.global_att_blocks = global_att_blocks

        # Windowed positional embedding (https://arxiv.org/abs/2311.05613)
        self.window_pos_embed_bkg_spatial_size = window_pos_embed_bkg_spatial_size
        self.pos_embed = nn.Parameter(torch.zeros(1, embed_dim, *self.window_pos_embed_bkg_spatial_size))
        self.pos_embed_window = nn.Parameter(torch.zeros(1, embed_dim, self.window_spec[0], self.window_spec[0]))

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule

        cur_stage = 1
        self.blocks = nn.ModuleList()

        for i in range(depth):
            dim_out = embed_dim
            # lags by a block, so first block of
            # next stage uses an initial window size
            # of previous stage and final window size of current stage
            window_size = self.window_spec[cur_stage - 1]

            if self.global_att_blocks is not None:
                window_size = 0 if i in self.global_att_blocks else window_size

            if i - 1 in self.stage_ends:
                dim_out = int(embed_dim * dim_mul)
                num_heads = int(num_heads * head_mul)
                cur_stage += 1

            block = MultiScaleBlock(
                dim=embed_dim,
                dim_out=dim_out,
                num_heads=num_heads,
                drop_path=dpr[i],
                q_stride=self.q_stride if i in self.q_pool_blocks else None,
                window_size=window_size,
            )

            embed_dim = dim_out
            self.blocks.append(block)

        self.channel_list = (
            [self.blocks[i].dim_out for i in self.stage_ends[::-1]]
            if return_interm_layers
            else [self.blocks[-1].dim_out]
        )

    def _get_pos_embed(self, hw: Tuple[int, int]) -> torch.Tensor:
        """Generate positional embeddings by interpolating and combining window and background embeddings."""
        h, w = hw
        window_embed = self.pos_embed_window
        pos_embed = F.interpolate(self.pos_embed, size=(h, w), mode="bicubic")
        pos_embed = pos_embed + window_embed.tile([x // y for x, y in zip(pos_embed.shape, window_embed.shape)])
        pos_embed = pos_embed.permute(0, 2, 3, 1)
        return pos_embed

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        """Performs hierarchical vision transformer forward pass, returning multiscale feature maps."""
        x = self.patch_embed(x)
        # x: (B, H, W, C)

        # Add pos embed
        x = x + self._get_pos_embed(x.shape[1:3])

        outputs = []
        for i, blk in enumerate(self.blocks):
            x = blk(x)
            if (i == self.stage_ends[-1]) or (i in self.stage_ends and self.return_interm_layers):
                feats = x.permute(0, 3, 1, 2)
                outputs.append(feats)

        return outputs
