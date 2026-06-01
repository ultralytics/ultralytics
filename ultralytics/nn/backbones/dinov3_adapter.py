"""DEIMv2 DINOv3 backbone adapter (inference-only build)."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .dinov3 import DinoVisionTransformer

__all__ = ["DINOv3STAs"]


class SpatialPriorModulev2(nn.Module):
    """Lite spatial prior branch used by DEIMv2."""

    def __init__(self, inplanes: int = 16):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, inplanes, kernel_size=3, stride=2, padding=1, bias=False),
            nn.SyncBatchNorm(inplanes),
            nn.GELU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(inplanes, 2 * inplanes, kernel_size=3, stride=2, padding=1, bias=False),
            nn.SyncBatchNorm(2 * inplanes),
        )
        self.conv3 = nn.Sequential(
            nn.GELU(),
            nn.Conv2d(2 * inplanes, 4 * inplanes, kernel_size=3, stride=2, padding=1, bias=False),
            nn.SyncBatchNorm(4 * inplanes),
        )
        self.conv4 = nn.Sequential(
            nn.GELU(),
            nn.Conv2d(4 * inplanes, 4 * inplanes, kernel_size=3, stride=2, padding=1, bias=False),
            nn.SyncBatchNorm(4 * inplanes),
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        c1 = self.stem(x)
        c2 = self.conv2(c1)
        c3 = self.conv3(c2)
        c4 = self.conv4(c3)
        return c2, c3, c4


class DINOv3STAs(nn.Module):
    """DEIMv2 DINOv3 + STA fusion backbone returning three pyramid features."""

    def __init__(
        self,
        name: str = "dinov3_vits16",
        interaction_indexes: tuple[int, ...] | list[int] = (5, 8, 11),
        patch_size: int = 16,
        use_sta: bool = True,
        conv_inplane: int = 32,
        hidden_dim: int = 224,
    ):
        super().__init__()
        self.dinov3 = DinoVisionTransformer(name=name)
        self.interaction_indexes = list(interaction_indexes)
        self.patch_size = patch_size

        self.use_sta = use_sta
        if use_sta:
            self.sta = SpatialPriorModulev2(inplanes=conv_inplane)
        else:
            conv_inplane = 0

        embed_dim = self.dinov3.embed_dim
        self.convs = nn.ModuleList(
            [
                nn.Conv2d(embed_dim + conv_inplane * 2, hidden_dim, kernel_size=1, stride=1, padding=0, bias=False),
                nn.Conv2d(embed_dim + conv_inplane * 4, hidden_dim, kernel_size=1, stride=1, padding=0, bias=False),
                nn.Conv2d(embed_dim + conv_inplane * 4, hidden_dim, kernel_size=1, stride=1, padding=0, bias=False),
            ]
        )
        self.norms = nn.ModuleList(
            [nn.SyncBatchNorm(hidden_dim), nn.SyncBatchNorm(hidden_dim), nn.SyncBatchNorm(hidden_dim)]
        )
        self.out_channels = [hidden_dim, hidden_dim, hidden_dim]

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        h_base = x.shape[2] // 16
        w_base = x.shape[3] // 16
        bs = x.shape[0]

        all_layers = self.dinov3.get_intermediate_layers(x, n=self.interaction_indexes, return_class_token=True)
        if len(all_layers) == 1:
            all_layers = [all_layers[0], all_layers[0], all_layers[0]]

        sem_feats = []
        num_scales = len(all_layers) - 2
        for i, sem_layer in enumerate(all_layers):
            feat, _ = sem_layer
            sem = feat.transpose(1, 2).view(bs, -1, h_base, w_base).contiguous()
            resize_h = int(h_base * 2 ** (num_scales - i))
            resize_w = int(w_base * 2 ** (num_scales - i))
            sem = F.interpolate(sem, size=[resize_h, resize_w], mode="bilinear", align_corners=False)
            sem_feats.append(sem)

        if self.use_sta:
            detail_feats = self.sta(x)
            fused_feats = [torch.cat([s, d], dim=1) for s, d in zip(sem_feats, detail_feats)]
        else:
            fused_feats = sem_feats

        c2 = self.norms[0](self.convs[0](fused_feats[0]))
        c3 = self.norms[1](self.convs[1](fused_feats[1]))
        c4 = self.norms[2](self.convs[2](fused_feats[2]))
        return [c2, c3, c4]
