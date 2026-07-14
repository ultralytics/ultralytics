"""DINOv3 ConvNeXt building blocks and backbone."""

from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvNeXtLayerNorm(nn.Module):
    """LayerNorm with channels_first / channels_last support."""

    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(normalized_shape))
        self.bias = nn.Parameter(torch.empty(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        self.normalized_shape = (normalized_shape,)

    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        # channels_first
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        return self.weight[:, None, None] * x + self.bias[:, None, None]


class ConvNeXtBlock(nn.Module):
    """DINOv3 ConvNeXt block: dwconv -> LN -> pwconv1 -> GELU -> pwconv2 -> gamma + residual."""

    def __init__(self, dim, drop_path=0.0, layer_scale_init_value=1e-6):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim)
        self.norm = ConvNeXtLayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, 4 * dim)
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.gamma = (
            nn.Parameter(layer_scale_init_value * torch.ones((dim)), requires_grad=True)
            if layer_scale_init_value > 0
            else None
        )
        self.drop_path = nn.Identity()  # inference only

    def forward(self, x):
        residual = x
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1)  # NCHW -> NHWC
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma is not None:
            x = self.gamma * x
        x = x.permute(0, 3, 1, 2)  # NHWC -> NCHW
        return residual + self.drop_path(x)


class DINOv3ConvNeXt(nn.Module):
    """Official DINOv3 ConvNeXt backbone (matches checkpoint keys exactly)."""

    def __init__(
        self,
        depths: list[int],
        dims: list[int],
        patch_size: int | None = None,
    ):
        super().__init__()
        # Stem + 3 downsampling layers
        self.downsample_layers = nn.ModuleList()
        stem = nn.Sequential(
            nn.Conv2d(3, dims[0], kernel_size=4, stride=4),
            ConvNeXtLayerNorm(dims[0], eps=1e-6, data_format="channels_first"),
        )
        self.downsample_layers.append(stem)
        for i in range(3):
            self.downsample_layers.append(nn.Sequential(
                ConvNeXtLayerNorm(dims[i], eps=1e-6, data_format="channels_first"),
                nn.Conv2d(dims[i], dims[i + 1], kernel_size=2, stride=2),
            ))
        # Stages
        self.stages = nn.ModuleList()
        for i in range(4):
            self.stages.append(nn.Sequential(*[
                ConvNeXtBlock(dim=dims[i]) for _ in range(depths[i])
            ]))
        # Norms: Identity for stages 0-2, LayerNorm for stage 3 (matches checkpoint)
        self.norms = nn.ModuleList([nn.Identity() for _ in range(3)])
        self.norm = nn.LayerNorm(dims[-1], eps=1e-6)
        self.norms.append(self.norm)
        # Housekeeping
        self.patch_size = patch_size
        self.n_storage_tokens = 0
        self.embed_dim = dims[-1]

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        """Forward features returning same dict format as DINOv2 hub model."""
        h, w = x.shape[-2:]
        for i in range(4):
            x = self.downsample_layers[i](x)
            x = self.stages[i](x)
        x_pool = x.mean([-2, -1])                               # [B, C]
        x = torch.flatten(x, 2).transpose(1, 2)                 # [B, HW, C]
        x_norm = self.norm(torch.cat([x_pool.unsqueeze(1), x], dim=1))  # [B, 1+HW, C]
        return {
            "x_norm_clstoken": x_norm[:, 0],
            "x_norm_patchtokens": x_norm[:, self.n_storage_tokens + 1:],
            "x_prenorm": x,
        }

    def forward_features(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        """Alias matching DINOv2 hub API."""
        return self.forward(x)


def load_convnext_sd(name: str, weight: str | Path | None, ckpt_dir: Path) -> dict:
    """Load a DINOv3-ConvNeXt state dict: explicit ``weight`` path, else glob the bundled ckpt."""
    if weight:
        ckpt_path = Path(weight)
    else:
        matches = sorted(ckpt_dir.glob(f"dinov3_convnext_{name}_pretrain_lvd1689m-*.pth"))
        if not matches:
            raise FileNotFoundError(f"No checkpoint for {name} in {ckpt_dir}")
        ckpt_path = matches[0]
    return torch.load(ckpt_path, map_location="cpu", weights_only=True)
