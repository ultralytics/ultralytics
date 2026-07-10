# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

"""Layer-by-layer building blocks for the DINOv3 ViT backbone.

These modules let a DINOv3+STA backbone be expressed in YAML row by row, matching the
convention of yolo26-ultravit-cls.yaml. They wrap the primitives already living in
`ultralytics.nn.backbones.dinov3.layers` (SelfAttention, LayerScale, Mlp, SwiGLUFFN) and
add just enough glue to route tokens through `nn.Sequential`:

    DinoV3PatchStem       Conv2d patch embed + prepend cls and n_storage register tokens.
    DinoV3Block           Pre-norm SA + LayerScale + FFN on (B, N, dim) tokens; RoPE is
                          rebuilt inside forward from N under a square-grid assumption.
    DinoV3TokenToSpatial  LayerNorm + strip cls/storage + reshape to (B, dim, H, W).
    DinoV3Downsample2x    Bilinear resize by 0.5 on (B, C, H, W).

Blocks are dim-preserving: c_out == c_in == dim. The YAML parser injects dim from ch[f].
"""

from __future__ import annotations

import math
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from ultralytics.nn.backbones.dinov3.layers import LayerScale, Mlp, SelfAttention, SwiGLUFFN
from ultralytics.nn.modules.conv import Conv

__all__ = ["DinoV3PatchStem", "DinoV3Block", "DinoV3TokenToSpatial", "DinoV3Downsample2x", "ConvSyncBN"]


class ConvSyncBN(Conv):
    """Ultralytics Conv wrapper whose BN is nn.SyncBatchNorm instead of nn.BatchNorm2d.

    Purpose: match the source DEIMDINOv3STAs SPM/fusion convention (SyncBN with eps=1e-5, momentum=0.1) and slip past
    `initialize_weights`, which mutates only modules whose `type(m) is nn.BatchNorm2d`. SyncBatchNorm is a sibling
    class, not a subclass, so the strict-identity check skips it and the eps stays at 1e-5. State_dict keys are
    identical to nn.BatchNorm2d, so existing flat checkpoints load without any remap.
    """

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        """Build a Conv but replace its BatchNorm2d with a SyncBatchNorm of the same width."""
        super().__init__(c1, c2, k, s, p, g, d, act)
        self.bn = nn.SyncBatchNorm(c2)


_FFN_MAP = {
    "mlp": Mlp,
    "swiglu": SwiGLUFFN,
    "swiglu64": partial(SwiGLUFFN, align_to=64),
}


class DinoV3PatchStem(nn.Module):
    """Patch embed + prepend cls and register tokens for a DINOv3 ViT.

    Emits a token tensor of shape (B, 1 + n_storage + HW, c2) where HW = (H/patch) * (W/patch). The parser treats this
    like a base_modules entry: first YAML arg is c2 (embed_dim), c1 is injected from ch[f].

    Attributes:
        proj (nn.Conv2d): Patch projection Conv2d(c1, c2, k=patch, s=patch).
        cls_token (nn.Parameter): (1, 1, c2) learnable class token.
        storage_tokens (nn.Parameter): (1, n_storage, c2) learnable register tokens.
        mask_token (nn.Parameter): (1, c2) learnable mask token, kept to preserve state_dict layout of pretrained DINOv3
            checkpoints.
        n_storage (int): Number of register tokens.
        patch_size (int): Patch size of the Conv2d projection.
    """

    def __init__(self, c1: int, c2: int, patch_size: int = 16, n_storage: int = 4):
        """Initialize DinoV3PatchStem.

        Args:
            c1 (int): Input image channels.
            c2 (int): Embedding dimension.
            patch_size (int): Square patch size for the Conv2d projection.
            n_storage (int): Number of register tokens.
        """
        super().__init__()
        self.n_storage = n_storage
        self.patch_size = patch_size
        self.proj = nn.Conv2d(c1, c2, kernel_size=patch_size, stride=patch_size)
        self.cls_token = nn.Parameter(torch.empty(1, 1, c2))
        self.storage_tokens = nn.Parameter(torch.empty(1, n_storage, c2))
        self.mask_token = nn.Parameter(torch.empty(1, c2))
        self._init_weights()

    def _init_weights(self) -> None:
        """Match the DinoVisionTransformer init scheme."""
        k = 1 / (self.proj.in_channels * self.patch_size**2)
        nn.init.uniform_(self.proj.weight, -math.sqrt(k), math.sqrt(k))
        if self.proj.bias is not None:
            nn.init.uniform_(self.proj.bias, -math.sqrt(k), math.sqrt(k))
        nn.init.normal_(self.cls_token, std=0.02)
        nn.init.normal_(self.storage_tokens, std=0.02)
        nn.init.zeros_(self.mask_token)

    def forward(self, x: Tensor) -> Tensor:
        """Patch embed and prepend cls + storage tokens.

        Args:
            x (Tensor): (B, C, H, W) image tensor.

        Returns:
            (Tensor): (B, 1 + n_storage + HW, D) token tensor.
        """
        B = x.shape[0]
        x = self.proj(x).flatten(2).transpose(1, 2)  # (B, HW, D)
        cls = self.cls_token + 0 * self.mask_token  # keeps mask_token reachable in autograd
        return torch.cat(
            [cls.expand(B, -1, -1), self.storage_tokens.expand(B, -1, -1), x],
            dim=1,
        )


class DinoV3Block(nn.Module):
    """DINOv3 pre-norm SelfAttention + LayerScale + FFN block on (B, N, dim) tokens.

    RoPE table is rebuilt per forward from N under a square-grid assumption (H = W = int(sqrt(N - 1 - n_storage))). The
    first 1 + n_storage tokens skip RoPE via the prefix mechanism inside SelfAttention.apply_rope.

    Attributes:
        dim (int): Embedding dimension.
        num_heads (int): Number of attention heads.
        head_dim (int): Per-head dimension, dim // num_heads.
        n_storage (int): Number of register tokens (must match PatchStem).
        norm1 (nn.LayerNorm): Pre-attention norm.
        attn (SelfAttention): Multi-head self-attention with masked K bias.
        ls1 (LayerScale): LayerScale on the attention residual.
        norm2 (nn.LayerNorm): Pre-FFN norm.
        mlp (nn.Module): FFN (Mlp or SwiGLUFFN).
        ls2 (LayerScale): LayerScale on the FFN residual.
    """

    _ROPE_BASE = 100.0

    def __init__(
        self,
        dim: int,
        num_heads: int,
        ffn_ratio: float,
        ffn_layer: str = "mlp",
        ls_init: float = 1.0e-5,
        n_storage: int = 4,
    ):
        """Initialize DinoV3Block.

        Args:
            dim (int): Embedding dimension.
            num_heads (int): Number of attention heads (dim must be divisible by num_heads).
            ffn_ratio (float): FFN expansion ratio.
            ffn_layer (str): One of 'mlp', 'swiglu', 'swiglu64'.
            ls_init (float): LayerScale initial value.
            n_storage (int): Register token count, used to skip RoPE on cls + register tokens.
        """
        super().__init__()
        assert dim % num_heads == 0, f"dim={dim} not divisible by num_heads={num_heads}"
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.n_storage = n_storage

        norm_layer = partial(nn.LayerNorm, eps=1e-5)
        ffn_cls = _FFN_MAP[ffn_layer]

        self.norm1 = norm_layer(dim)
        self.attn = SelfAttention(
            dim, num_heads=num_heads, qkv_bias=True, proj_bias=True, mask_k_bias=True
        )
        self.ls1 = LayerScale(dim, init_values=ls_init)
        self.norm2 = norm_layer(dim)
        self.mlp = ffn_cls(
            in_features=dim,
            hidden_features=int(dim * ffn_ratio),
            act_layer=nn.GELU,
            bias=True,
        )
        self.ls2 = LayerScale(dim, init_values=ls_init)

        # RoPE periods depend on head_dim and base, not H/W. Persistent buffer so a remapper
        # can bit-copy the source checkpoint's `rope_embed.periods` (which may have been
        # fp16-quantized on save) into every block — otherwise freshly-computed fp32 periods
        # drift by ~1e-3 vs the saved fp16 values, and the drift explodes over 12 layers.
        n_periods = self.head_dim // 4
        periods = self._ROPE_BASE ** (
            2 * torch.arange(n_periods, dtype=torch.float32) / (self.head_dim // 2)
        )
        self.register_buffer("_periods", periods, persistent=True)

    def _rope(self, HW: int, device: torch.device, dtype: torch.dtype) -> tuple[Tensor, Tensor]:
        """Build (sin, cos) RoPE tables for a square HW grid."""
        H = int(HW**0.5)
        W = HW // H
        coords_h = torch.arange(0.5, H, dtype=torch.float32, device=device) / H
        coords_w = torch.arange(0.5, W, dtype=torch.float32, device=device) / W
        coords = torch.stack(torch.meshgrid(coords_h, coords_w, indexing="ij"), dim=-1)
        coords = coords.flatten(0, 1)  # (HW, 2)
        coords = 2.0 * coords - 1.0
        angles = 2 * math.pi * coords[:, :, None] / self._periods[None, None, :]  # (HW, 2, D/4)
        angles = angles.flatten(1, 2).tile(2)  # (HW, D)
        return angles.sin().to(dtype), angles.cos().to(dtype)

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass: pre-norm SA + LS residual, then pre-norm FFN + LS residual."""
        HW = x.shape[1] - 1 - self.n_storage
        rope = self._rope(HW, x.device, x.dtype)
        x = x + self.ls1(self.attn(self.norm1(x), rope=rope))
        x = x + self.ls2(self.mlp(self.norm2(x)))
        return x


class DinoV3TokenToSpatial(nn.Module):
    """Final LayerNorm + strip cls/storage + reshape (B, N, D) → (B, D, H, W).

    Assumes a square patch grid, deriving H = W = int(sqrt(N - 1 - n_storage)). Emits a (B, D, H, W) tensor suitable for
    the standard Ultralytics 4D pipeline.

    Attributes:
        dim (int): Embedding dimension (== channel count of the emitted spatial tensor).
        n_storage (int): Number of register tokens to strip alongside the cls token.
        norm (nn.LayerNorm): Final normalization applied on tokens before reshape.
    """

    def __init__(self, dim: int, n_storage: int = 4):
        """Initialize DinoV3TokenToSpatial.

        Args:
            dim (int): Embedding dimension.
            n_storage (int): Register token count to strip.
        """
        super().__init__()
        self.dim = dim
        self.n_storage = n_storage
        self.norm = nn.LayerNorm(dim, eps=1e-5)

    def forward(self, x: Tensor) -> Tensor:
        """LayerNorm, drop cls + register tokens, reshape to spatial."""
        x = self.norm(x)[:, 1 + self.n_storage :, :]  # (B, HW, D)
        B, HW, D = x.shape
        H = int(HW**0.5)
        W = HW // H
        return x.transpose(1, 2).reshape(B, D, H, W).contiguous()


class DinoV3Downsample2x(nn.Module):
    """Bilinear resize by 0.5 on (B, C, H, W)."""

    def forward(self, x: Tensor) -> Tensor:
        """Downsample spatial dims by 2 via bilinear interpolation."""
        return F.interpolate(x, scale_factor=0.5, mode="bilinear", align_corners=False)
