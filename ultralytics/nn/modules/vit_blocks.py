# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

"""Vision Transformer (ViT) building blocks for flat YAML expression.

Provides self-contained ViT primitives so a transformer backbone can be expressed row-by-row in YAML. Everything a
flat ViT backbone needs (LayerScale, MLP, SwiGLU, self-attention with 2D rotary position embedding and zero-key-bias
attention) lives in this file, so the module set has no cross-package dependencies:

    LayerScale         Learnable per-channel gain applied to a residual branch.
    MLP, SwiGLU        Standard and SwiGLU-gated feed-forward blocks.
    SelfAttention      Multi-head self-attention with 2D RoPE and optional zero-key bias.
    VITPatchStem       Conv2d patch embed that prepends a cls token and n_registers extra tokens.
    VITBlock           Pre-norm SA + LayerScale + FFN on (B, N, dim) tokens with per-forward RoPE.
    VITTokenToSpatial  LayerNorm, strip cls/register tokens, reshape to (B, dim, H, W).
    VITDownsample2x    Bilinear resize by 0.5 on (B, C, H, W).
    ConvSyncBN         Conv wrapper whose BN is nn.SyncBatchNorm and survives initialize_weights.

ViT blocks are dim-preserving (c_out == c_in == dim); the YAML parser injects dim from ch[f].
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from ultralytics.nn.modules.conv import Conv

__all__ = (
    "MLP",
    "ConvSyncBN",
    "LayerScale",
    "SelfAttention",
    "SwiGLU",
    "VITBlock",
    "VITDownsample2x",
    "VITPatchStem",
    "VITTokenToSpatial",
)


def _rotate_half(x):
    """Return [-x[..., d/2:], x[..., :d/2]] concatenated along the last dim."""
    a, b = x.chunk(2, dim=-1)
    return torch.cat((-b, a), dim=-1)


def _apply_rotary(x, sin, cos):
    """Apply rotary position embedding to x with precomputed sin and cos tables."""
    return x * cos + _rotate_half(x) * sin


class _QKVZeroKeyBias(nn.Linear):
    """nn.Linear for a fused Q|K|V projection whose K portion of the bias is forced to zero.

    Modern ViT variants learn Q and V biases but keep the K bias at zero. Applying a mask to the full-length bias
    (rather than deleting the K entries) keeps state_dict shape identical to a plain nn.Linear, so pretrained
    checkpoints round-trip without a remap.

    Attributes:
        bias_mask (torch.Tensor): Buffer of shape (3*C,) with zeros over the K slice and ones over Q and V.
    """

    def __init__(self, *args, **kwargs):
        """Initialize QKV linear with a K-portion-zeroed bias mask.

        Args:
            *args: Positional arguments forwarded to nn.Linear.
            **kwargs: Keyword arguments forwarded to nn.Linear.
        """
        super().__init__(*args, **kwargs)
        assert self.out_features % 3 == 0, "out_features must be divisible by 3 for a fused Q|K|V split"
        if self.bias is not None:
            c = self.out_features // 3
            mask = torch.ones_like(self.bias)
            mask[c : 2 * c] = 0.0
            self.register_buffer("bias_mask", mask)

    def forward(self, x):
        """Apply the linear projection with the K portion of the bias forced to zero.

        Args:
            x (torch.Tensor): Input tensor of shape (..., in_features).

        Returns:
            (torch.Tensor): Projected tensor of shape (..., 3*C) with the K bias masked to zero.
        """
        bias = self.bias * self.bias_mask.to(self.bias.dtype) if self.bias is not None else None
        return F.linear(x, self.weight, bias)


class SelfAttention(nn.Module):
    """Multi-head self-attention with 2D RoPE and optional zero-key bias.

    Delegates the actual attention math to torch.nn.functional.scaled_dot_product_attention so it picks the fastest
    available kernel (FlashAttention, memory-efficient, or math) at runtime.

    Attributes:
        num_heads (int): Number of attention heads.
        qkv (nn.Linear): Fused Q|K|V projection (zero-key-bias variant when zero_key_bias=True).
        proj (nn.Linear): Output projection.
    """

    def __init__(self, dim, num_heads=8, qkv_bias=False, proj_bias=True, zero_key_bias=False):
        """Initialize SelfAttention layer.

        Args:
            dim (int): Embedding dimension.
            num_heads (int): Number of attention heads.
            qkv_bias (bool): Whether the fused Q|K|V projection has a bias.
            proj_bias (bool): Whether the output projection has a bias.
            zero_key_bias (bool): Whether to force the K portion of the QKV bias to zero.
        """
        super().__init__()
        assert dim % num_heads == 0, f"dim={dim} not divisible by num_heads={num_heads}"
        self.num_heads = num_heads
        qkv_cls = _QKVZeroKeyBias if zero_key_bias else nn.Linear
        self.qkv = qkv_cls(dim, 3 * dim, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim, bias=proj_bias)

    def forward(self, x, rope=None):
        """Compute scaled dot-product attention with optional RoPE applied to q and k.

        Args:
            x (torch.Tensor): Input tokens of shape (B, N, C).
            rope (tuple, optional): Precomputed (sin, cos) tables of shape (HW, C/num_heads).

        Returns:
            (torch.Tensor): Output tokens of shape (B, N, C).
        """
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads)
        q, k, v = (t.transpose(1, 2) for t in qkv.unbind(2))
        if rope is not None:
            q, k = _rope_qk(q, k, *rope)
        y = F.scaled_dot_product_attention(q, k, v)
        return self.proj(y.transpose(1, 2).reshape(B, N, C))


def _rope_qk(q, k, sin, cos):
    """Apply RoPE to q and k, leaving any leading prefix tokens (cls, register) unrotated.

    Args:
        q (torch.Tensor): Query tensor of shape (B, heads, N, head_dim).
        k (torch.Tensor): Key tensor of shape (B, heads, N, head_dim).
        sin (torch.Tensor): Sine table of shape (HW, head_dim).
        cos (torch.Tensor): Cosine table of shape (HW, head_dim).

    Returns:
        q (torch.Tensor): Rotated queries with the prefix untouched.
        k (torch.Tensor): Rotated keys with the prefix untouched.
    """
    prefix = q.shape[-2] - sin.shape[-2]
    assert prefix >= 0, "sin table cannot exceed sequence length"
    q_dtype, k_dtype = q.dtype, k.dtype
    q_pre, q_rot = q.to(sin.dtype).split((prefix, q.shape[-2] - prefix), dim=-2)
    k_pre, k_rot = k.to(sin.dtype).split((prefix, k.shape[-2] - prefix), dim=-2)
    q = torch.cat((q_pre, _apply_rotary(q_rot, sin, cos)), dim=-2)
    k = torch.cat((k_pre, _apply_rotary(k_rot, sin, cos)), dim=-2)
    return q.to(q_dtype), k.to(k_dtype)


class LayerScale(nn.Module):
    """Learnable per-channel gain applied to a residual branch.

    Introduced by CaiT to stabilize deep-block residuals in ViT variants: y = x * scale, where scale is a length-dim
    parameter vector initialized to a small value.

    Attributes:
        scale (nn.Parameter): Per-channel gain of shape (dim,).
    """

    def __init__(self, dim, init_value=1.0e-5):
        """Initialize LayerScale with scale pre-filled to init_value.

        Args:
            dim (int): Embedding dimension.
            init_value (float): Initial value for every entry of scale.
        """
        super().__init__()
        self.scale = nn.Parameter(torch.full((dim,), float(init_value)))

    def forward(self, x):
        """Multiply x by scale along the last dimension.

        Args:
            x (torch.Tensor): Input tensor of shape (..., dim).

        Returns:
            (torch.Tensor): Input scaled by the learnable gain.
        """
        return x * self.scale


class MLP(nn.Module):
    """Two-layer feed-forward block with GELU activation.

    Attributes:
        fc1 (nn.Linear): Input projection from dim to hidden_dim.
        act (nn.GELU): Activation applied between fc1 and fc2.
        fc2 (nn.Linear): Output projection from hidden_dim back to dim.
    """

    def __init__(self, dim, hidden_dim, bias=True):
        """Initialize MLP layer.

        Args:
            dim (int): Input and output feature count (dim-preserving block).
            hidden_dim (int): Hidden width between fc1 and fc2.
            bias (bool): Whether fc1 and fc2 include a bias term.
        """
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden_dim, bias=bias)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_dim, dim, bias=bias)

    def forward(self, x):
        """Apply fc1, GELU, fc2 to the input tensor.

        Args:
            x (torch.Tensor): Input tensor of shape (..., dim).

        Returns:
            (torch.Tensor): Output tensor of shape (..., dim).
        """
        return self.fc2(self.act(self.fc1(x)))


class SwiGLU(nn.Module):
    """SwiGLU-gated feed-forward block computing down(silu(gate(x)) * up(x)).

    The reference hidden width is rescaled to (2/3) * hidden_dim and rounded up to a multiple of align, so the total
    parameter count matches a plain 4x-expansion MLP.

    Attributes:
        gate (nn.Linear): Gating projection from dim to hidden.
        up (nn.Linear): Value projection from dim to hidden.
        down (nn.Linear): Output projection from hidden back to dim.
    """

    def __init__(self, dim, hidden_dim, bias=True, align=8):
        """Initialize SwiGLU layer.

        Args:
            dim (int): Input and output feature count (dim-preserving block).
            hidden_dim (int): Reference hidden width; rescaled to 2/3 and aligned up to a multiple of align.
            bias (bool): Whether each linear includes a bias term.
            align (int): Multiple to align the SwiGLU hidden width to.
        """
        super().__init__()
        h = int(hidden_dim * 2 / 3)
        h += (-h) % align
        self.gate = nn.Linear(dim, h, bias=bias)
        self.up = nn.Linear(dim, h, bias=bias)
        self.down = nn.Linear(h, dim, bias=bias)

    def forward(self, x):
        """Apply the SwiGLU gate then project through down.

        Args:
            x (torch.Tensor): Input tensor of shape (..., dim).

        Returns:
            (torch.Tensor): Output tensor of shape (..., dim).
        """
        return self.down(F.silu(self.gate(x)) * self.up(x))


def _build_ffn(name, dim, hidden_dim):
    """Instantiate the FFN block selected by the YAML string name.

    Args:
        name (str): FFN kind, one of 'mlp', 'swiglu', 'swiglu64'.
        dim (int): Input and output feature count.
        hidden_dim (int): Reference hidden width (SwiGLU rescales this by 2/3 internally).

    Returns:
        (nn.Module): Instantiated FFN block.
    """
    if name == "mlp":
        return MLP(dim, hidden_dim)
    if name == "swiglu":
        return SwiGLU(dim, hidden_dim)
    if name == "swiglu64":
        return SwiGLU(dim, hidden_dim, align=64)
    raise ValueError(f"unknown ffn_layer: {name!r} (expected 'mlp' | 'swiglu' | 'swiglu64')")


class VITPatchStem(nn.Module):
    """Patch embed that prepends a cls token and n_registers extra tokens.

    Emits a token tensor of shape (B, 1 + n_registers + HW, c2) where HW = (H // patch) * (W // patch). The parser
    treats this like a base_modules entry: the first YAML arg is c2 (embed_dim), c1 is injected from ch[f].

    Attributes:
        proj (nn.Conv2d): Patch projection Conv2d(c1, c2, k=patch, s=patch).
        cls_token (nn.Parameter): Learnable class token of shape (1, 1, c2).
        register_tokens (nn.Parameter): Learnable register tokens of shape (1, n_registers, c2), following the register
            token idea of Darcet et al. 2024.
        mask_token (nn.Parameter): Learnable mask token of shape (1, c2), retained for checkpoint compatibility with
            pretrained ViT weights that stored it.
        n_registers (int): Number of register tokens prepended after the cls token.
        patch_size (int): Square patch size of the Conv2d projection.
    """

    def __init__(self, c1, c2, patch_size=16, n_registers=4):
        """Initialize VITPatchStem layer.

        Args:
            c1 (int): Number of input image channels.
            c2 (int): Embedding dimension.
            patch_size (int): Square patch size for the Conv2d projection.
            n_registers (int): Number of register tokens prepended after the cls token.
        """
        super().__init__()
        self.n_registers = n_registers
        self.patch_size = patch_size
        self.proj = nn.Conv2d(c1, c2, kernel_size=patch_size, stride=patch_size)
        self.cls_token = nn.Parameter(torch.empty(1, 1, c2))
        self.register_tokens = nn.Parameter(torch.empty(1, n_registers, c2))
        self.mask_token = nn.Parameter(torch.empty(1, c2))
        self._init_weights()

    def _init_weights(self):
        """Uniform (fan-in aware) init for the projection and N(0, 0.02^2) init for prepended tokens."""
        bound = math.sqrt(1 / (self.proj.in_channels * self.patch_size**2))
        nn.init.uniform_(self.proj.weight, -bound, bound)
        if self.proj.bias is not None:
            nn.init.uniform_(self.proj.bias, -bound, bound)
        nn.init.normal_(self.cls_token, std=0.02)
        nn.init.normal_(self.register_tokens, std=0.02)
        nn.init.zeros_(self.mask_token)

    def forward(self, x):
        """Patch-embed the input image and prepend cls plus register tokens.

        Args:
            x (torch.Tensor): Input image tensor of shape (B, C, H, W).

        Returns:
            (torch.Tensor): Token tensor of shape (B, 1 + n_registers + HW, c2).
        """
        b = x.shape[0]
        patches = self.proj(x).flatten(2).transpose(1, 2)  # (B, HW, D)
        cls = self.cls_token + 0.0 * self.mask_token  # keeps mask_token reachable in autograd
        return torch.cat((cls.expand(b, -1, -1), self.register_tokens.expand(b, -1, -1), patches), dim=1)


class VITBlock(nn.Module):
    """Pre-norm SelfAttention + LayerScale + FFN block on (B, N, dim) tokens.

    Builds 2D RoPE tables inside forward from N under a square-grid assumption (H = W = int(sqrt(N - 1 - n_registers))).
    The leading 1 + n_registers tokens skip RoPE via the prefix mechanism inside _rope_qk.

    Attributes:
        n_registers (int): Register token count (must match the stem).
        head_dim (int): Per-head dimension, dim // num_heads.
        norm_attn (nn.LayerNorm): Pre-attention norm.
        self_attn (SelfAttention): Multi-head self-attention with zero-key bias.
        ls_attn (LayerScale): LayerScale on the attention residual.
        norm_ffn (nn.LayerNorm): Pre-FFN norm.
        ffn (nn.Module): FFN block (MLP or SwiGLU).
        ls_ffn (LayerScale): LayerScale on the FFN residual.
        rope_freqs (torch.Tensor): Buffer of RoPE inverse periods of shape (head_dim // 4,).
    """

    def __init__(self, dim, num_heads, ffn_ratio, ffn_layer="mlp", ls_init=1.0e-5, n_registers=4, rope_base=100.0):
        """Initialize VITBlock layer.

        Args:
            dim (int): Embedding dimension.
            num_heads (int): Number of attention heads.
            ffn_ratio (float): FFN expansion ratio; hidden width is int(dim * ffn_ratio).
            ffn_layer (str): FFN kind, one of 'mlp', 'swiglu', 'swiglu64'.
            ls_init (float): LayerScale initial value.
            n_registers (int): Register token count skipped by RoPE.
            rope_base (float): Base used for the RoPE frequency schedule.
        """
        super().__init__()
        assert dim % num_heads == 0, f"dim={dim} not divisible by num_heads={num_heads}"
        self.n_registers = n_registers
        self.head_dim = dim // num_heads

        self.norm_attn = nn.LayerNorm(dim, eps=1e-5)
        self.self_attn = SelfAttention(dim, num_heads=num_heads, qkv_bias=True, proj_bias=True, zero_key_bias=True)
        self.ls_attn = LayerScale(dim, init_value=ls_init)
        self.norm_ffn = nn.LayerNorm(dim, eps=1e-5)
        self.ffn = _build_ffn(ffn_layer, dim=dim, hidden_dim=int(dim * ffn_ratio))
        self.ls_ffn = LayerScale(dim, init_value=ls_init)

        # Persistent buffer so a remapper can bit-copy fp16-quantized rope periods from a source
        # checkpoint into every block; freshly computed fp32 periods drift by ~1e-3 vs the saved
        # values, and the drift compounds across the block stack.
        n_freqs = self.head_dim // 4
        freqs = rope_base ** (2 * torch.arange(n_freqs, dtype=torch.float32) / (self.head_dim // 2))
        self.register_buffer("rope_freqs", freqs, persistent=True)

    def _build_rope(self, hw, device, dtype):
        """Build (sin, cos) 2D RoPE tables for a square hw token grid.

        Args:
            hw (int): Number of spatial tokens (H * W).
            device (torch.device): Device on which to build the tables.
            dtype (torch.dtype): Output dtype for the tables.

        Returns:
            sin (torch.Tensor): Sine table of shape (HW, head_dim).
            cos (torch.Tensor): Cosine table of shape (HW, head_dim).
        """
        h = int(hw**0.5)
        w = hw // h
        yy = (torch.arange(0.5, h, dtype=torch.float32, device=device) / h) * 2.0 - 1.0
        xx = (torch.arange(0.5, w, dtype=torch.float32, device=device) / w) * 2.0 - 1.0
        coords = torch.stack(torch.meshgrid(yy, xx, indexing="ij"), dim=-1).flatten(0, 1)  # (HW, 2)
        angles = 2 * math.pi * coords[:, :, None] / self.rope_freqs[None, None, :]  # (HW, 2, head_dim/4)
        angles = angles.flatten(1, 2).tile(2)  # (HW, head_dim)
        return angles.sin().to(dtype), angles.cos().to(dtype)

    def forward(self, x):
        """Apply pre-norm SA + LS residual, then pre-norm FFN + LS residual.

        Args:
            x (torch.Tensor): Input tokens of shape (B, N, dim).

        Returns:
            (torch.Tensor): Output tokens of shape (B, N, dim).
        """
        hw = x.shape[1] - 1 - self.n_registers
        rope = self._build_rope(hw, x.device, x.dtype)
        x = x + self.ls_attn(self.self_attn(self.norm_attn(x), rope=rope))
        x = x + self.ls_ffn(self.ffn(self.norm_ffn(x)))
        return x


class VITTokenToSpatial(nn.Module):
    """Final LayerNorm, strip cls and register tokens, reshape (B, N, D) to (B, D, H, W).

    Assumes a square patch grid, deriving H = W = int(sqrt(N - 1 - n_registers)). Emits (B, D, H, W) suitable for
    downstream 4D convolutional pipelines.

    Attributes:
        n_registers (int): Register token count stripped alongside the cls token.
        norm (nn.LayerNorm): LayerNorm applied on tokens before spatial reshape.
    """

    def __init__(self, dim, n_registers=4):
        """Initialize VITTokenToSpatial layer.

        Args:
            dim (int): Embedding dimension (equal to the emitted spatial tensor channel count).
            n_registers (int): Register token count to strip.
        """
        super().__init__()
        self.n_registers = n_registers
        self.norm = nn.LayerNorm(dim, eps=1e-5)

    def forward(self, x):
        """Apply LayerNorm, drop cls and register tokens, reshape to a 4D spatial tensor.

        Args:
            x (torch.Tensor): Input tokens of shape (B, N, D).

        Returns:
            (torch.Tensor): Spatial tensor of shape (B, D, H, W).
        """
        tokens = self.norm(x)[:, 1 + self.n_registers :, :]  # (B, HW, D)
        b, hw, d = tokens.shape
        h = int(hw**0.5)
        w = hw // h
        return tokens.transpose(1, 2).reshape(b, d, h, w).contiguous()


class VITDownsample2x(nn.Module):
    """Bilinear resize by 0.5 on a (B, C, H, W) spatial tensor."""

    def forward(self, x):
        """Downsample spatial dims by 2 via bilinear interpolation.

        Args:
            x (torch.Tensor): Input tensor of shape (B, C, H, W).

        Returns:
            (torch.Tensor): Downsampled tensor of shape (B, C, H/2, W/2).
        """
        return F.interpolate(x, scale_factor=0.5, mode="bilinear", align_corners=False)


class ConvSyncBN(Conv):
    """Ultralytics Conv wrapper whose BN is nn.SyncBatchNorm instead of nn.BatchNorm2d.

    Preserves BN eps=1e-5 and momentum=0.1 by slipping past initialize_weights, which mutates only modules whose
    `type(m) is nn.BatchNorm2d`. SyncBatchNorm is a sibling class (not a subclass), so the strict-identity check skips
    it and the class defaults survive. State_dict keys match plain nn.BatchNorm2d, so existing checkpoints load without
    a remap.

    Attributes:
        conv (nn.Conv2d): Convolutional layer.
        bn (nn.SyncBatchNorm): Synchronized batch normalization layer.
        act (nn.Module): Activation function layer.
    """

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        """Initialize ConvSyncBN layer with given parameters.

        Args:
            c1 (int): Number of input channels.
            c2 (int): Number of output channels.
            k (int): Kernel size.
            s (int): Stride.
            p (int, optional): Padding.
            g (int): Groups.
            d (int): Dilation.
            act (bool | nn.Module): Activation function.
        """
        super().__init__(c1, c2, k, s, p, g, d, act)
        self.bn = nn.SyncBatchNorm(c2)
