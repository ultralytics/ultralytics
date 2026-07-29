# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

"""Vision Transformer (ViT) and UltraViT building blocks.

Provides UltraViT student blocks for encoder distillation and self-contained ViT primitives so a transformer backbone
can be expressed row-by-row in YAML. Everything a flat ViT backbone needs (LayerScale, MLP, SwiGLU, self-attention with
2D rotary position embedding and zero-key-bias attention) lives in this file:

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
import os
from contextlib import nullcontext
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F

from ultralytics.nn.modules.conv import Conv, RepConv
from ultralytics.utils import deprecation_warn
from ultralytics.utils.ops import make_divisible
from ultralytics.utils.torch_utils import TORCH_2_3, fuse_conv_and_bn

if TORCH_2_3:
    from torch.nn.attention import SDPBackend, sdpa_kernel

    # On this backbone's fp16 activations cuDNN's fused kernel pairs a finite forward with a NaN query gradient
    # (dK and dV stay finite), so GradScaler skips every AMP step and training stalls at step 0. Flash,
    # mem-efficient and math all agree with fp32 on the same inputs, so restrict dispatch to those three.
    _sdpa_backends = partial(sdpa_kernel, [SDPBackend.FLASH_ATTENTION, SDPBackend.EFFICIENT_ATTENTION, SDPBackend.MATH])
else:  # torch<2.3 predates the backend selector and never dispatches attention to cuDNN
    _sdpa_backends = nullcontext

__all__ = (
    "AnchorPoolQueryMix",
    "ConvSyncBN",
    "FastViTBlock",
    "FracRoPE2D",
    "LayerScale",
    "MLP",
    "MHSABlock",
    "PooledMHSABlock",
    "RepUltraViTBlock",
    "SelfAttention",
    "SwiGLU",
    "UltraViTBlock",
    "VITBlock",
    "VITDownsample2x",
    "VITPatchStem",
    "VITTokenToSpatial",
    "WindowMHSABlock",
)


# Length-aware SDPA temperature. A P5 token grid trained at ~224px (49 tokens for the /16 stem) but run at 640px
# (400 tokens) diffuses the fixed 1/sqrt(d) softmax over 8x more keys. Scaling the logits by sqrt(log N / log N_ref)
# restores the training-time peakiness. Env-gated so the CE/distill graph is byte-identical unless opted in; at
# N == N_ref the factor is exactly 1 (no-op at training resolution). q.shape[-2] is a concrete int under static-shape
# export, so this folds into SDPA's scalar scale (one constant Mul, no branch) and passes RKNN/Paddle/CoreML.
_LOGN_ATTN = os.getenv("ULTRAVIT_LOGN_ATTN", "0") == "1"
_INV_LOG_REF = 1.0 / math.log(49)  # 49 = /16-stem P5 tokens at the ~224px training grid


class UltraViTBlock(nn.Module):
    """UltraViT stages 1-3 block: RepMixer (inference form) + ConvFFN. Dim-preserving 4D in/out.

    Paper: arXiv:2303.14189 §3 (FastViT, Vasu et al. 2023). Reparameterized inference form collapses the train-time
    RepMixer to `x + DWConv3x3+BN(x)`. ConvFFN inverted-bottleneck: PW → GELU → DW3x3+BN → PW (or, with
    `fastvit_ffn=True`, the paper-exact DW7x7+BN → PW → GELU → PW, at input width instead of hidden width). No LayerNorm
    in stages 1-3 (FastViT paper §3.2 uses BN here for speed).

    Attributes:
        mixer_dw (nn.Conv2d): Depthwise 3x3 mixing conv.
        mixer_bn (nn.BatchNorm2d): BN after mixer.
        fastvit_ffn (bool): FFN order/kernel switch, see `_ffn`.
        ffn_pw1 (nn.Conv2d): 1x1 PW conv to hidden dim.
        ffn_dw (nn.Conv2d): DW mixing conv, 3x3 at hidden dim (default) or 7x7 at c (fastvit_ffn).
        ffn_bn (nn.BatchNorm2d): BN after ffn_dw.
        ffn_pw2 (nn.Conv2d): 1x1 PW conv back to c.
        act (nn.Module): FFN activation, GELU or SiLU.
        ls1 (nn.Parameter): Optional LayerScale on the mixer residual (timm FastViT trains 1e-5 on every residual).
            Created only when `ls > 0`; `forward` guards on it so pre-LayerScale checkpoints still load and run.
        ls2 (nn.Parameter): Optional LayerScale on the FFN residual.
    """

    def __init__(
        self, c: int, mlp_ratio: float = 3.0, silu: bool = False, ls: float = 0.0, fastvit_ffn: bool = False
    ):
        """Initialize UltraViTBlock with dim c, FFN expansion ratio, activation choice, LayerScale init, and FFN order.
        """
        super().__init__()
        self.mixer_dw = nn.Conv2d(c, c, 3, padding=1, groups=c, bias=False)
        self.mixer_bn = nn.BatchNorm2d(c)
        hidden = int(c * mlp_ratio)
        self.fastvit_ffn = fastvit_ffn
        if fastvit_ffn:  # paper-exact ConvFFN: DW7x7+BN at c width, before the PW expansion
            self.ffn_dw = nn.Conv2d(c, c, 7, padding=3, groups=c, bias=False)
            self.ffn_bn = nn.BatchNorm2d(c)
        else:  # current default: DW3x3+BN at hidden width, after the PW expansion
            self.ffn_dw = nn.Conv2d(hidden, hidden, 3, padding=1, groups=hidden, bias=False)
            self.ffn_bn = nn.BatchNorm2d(hidden)
        self.ffn_pw1 = nn.Conv2d(c, hidden, 1, bias=False)
        self.ffn_pw2 = nn.Conv2d(hidden, c, 1, bias=False)
        # SiLU fuses into conv epilogues under TensorRT; GELU lowers to standalone fp32 erf+cast kernels
        # (measured 9-26% of engine time). GELU stays the default so pre-silu checkpoints keep their activation.
        self.act = nn.SiLU() if silu else nn.GELU()
        if ls:
            self.ls1 = nn.Parameter(ls * torch.ones(c, 1, 1))
            self.ls2 = nn.Parameter(ls * torch.ones(c, 1, 1))

    def _ffn(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the ConvFFN; `fastvit_ffn` selects pre-expansion DW7x7 (paper-exact) vs post-expansion DW3x3."""
        if getattr(self, "fastvit_ffn", False):  # getattr: pre-fastvit_ffn checkpoints still load and run
            h = self.ffn_dw(x)
            h = self.ffn_bn(h) if hasattr(self, "ffn_bn") else h
            return self.ffn_pw2(self.act(self.ffn_pw1(h)))
        h = self.act(self.ffn_pw1(x))
        h = self.ffn_dw(h)
        h = self.ffn_bn(h) if hasattr(self, "ffn_bn") else h
        return self.ffn_pw2(self.act(h))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass: residual mixer + residual FFN, each optionally LayerScale-gated."""
        m = self.mixer_bn(self.mixer_dw(x))
        ls1 = getattr(self, "ls1", None)
        x = x + (m if ls1 is None else ls1 * m)
        f = self._ffn(x)
        ls2 = getattr(self, "ls2", None)
        return x + (f if ls2 is None else ls2 * f)


class RepUltraViTBlock(UltraViTBlock):
    """Use a FastViT-style reparameterized token mixer with the UltraViT ConvFFN."""

    def __init__(self, c: int, mlp_ratio: float = 3.0, silu: bool = False, ls: float = 0.0, fastvit_ffn: bool = False):
        """Initialize the train-time multi-branch mixer and ConvFFN."""
        super().__init__(c, mlp_ratio, silu, ls, fastvit_ffn)
        del self.mixer_dw, self.mixer_bn
        self.mixer = RepConv(c, c, g=c, act=False, bn=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the reparameterized mixer followed by the ConvFFN."""
        if isinstance(self.mixer, nn.Conv2d):
            x = self.mixer(x)
        else:
            m = self.mixer(x) - x
            ls1 = getattr(self, "ls1", None)
            x = x + (m if ls1 is None else ls1 * m)
        f = self._ffn(x)
        ls2 = getattr(self, "ls2", None)
        return x + (f if ls2 is None else ls2 * f)

    @torch.no_grad()
    def fuse(self):
        """Fold the token-mixer branches, residual, and ConvFFN normalization for deploy."""
        if isinstance(self.mixer, nn.Conv2d):
            return
        self.mixer.fuse_convs()
        c = self.mixer.conv
        scale = getattr(self, "ls1", None)
        scale = torch.ones(c.out_channels, 1, 1, device=c.weight.device, dtype=c.weight.dtype) if scale is None else scale
        identity = torch.zeros_like(c.weight)
        identity[:, :, 1, 1] = 1
        weight = c.weight * scale[:, None] + identity * (1 - scale[:, None])
        mixer = nn.Conv2d(c.in_channels, c.out_channels, 3, padding=1, groups=c.groups, bias=True).to(c.weight)
        mixer.weight.copy_(weight)
        mixer.bias.copy_(c.bias * scale.flatten())
        self.mixer = mixer
        self.ffn_dw = fuse_conv_and_bn(self.ffn_dw, self.ffn_bn)
        del self.ffn_bn


class MHSABlock(nn.Module):
    """Pre-norm ViT block with SDPA and a token-Linear, SwiGLU, or NCHW ConvMlp FFN. Dim-preserving 4D in/out.

    Uses explicit QKV Linear + `F.scaled_dot_product_attention` instead of `nn.MultiheadAttention` (the AIFI bloat
    source — MHA-based AIFI ViT wraps to ~1327 ONNX nodes @ opset 17). SDPA decomposes in opset 17 to
    `MatMul+Softmax+MatMul+Mul(scale)`; the win is skipping PyTorch's MHA wrapper, not graph fusion.

    Used for UltraViT (and legacy FastViT) stage 4 global attention at the coarsest scale. `qkv_bias`, `proj_bias`,
    `swiglu`, and `n_storage_tokens` are DINOv3-style additions, Lane B only: keep off any Lane A yaml until each is
    verified against the broad edge-export set.

    Attributes:
        num_heads (int): Number of attention heads. c must be divisible by num_heads. YAMLs that pin `head_dim` pass 0
            here so no dead head count sits in the config; the value is then derived, never read.
        head_dim (int): Per-head dim. When the `head_dim` arg is nonzero it pins this value and derives num_heads = c //
            head_dim (Apple FastViT policy, head_dim 32), so head width no longer shrinks with model scale.
        temperature (nn.Parameter): Per-head scalar for cross-covariance attention (XCiT), created only when `xca=True`.
            Its presence switches `forward` to channel attention (map is head_dim x head_dim, invariant to token count),
            so a frozen backbone meets no length-coupled softmax when transferred to a larger detection grid.
        storage_tokens (nn.Parameter): Learnable (1, n, C) tokens providing extra K/V context for SDPA (DINOv3
            registers), created only when `n_storage_tokens > 0`. Their own query rows are dropped before SDPA (SDPA
            output for a query row depends only on that row, so the discarded rows are never computed) on the plain
            path, or from the attention output on the XCA path (there they still contribute to the covariance stats).
            Scoped to this block only, not carried to the next stage, since each `MHSABlock` repeat is a standalone
            4D-in/4D-out module rather than one shared token sequence spanning depth like DINOv3's.
        ln1 (nn.LayerNorm): Pre-attention norm.
        qkv (nn.Linear): Fused QKV projection, bias per `qkv_bias`.
        proj (nn.Linear): Post-attention projection, bias per `proj_bias`.
        pe (nn.Conv2d): Optional zero-initialized depthwise 7x7 conditional positional encoding.
        ln2 (nn.LayerNorm): Pre-FFN norm (token-Linear/SwiGLU FFN only).
        swiglu (bool): FFN form switch (token-Linear vs SwiGLU), set only on the token-FFN path (`conv_ffn=False`).
        fc1 (nn.Linear): FFN first layer (token FFN), or the fused value+gate projection when `swiglu=True` (one Linear
            to `2 * swiglu_hidden`, split via `chunk`, matching this block's own `qkv` fusion).
        fc2 (nn.Linear): FFN second layer (token-Linear or SwiGLU FFN only).
        ffn_dw (nn.Conv2d): 7x7 DW local-mixing conv opening the ConvMlp FFN (timm FastViT AttentionBlock form), created
            only when `conv_ffn=True`; `forward` guards on it for pre-ConvMlp checkpoints.
        ffn_bn (nn.BatchNorm2d): ConvMlp norm after the DW conv.
        ffn_pw1 (nn.Conv2d): ConvMlp 1x1 conv to hidden dim.
        ffn_pw2 (nn.Conv2d): ConvMlp 1x1 conv back to c.
        act (nn.Module): FFN activation, GELU or SiLU (see UltraViTBlock on the TensorRT fusion difference).
        ls1 (nn.Parameter): Optional LayerScale on the attention residual (does not fold away at inference).
        ls2 (nn.Parameter): Optional LayerScale on the FFN residual, shaped (C, 1, 1) for the ConvMlp path and (C,) for
            the token path.
    """

    def __init__(
        self,
        c: int,
        num_heads: int = 6,
        mlp_ratio: float = 4.0,
        silu: bool = False,
        ls: float = 0.0,
        conv_ffn: bool = False,
        head_dim: int = 0,
        xca: bool = False,
        qkv_bias: bool = False,
        proj_bias: bool = False,
        swiglu: bool = False,
        n_storage_tokens: int = 0,
        cpe: bool = False,
    ):
        """Initialize MHSABlock."""
        super().__init__()
        if head_dim:
            assert c % head_dim == 0, f"MHSABlock: c={c} not divisible by head_dim={head_dim}"
            num_heads = c // head_dim
        assert c % num_heads == 0, f"MHSABlock: c={c} not divisible by num_heads={num_heads}"
        assert not (swiglu and conv_ffn), "MHSABlock: swiglu is a token-FFN, mutually exclusive with conv_ffn"
        self.num_heads = num_heads
        self.head_dim = c // num_heads
        if cpe:
            self.pe = nn.Conv2d(c, c, 7, padding=3, groups=c, bias=True)
            nn.init.zeros_(self.pe.weight)
            nn.init.zeros_(self.pe.bias)
        if xca:  # cross-covariance attention: map is head_dim x head_dim (token-count invariant), learnable per-head temperature (XCiT)
            self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))
        self.n_storage_tokens = n_storage_tokens
        if n_storage_tokens:  # DINOv3-style registers: extra learnable K/V/Q slots for attention to route through
            self.storage_tokens = nn.Parameter(torch.zeros(1, n_storage_tokens, c))
            nn.init.normal_(self.storage_tokens, std=0.02)
        self.ln1 = nn.LayerNorm(c)
        self.qkv = nn.Linear(c, 3 * c, bias=qkv_bias)
        self.proj = nn.Linear(c, c, bias=proj_bias)
        hidden = int(c * mlp_ratio)
        self.act = nn.SiLU() if silu else nn.GELU()
        if conv_ffn:
            self.ffn_dw = nn.Conv2d(c, c, 7, padding=3, groups=c, bias=False)
            self.ffn_bn = nn.BatchNorm2d(c)
            self.ffn_pw1 = nn.Conv2d(c, hidden, 1)
            self.ffn_pw2 = nn.Conv2d(hidden, c, 1)
        else:
            self.ln2 = nn.LayerNorm(c)
            self.swiglu = swiglu
            if swiglu:
                # PaLM-style gated FFN, one fused Linear + chunk (matches this block's own qkv fusion and
                # nn/modules/block.py's SwiGLUFFN); hidden trimmed by 2/3 and rounded up to a multiple of 8 so the
                # extra (3rd) projection stays param-matched to the 2-linear MLP it replaces (EUPE/DINOv3 convention).
                swiglu_hidden = make_divisible(int(hidden * 2 / 3), 8)
                self.fc1 = nn.Linear(c, 2 * swiglu_hidden)
                self.fc2 = nn.Linear(swiglu_hidden, c)
            else:
                self.fc1 = nn.Linear(c, hidden)
                self.fc2 = nn.Linear(hidden, c)
        if ls:
            self.ls1 = nn.Parameter(ls * torch.ones(c))
            # ls2 shape follows the FFN form chosen at construction: (C, 1, 1) broadcasts in NCHW for ConvMlp,
            # (C,) for tokens. Avoids a per-forward view that would add a Reshape node to traced graphs.
            self.ls2 = nn.Parameter(ls * torch.ones(c, 1, 1) if conv_ffn else ls * torch.ones(c))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass: 4D → tokens (+ storage) → SA → drop storage → FFN (token Linear/SwiGLU or ConvMlp) → 4D."""
        b, c, h, w = x.shape
        pe = getattr(self, "pe", None)
        if pe is not None:
            x = x + pe(x)
        t = x.flatten(2).transpose(1, 2)  # (B, N, C)
        xca = getattr(self, "temperature", None) is not None
        n_storage_tokens = getattr(self, "n_storage_tokens", 0)  # getattr: pre-registers checkpoints still load
        t_kv = torch.cat([self.storage_tokens.expand(b, -1, -1), t], dim=1) if n_storage_tokens else t
        n = self.ln1(t_kv)
        qkv = self.qkv(n).reshape(b, -1, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # indexed split traces to aten::select, x2paddle maps it but not aten::unbind
        if n_storage_tokens and not xca:  # storage tokens are K/V-only; SDPA output for their query rows is
            q = q[:, :, n_storage_tokens:]  # unused, so drop them from q before SDPA instead of after
        if xca:  # XCA: attention over channels, invariant to token count (needs every token's contribution to the
            qn = F.normalize(q.transpose(-2, -1), dim=-1)  # (B, heads, head_dim, N), L2-normed over tokens
            kn = F.normalize(k.transpose(-2, -1), dim=-1)  # covariance stats, so q keeps any storage-token rows)
            attn = (qn @ kn.transpose(-2, -1)) * self.temperature  # (B, heads, head_dim, head_dim)
            a = (attn.softmax(dim=-1) @ v.transpose(-2, -1)).permute(0, 3, 1, 2).reshape(b, -1, c)
            if n_storage_tokens:  # output N follows v (storage+patch); drop storage rows here instead
                a = a[:, n_storage_tokens:]
        else:
            if _LOGN_ATTN:  # length-aware temperature: sharpen softmax as the grid grows past the 49-token train grid
                scale = self.head_dim**-0.5 * (math.log(max(int(q.shape[-2]), 2)) * _INV_LOG_REF) ** 0.5
                a = F.scaled_dot_product_attention(q, k, v, scale=scale)
            else:
                a = F.scaled_dot_product_attention(q, k, v)
            a = a.transpose(1, 2).reshape(b, -1, c)
        a = self.proj(a)
        ls1 = getattr(self, "ls1", None)
        t = t + (a if ls1 is None else ls1 * a)
        ls2 = getattr(self, "ls2", None)
        if getattr(self, "ffn_dw", None) is not None:
            x = t.transpose(1, 2).reshape(b, c, h, w)
            f = self.ffn_pw2(self.act(self.ffn_pw1(self.ffn_bn(self.ffn_dw(x)))))
            return x + (f if ls2 is None else ls2 * f)
        n2 = self.ln2(t)
        if getattr(self, "swiglu", False):  # getattr: pre-swiglu checkpoints still load and run
            x1, x2 = self.fc1(n2).chunk(2, dim=-1)
            f = self.fc2(F.silu(x1) * x2)  # functional silu: export/fuse can't flip it inplace on the chunk view (block.py SwiGLUFFN)
        else:
            f = self.fc2(self.act(self.fc1(n2)))
        t = t + (f if ls2 is None else ls2 * f)
        return t.transpose(1, 2).reshape(b, c, h, w)


class WindowMHSABlock(MHSABlock):
    """Run MHSABlock independently in a square grid of spatial windows."""

    def __init__(
        self,
        c: int,
        num_heads: int = 6,
        mlp_ratio: float = 4.0,
        silu: bool = False,
        ls: float = 0.0,
        conv_ffn: bool = False,
        head_dim: int = 0,
        num_windows: int = 2,
        xca: bool = False,
        qkv_bias: bool = False,
        proj_bias: bool = False,
        swiglu: bool = False,
        n_storage_tokens: int = 0,
        cpe: bool = False,
    ):
        """Initialize windowed attention with a fixed grid count."""
        super().__init__(
            c,
            num_heads,
            mlp_ratio,
            silu,
            ls,
            conv_ffn,
            head_dim,
            xca,
            qkv_bias,
            proj_bias,
            swiglu,
            n_storage_tokens,
            cpe,
        )
        self.num_windows = num_windows

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply attention and ConvMlp inside each spatial window."""
        b, c, h, w = x.shape
        assert h % self.num_windows == 0 and w % self.num_windows == 0, (
            f"WindowMHSABlock: {h}x{w} is not divisible by {self.num_windows}"
        )
        hh, ww = h // self.num_windows, w // self.num_windows
        x = x.reshape(b, c, self.num_windows, hh, self.num_windows, ww)
        x = x.permute(0, 2, 4, 1, 3, 5).reshape(b * self.num_windows**2, c, hh, ww)
        x = super().forward(x)
        x = x.reshape(b, self.num_windows, self.num_windows, c, hh, ww)
        return x.permute(0, 3, 1, 4, 2, 5).reshape(b, c, h, w)


class PooledMHSABlock(nn.Module):
    """Update dense tokens with attention over a pooled spatial grid and a ConvMlp."""

    def __init__(
        self,
        c: int,
        num_heads: int = 0,
        mlp_ratio: float = 4.0,
        silu: bool = False,
        ls: float = 0.0,
        head_dim: int = 32,
        pool_stride: int = 2,
    ):
        """Initialize pooled key/value attention."""
        super().__init__()
        if head_dim:
            assert c % head_dim == 0, f"PooledMHSABlock: c={c} not divisible by head_dim={head_dim}"
            num_heads = c // head_dim
        assert c % num_heads == 0, f"PooledMHSABlock: c={c} not divisible by num_heads={num_heads}"
        self.num_heads = num_heads
        self.head_dim = c // num_heads
        self.pool_dw = nn.Conv2d(c, c, 3, stride=pool_stride, padding=1, groups=c, bias=False)
        self.pool_bn = nn.BatchNorm2d(c)
        self.lnq = nn.LayerNorm(c)
        self.lnkv = nn.LayerNorm(c)
        self.q = nn.Linear(c, c)
        self.kv = nn.Linear(c, 2 * c)
        self.proj = nn.Linear(c, c)
        hidden = int(c * mlp_ratio)
        self.ffn_dw = nn.Conv2d(c, c, 7, padding=3, groups=c, bias=False)
        self.ffn_bn = nn.BatchNorm2d(c)
        self.ffn_pw1 = nn.Conv2d(c, hidden, 1)
        self.ffn_pw2 = nn.Conv2d(hidden, c, 1)
        self.act = nn.SiLU() if silu else nn.GELU()
        if ls:
            self.ls1 = nn.Parameter(ls * torch.ones(c))
            self.ls2 = nn.Parameter(ls * torch.ones(c, 1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply dense-query attention over pooled keys and values."""
        b, c, h, w = x.shape
        t = x.flatten(2).transpose(1, 2)
        p = self.pool_bn(self.pool_dw(x)).flatten(2).transpose(1, 2)
        q = self.q(self.lnq(t)).reshape(b, -1, self.num_heads, self.head_dim).transpose(1, 2)
        kv = self.kv(self.lnkv(p)).reshape(b, -1, 2, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        a = F.scaled_dot_product_attention(q, kv[0], kv[1]).transpose(1, 2).reshape(b, -1, c)
        a = self.proj(a)
        ls1 = getattr(self, "ls1", None)
        t = t + (a if ls1 is None else ls1 * a)
        x = t.transpose(1, 2).reshape(b, c, h, w)
        f = self.ffn_pw2(self.act(self.ffn_pw1(self.ffn_bn(self.ffn_dw(x)))))
        ls2 = getattr(self, "ls2", None)
        return x + (f if ls2 is None else ls2 * f)


def _frac_rope_tbl(h: int, w: int, head_dim: int, base: float = 100.0):
    """Build a normalized-coordinate 2D rotary cos/sin table for an h x w token grid.

    Angles are keyed to fractional coordinates (x/W, y/H) in [0, 1] rather than integer indices, so the relative phase
    between two tokens is a function of their normalized separation and stays constant as the grid grows. Row (y)
    frequencies fill the first head_dim half, col (x) frequencies the second, matching the rotate-half convention `[-x2,
    x1]`. Returns dense (N, head_dim) cos and sin tensors so the traced graph carries one initializer per table (not
    per-frequency Constants), dodging the ECViT 554-Constant blowup.

    Args:
        h (int): Token grid height.
        w (int): Token grid width.
        head_dim (int): Per-head dim. Must be divisible by 4 (row/col halves, each rotary needs cos/sin pairs).
        base (float, optional): Rotary frequency base.

    Returns:
        cos (torch.Tensor): Cosine table of shape (N, head_dim) with N = h * w.
        sin (torch.Tensor): Sine table of shape (N, head_dim) with N = h * w.
    """
    assert head_dim % 4 == 0, f"FracRoPE2D: head_dim={head_dim} must be divisible by 4"
    n_freq = head_dim // 4  # distinct freqs per axis; each axis fills head_dim // 2 dims as (freqs, freqs)
    freqs = 1.0 / (base ** (torch.arange(n_freq, dtype=torch.float32) / n_freq))
    gy, gx = torch.meshgrid(torch.linspace(0, 1, h), torch.linspace(0, 1, w), indexing="ij")
    ay = gy.reshape(-1, 1) * freqs[None, :]  # (N, n_freq) row angles
    ax = gx.reshape(-1, 1) * freqs[None, :]  # (N, n_freq) col angles
    ang = torch.cat([ay, ay, ax, ax], dim=-1)  # (N, head_dim): row half then col half, doubled for rotate-half
    return torch.cos(ang), torch.sin(ang)


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Rotate the last dim by splitting in half and mapping (x1, x2) -> (-x2, x1) for RoPE."""
    d = x.shape[-1] // 2
    return torch.cat([-x[..., d:], x[..., :d]], dim=-1)


class FracRoPE2D(MHSABlock):
    """MHSABlock with fractional-coordinate 2D rotary position embedding on Q/K. Dim-preserving 4D in/out.

    Keys the RoPE rotation angles to NORMALIZED fractional coordinates (x/W, y/H) instead of integer indices, so the
    relative phase between two tokens depends on their normalized separation and is resolution-invariant by
    construction. The 224-trained attention geometry stays correct at 640 with no positional interpolation and no
    learned table, leaving only content magnitude for a short hi-res finetune to adapt. Row frequencies occupy one
    head_dim half, col frequencies the other; cos/sin are baked as one dense registered buffer each (not per-frequency
    Constants) so ONNX carries them as initializers.

    Mutually exclusive with the `xca` branch of MHSABlock (rotary phase applies to the token-attention path). The
    dense-buffer cos/sin are resolution-dependent: `forward` rebuilds them on the fly when the running grid differs from
    the cached grid (so a single build serves 224 and 640), and `switch_to_deploy(hw)` rebakes them at the export imgsz.
    The indexed qkv[0..2] split of MHSABlock is preserved for x2paddle.

    Attributes:
        rope_hw (tuple): Cached (h, w) the current cos/sin buffers were built for.
        rope_cos (torch.Tensor): Dense (N, head_dim) cosine table registered as a buffer.
        rope_sin (torch.Tensor): Dense (N, head_dim) sine table registered as a buffer.
        rope_base (float): Rotary frequency base used to build the tables.
    """

    def __init__(
        self,
        c: int,
        num_heads: int = 6,
        mlp_ratio: float = 4.0,
        silu: bool = False,
        ls: float = 0.0,
        conv_ffn: bool = False,
        head_dim: int = 0,
        hw: int = 7,
        rope_base: float = 100.0,
    ):
        """Initialize FracRoPE2D, building the cos/sin tables for an hw x hw build grid; xca is forced off."""
        super().__init__(c, num_heads, mlp_ratio, silu, ls, conv_ffn, head_dim, xca=False)
        assert self.head_dim % 4 == 0, f"FracRoPE2D: head_dim={self.head_dim} must be divisible by 4"
        self.rope_base = rope_base
        self.rope_hw = (hw, hw)
        cos, sin = _frac_rope_tbl(hw, hw, self.head_dim, rope_base)
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)

    def _rope(self, h: int, w: int):
        """Return cos/sin tables for the h x w grid, rebuilding the buffers if the grid changed."""
        if self.rope_hw != (h, w):
            cos, sin = _frac_rope_tbl(h, w, self.head_dim, self.rope_base)
            self.rope_cos = cos.to(self.rope_cos.device, self.rope_cos.dtype)
            self.rope_sin = sin.to(self.rope_sin.device, self.rope_sin.dtype)
            self.rope_hw = (h, w)
        return self.rope_cos, self.rope_sin

    def switch_to_deploy(self, hw: int):
        """Rebake the cos/sin buffers for an hw x hw export grid so ONNX/TRT carry them at the export imgsz."""
        self._rope(hw, hw)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass: 4D -> tokens -> RoPE(q,k) -> SDPA -> FFN (token Linear or NCHW ConvMlp) -> 4D."""
        b, c, h, w = x.shape
        t = x.flatten(2).transpose(1, 2)  # (B, N, C)
        n = self.ln1(t)
        qkv = self.qkv(n).reshape(b, -1, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # indexed split traces to aten::select, x2paddle maps it but not aten::unbind
        cos, sin = self._rope(h, w)  # (N, head_dim), broadcasts over (B, heads, N, head_dim)
        q = q * cos + _rotate_half(q) * sin
        k = k * cos + _rotate_half(k) * sin
        if _LOGN_ATTN:  # length-aware temperature: sharpen softmax as the grid grows past the 49-token train grid
            scale = self.head_dim**-0.5 * (math.log(max(int(q.shape[-2]), 2)) * _INV_LOG_REF) ** 0.5
            a = F.scaled_dot_product_attention(q, k, v, scale=scale)
        else:
            a = F.scaled_dot_product_attention(q, k, v)
        a = a.transpose(1, 2).reshape(b, -1, c)
        a = self.proj(a)
        ls1 = getattr(self, "ls1", None)
        t = t + (a if ls1 is None else ls1 * a)
        ls2 = getattr(self, "ls2", None)
        if getattr(self, "ffn_dw", None) is not None:
            x = t.transpose(1, 2).reshape(b, c, h, w)
            f = self.ffn_pw2(self.act(self.ffn_pw1(self.ffn_bn(self.ffn_dw(x)))))
            return x + (f if ls2 is None else ls2 * f)
        f = self.fc2(self.act(self.fc1(self.ln2(t))))
        t = t + (f if ls2 is None else ls2 * f)
        return t.transpose(1, 2).reshape(b, c, h, w)


class AnchorPoolQueryMix(nn.Module):
    """Anchor-pooled two-stage attention block with dense 4D output."""

    def __init__(
        self,
        c: int,
        m: int = 49,
        mlp_ratio: float = 4.0,
        silu: bool = False,
        ls: float = 0.0,
        head_dim: int = 32,
        pool_stride: int = 3,
    ):
        """Initialize the anchor-pooled attention block."""
        super().__init__()
        assert c % head_dim == 0, f"AnchorPoolQueryMix: c={c} not divisible by head_dim={head_dim}"
        self.num_heads = c // head_dim
        self.head_dim = head_dim
        self.dim = c
        g = round(m**0.5)
        self.m = g * g
        self.anchors = nn.Parameter(self._coord_seed(g, c))
        self.pool_dw = nn.Conv2d(c, c, 3, stride=pool_stride, padding=1, groups=c, bias=False)
        self.pool_bn = nn.BatchNorm2d(c)
        self.lnA = nn.LayerNorm(c)
        self.lnP = nn.LayerNorm(c)
        self.lnB = nn.LayerNorm(c)
        self.qA = nn.Linear(c, c, bias=False)
        self.kvA = nn.Linear(c, 2 * c, bias=False)
        self.qB = nn.Linear(c, c, bias=False)
        self.kvB = nn.Linear(c, 2 * c, bias=False)
        self.projA = nn.Linear(c, c, bias=False)
        self.projB = nn.Linear(c, c, bias=False)
        hidden = int(c * mlp_ratio)
        self.ffn_dw = nn.Conv2d(c, c, 7, padding=3, groups=c, bias=False)
        self.ffn_bn = nn.BatchNorm2d(c)
        self.ffn_pw1 = nn.Conv2d(c, hidden, 1)
        self.ffn_pw2 = nn.Conv2d(hidden, c, 1)
        self.act = nn.SiLU() if silu else nn.GELU()
        if ls:
            self.ls1 = nn.Parameter(ls * torch.ones(c))
            self.ls2 = nn.Parameter(ls * torch.ones(c, 1, 1))

    @staticmethod
    def _coord_seed(g: int, c: int) -> torch.Tensor:
        """Build a 2D sinusoidal seed for the anchor slots."""
        d = c // 2
        div = torch.exp(torch.arange(0, d, 2) * (-math.log(10000.0) / max(d, 1)))
        ys, xs = torch.meshgrid(torch.arange(g), torch.arange(g), indexing="ij")
        pos = torch.zeros(g * g, c)
        for coord, off in ((ys.flatten(), 0), (xs.flatten(), d)):
            a = coord.unsqueeze(1).float() * div
            pos[:, off : off + a.shape[1] * 2 : 2] = torch.sin(a)
            pos[:, off + 1 : off + a.shape[1] * 2 : 2] = torch.cos(a)
        return pos.unsqueeze(0) * 0.02

    def _sdpa(self, q: torch.Tensor, kv: torch.Tensor, b: int) -> torch.Tensor:
        """Apply multi-head attention from query tokens to fused K/V tokens."""
        nq = q.shape[1]
        q = q.reshape(b, nq, self.num_heads, self.head_dim).transpose(1, 2)
        kv = kv.reshape(b, -1, 2, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]
        a = F.scaled_dot_product_attention(q, k, v)
        return a.transpose(1, 2).reshape(b, nq, self.dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run anchor gather, dense scatter, and ConvMlp FFN."""
        b, c, h, w = x.shape
        p = self.pool_bn(self.pool_dw(x)).flatten(2).transpose(1, 2)
        t = x.flatten(2).transpose(1, 2)
        anchors = self.lnA(self.anchors).expand(b, -1, -1)
        a = self.projA(self._sdpa(self.qA(anchors), self.kvA(self.lnP(p)), b))
        d = self.projB(self._sdpa(self.qB(self.lnB(t)), self.kvB(a), b))
        ls1 = getattr(self, "ls1", None)
        t = t + (d if ls1 is None else ls1 * d)
        x = t.transpose(1, 2).reshape(b, c, h, w)
        f = self.ffn_pw2(self.act(self.ffn_pw1(self.ffn_bn(self.ffn_dw(x)))))
        ls2 = getattr(self, "ls2", None)
        return x + (f if ls2 is None else ls2 * f)

    def forward_fuse(self, x: torch.Tensor) -> torch.Tensor:
        """Run the block after BN folding and anchor query caching."""
        b, c, h, w = x.shape
        p = self.pool_dw(x).flatten(2).transpose(1, 2)
        t = x.flatten(2).transpose(1, 2)
        a = self.projA(self._sdpa(self.anchors_flat.expand(b, -1, -1), self.kvA(self.lnP(p)), b))
        d = self.projB(self._sdpa(self.qB(self.lnB(t)), self.kvB(a), b))
        ls1 = getattr(self, "ls1", None)
        t = t + (d if ls1 is None else ls1 * d)
        x = t.transpose(1, 2).reshape(b, c, h, w)
        f = self.ffn_pw2(self.act(self.ffn_pw1(self.ffn_dw(x))))
        ls2 = getattr(self, "ls2", None)
        return x + (f if ls2 is None else ls2 * f)

    @torch.no_grad()
    def fuse(self):
        """Fold BN layers and cache the anchor query projection."""
        if hasattr(self, "anchors_flat"):
            return
        self.pool_dw = fuse_conv_and_bn(self.pool_dw, self.pool_bn)
        self.ffn_dw = fuse_conv_and_bn(self.ffn_dw, self.ffn_bn)
        self.register_buffer("anchors_flat", self.qA(self.lnA(self.anchors)))
        delattr(self, "pool_bn")
        delattr(self, "ffn_bn")


class FastViTBlock(UltraViTBlock):
    """Deprecated alias of UltraViTBlock, kept so legacy fastvit YAMLs and pickled checkpoints keep loading."""

    _warned = False  # warn once per session, not once per constructed block

    def __init__(self, *args, **kwargs):
        """Initialize UltraViTBlock under its deprecated name with a rename warning."""
        if not FastViTBlock._warned:
            FastViTBlock._warned = True
            deprecation_warn("FastViTBlock", "UltraViTBlock")
        super().__init__(*args, **kwargs)


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
        if torch.onnx.is_in_onnx_export() and q.dtype == torch.float16:
            softmax_scale = 2.0
            logits = torch.matmul(q * (q.shape[-1] ** -0.5 / softmax_scale), k.transpose(-2, -1))
            logits = ((logits - logits.amax(dim=-1, keepdim=True)) * softmax_scale).clamp(min=-20)
            attn = logits.exp()
            y = torch.matmul(attn / attn.sum(dim=-1, keepdim=True), v)
        else:
            with _sdpa_backends():
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
