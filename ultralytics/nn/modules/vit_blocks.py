# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

"""ViT-like student blocks for encoder distillation (UltraViT).

Simple-component constraint: Conv2d, BatchNorm2d, LayerNorm, GELU/SiLU, Linear, F.scaled_dot_product_attention.
No `nn.MultiheadAttention` (source of AIFI's 1327-node ONNX bloat). 2D RoPE only through `RoPE2DBlock`, which bakes
dense cos/sin tables into one initializer each rather than per-frequency Constants (ECViT-t hits 554 Constant nodes).

Registered in `ultralytics.nn.modules.__init__` and imported by `ultralytics.nn.tasks` so `parse_model` resolves
them through `globals()[m]`. All blocks are dim-preserving (C_in == C_out, H/W unchanged).

Must-build export paths pass across the UltraViT YAMLs: TorchScript, ONNX opset17, OpenVINO, CoreML, TFLite, TensorRT,
PaddlePaddle (x2paddle>=1.6.0, needs the indexed QKV split in MHSABlock), RKNN (rknn-toolkit2>=2.3.2). RKNN still
requires an isolated venv (its AutoUpdate downgrades torch 2.9→2.4 + cudnn 9.10→9.1, contaminating the primary env).
"""

from __future__ import annotations

import math
import os

import torch
import torch.nn.functional as F
from torch import nn

from ultralytics.utils import deprecation_warn
from ultralytics.utils.ops import make_divisible
from ultralytics.utils.torch_utils import fuse_conv_and_bn

from .conv import RepConv

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

    def __init__(self, c: int, mlp_ratio: float = 3.0, silu: bool = False, ls: float = 0.0, fastvit_ffn: bool = False):
        """Initialize UltraViTBlock with dim c, FFN expansion ratio, activation choice, LayerScale init, and FFN order."""
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
        """Fold the token-mixer branches, LayerScale, and ConvFFN normalization for deploy."""
        if isinstance(self.mixer, nn.Conv2d):
            return
        self.mixer.fuse_convs()
        c = self.mixer.conv
        scale = getattr(self, "ls1", None)
        scale = (
            torch.ones(c.out_channels, 1, 1, device=c.weight.device, dtype=c.weight.dtype) if scale is None else scale
        )
        identity = torch.zeros_like(c.weight)
        identity[:, :, 1, 1] = 1
        weight = c.weight * scale[:, None] + identity * (1 - scale[:, None])
        mixer = nn.Conv2d(c.in_channels, c.out_channels, 3, padding=1, groups=c.groups, bias=True).to(c.weight)
        mixer.weight.copy_(weight)
        mixer.bias.copy_(c.bias * scale.flatten())
        self.mixer = mixer
        self.ffn_dw = fuse_conv_and_bn(self.ffn_dw, self.ffn_bn)
        del self.ffn_bn
        if hasattr(self, "ls1"):
            del self.ls1
        if hasattr(self, "ls2"):
            self.ffn_pw2.weight.mul_(self.ls2[:, None])
            del self.ls2


class RepCPE(nn.Module):
    """Apply conditional positional encoding and fold its residual into one deploy convolution.

    Attributes:
        pe (nn.Conv2d): Training-time depthwise positional convolution.
        reparam_conv (nn.Conv2d): Deploy convolution containing the positional and identity kernels.
    """

    def __init__(self, c: int, kernel_size: int = 7):
        """Initialize the depthwise positional convolution.

        Args:
            c (int): Number of input and output channels.
            kernel_size (int): Spatial convolution kernel size.
        """
        super().__init__()
        self.pe = nn.Conv2d(c, c, kernel_size, padding=kernel_size // 2, groups=c, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the training or reparameterized positional convolution."""
        return self.reparam_conv(x) if hasattr(self, "reparam_conv") else x + self.pe(x)

    @torch.no_grad()
    def fuse(self):
        """Fold the identity kernel into the positional convolution."""
        self.pe.weight[:, 0, self.pe.kernel_size[0] // 2, self.pe.kernel_size[1] // 2].add_(1)
        self.reparam_conv = self.pe
        del self.pe


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
        pe (RepCPE): Optional reparameterizable conditional positional encoding.
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
        ls1 (nn.Parameter): Optional LayerScale on the attention residual, folded into `proj` for deploy.
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
            self.pe = RepCPE(c)
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

    def _apply_rope(self, q: torch.Tensor, k: torch.Tensor, h: int, w: int):
        """Rotary extension point, a no-op here. RoPE2DBlock subclasses rotate q/k by their positional tables."""
        return q, k

    @torch.no_grad()
    def fuse(self):
        """Fold LayerScale into the attention and FFN output projections for deploy."""
        if hasattr(self, "pe"):
            self.pe.fuse()
        if hasattr(self, "ffn_bn"):
            self.ffn_dw = fuse_conv_and_bn(self.ffn_dw, self.ffn_bn)
            del self.ffn_bn
        if hasattr(self, "ls1"):
            self.proj.weight.mul_(self.ls1[:, None])
            if self.proj.bias is not None:
                self.proj.bias.mul_(self.ls1)
            del self.ls1
        if hasattr(self, "ls2"):
            projection = self.ffn_pw2 if hasattr(self, "ffn_pw2") else self.fc2
            scale = self.ls2.flatten()
            projection.weight.mul_(scale[:, None] if isinstance(projection, nn.Linear) else scale[:, None, None, None])
            if projection.bias is not None:
                projection.bias.mul_(scale)
            del self.ls2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass: 4D → tokens (+ storage) → SA → drop storage → FFN (token Linear/SwiGLU or ConvMlp) → 4D."""
        b, c, h, w = x.shape
        if hasattr(self, "pe"):
            x = self.pe(x)
        t = x.flatten(2).transpose(1, 2)  # (B, N, C)
        xca = getattr(self, "temperature", None) is not None
        n_storage_tokens = getattr(self, "n_storage_tokens", 0)  # getattr: pre-registers checkpoints still load
        t_kv = torch.cat([self.storage_tokens.expand(b, -1, -1), t], dim=1) if n_storage_tokens else t
        n = self.ln1(t_kv)
        qkv = self.qkv(n).reshape(b, -1, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # indexed split traces to aten::select, x2paddle maps it but not aten::unbind
        if n_storage_tokens and not xca:  # storage tokens are K/V-only, SDPA output for their query rows is
            q = q[:, :, n_storage_tokens:]  # unused, so drop them from q before SDPA instead of after
        q, k = self._apply_rope(q, k, h, w)  # no-op unless RoPE2DBlock. Runs after the q drop so only k keeps a prefix
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
            f = self.ffn_dw(x)
            f = self.ffn_bn(f) if hasattr(self, "ffn_bn") else f
            f = self.ffn_pw2(self.act(self.ffn_pw1(f)))
            return x + (f if ls2 is None else ls2 * f)
        n2 = self.ln2(t)
        if getattr(self, "swiglu", False):  # getattr: pre-swiglu checkpoints still load and run
            x1, x2 = self.fc1(n2).chunk(2, dim=-1)
            f = self.fc2(
                F.silu(x1) * x2
            )  # functional silu: export/fuse can't flip it inplace on the chunk view (block.py SwiGLUFFN)
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


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Rotate the last dim by splitting in half and mapping (x1, x2) -> (-x2, x1) for RoPE."""
    d = x.shape[-1] // 2
    return torch.cat([-x[..., d:], x[..., :d]], dim=-1)


def _rope_coords(h: int, w: int, device=None) -> torch.Tensor:
    """Build DINOv3 patch-center coordinates for an h x w token grid, per-axis normalized to [-1, 1].

    Each axis is divided by its own extent, so the coordinate field is resolution independent and a grid trained at 224
    keeps its geometry at 640. Matches `get_patches_center_coordinates` in the released DINOv3 and the EUPE
    `RopePositionEmbedding` that our distillation teacher runs.

    Args:
        h (int): Token grid height.
        w (int): Token grid width.
        device (torch.device, optional): Device to build on.

    Returns:
        (torch.Tensor): Coordinates of shape (N, 2) ordered (y, x) with N = h * w.
    """
    cy = torch.arange(0.5, h, dtype=torch.float32, device=device) / h
    cx = torch.arange(0.5, w, dtype=torch.float32, device=device) / w
    return 2.0 * torch.stack(torch.meshgrid(cy, cx, indexing="ij"), dim=-1).flatten(0, 1) - 1.0


class RoPE2DBlock(MHSABlock):
    """MHSABlock with 2D rotary position embedding on Q/K. Dim-preserving 4D in/out.

    Owns the dense cos/sin buffers, the rebuild-on-grid-change cache, and the deploy rebake. Subclasses supply only
    `_rope_tbl(h, w)`, so every rotary variant shares MHSABlock's single `forward` through its `_apply_rope` hook
    instead of carrying a near-duplicate copy. Tables are dense (N, head_dim) buffers, one initializer per table in a
    traced graph, which avoids the per-frequency Constant blowup that made ECViT-t emit 554 Constant nodes.

    `xca` is rejected, since cross-covariance attention maps channels rather than positions. Storage tokens are
    supported: they carry no patch position, so `_apply_rope` leaves their K/V prefix unrotated and rotates only the
    patch rows, which is what DINOv3, EUPE and RADIO all do. Their query rows are already dropped upstream.

    Attributes:
        rope_hw (tuple): Grid the buffers were baked at, or the last grid a forward ran at.
        rope_deploy (bool): When True, forward reads the baked buffers instead of rebuilding every call.
        rope_cos (torch.Tensor): Dense cosine table registered as a non-persistent buffer.
        rope_sin (torch.Tensor): Dense sine table registered as a non-persistent buffer.
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
        qkv_bias: bool = False,
        proj_bias: bool = False,
        swiglu: bool = False,
        n_storage_tokens: int = 0,
        hw: int = 7,
    ):
        """Initialize shared rotary state. Subclasses materialize their own tables at the end of their init."""
        super().__init__(
            c,
            num_heads,
            mlp_ratio,
            silu,
            ls,
            conv_ffn,
            head_dim,
            xca=False,
            qkv_bias=qkv_bias,
            proj_bias=proj_bias,
            swiglu=swiglu,
            n_storage_tokens=n_storage_tokens,
        )
        assert self.head_dim % 4 == 0, f"{type(self).__name__}: head_dim={self.head_dim} must be divisible by 4"
        self.rope_hw = (hw, hw)  # build grid, so a bake before any forward still uses a real size
        self.rope_deploy = False
        self.register_buffer("rope_cos", torch.zeros(1), persistent=False)
        self.register_buffer("rope_sin", torch.zeros(1), persistent=False)

    def _rope_tbl(self, h: int, w: int):
        """Return the (cos, sin) angle tables for an h x w grid, implemented by each rotary variant."""
        raise NotImplementedError

    def _rope(self, h: int, w: int):
        """Read the baked tables once deployed, otherwise rebuild while the frequencies still train."""
        if not self.rope_deploy:
            self.rope_hw = (h, w)  # track the live grid so a later bake uses the size actually being run
            return self._rope_tbl(h, w)
        if self.rope_hw != (h, w):
            self.switch_to_deploy((h, w))  # deployed graph met a new grid, rebake rather than serve a stale table
        return self.rope_cos, self.rope_sin

    def switch_to_deploy(self, hw=None):
        """Bake the cos/sin tables into the buffers so ONNX and TRT carry them as constants.

        Args:
            hw (int | tuple, optional): Export token grid, a square side or an explicit (h, w). Defaults to the grid the
                block was built for or last ran at.
        """
        h, w = self.rope_hw if hw is None else (hw, hw) if isinstance(hw, int) else tuple(hw)
        cos, sin = self._rope_tbl(h, w)
        self.rope_cos, self.rope_sin = cos.detach(), sin.detach()
        self.rope_hw, self.rope_deploy = (h, w), True

    def _apply_rope(self, q: torch.Tensor, k: torch.Tensor, h: int, w: int):
        """Rotate q and the patch rows of k at table precision, leaving any storage-token K/V prefix unrotated."""
        cos, sin = self._rope(h, w)
        # Rotate in the tables' dtype and cast the result back, which is what EUPE's apply_rope does. Running the
        # rotation at fp16 instead drifts the token norms 1.5x further (3.0e-03 against 2.0e-03, measured).
        dt, (q, k) = q.dtype, (q.to(cos.dtype), k.to(cos.dtype))
        p = self.n_storage_tokens  # storage tokens hold no patch position, so they stay outside the rotation
        kp = k[:, :, p:] * cos + _rotate_half(k[:, :, p:]) * sin
        k = torch.cat([k[:, :, :p], kp], 2) if p else kp
        # q lost its storage rows upstream, so it is patch-only and rotates whole
        return (q * cos + _rotate_half(q) * sin).to(dt), k.to(dt)


class DINOv3RoPE2D(RoPE2DBlock):
    """RoPE2DBlock using the released DINOv3 axial formulation. Dim-preserving 4D in/out.

    Reproduces `RopePositionEmbedding` of the DINOv3 lineage, verified line by line against the EUPE teacher
    implementation and the transformers `DINOv3ViTRopePositionEmbedding`. Patch-center coordinates are normalized per
    axis to [-1, 1], periods form the geometric ladder `base ** (2k / (head_dim / 2))`, angles are `2 * pi * coord /
    period`, and the `[y, x, y, x]` duplication keeps each rotate-half pair on a single angle.

    The released ViT-L/16 config also samples one log-uniform coordinate rescale per training step, shared by every
    layer because DINOv3 holds a single rope module for the whole trunk. This block is one of eight independent P5
    repeats with no shared owner, so a per-block sample would not reproduce that behavior. The augmentation is left out
    rather than approximated.

    Attributes:
        periods (torch.Tensor): Geometric period ladder of shape (head_dim / 4,), a non-persistent buffer.
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
        qkv_bias: bool = False,
        proj_bias: bool = False,
        swiglu: bool = False,
        n_storage_tokens: int = 0,
        hw: int = 7,
        rope_base: float = 100.0,
    ):
        """Initialize DINOv3RoPE2D with the geometric period ladder. The xca branch is forced off."""
        super().__init__(
            c, num_heads, mlp_ratio, silu, ls, conv_ffn, head_dim, qkv_bias, proj_bias, swiglu, n_storage_tokens, hw
        )
        n_freq = self.head_dim // 4
        periods = rope_base ** (2 * torch.arange(n_freq, dtype=torch.float32) / (self.head_dim // 2))
        self.register_buffer("periods", periods, persistent=False)

    def _rope_tbl(self, h: int, w: int):
        """Build the axial table from patch-center coordinates and the geometric period ladder."""
        coords = _rope_coords(h, w, self.periods.device)  # (N, 2) ordered (y, x)
        ang = (2 * math.pi * coords[:, :, None] / self.periods[None, None, :]).flatten(1, 2).tile(2)
        return torch.cos(ang), torch.sin(ang)


class MixedRoPE2D(RoPE2DBlock):
    """RoPE2DBlock with learned per-head mixed x/y rotary frequencies. Dim-preserving 4D in/out.

    Faithful to RoPE-Mixed of Heo, Park, Han, and Yun (ECCV 2024) by default: an integer token grid, frequency
    magnitudes `1 / base ** (4k / head_dim)` at base 10, and per-head random rotation of the axial basis at init. Each
    head learns its own pair of frequency vectors and each angle mixes both axes as `y * freq_y + x * freq_x`, so a head
    can tune to diagonal or anisotropic structure instead of the fixed axial split of DINOv3RoPE2D.

    Why this arm, from the paper's own tables. On MS-COCO with DINO-ViTDet at 12 epochs, swapping only the backbone
    positional encoding moves ViT-B from 49.4 to 51.2 AP and ViT-L from 51.1 to 52.9 AP, +1.8 in both cases and ahead of
    axial RoPE at +1.4 and +1.1. On multi-resolution ImageNet a 224-trained ViT-S goes from 80.4 at 224 and 75.4 at 512
    to 80.9 and 79.1. That combination, an isolated backbone-only detector gain plus resolution extrapolation, is what
    this project needs from a positional arm.

    The released code rotates Q/K with complex `torch.polar` on interleaved dim pairs. This block uses the real
    rotate-half form on half-split pairs instead, which is the same rotation up to a fixed dim permutation that the qkv
    projection absorbs (verified equal to 2.4e-07). Complex ops do not survive the export set, so this is a
    representation choice rather than a change of method.

    Frequencies stay learnable through fusion. `switch_to_deploy` materializes the tables into the inherited buffers and
    sets `rope_deploy`, so a traced graph reads a constant table instead of the parameters.

    Attributes:
        freqs (nn.Parameter): Learned frequencies of shape (2, num_heads, head_dim / 2), indexed (y, x).
        rope_deploy (bool): When True, forward reads the baked buffers instead of rebuilding from `freqs`.

    References:
        Rotary Position Embedding for Vision Transformer, ECCV 2024.
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
        qkv_bias: bool = False,
        proj_bias: bool = False,
        swiglu: bool = False,
        n_storage_tokens: int = 0,
        hw: int = 7,
        rope_base: float = 10.0,
    ):
        """Initialize MixedRoPE2D with a per-head random rotation of the axial basis (RoPE-ViT init)."""
        super().__init__(
            c, num_heads, mlp_ratio, silu, ls, conv_ffn, head_dim, qkv_bias, proj_bias, swiglu, n_storage_tokens, hw
        )
        d = self.head_dim  # magnitude ladder and the integer grid below mirror compute_axial_cis / init_t_xy in
        # ultralytics/models/sam/modules/utils.py, same naver-ai/rope-vit lineage, kept separate because that path is
        # complex-valued and does not export
        mag = 1.0 / rope_base ** (torch.arange(0, d, 4, dtype=torch.float32)[: d // 4] / d)
        a = torch.rand(self.num_heads, 1) * 2 * math.pi  # per-head rotation, breaks the tie between heads at init
        self.freqs = nn.Parameter(
            torch.stack(
                [
                    torch.cat([mag * torch.sin(a), mag * torch.sin(a + math.pi / 2)], dim=-1),
                    torch.cat([mag * torch.cos(a), mag * torch.cos(a + math.pi / 2)], dim=-1),
                ]
            )
        )

    def _rope_tbl(self, h: int, w: int):
        """Build (num_heads, N, head_dim) tables by mixing both axes with the learned per-head frequencies."""
        t = torch.arange(h * w, dtype=torch.float32, device=self.freqs.device)
        coords = torch.stack([torch.div(t, w, rounding_mode="floor"), t % w], dim=-1)  # published integer token grid
        ang = coords[:, 0, None] * self.freqs[0, :, None] + coords[:, 1, None] * self.freqs[1, :, None]
        ang = ang.tile(2)  # (num_heads, N, head_dim), broadcasts over (B, num_heads, N, head_dim)
        return torch.cos(ang), torch.sin(ang)


class FastViTBlock(UltraViTBlock):
    """Deprecated alias of UltraViTBlock, kept so legacy fastvit YAMLs and pickled checkpoints keep loading."""

    _warned = False  # warn once per session, not once per constructed block

    def __init__(self, *args, **kwargs):
        """Initialize UltraViTBlock under its deprecated name with a rename warning."""
        if not FastViTBlock._warned:
            FastViTBlock._warned = True
            deprecation_warn("FastViTBlock", "UltraViTBlock")
        super().__init__(*args, **kwargs)
