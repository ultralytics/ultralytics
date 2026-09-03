# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

"""UltraViT building blocks for convolutional mixing, attention, and feature aggregation."""

from __future__ import annotations

import math
import os

import torch
import torch.nn as nn
import torch.nn.functional as F

from ultralytics.nn.modules.conv import RepConv
from ultralytics.utils.torch_utils import fuse_conv_and_bn

__all__ = ("MHSABlock", "RepUltraViTBlock", "UltraViTBlock")


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
        """Initialize the block.

        Args:
            c (int): Number of input and output channels.
            mlp_ratio (float): Expansion ratio of the FFN hidden width.
            silu (bool): Use SiLU instead of GELU in the FFN.
            ls (float): LayerScale initialization value; no LayerScale parameters are created when zero.
            fastvit_ffn (bool): Use the paper-exact pre-expansion DW7x7 FFN instead of the post-expansion DW3x3 one.
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
        """Apply the ConvFFN; `fastvit_ffn` selects pre-expansion DW7x7 (paper-exact) vs post-expansion DW3x3.

        Args:
            x (torch.Tensor): Input features with shape (B, C, H, W).

        Returns:
            (torch.Tensor): FFN output with shape (B, C, H, W).
        """
        if getattr(self, "fastvit_ffn", False):  # getattr: pre-fastvit_ffn checkpoints still load and run
            h = self.ffn_dw(x)
            h = self.ffn_bn(h) if hasattr(self, "ffn_bn") else h
            return self.ffn_pw2(self.act(self.ffn_pw1(h)))
        h = self.act(self.ffn_pw1(x))
        h = self.ffn_dw(h)
        h = self.ffn_bn(h) if hasattr(self, "ffn_bn") else h
        return self.ffn_pw2(self.act(h))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the residual token mixer followed by the residual FFN, each optionally LayerScale-gated.

        Args:
            x (torch.Tensor): Input features with shape (B, C, H, W).

        Returns:
            (torch.Tensor): Output features with shape (B, C, H, W).
        """
        m = self.mixer_bn(self.mixer_dw(x))
        ls1 = getattr(self, "ls1", None)
        x = x + (m if ls1 is None else ls1 * m)
        f = self._ffn(x)
        ls2 = getattr(self, "ls2", None)
        return x + (f if ls2 is None else ls2 * f)


class RepUltraViTBlock(UltraViTBlock):
    """Use a FastViT-style reparameterized token mixer with the UltraViT ConvFFN."""

    def __init__(self, c: int, mlp_ratio: float = 3.0, silu: bool = False, ls: float = 0.0, fastvit_ffn: bool = False):
        """Initialize the train-time multi-branch mixer and ConvFFN.

        Args:
            c (int): Number of input and output channels.
            mlp_ratio (float): Expansion ratio of the FFN hidden width.
            silu (bool): Use SiLU instead of GELU in the FFN.
            ls (float): LayerScale initialization value; no LayerScale parameters are created when zero.
            fastvit_ffn (bool): Use the paper-exact pre-expansion DW7x7 FFN instead of the post-expansion DW3x3 one.
        """
        super().__init__(c, mlp_ratio, silu, ls, fastvit_ffn)
        del self.mixer_dw, self.mixer_bn
        self.mixer = RepConv(c, c, g=c, act=False, bn=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the reparameterized mixer followed by the ConvFFN.

        Args:
            x (torch.Tensor): Input features with shape (B, C, H, W).

        Returns:
            (torch.Tensor): Output features with shape (B, C, H, W).

        Notes:
            After fuse() the mixer is a plain Conv2d that already absorbs the residual and LayerScale, so the
            branch below applies it directly.
        """
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


class MHSABlock(nn.Module):
    """Pre-norm SDPA attention block with an NCHW ConvMLP FFN. Dim-preserving 4D in/out.

    Uses explicit QKV Linear + `F.scaled_dot_product_attention` instead of `nn.MultiheadAttention` (the AIFI bloat
    source — MHA-based AIFI ViT wraps to ~1327 ONNX nodes @ opset 17). SDPA decomposes in opset 17 to
    `MatMul+Softmax+MatMul+Mul(scale)`; the win is skipping PyTorch's MHA wrapper, not graph fusion.

    Used for UltraViT (and legacy FastViT) stage 4 global attention at the coarsest scale.

    Attributes:
        num_heads (int): Number of attention heads. c must be divisible by num_heads. YAMLs that pin `head_dim` pass 0
            here so no dead head count sits in the config; the value is then derived, never read.
        head_dim (int): Per-head dim. When the `head_dim` arg is nonzero it pins this value and derives num_heads = c //
            head_dim (Apple FastViT policy, head_dim 32), so head width no longer shrinks with model scale.
        ln1 (nn.LayerNorm): Pre-attention norm.
        qkv (nn.Linear): Bias-free fused QKV projection.
        proj (nn.Linear): Bias-free post-attention projection.
        ffn_dw (nn.Conv2d): 7x7 depthwise local-mixing convolution opening the ConvMLP FFN.
        ffn_bn (nn.BatchNorm2d): ConvMlp norm after the DW conv.
        ffn_pw1 (nn.Conv2d): ConvMlp 1x1 conv to hidden dim.
        ffn_pw2 (nn.Conv2d): ConvMlp 1x1 conv back to c.
        act (nn.Module): FFN activation, GELU or SiLU (see UltraViTBlock on the TensorRT fusion difference).
        ls1 (nn.Parameter): Optional LayerScale on the attention residual (does not fold away at inference).
        ls2 (nn.Parameter): Optional LayerScale on the FFN residual, shaped (C, 1, 1).
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
    ):
        """Initialize the block.

        Args:
            c (int): Number of input and output channels.
            num_heads (int): Number of attention heads; ignored when head_dim is nonzero.
            mlp_ratio (float): Expansion ratio of the FFN hidden width.
            silu (bool): Use SiLU instead of GELU in the FFN.
            ls (float): LayerScale initialization value; no LayerScale parameters are created when zero.
            conv_ffn (bool): Must be True; the block only supports the ConvMLP FFN.
            head_dim (int): Per-head dimension; when nonzero it pins the head width and derives num_heads.
        """
        super().__init__()
        assert conv_ffn, "MHSABlock requires conv_ffn=True"
        if head_dim:
            assert c % head_dim == 0, f"MHSABlock: c={c} not divisible by head_dim={head_dim}"
            num_heads = c // head_dim
        assert c % num_heads == 0, f"MHSABlock: c={c} not divisible by num_heads={num_heads}"
        self.num_heads = num_heads
        self.head_dim = c // num_heads
        self.ln1 = nn.LayerNorm(c)
        self.qkv = nn.Linear(c, 3 * c, bias=False)
        self.proj = nn.Linear(c, c, bias=False)
        hidden = int(c * mlp_ratio)
        self.act = nn.SiLU() if silu else nn.GELU()
        self.ffn_dw = nn.Conv2d(c, c, 7, padding=3, groups=c, bias=False)
        self.ffn_bn = nn.BatchNorm2d(c)
        self.ffn_pw1 = nn.Conv2d(c, hidden, 1)
        self.ffn_pw2 = nn.Conv2d(hidden, c, 1)
        if ls:
            self.ls1 = nn.Parameter(ls * torch.ones(c))
            self.ls2 = nn.Parameter(ls * torch.ones(c, 1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the residual self-attention followed by the residual ConvMLP FFN.

        Args:
            x (torch.Tensor): Input features with shape (B, C, H, W).

        Returns:
            (torch.Tensor): Output features with shape (B, C, H, W).
        """
        b, c, h, w = x.shape
        t = x.flatten(2).transpose(1, 2)  # (B, N, C)
        n = self.ln1(t)
        qkv = self.qkv(n).reshape(b, -1, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # indexed split traces to aten::select, x2paddle maps it but not aten::unbind
        if _LOGN_ATTN:  # length-aware temperature: sharpen softmax as the grid grows past the 49-token train grid
            scale = self.head_dim**-0.5 * (math.log(max(int(q.shape[-2]), 2)) * _INV_LOG_REF) ** 0.5
            a = F.scaled_dot_product_attention(q, k, v, scale=scale)
        else:
            a = F.scaled_dot_product_attention(q, k, v)
        a = a.transpose(1, 2).reshape(b, -1, c)
        a = self.proj(a)
        ls1 = getattr(self, "ls1", None)
        t = t + (a if ls1 is None else ls1 * a)
        x = t.transpose(1, 2).reshape(b, c, h, w)
        f = self.ffn_pw2(self.act(self.ffn_pw1(self.ffn_bn(self.ffn_dw(x)))))
        ls2 = getattr(self, "ls2", None)
        return x + (f if ls2 is None else ls2 * f)
