# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

"""R3 ViT-like student blocks for encoder distillation (UltraViT).

Simple-component constraint: Conv2d, BatchNorm2d, LayerNorm, GELU/SiLU, Linear, F.scaled_dot_product_attention.
No `nn.MultiheadAttention` (source of AIFI's 1327-node ONNX bloat). No 2D RoPE (ECViT-t hits 554 Constant nodes).

Registered in `ultralytics.nn.modules.__init__` and imported by `ultralytics.nn.tasks` so `parse_model` resolves
them through `globals()[m]`. All blocks are dim-preserving (C_in == C_out, H/W unchanged).

Export validation (2026-04-23 R3.3, RTX PRO 6000 Blackwell, imgsz=224, bs=1 fp16):
    yolo26s-fastvit-cls    5.05 M   228 ONNX nodes   1.948 ms   (conv baseline 1.83 ms, 234 nodes)
    yolo26l-fastvit-cls   14.77 M   804 ONNX nodes   2.652 ms

Must-build export paths pass across the UltraViT and legacy FastViT YAMLs: TorchScript, ONNX opset17, OpenVINO, CoreML, TFLite, TensorRT,
PaddlePaddle (x2paddle>=1.6.0, needs the indexed QKV split in MHSABlock), RKNN (rknn-toolkit2>=2.3.2). RKNN still
requires an isolated venv (its AutoUpdate downgrades torch 2.9→2.4 + cudnn 9.10→9.1, contaminating the primary env).
"""

from __future__ import annotations

import math
import os

import torch
import torch.nn as nn
import torch.nn.functional as F

from ultralytics.utils import deprecation_warn

# Length-aware SDPA temperature (A1). A P5 token grid trained at ~224px (49 tokens for the /16 stem) but run at 640px
# (400 tokens) diffuses the fixed 1/sqrt(d) softmax over 8x more keys. Scaling the logits by sqrt(log N / log N_ref)
# restores the training-time peakiness. Env-gated so the CE/distill graph is byte-identical unless opted in; at
# N == N_ref the factor is exactly 1 (no-op at training resolution). q.shape[-2] is a concrete int under static-shape
# export, so this folds into SDPA's scalar scale (one constant Mul, no branch) and passes RKNN/Paddle/CoreML.
_LOGN_ATTN = os.getenv("ULTRAVIT_LOGN_ATTN", "0") == "1"
_INV_LOG_REF = 1.0 / math.log(49)  # 49 = /16-stem P5 tokens at the ~224px training grid


class UltraViTBlock(nn.Module):
    """UltraViT stages 1-3 block: RepMixer (inference form) + ConvFFN. Dim-preserving 4D in/out.

    Paper: arXiv:2303.14189 §3 (FastViT, Vasu et al. 2023). Reparameterized inference form collapses the train-time
    RepMixer to `x + DWConv3x3+BN(x)`. ConvFFN inverted-bottleneck: PW → GELU → DW3x3+BN → PW. No LayerNorm in stages
    1-3 (FastViT paper §3.2 uses BN here for speed).

    Attributes:
        mixer_dw (nn.Conv2d): Depthwise 3x3 mixing conv.
        mixer_bn (nn.BatchNorm2d): BN after mixer.
        ffn_pw1 (nn.Conv2d): 1x1 PW conv to hidden dim.
        ffn_dw (nn.Conv2d): 3x3 DW conv at hidden dim.
        ffn_bn (nn.BatchNorm2d): BN on hidden.
        ffn_pw2 (nn.Conv2d): 1x1 PW conv back to c.
        act (nn.Module): FFN activation, GELU or SiLU.
        ls1 (nn.Parameter): Optional LayerScale on the mixer residual (timm FastViT trains 1e-5 on every residual).
            Created only when `ls > 0`; `forward` guards on it so pre-LayerScale checkpoints still load and run.
        ls2 (nn.Parameter): Optional LayerScale on the FFN residual.
    """

    def __init__(self, c: int, mlp_ratio: float = 3.0, silu: bool = False, ls: float = 0.0):
        """Initialize UltraViTBlock with dim c, FFN expansion ratio, activation choice, and LayerScale init."""
        super().__init__()
        self.mixer_dw = nn.Conv2d(c, c, 3, padding=1, groups=c, bias=False)
        self.mixer_bn = nn.BatchNorm2d(c)
        hidden = int(c * mlp_ratio)
        self.ffn_pw1 = nn.Conv2d(c, hidden, 1, bias=False)
        self.ffn_dw = nn.Conv2d(hidden, hidden, 3, padding=1, groups=hidden, bias=False)
        self.ffn_bn = nn.BatchNorm2d(hidden)
        self.ffn_pw2 = nn.Conv2d(hidden, c, 1, bias=False)
        # SiLU fuses into conv epilogues under TensorRT; GELU lowers to standalone fp32 erf+cast kernels
        # (measured 9-26% of engine time). GELU stays the default so pre-silu checkpoints keep their activation.
        self.act = nn.SiLU() if silu else nn.GELU()
        if ls:
            self.ls1 = nn.Parameter(ls * torch.ones(c, 1, 1))
            self.ls2 = nn.Parameter(ls * torch.ones(c, 1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass: residual mixer + residual FFN, each optionally LayerScale-gated."""
        m = self.mixer_bn(self.mixer_dw(x))
        ls1 = getattr(self, "ls1", None)
        x = x + (m if ls1 is None else ls1 * m)
        h = self.act(self.ffn_pw1(x))
        h = self.act(self.ffn_bn(self.ffn_dw(h)))
        f = self.ffn_pw2(h)
        ls2 = getattr(self, "ls2", None)
        return x + (f if ls2 is None else ls2 * f)


class MHSABlock(nn.Module):
    """Pre-norm ViT block with SDPA and a token-Linear or NCHW ConvMlp FFN. Dim-preserving 4D in/out.

    Uses explicit QKV Linear + `F.scaled_dot_product_attention` instead of `nn.MultiheadAttention` (the AIFI bloat
    source — MHA-based AIFI ViT wraps to ~1327 ONNX nodes @ opset 17). SDPA decomposes in opset 17 to
    `MatMul+Softmax+MatMul+Mul(scale)`; the win is skipping PyTorch's MHA wrapper, not graph fusion.

    Used for UltraViT (and legacy FastViT) stage 4 global attention at the coarsest scale.

    Attributes:
        num_heads (int): Number of attention heads. c must be divisible by num_heads. YAMLs that pin `head_dim` pass 0
            here so no dead head count sits in the config; the value is then derived, never read.
        head_dim (int): Per-head dim. When the `head_dim` arg is nonzero it pins this value and derives num_heads = c //
            head_dim (Apple FastViT policy, head_dim 32), so head width no longer shrinks with model scale.
        pe (nn.Conv2d): Depthwise 7x7 conditional positional encoding (FastViT RepCPE / CPVT), applied before attention.
            Zero-initialized so a fresh block starts as identity (ReZero) and the residual learns the position signal
            from zero. `forward` guards on it so checkpoints saved before `pe` existed still load and run.
        ln1 (nn.LayerNorm): Pre-attention norm.
        qkv (nn.Linear): Fused QKV projection.
        proj (nn.Linear): Post-attention projection.
        ln2 (nn.LayerNorm): Pre-FFN norm (token-Linear FFN only).
        fc1 (nn.Linear): FFN first layer (token-Linear FFN only).
        fc2 (nn.Linear): FFN second layer (token-Linear FFN only).
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
    ):
        """Initialize MHSABlock."""
        super().__init__()
        if head_dim:
            assert c % head_dim == 0, f"MHSABlock: c={c} not divisible by head_dim={head_dim}"
            num_heads = c // head_dim
        assert c % num_heads == 0, f"MHSABlock: c={c} not divisible by num_heads={num_heads}"
        self.num_heads = num_heads
        self.head_dim = c // num_heads
        self.pe = nn.Conv2d(c, c, 7, padding=3, groups=c, bias=True)
        nn.init.zeros_(self.pe.weight)
        nn.init.zeros_(self.pe.bias)
        self.ln1 = nn.LayerNorm(c)
        self.qkv = nn.Linear(c, 3 * c, bias=False)
        self.proj = nn.Linear(c, c, bias=False)
        hidden = int(c * mlp_ratio)
        self.act = nn.SiLU() if silu else nn.GELU()
        if conv_ffn:
            self.ffn_dw = nn.Conv2d(c, c, 7, padding=3, groups=c, bias=False)
            self.ffn_bn = nn.BatchNorm2d(c)
            self.ffn_pw1 = nn.Conv2d(c, hidden, 1)
            self.ffn_pw2 = nn.Conv2d(hidden, c, 1)
        else:
            self.ln2 = nn.LayerNorm(c)
            self.fc1 = nn.Linear(c, hidden)
            self.fc2 = nn.Linear(hidden, c)
        if ls:
            self.ls1 = nn.Parameter(ls * torch.ones(c))
            # ls2 shape follows the FFN form chosen at construction: (C, 1, 1) broadcasts in NCHW for ConvMlp,
            # (C,) for tokens. Avoids a per-forward view that would add a Reshape node to traced graphs.
            self.ls2 = nn.Parameter(ls * torch.ones(c, 1, 1) if conv_ffn else ls * torch.ones(c))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass: 4D → tokens → SA → FFN (token Linear or NCHW ConvMlp) → 4D."""
        b, c, h, w = x.shape
        if getattr(self, "pe", None) is not None:
            x = x + self.pe(x)  # RepCPE conditional position before attention
        t = x.flatten(2).transpose(1, 2)  # (B, N, C)
        n = self.ln1(t)
        qkv = self.qkv(n).reshape(b, -1, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # indexed split traces to aten::select, x2paddle maps it but not aten::unbind
        if _LOGN_ATTN:  # length-aware temperature: sharpen softmax as the token grid grows past the 49-token train grid
            scale = self.head_dim**-0.5 * (math.log(max(int(q.shape[-2]), 2)) * _INV_LOG_REF) ** 0.5
            a = F.scaled_dot_product_attention(q, k, v, scale=scale)
        else:
            a = F.scaled_dot_product_attention(q, k, v)
        a = self.proj(a.transpose(1, 2).reshape(b, -1, c))
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


class FastViTBlock(UltraViTBlock):
    """Deprecated alias of UltraViTBlock, kept so legacy fastvit YAMLs and pickled checkpoints keep loading."""

    _warned = False  # warn once per session, not once per constructed block

    def __init__(self, *args, **kwargs):
        """Initialize UltraViTBlock under its deprecated name with a rename warning."""
        if not FastViTBlock._warned:
            FastViTBlock._warned = True
            deprecation_warn("FastViTBlock", "UltraViTBlock")
        super().__init__(*args, **kwargs)
