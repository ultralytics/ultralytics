# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

"""R3 ViT-like student blocks for encoder distillation (FastViT).

Simple-component constraint: Conv2d, BatchNorm2d, LayerNorm, GELU/SiLU, Linear, F.scaled_dot_product_attention.
No `nn.MultiheadAttention` (source of AIFI's 1327-node ONNX bloat). No 2D RoPE (ECViT-t hits 554 Constant nodes).

Registered in `ultralytics.nn.modules.__init__` and imported by `ultralytics.nn.tasks` so `parse_model` resolves
them through `globals()[m]`. All blocks are dim-preserving (C_in == C_out, H/W unchanged).

Export validation (2026-04-23 R3.3, RTX PRO 6000 Blackwell, imgsz=224, bs=1 fp16):
    yolo26s-fastvit-cls    5.05 M   228 ONNX nodes   1.948 ms   (conv baseline 1.83 ms, 234 nodes)
    yolo26l-fastvit-cls   14.77 M   804 ONNX nodes   2.652 ms

Must-build export paths pass across the FastViT YAMLs: TorchScript, ONNX opset17, OpenVINO, CoreML, TFLite, TensorRT.
PaddlePaddle fails (RepMixer/SDPA op-coverage gap in Paddle converter). RKNN runs only in an isolated venv
(rknn-toolkit2 AutoUpdate downgrades torch 2.9→2.4 + cudnn 9.10→9.1, contaminating the primary env).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class FastViTBlock(nn.Module):
    """FastViT stages 1-3 block: RepMixer (inference form) + ConvFFN. Dim-preserving 4D in/out.

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
    """

    def __init__(self, c: int, mlp_ratio: float = 3.0, silu: bool = False):
        """Initialize FastViTBlock with dim c, FFN expansion ratio, and activation choice."""
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass: residual mixer + residual FFN."""
        x = x + self.mixer_bn(self.mixer_dw(x))
        h = self.act(self.ffn_pw1(x))
        h = self.act(self.ffn_bn(self.ffn_dw(h)))
        x = x + self.ffn_pw2(h)
        return x


class MHSABlock(nn.Module):
    """Pre-norm ViT block with SDPA + LN + Linear FFN. Dim-preserving 4D in/out.

    Uses explicit QKV Linear + `F.scaled_dot_product_attention` instead of `nn.MultiheadAttention` (the AIFI bloat
    source — MHA-based AIFI ViT wraps to ~1327 ONNX nodes @ opset 17). SDPA decomposes in opset 17 to
    `MatMul+Softmax+MatMul+Mul(scale)`; the win is skipping PyTorch's MHA wrapper, not graph fusion.

    Used for FastViT stage 4 global attention at the coarsest scale.

    Attributes:
        num_heads (int): Number of attention heads. c must be divisible by num_heads.
        head_dim (int): c // num_heads.
        pe (nn.Conv2d): Depthwise 7x7 conditional positional encoding (FastViT RepCPE / CPVT), applied before attention.
            Zero-initialized so a fresh block starts as identity (ReZero) and the residual learns the position signal
            from zero. `forward` guards on it so checkpoints saved before `pe` existed still load and run.
        ln1 (nn.LayerNorm): Pre-attention norm.
        qkv (nn.Linear): Fused QKV projection.
        proj (nn.Linear): Post-attention projection.
        ln2 (nn.LayerNorm): Pre-FFN norm.
        fc1 (nn.Linear): FFN first layer.
        fc2 (nn.Linear): FFN second layer.
        act (nn.Module): FFN activation, GELU or SiLU (see FastViTBlock on the TensorRT fusion difference).
    """

    def __init__(self, c: int, num_heads: int = 6, mlp_ratio: float = 4.0, silu: bool = False):
        """Initialize MHSABlock."""
        super().__init__()
        assert c % num_heads == 0, f"MHSABlock: c={c} not divisible by num_heads={num_heads}"
        self.num_heads = num_heads
        self.head_dim = c // num_heads
        self.pe = nn.Conv2d(c, c, 7, padding=3, groups=c, bias=True)
        nn.init.zeros_(self.pe.weight)
        nn.init.zeros_(self.pe.bias)
        self.ln1 = nn.LayerNorm(c)
        self.qkv = nn.Linear(c, 3 * c, bias=False)
        self.proj = nn.Linear(c, c, bias=False)
        self.ln2 = nn.LayerNorm(c)
        hidden = int(c * mlp_ratio)
        self.fc1 = nn.Linear(c, hidden)
        self.act = nn.SiLU() if silu else nn.GELU()
        self.fc2 = nn.Linear(hidden, c)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass: 4D → tokens → SA + FFN → 4D."""
        b, c, h, w = x.shape
        if getattr(self, "pe", None) is not None:
            x = x + self.pe(x)  # RepCPE conditional position before attention
        t = x.flatten(2).transpose(1, 2)  # (B, N, C)
        n = self.ln1(t)
        qkv = self.qkv(n).reshape(b, -1, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        a = F.scaled_dot_product_attention(q, k, v)
        a = a.transpose(1, 2).reshape(b, -1, c)
        t = t + self.proj(a)
        t = t + self.fc2(self.act(self.fc1(self.ln2(t))))
        return t.transpose(1, 2).reshape(b, c, h, w)
