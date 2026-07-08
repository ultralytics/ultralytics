# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

"""ViT-like student blocks for encoder distillation (UltraViT).

Simple-component constraint: Conv2d, BatchNorm2d, LayerNorm, GELU/SiLU, Linear, F.scaled_dot_product_attention.
No `nn.MultiheadAttention` (source of AIFI's 1327-node ONNX bloat). No 2D RoPE (ECViT-t hits 554 Constant nodes).

Registered in `ultralytics.nn.modules.__init__` and imported by `ultralytics.nn.tasks` so `parse_model` resolves
them through `globals()[m]`. All blocks are dim-preserving (C_in == C_out, H/W unchanged).

Export validation (2026-04-23, RTX PRO 6000 Blackwell, imgsz=224, bs=1 fp16):
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
        temperature (nn.Parameter): Per-head scalar for cross-covariance attention (XCiT), created only when `xca=True`.
            Its presence switches `forward` to channel attention (map is head_dim x head_dim, invariant to token count),
            so a frozen backbone meets no length-coupled softmax when transferred to a larger detection grid.
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
        xca: bool = False,
    ):
        """Initialize MHSABlock."""
        super().__init__()
        if head_dim:
            assert c % head_dim == 0, f"MHSABlock: c={c} not divisible by head_dim={head_dim}"
            num_heads = c // head_dim
        assert c % num_heads == 0, f"MHSABlock: c={c} not divisible by num_heads={num_heads}"
        self.num_heads = num_heads
        self.head_dim = c // num_heads
        if xca:  # cross-covariance attention: map is head_dim x head_dim (token-count invariant), learnable per-head temperature (XCiT)
            self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))
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
        t = x.flatten(2).transpose(1, 2)  # (B, N, C)
        n = self.ln1(t)
        qkv = self.qkv(n).reshape(b, -1, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # indexed split traces to aten::select, x2paddle maps it but not aten::unbind
        if getattr(self, "temperature", None) is not None:  # XCA: attention over channels, invariant to token count
            qn = F.normalize(q.transpose(-2, -1), dim=-1)  # (B, heads, head_dim, N), L2-normed over tokens
            kn = F.normalize(k.transpose(-2, -1), dim=-1)
            attn = (qn @ kn.transpose(-2, -1)) * self.temperature  # (B, heads, head_dim, head_dim)
            a = (attn.softmax(dim=-1) @ v.transpose(-2, -1)).permute(0, 3, 1, 2).reshape(b, -1, c)
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
        f = self.fc2(self.act(self.fc1(self.ln2(t))))
        t = t + (f if ls2 is None else ls2 * f)
        return t.transpose(1, 2).reshape(b, c, h, w)


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
        from ultralytics.utils.torch_utils import fuse_conv_and_bn

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
