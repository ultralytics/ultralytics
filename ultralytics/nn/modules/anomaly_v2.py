# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""YOLOA v2 modules: bbox-mask renderer, heatmap bias fusion, optional SegBranch.

Soft-hint fusion: a 1-channel mask is turned into a bounded per-pixel bias added
(broadcast over channels) to PAN features before the Detect head. PAN feature
addition keeps the Detect head unmodified, lets reg and cls both see the bias
(empirical question — see spec §2), and is bounded vs the previous multiplicative
amplifier that forced detections.

See docs_yoloa_v2/specs/2026-06-02-softhint-fusion-design.md.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .conv import Conv

__all__ = (
    "BboxMaskRenderer",
    "HeatmapBiasFusion",
    "HeatmapSoftFusion",
    "HeatmapFiLMFusion",
    "QueryFiLMFusion",
    "SegBranch",
    "binary_seg_loss",
    "query_film_loss",
    "BackboneMemoryBank",
    "LearnedScorer",
    "FeatureDiscriminatorScorer",
)


class BboxMaskRenderer(nn.Module):
    """Render normalized YOLO-format bboxes into a 1xHxW mask.

    Two modes:
      - "rect":  hard rectangle (inside bbox = 1, outside = 0)
      - "gauss": per-bbox 2D Gaussian centered at bbox center,
                 sigma_x = w * sigma_factor, sigma_y = h * sigma_factor;
                 multiple bboxes in the same image are combined with max.

    ``sigma_factor`` may be a scalar (fixed width) or a ``[lo, hi]`` range. With a range, each
    bbox draws its own factor ~ U(lo, hi) while training (prior-width augmentation toward the
    diffuse inference heatmap) and uses the midpoint deterministically at eval.

    Output spatial size is fixed at construction (default 80 to match P3).
    """

    def __init__(self, mask_size: int = 80, mode: str = "rect", sigma_factor: float | list = 0.25):
        super().__init__()
        assert mode in ("rect", "gauss"), f"mode must be 'rect' or 'gauss', got {mode!r}"
        self.mask_size = int(mask_size)
        self.mode = mode
        if isinstance(sigma_factor, (list, tuple)):
            self.sigma_lo, self.sigma_hi = float(sigma_factor[0]), float(sigma_factor[1])
        else:
            self.sigma_lo = self.sigma_hi = float(sigma_factor)
        self.sigma_factor = self.sigma_lo  # back-compat scalar handle (== lo)
        ys, xs = torch.meshgrid(
            torch.arange(self.mask_size, dtype=torch.float32),
            torch.arange(self.mask_size, dtype=torch.float32),
            indexing="ij",
        )
        self.register_buffer("grid_x", xs + 0.5, persistent=False)
        self.register_buffer("grid_y", ys + 0.5, persistent=False)

    def __setstate__(self, state):
        """Backfill sigma_lo/sigma_hi for checkpoints pickled before the [lo, hi] range knob."""
        super().__setstate__(state)
        if not hasattr(self, "sigma_lo"):
            sf = float(getattr(self, "sigma_factor", 0.25))
            self.sigma_lo = self.sigma_hi = sf

    def forward(self, bboxes: torch.Tensor, batch_idx: torch.Tensor, batch_size: int) -> torch.Tensor:
        H = self.mask_size
        device = self.grid_x.device
        dtype = self.grid_x.dtype
        mask = torch.zeros(batch_size, 1, H, H, device=device, dtype=dtype)
        if bboxes.numel() == 0:
            return mask

        bboxes = bboxes.to(device=device, dtype=dtype)
        batch_idx = batch_idx.to(device=device, dtype=torch.long)

        cx = bboxes[:, 0] * H
        cy = bboxes[:, 1] * H
        w = bboxes[:, 2] * H
        h = bboxes[:, 3] * H

        if self.mode == "rect":
            x1 = (cx - w / 2)[:, None, None]
            x2 = (cx + w / 2)[:, None, None]
            y1 = (cy - h / 2)[:, None, None]
            y2 = (cy + h / 2)[:, None, None]
            inside = (
                (self.grid_x[None] >= x1)
                & (self.grid_x[None] < x2)
                & (self.grid_y[None] >= y1)
                & (self.grid_y[None] < y2)
            ).to(dtype)
        else:  # gauss
            # Per-bbox sigma factor: random in [lo, hi] while training, midpoint at eval.
            if self.training and self.sigma_hi > self.sigma_lo:
                sf = torch.empty_like(w).uniform_(self.sigma_lo, self.sigma_hi)
            else:
                sf = 0.5 * (self.sigma_lo + self.sigma_hi)
            sigma_x = (w * sf).clamp(min=0.5)
            sigma_y = (h * sf).clamp(min=0.5)
            dx = self.grid_x[None] - cx[:, None, None]
            dy = self.grid_y[None] - cy[:, None, None]
            inside = torch.exp(
                -(dx**2 / (2 * sigma_x[:, None, None] ** 2) + dy**2 / (2 * sigma_y[:, None, None] ** 2))
            )

        for b in range(batch_size):
            sel = batch_idx == b
            if sel.any():
                mask[b, 0] = inside[sel].max(dim=0).values
        return mask

    def render_per_instance(
        self, bboxes: torch.Tensor, batch_idx: torch.Tensor, batch_size: int, sigma_factor: float | None = None
    ) -> list[torch.Tensor]:
        """Render each bbox as its own gauss mask, grouped per image (NO per-image max merge).

        Training-only target for query-to-instance matching. Unlike ``forward``, the per-bbox
        masks are kept separate so each can be Hungarian-matched to a query attention map.

        Args:
            bboxes: (N, 4) normalized YOLO ``[cx, cy, w, h]``.
            batch_idx: (N,) image index per bbox.
            batch_size: number of images B.
            sigma_factor: fixed gauss width factor; defaults to ``self.sigma_lo`` (independent of
                the fusion-prior sigma so the query GT stays sharp).

        Returns:
            list of length B, each ``(N_b, H, H)`` in [0, 1] (empty ``(0, H, H)`` for images with
            no bboxes).
        """
        H = self.mask_size
        device = self.grid_x.device
        dtype = self.grid_x.dtype
        empty = torch.zeros(0, H, H, device=device, dtype=dtype)
        if bboxes.numel() == 0:
            return [empty for _ in range(batch_size)]
        bboxes = bboxes.to(device=device, dtype=dtype)
        batch_idx = batch_idx.to(device=device, dtype=torch.long)
        cx = bboxes[:, 0] * H
        cy = bboxes[:, 1] * H
        w = bboxes[:, 2] * H
        h = bboxes[:, 3] * H
        sf = float(self.sigma_lo if sigma_factor is None else sigma_factor)
        sigma_x = (w * sf).clamp(min=0.5)
        sigma_y = (h * sf).clamp(min=0.5)
        dx = self.grid_x[None] - cx[:, None, None]
        dy = self.grid_y[None] - cy[:, None, None]
        inst = torch.exp(
            -(dx**2 / (2 * sigma_x[:, None, None] ** 2) + dy**2 / (2 * sigma_y[:, None, None] ** 2))
        )  # (N, H, H)
        return [inst[batch_idx == b] for b in range(batch_size)]

    def extra_repr(self) -> str:
        return f"mask_size={self.mask_size}, mode={self.mode!r}, sigma_factor=[{self.sigma_lo}, {self.sigma_hi}]"


class HeatmapBiasFusion(nn.Module):
    """Soft-hint fusion: 1-ch mask -> bounded per-pixel bias broadcast onto PAN features.

    Output shape ``(B, 1, H, W)`` — the caller broadcasts (adds) it to a PAN feature
    of shape ``(B, C, H, W)``. The conv stack is SHARED across PAN scales; the caller
    is responsible for resizing the mask to each scale before calling forward.

    Per-scale magnitude is controlled by ``beta[i]``, initialized to zero so training
    starts as pure passthrough (vanilla YOLO). Without a hard cap, beta can in
    principle grow large; that is intentional — the detection loss decides how much
    to lean on the heatmap.

    Output per pixel is in ``[-beta_i, +beta_i]`` via tanh.
    """

    def __init__(self, num_scales: int = 3, c_mid: int = 8):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, c_mid, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(c_mid, 1, 3, padding=1),
        )
        self.beta = nn.Parameter(torch.zeros(num_scales))

    def forward(self, mask: torch.Tensor, scale_idx: int) -> torch.Tensor:
        """Return bias (B, 1, H, W) for the given PAN scale.

        Args:
            mask: (B, 1, H, W) already resized to the target PAN scale.
            scale_idx: index into ``self.beta``.

        Returns:
            Bias tensor (B, 1, H, W) in ``[-beta_i, +beta_i]``.
        """
        return self.beta[scale_idx] * torch.tanh(self.conv(mask))


class HeatmapSoftFusion(nn.Module):
    """Soft-hint fusion: 1-ch mask → spatial softmax(x/t) → conv → BN → tanh → *beta.

    A learned-temperature spatial softmax normalises the prior distribution before a
    shared 2-layer conv stack, making the fusion robust to category-to-category
    variation in raw heatmap value ranges. BatchNorm stabilises the conv output before
    tanh clamping.

    Output shape ``(B, 1, H, W)`` — the caller broadcasts (adds) it to a PAN feature.
    ``beta[i]`` is zero-init so training starts as pure YOLO passthrough.
    """

    def __init__(self, num_scales: int = 3, c_mid: int = 8):
        super().__init__()
        self.log_t = nn.Parameter(torch.zeros(1))  # t = exp(log_t), init t=1
        self.conv = nn.Sequential(
            nn.Conv2d(1, c_mid, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(c_mid, 1, 3, padding=1),
        )
        self.bn = nn.BatchNorm2d(1)
        self.beta = nn.Parameter(torch.zeros(num_scales))

    def forward(self, mask: torch.Tensor, scale_idx: int) -> torch.Tensor:
        """Return bias (B, 1, H, W) for the given PAN scale.

        Args:
            mask: (B, 1, H, W) already resized to the target PAN scale.
            scale_idx: index into ``self.beta``.

        Returns:
            Bias tensor (B, 1, H, W) in ``[-beta_i, +beta_i]``.
        """
        B, _, H, W = mask.shape
        t = torch.exp(self.log_t).clamp(min=1e-3)
        # spatial softmax normalises the distribution shape; rescale by H*W
        # keeps pixel values in roughly [0,1] so conv gradients don't vanish
        x = mask.view(B, -1)                   # (B, H*W)
        x = F.softmax(x / t, dim=-1)           # (B, H*W), sums to 1
        x = x.view(B, 1, H, W) * (H * W)       # (B, 1, H, W), ~[0,1] range
        x = self.conv(x)
        x = self.bn(x)
        return self.beta[scale_idx] * torch.tanh(x)


class HeatmapFiLMFusion(nn.Module):
    """Residual grouped-FiLM fusion: the prior modulates a projected copy of each PAN feature.

    Richer than ``HeatmapBiasFusion`` (1-channel additive bias broadcast over all channels). Each
    PAN feature is projected into a modulation space, split into ``num_groups`` channel groups, and
    each group is scaled by a prior-derived spatial map (grouped FiLM: ``V * (1 + gamma)``, or the
    bounded gate ``V * (1 + tanh(gamma))`` in ``(0, 2)`` when ``gamma_bound``). The modulated feature
    is projected back and added as a LayerScale-gated residual, so the main detection path is
    preserved and the branch starts as a near-identity (``alpha`` init ~0).

    The prior conv (mask -> per-group ``gamma``) is shared across PAN scales since its input is
    always the 1-channel mask; ``proj_in`` / ``proj_out`` are per-scale because PAN channel counts
    differ. ``forward`` returns the residual increment ``alpha * dP`` of shape ``(B, C, H, W)``;
    the caller adds it to the PAN feature and applies mask-dropout, mirroring the bias path.
    """

    def __init__(
        self,
        pan_channels: list[int],
        num_groups: int = 16,
        group_dim: int = 16,
        prior_mid: int = 32,
        alpha_init: float = 1e-4,
        gamma_bound: bool = False,
    ):
        super().__init__()
        self.num_groups = int(num_groups)
        self.group_dim = int(group_dim)
        self.c_mod = self.num_groups * self.group_dim
        self.gamma_bound = bool(gamma_bound)
        # Prior -> per-group spatial scale gamma. Shared across scales (input is the 1-ch mask).
        self.prior_conv = nn.Sequential(
            nn.Conv2d(1, prior_mid, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(prior_mid, self.num_groups, 3, padding=1),
        )
        # Per-scale projections into / out of the modulation space (PAN channels differ per scale).
        self.proj_in = nn.ModuleList(Conv(c, self.c_mod, k=1) for c in pan_channels)
        self.proj_out = nn.ModuleList(nn.Conv2d(self.c_mod, c, 1) for c in pan_channels)
        # LayerScale: per-scale, per-channel residual gate, init ~0 so the branch starts inert.
        self.alpha = nn.ParameterList(
            nn.Parameter(torch.full((c,), float(alpha_init))) for c in pan_channels
        )

    def forward(self, feat: torch.Tensor, mask: torch.Tensor, scale_idx: int) -> torch.Tensor:
        """Return the residual increment ``(B, C, H, W)`` to add to ``feat``.

        Args:
            feat: PAN feature ``(B, C, H, W)`` at scale ``scale_idx``.
            mask: prior ``(B, 1, H, W)`` already resized to ``feat``'s spatial size.
            scale_idx: index into the per-scale projections / LayerScale.

        Returns:
            Residual increment ``(B, C, H, W)``; ~0 at init via the LayerScale gate.
        """
        b, _, h, w = feat.shape
        gamma = self.prior_conv(mask)  # (B, num_groups, H, W)
        # Bounded gate 1+tanh(gamma) in (0, 2) (multiplicative suppress/amplify, training-stable)
        # vs raw 1+gamma (unbounded). gamma_bound selects between them.
        scale = (1.0 + torch.tanh(gamma)) if self.gamma_bound else (1.0 + gamma)
        v = self.proj_in[scale_idx](feat)  # (B, c_mod, H, W)
        v = v.view(b, self.num_groups, self.group_dim, h, w)
        v = v * scale.unsqueeze(2)  # grouped FiLM scale
        v = v.reshape(b, self.c_mod, h, w)
        d = self.proj_out[scale_idx](v)  # (B, C, H, W)
        return self.alpha[scale_idx].view(1, -1, 1, 1) * d


def _sincos_pos2d(h: int, w: int, dim: int, device, dtype) -> torch.Tensor:
    """Fixed (non-learnable) 2D sinusoidal positional encoding, shape ``(1, dim, h, w)``.

    DETR-style: half the channels encode the row (y) coordinate, half the column (x), each via
    geometric-frequency sin/cos. Parameter-free and resolution-agnostic (computed from the formula).
    """
    d4 = dim // 4
    omega = 1.0 / (10000.0 ** (torch.arange(d4, device=device, dtype=torch.float32) / d4))  # (d4,)
    y = torch.arange(h, device=device, dtype=torch.float32)[:, None] * omega[None, :]  # (h, d4)
    x = torch.arange(w, device=device, dtype=torch.float32)[:, None] * omega[None, :]  # (w, d4)
    pe_y = torch.cat([y.sin(), y.cos()], dim=1)  # (h, dim//2)
    pe_x = torch.cat([x.sin(), x.cos()], dim=1)  # (w, dim//2)
    pe = torch.cat([pe_y[:, None, :].expand(h, w, dim // 2),
                    pe_x[None, :, :].expand(h, w, dim // 2)], dim=2)  # (h, w, dim)
    return pe.permute(2, 0, 1).unsqueeze(0).to(dtype)  # (1, dim, h, w)


class QueryFiLMFusion(nn.Module):
    """Query-based grouped-FiLM modulation of the P3 feature (deployable, ONNX-clean).

    From P3 ``(B, C, H, W)`` and a 1-channel prior heatmap, a conv encoder produces ``K`` query
    attention maps ``A`` and a ``query_dim`` feature map ``F``. Each query masked-average-pools its
    own feature ``Q_k``, predicts an objectness, and emits group-wise FiLM params ``(gamma_k,
    beta_k)``. The params are written back to space weighted by ``A_k * sigmoid(obj_k)`` and applied
    as grouped FiLM: ``P_out = P_group * (1 + alpha*gamma) + alpha*beta``.

    Identity at init: ``alpha`` is a scalar parameter init 0 and the FiLM MLP's last layer is
    zero-init, so the returned increment is bit-exact zero at start (vanilla YOLO).

    The forward returns the *increment* (like ``HeatmapFiLMFusion``) so the caller does ``p + delta``
    and applies mask-dropout. The deployable path uses only Conv/Gelu/Sigmoid/Reshape/Transpose/
    ReduceSum/Div/Clip/MatMul/Gemm/Mul/Add — no Hungarian, top-k, or GT rendering (those are
    training-only, in ``query_film_loss``).
    """

    def __init__(
        self,
        p3_channels: int,
        num_queries_k: int = 16,
        query_dim: int = 128,
        num_groups: int = 16,
        enc_mid: int = 64,
        film_mid: int = 64,
        alpha_init: float = 0.0,
        softmax_attn: bool = False,
        pos_enc: bool = False,
    ):
        super().__init__()
        assert p3_channels % num_groups == 0, (
            f"p3_channels ({p3_channels}) must be divisible by num_groups ({num_groups})"
        )
        self.c = int(p3_channels)
        self.k = int(num_queries_k)
        self.d = int(query_dim)
        self.g = int(num_groups)
        # Slot-attention-style normalization: when on, attention competes across queries per
        # pixel (softmax over the K+1 axis, with a null/background sink slot that absorbs
        # unclaimed pixels), so two queries can't both own the same region. When off, each
        # query is an independent sigmoid (queries may overlap; v0 behavior).
        self.softmax_attn = bool(softmax_attn)
        # DETR-style fixed 2D sinusoidal pos-enc added to the attention key (not the pooled value),
        # so each query's learnable attn vector can specialize by position as well as by content.
        self.pos_enc = bool(pos_enc)
        if self.pos_enc:
            assert self.d % 4 == 0, f"query_dim ({self.d}) must be divisible by 4 for 2D sincos pos-enc"
        attn_out = self.k + 1 if self.softmax_attn else self.k  # +1 = null/background slot
        self.enc = nn.Sequential(
            nn.Conv2d(self.c + 1, enc_mid, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(enc_mid, self.d, 3, padding=1),
        )
        self.attn = nn.Conv2d(self.d, attn_out, 1)  # query (+null) attention logits
        self.obj = nn.Linear(self.d, 1)  # per-query objectness logit
        self.film_mlp = nn.Sequential(
            nn.Linear(self.d, film_mid),
            nn.GELU(),
            nn.Linear(film_mid, 2 * self.g),
        )
        # Zero-init the FiLM head so gamma/beta start at 0 (second identity guarantee).
        nn.init.zeros_(self.film_mlp[-1].weight)
        nn.init.zeros_(self.film_mlp[-1].bias)
        # Scalar LayerScale, init 0 -> increment is exactly 0 at start.
        self.alpha = nn.Parameter(torch.full((1,), float(alpha_init)))
        # Learnable softmax temperature (tau = exp(log_tau), init 1.0); only used when softmax_attn.
        self.log_tau = nn.Parameter(torch.zeros(1)) if self.softmax_attn else None
        # Eval-only debug hook (query knockout): if set to a list of query indices, only those
        # queries' write-back contributes to the FiLM (gamma, beta); the rest are zeroed. None =
        # all queries (no-op). Not used in training/export — purely for diagnostics.
        self._keep_queries: list[int] | None = None

    def forward(self, p3: torch.Tensor, heatmap: torch.Tensor, return_aux: bool = False):
        """Return the residual increment ``(B, C, H, W)`` for the P3 feature.

        Args:
            p3: P3 PAN feature ``(B, C, H, W)``.
            heatmap: 1-channel prior ``(B, 1, H, W)`` at P3 resolution.
            return_aux: also return ``{"A", "attn_logits", "obj_logits"}`` for the training loss.

        Returns:
            ``delta`` (``(B, C, H, W)``), or ``(delta, aux)`` when ``return_aux``.
        """
        b, _, h, w = p3.shape
        hw = h * w
        feat = self.enc(torch.cat([p3, heatmap], dim=1))  # (B, D, H, W)
        # DETR-style: pos-enc enters only the attention key, not the pooled value (ff stays feat).
        key = feat + _sincos_pos2d(h, w, self.d, feat.device, feat.dtype) if self.pos_enc else feat
        logits = self.attn(key)  # (B, K or K+1, H, W)
        if self.softmax_attn:
            # Compete across the K real queries + 1 null slot, then drop the null slot. The
            # leftover (null) mass lets background pixels avoid being forced onto a query.
            tau = self.log_tau.exp().clamp_min(1e-2)
            a = (logits / tau).softmax(dim=1)[:, : self.k]  # (B, K, H, W)
            attn_logits = logits[:, : self.k]  # real-query logits for the aux loss
            fg_pred = a.sum(dim=1)  # (B, H, W) foreground occupancy = 1 - null prob (fg/bg loss)
        else:
            a = logits.sigmoid()  # (B, K, H, W), independent per-query (v0)
            attn_logits = logits
            fg_pred = None  # no null slot without softmax -> no fg/bg supervision
        af = a.reshape(b, self.k, hw)  # (B, K, HW)
        ff = feat.reshape(b, self.d, hw).transpose(1, 2)  # (B, HW, D)
        q = torch.bmm(af, ff) / af.sum(2, keepdim=True).clamp_min(1.0)  # (B, K, D) masked-avg-pool
        obj_logits = self.obj(q).squeeze(-1)  # (B, K)
        objs = obj_logits.sigmoid()  # (B, K)
        gb = self.film_mlp(q)  # (B, K, 2G), == 0 at init
        gamma_k, beta_k = gb[..., : self.g], gb[..., self.g :]  # (B, K, G)
        weight = (a * objs[..., None, None]).reshape(b, self.k, hw)  # (B, K, HW)
        if self._keep_queries is not None:  # eval-only query knockout (diagnostics)
            keep = torch.zeros(self.k, device=weight.device, dtype=weight.dtype)
            keep[self._keep_queries] = 1.0
            weight = weight * keep.view(1, self.k, 1)
        gamma = torch.bmm(gamma_k.transpose(1, 2), weight).reshape(b, self.g, h, w)  # (B, G, H, W)
        beta = torch.bmm(beta_k.transpose(1, 2), weight).reshape(b, self.g, h, w)  # (B, G, H, W)
        pg = p3.reshape(b, self.g, self.c // self.g, h, w)  # (B, G, C/G, H, W)
        scale = (self.alpha * gamma).unsqueeze(2)  # (B, G, 1, H, W)
        shift = (self.alpha * beta).unsqueeze(2)  # (B, G, 1, H, W)
        delta = (scale * pg + shift).reshape(b, self.c, h, w)  # increment, == 0 at init
        if return_aux:
            return delta, {"A": a, "attn_logits": attn_logits, "obj_logits": obj_logits, "fg_pred": fg_pred}
        return delta

    def extra_repr(self) -> str:
        return (f"c={self.c}, k={self.k}, d={self.d}, g={self.g}, "
                f"softmax_attn={self.softmax_attn}, pos_enc={self.pos_enc}")


class SegBranch(nn.Module):
    """Lightweight semantic-segmentation head that predicts a 1-channel anomaly heatmap.

    Consumes the P3 and P4 PAN features and emits per-pixel logits at P3 resolution
    (e.g. 80x80 for 640 input). A P4 auxiliary head provides deep supervision during
    training.

    When ``prior_cond`` is set, a 1-channel prior heatmap is concatenated onto each scale's
    feature (resized to its H×W) before the conv, turning the branch into a prior-conditioned
    denoiser/refiner: it cleans a noisy input prior toward the GT mask. ``forward(x, prior=None)``
    with ``prior=None`` reproduces the prior-free behavior (and a ``prior_cond`` head fed
    ``None`` falls back to a zero prior, i.e. segment-from-features). Concat + interpolate keep
    the deployable forward ONNX-clean.
    """

    def __init__(self, ch: tuple, nc: int = 1, c_mid: int | None = None, prior_cond: bool = False):
        super().__init__()
        self.nc = nc
        self.prior_cond = bool(prior_cond)
        extra = 1 if self.prior_cond else 0  # +1 input channel for the concatenated prior
        c_mid = ch[0] if c_mid is None else c_mid
        self.classifier = nn.Sequential(Conv(ch[0] + extra, c_mid, 3), nn.Conv2d(c_mid, nc, 1))
        self.aux_head = (
            nn.Sequential(Conv(ch[1] + extra, c_mid, 3), nn.Conv2d(c_mid, nc, 1)) if len(ch) > 1 else None
        )

    def _cat_prior(self, feat: torch.Tensor, prior: torch.Tensor | None) -> torch.Tensor:
        """Concat the 1-channel prior (resized to ``feat``'s H×W) onto ``feat`` when prior_cond."""
        if not self.prior_cond:
            return feat
        if prior is None:  # prior_cond head with no prior -> zero prior (segment from features)
            prior = feat.new_zeros(feat.shape[0], 1, feat.shape[2], feat.shape[3])
        elif prior.shape[2:] != feat.shape[2:]:
            prior = F.interpolate(prior, size=feat.shape[2:], mode="bilinear", align_corners=False)
        return torch.cat([feat, prior], dim=1)

    def forward(self, x: list[torch.Tensor], prior: torch.Tensor | None = None):
        logits = self.classifier(self._cat_prior(x[0], prior))
        if self.training and self.aux_head is not None:
            return logits, self.aux_head(self._cat_prior(x[1], prior))
        return logits


def binary_seg_loss(
    logits: torch.Tensor, target: torch.Tensor, aux_logits: torch.Tensor | None = None, aux_weight: float = 0.4
) -> torch.Tensor:
    """BCE + soft-Dice loss for a single-channel anomaly heatmap."""
    if target.shape[2:] != logits.shape[2:]:
        target = F.interpolate(target, size=logits.shape[2:], mode="nearest")
    target = target.to(logits.dtype)
    bce = F.binary_cross_entropy_with_logits(logits, target)
    prob = logits.sigmoid()
    inter = (prob * target).sum(dim=(1, 2, 3))
    card = prob.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3))
    dice = (1.0 - (2.0 * inter + 1.0) / (card + 1.0)).mean()
    loss = bce + dice
    if aux_logits is not None:
        aux_t = target
        if aux_t.shape[2:] != aux_logits.shape[2:]:
            aux_t = F.interpolate(target, size=aux_logits.shape[2:], mode="nearest")
        loss = loss + aux_weight * F.binary_cross_entropy_with_logits(aux_logits, aux_t)
    return loss


def query_film_loss(
    a: torch.Tensor,
    attn_logits: torch.Tensor,
    obj_logits: torch.Tensor,
    gt_masks: list[torch.Tensor],
    fg_pred: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    """Training-only supervision for QueryFiLM queries (Hungarian-matched to GT instances).

    Computes four scalar losses:
      - ``mask``: Dice + BCE between each matched query map and its GT instance mask.
      - ``obj``: BCEWithLogits on objectness (matched query -> 1, unmatched -> 0).
      - ``overlap``: mean off-diagonal pairwise overlap of the query maps (collapse guard).
      - ``fg``: BCE + Dice between foreground occupancy ``Σ_k A_k`` and the GT-instance union,
        i.e. background pixels are pushed onto the null slot (``Σ_k A_k -> 0``) and foreground onto
        the real queries. Only meaningful under softmax-over-(K+1); ``0`` when ``fg_pred is None``.

    Args:
        a: query attention maps ``(B, K, H, W)`` in [0, 1].
        attn_logits: raw query logits ``(B, K, H, W)`` (BCE uses logits for stability).
        obj_logits: per-query objectness logits ``(B, K)``.
        gt_masks: length-B list, each ``(N_b, H, W)`` per-instance GT gauss masks in [0, 1].
        fg_pred: foreground occupancy ``Σ_k A_k`` ``(B, H, W)`` in [0, 1] (softmax-attn only), or None.

    Returns:
        dict with scalar tensors ``{"mask", "obj", "overlap", "fg"}``.
    """
    from scipy.optimize import linear_sum_assignment

    b, k, h, w = a.shape
    hw = h * w
    device = a.device
    af = a.reshape(b, k, hw)  # (B, K, HW)

    # Overlap: pairwise inner product of query maps, normalized by HW, off-diagonal mean.
    gram = torch.bmm(af, af.transpose(1, 2)) / hw  # (B, K, K)
    eye = torch.eye(k, device=device, dtype=gram.dtype)[None]
    denom = max(k * (k - 1), 1)
    overlap = ((gram * (1.0 - eye)).sum(dim=(1, 2)) / denom).mean()

    obj_target = torch.zeros(b, k, device=device, dtype=obj_logits.dtype)
    mask_terms: list[torch.Tensor] = []
    for i in range(b):
        gt = gt_masks[i]  # (N_b, H, W)
        n = gt.shape[0]
        if n == 0:
            continue
        gt = gt.to(device=device, dtype=af.dtype)
        # Align GT mask resolution to the query maps (P3 grid). They match at the
        # production imgsz (640 -> P3 80x80 == mask_size); resize guards other sizes.
        if gt.shape[-2:] != (h, w):
            gt = F.interpolate(gt.unsqueeze(0), size=(h, w), mode="bilinear", align_corners=False).squeeze(0)
        gtf = gt.reshape(n, hw)  # (N, HW)
        # Soft-Dice cost between every (query, gt) pair: (K, N).
        inter = af[i] @ gtf.t()  # (K, N)
        card = af[i].sum(1, keepdim=True) + gtf.sum(1)[None]  # (K, N)
        dice = 1.0 - (2.0 * inter + 1.0) / (card + 1.0)  # (K, N), lower = better match
        # .float() guards against bf16/fp16 under autocast (numpy has no bfloat16).
        rows, cols = linear_sum_assignment(dice.detach().float().cpu().numpy())
        rows = torch.as_tensor(rows, device=device, dtype=torch.long)
        cols = torch.as_tensor(cols, device=device, dtype=torch.long)
        obj_target[i, rows] = 1.0
        # Matched-pair mask loss: Dice (on probs) + BCE (on logits).
        dice_m = dice[rows, cols]  # (M,)
        bce_m = F.binary_cross_entropy_with_logits(
            attn_logits[i, rows].reshape(len(rows), hw), gtf[cols], reduction="none"
        ).mean(dim=1)  # (M,)
        mask_terms.append(dice_m + bce_m)

    obj_loss = F.binary_cross_entropy_with_logits(obj_logits, obj_target)
    if mask_terms:
        mask_loss = torch.cat(mask_terms).mean()
    else:
        mask_loss = torch.zeros((), device=device, dtype=af.dtype)

    # Foreground/background term: push Σ_k A_k toward the GT-instance union (background -> null).
    if fg_pred is not None:
        fg_t = torch.zeros_like(fg_pred)  # (B, H, W); stays 0 for good (no-instance) images
        for i in range(b):
            gt = gt_masks[i]
            if gt.shape[0] == 0:
                continue
            gt = gt.to(device=device, dtype=fg_pred.dtype)
            if gt.shape[-2:] != (h, w):
                gt = F.interpolate(gt.unsqueeze(0), size=(h, w), mode="bilinear", align_corners=False).squeeze(0)
            fg_t[i] = gt.max(0).values  # union of per-instance masks
        fg_p = fg_pred.clamp(1e-6, 1.0 - 1e-6)
        # Manual BCE on probabilities (F.binary_cross_entropy is banned under AMP autocast).
        bce = -(fg_t * fg_p.log() + (1.0 - fg_t) * (1.0 - fg_p).log()).mean()
        inter = (fg_pred * fg_t).sum(dim=(1, 2))
        card = fg_pred.sum(dim=(1, 2)) + fg_t.sum(dim=(1, 2))
        dice = (1.0 - (2.0 * inter + 1.0) / (card + 1.0)).mean()
        fg_loss = bce + dice
    else:
        fg_loss = torch.zeros((), device=device, dtype=af.dtype)
    return {"mask": mask_loss, "obj": obj_loss, "overlap": overlap, "fg": fg_loss}


class BackboneMemoryBank(nn.Module):
    """Memory-bank anomaly heatmap from backbone features (v1 ADMBHead inference logic).

    Stores L2-normalised normal-image backbone features. During inference, scores
    each spatial position via Noisy-OR cosine similarity against the bank, producing a
    (B, 1, H, W) heatmap that feeds into HeatmapBiasFusion as a prior.

    Two modes controlled by ``update``:
      - ``True`` (build): ``forward()`` returns zeros — bank not yet frozen.
      - ``False`` (inference): ``forward()`` returns anomaly scores in [0, 1].

    Calibration modes (``calibrate`` parameter):

      - ``"auto"`` (default): sample from the full accumulated bank and calibrate β
        so that the 90th-percentile normal feature scores ``calibration_target_score``.
        Coreset is skipped in this mode.
      - ``"compactness"``: first subsample via greedy k-centre coreset, then measure
        local neighbour density (compactness) on the coreset to calibrate β.
        Scores naturally map to [0, 1] because compactness reflects the true normal
        manifold tightness.
    """

    def __init__(
        self,
        temperature: float = 3.0,
        K: int = 5,
        max_bank_size: int | None = None,
        calibration_target_score: float = 0.2,
        calibration_target_quantile: float = 0.95,
        proj_dim: int = 0,
        hmap_stretch_strength: float = 0.0,
        holdout_max: int = 5000,
    ):
        super().__init__()
        self.temperature = float(temperature)
        self.K = int(K)
        self.max_bank_size = max_bank_size
        self.calibration_target_score = float(calibration_target_score)
        self.calibration_target_quantile = float(calibration_target_quantile)
        self.hmap_stretch_strength = float(hmap_stretch_strength)
        self.holdout_max = int(holdout_max)
        self._calibrated = False
        self.proj_dim = int(proj_dim)
        self.register_buffer("memory_bank", torch.empty(0, 0), persistent=True)
        self.register_buffer("_proj_weight", torch.empty(0, 0), persistent=True)  # lazy-init random projection
        self.feature_dim: int | None = None
        self.update = True
        self._bb_layer_indices: list[int] = []
        self._bank_chunks: list[torch.Tensor] = []
        self._compactness: float | None = None  # normal-manifold tightness from coreset
        self._threshold: float | None = None     # sigmoid threshold in d_norm space
        self.score_chunk_elems = 1 << 27  # max elements per similarity slice in _anomaly_scores

    @property
    def built(self) -> bool:
        return not self.update

    def __setstate__(self, state):
        """Backfill attributes added after the checkpoint was saved."""
        super().__setstate__(state)
        if not hasattr(self, "proj_dim"):
            self.proj_dim = 0
        if "_proj_weight" not in self._buffers:
            self.register_buffer("_proj_weight", torch.empty(0, 0), persistent=True)
        if not hasattr(self, "_compactness"):
            self._compactness = None
        if not hasattr(self, "_threshold"):
            self._threshold = None
        if not hasattr(self, "calibration_target_quantile"):
            self.calibration_target_quantile = 0.95
        if not hasattr(self, "hmap_stretch_strength"):
            self.hmap_stretch_strength = 0.0
        if not hasattr(self, "holdout_max"):
            self.holdout_max = 5000

    def _apply(self, fn, recurse=True):
        """Keep the de-buffered FIFO queue in sync with ``.to()``/``.float()``/``.half()``.

        ``init_queue``/``adopt_queue`` replace the registered buffer with a plain attribute
        (invisible to state_dict/ModelEMA/DDP); plain attributes are skipped by
        ``nn.Module._apply``, so a carried queue would otherwise stay on the old
        device/dtype after ``model.to(device)``.
        """
        module = super()._apply(fn, recurse)
        if "memory_bank" not in module._buffers and isinstance(module.memory_bank, torch.Tensor):
            module.memory_bank = fn(module.memory_bank)
        return module

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def load_bank(self, features: torch.Tensor) -> None:
        """Direct-set the memory bank from pre-extracted L2-normalised features [M, C]."""
        if features.numel() == 0:
            return
        if not torch.isfinite(features).all():
            raise ValueError(f"BackboneMemoryBank.load_bank: features contain NaN/Inf ({features.shape})")
        self.feature_dim = features.shape[1]
        self.memory_bank = F.normalize(features.to(self.memory_bank.device), p=2, dim=1)
        self._calibrated = True  # trigger lazy compactness/threshold recompute on next score
        self._compactness = None
        self._threshold = None

    def freeze_memory_bank(self) -> None:
        """Coreset-compress, calibrate via compactness + holdout, then freeze the bank."""
        if self._bank_chunks:
            self.memory_bank = torch.cat(self._bank_chunks, dim=0)
            self._bank_chunks.clear()
        mem = self._effective_bank()

        # Coreset subsample, collect holdout (features not selected into coreset)
        holdout = None
        if self.max_bank_size is not None and mem.shape[0] > self.max_bank_size:
            self.memory_bank, coreset_idx = self._coreset_subsample(
                mem, self.max_bank_size, return_indices=True)
            holdout_mask = torch.ones(mem.shape[0], dtype=torch.bool, device=mem.device)
            holdout_mask[coreset_idx] = False
            holdout = mem[holdout_mask]
        mem = self._effective_bank()

        if mem.shape[0] > 0:
            self._calibrate_compactness(mem)
            if holdout is not None and holdout.shape[0] > 0:
                if holdout.shape[0] > self.holdout_max:
                    idx = torch.randperm(holdout.shape[0], device=holdout.device)[:self.holdout_max]
                    holdout = holdout[idx]
                self._calibrate_threshold_from_holdout(holdout, mem)

        self.update = False

    def reset_memory_bank(self) -> None:
        """Clear the bank and return to build mode."""
        self.memory_bank = torch.empty(0, 0, device=self.memory_bank.device)
        self.feature_dim = None
        self._calibrated = False
        self._compactness = None
        self._threshold = None
        self.update = True
        self._bank_chunks: list[torch.Tensor] = []  # defer cat until freeze

    # ------------------------------------------------------------------
    # MoCo-style FIFO queue (training-time prior)
    # ------------------------------------------------------------------
    def init_queue(self, capacity: int, feature_dim: int, device=None) -> None:
        """Switch the bank to FIFO-queue mode for training-time priors.

        Replaces the registered ``memory_bank`` buffer with a plain-attribute tensor so it is
        invisible to ``state_dict``/``ModelEMA``/DDP buffer broadcast (which would otherwise
        EMA-blend unit vectors into garbage, crash on the shape change, or broadcast ~100 MB
        per step). Zero rows mark unfilled slots and are excluded by ``_effective_bank``.
        """
        dev = device if device is not None else self.memory_bank.device
        self._buffers.pop("memory_bank", None)
        self.memory_bank = torch.zeros(int(capacity), int(feature_dim), device=dev)
        self.feature_dim = int(feature_dim)
        self._queue_capacity = int(capacity)
        self._queue_ptr = 0
        self.update = False

    def adopt_queue(self, src: "BackboneMemoryBank") -> None:
        """Alias another bank's FIFO queue (shared storage, no copy).

        Points the EMA model's bank at the live model's queue so validation and checkpoint
        saves see the current contents. Safe because the queue is a plain attribute:
        ``ModelEMA.update`` (state_dict-based) never touches it.
        """
        self._buffers.pop("memory_bank", None)
        self.memory_bank = src.memory_bank
        self.feature_dim = src.feature_dim
        self.temperature = src.temperature
        self.K = src.K
        self._calibrated = True
        self.update = False

    @torch.no_grad()
    def enqueue(
        self,
        feat_dict: dict[int, torch.Tensor],
        exclude_mask: torch.Tensor | None = None,
        max_patches: int | None = None,
    ) -> int:
        """Push a batch of backbone patches into the FIFO queue, evicting the oldest.

        Args:
            feat_dict: Backbone features captured by the taps, as in ``accumulate_features``.
            exclude_mask: (B, 1, H, W) prior in [0, 1]; patches where it exceeds 0.05 (inside
                or near a GT box) are skipped so defect pixels never enter the bank.
            max_patches: Random subsample cap so the queue's refresh period spans
                ~capacity/max_patches steps instead of wrapping every batch.

        Returns:
            Number of patches enqueued.
        """
        if not feat_dict or getattr(self, "_queue_capacity", 0) <= 0:
            return 0
        fused = self._build_fused_feature(feat_dict)  # (B, C, H, W)
        c, h, w = fused.shape[1], fused.shape[2], fused.shape[3]
        if c != self.feature_dim:
            return 0
        flat = fused.permute(0, 2, 3, 1).reshape(-1, c).float()
        if exclude_mask is not None:
            m = F.interpolate(exclude_mask.to(device=fused.device, dtype=torch.float32), size=(h, w), mode="nearest")
            flat = flat[m.reshape(-1) < 0.05]
        if flat.shape[0] == 0:
            return 0
        if max_patches is not None and flat.shape[0] > max_patches:
            flat = flat[torch.randperm(flat.shape[0], device=flat.device)[:max_patches]]
        normed = F.normalize(flat, p=2, dim=1).to(self.memory_bank.device)
        n, cap, ptr = normed.shape[0], self._queue_capacity, self._queue_ptr
        if n >= cap:
            self.memory_bank.copy_(normed[:cap])
            self._queue_ptr = 0
            return cap
        end = ptr + n
        if end <= cap:
            self.memory_bank[ptr:end] = normed
        else:
            k = cap - ptr
            self.memory_bank[ptr:] = normed[:k]
            self.memory_bank[: n - k] = normed[k:]
        self._queue_ptr = end % cap
        return n

    def accumulate_features(self, feat_dict: dict[int, torch.Tensor]) -> None:
        """Extract and accumulate backbone features into the memory bank (build phase).

        Fused backbone features are L2-normalised per spatial position and appended
        to a chunk list; the full bank is materialised once in ``freeze_memory_bank``
        to avoid O(N²) reallocation from repeated ``torch.cat``.
        """
        if not feat_dict:
            return
        fused = self._build_fused_feature(feat_dict)  # (B, C, H, W)
        C, H, W = fused.shape[1], fused.shape[2], fused.shape[3]
        if self.feature_dim is None:
            self.feature_dim = C
        flat = fused.permute(0, 2, 3, 1).reshape(-1, C)  # [B*H*W, C]
        normed = F.normalize(flat, p=2, dim=1)
        self._bank_chunks.append(normed)

    def forward(self, feat_dict: dict[int, torch.Tensor]) -> torch.Tensor:
        """Produce (B, 1, H, W) anomaly heatmap from backbone features.

        During build mode or when the bank is empty, returns zeros.
        """
        b, device = self._resolve_batch_size(feat_dict)
        h = feat_dict[list(feat_dict.keys())[0]].shape[2] if feat_dict else 80
        w = feat_dict[list(feat_dict.keys())[0]].shape[3] if feat_dict else 80
        if self.update:
            return torch.zeros(b, 1, h, w, device=device)
        mem = self._effective_bank()
        if mem.shape[0] == 0:
            return torch.zeros(b, 1, h, w, device=device)
        fused = self._build_fused_feature(feat_dict)  # (B, C, H, W)
        if fused.shape[1] != self.feature_dim:
            return torch.zeros(b, 1, fused.shape[2], fused.shape[3], device=device)
        bh, bw = fused.shape[2], fused.shape[3]
        flat = fused.permute(0, 2, 3, 1).reshape(-1, self.feature_dim)
        scores = self._anomaly_scores(flat, mem)  # [B*H*W]
        hmap = scores.view(b, 1, bh, bw)
        s = self.hmap_stretch_strength
        if s:
            hmap = (hmap + s * hmap * hmap).clamp(0, 1)
        return hmap

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------
    @staticmethod
    def _resolve_batch_size(feat_dict: dict[int, torch.Tensor]) -> tuple[int, torch.device]:
        for v in feat_dict.values():
            return v.shape[0], v.device
        return 1, torch.device("cpu")

    def _effective_bank(self) -> torch.Tensor:
        """Return real memory-bank entries excluding zero-padding placeholders."""
        if self.update and self._bank_chunks:
            return torch.cat(self._bank_chunks, dim=0)
        mem = self.memory_bank
        if self.feature_dim is None or mem.numel() == 0 or mem.shape[0] == 0:
            return mem[:0]
        valid = mem.norm(dim=1) > 0
        return mem[valid]

    def _build_fused_feature(self, feat_dict: dict[int, torch.Tensor]) -> torch.Tensor:
        """Gather backbone features at configured layer indices, concat, optionally project down."""
        indices = self._bb_layer_indices
        if not indices:
            return feat_dict[list(feat_dict.keys())[0]]
        feats = [feat_dict[i] for i in indices if i in feat_dict]
        if not feats:
            return feat_dict[list(feat_dict.keys())[0]]
        target_size = feats[0].shape[-2:]
        aligned = [
            F.interpolate(f, size=target_size, mode="nearest") if f.shape[-2:] != target_size else f
            for f in feats
        ]
        fused = torch.cat(aligned, dim=1) if len(aligned) > 1 else aligned[0]
        # Random projection (Johnson-Lindenstrauss): fixed random matrix, ~preserves cosine structure.
        if self.proj_dim > 0 and fused.shape[1] > self.proj_dim:
            if self._proj_weight.numel() == 0:
                in_dim = fused.shape[1]
                # Random Gaussian projection: E[||Wx||^2] = ||x||^2 (Johnson-Lindenstrauss)
                w = torch.randn(in_dim, self.proj_dim, device=fused.device, dtype=torch.float32)
                self._proj_weight = w / (self.proj_dim ** 0.5)
            B, C, H, W = fused.shape
            orig_dtype = fused.dtype
            fused = fused.permute(0, 2, 3, 1).reshape(-1, C)
            fused = (fused.to(dtype=self._proj_weight.dtype) @ self._proj_weight).to(orig_dtype)
            fused = fused.reshape(B, H, W, -1).permute(0, 3, 1, 2)
        return fused

    def estimate_temperature(self) -> float:
        """Estimate β from the current bank WITHOUT modifying state (lightweight, for MoCo live estimate)."""
        import math

        mem = self._effective_bank()
        if mem.shape[0] < 2:
            return self.temperature
        with torch.no_grad():
            k = min(self.K, mem.shape[0])
            n_sample = min(512, mem.shape[0])
            idx = torch.randperm(mem.shape[0], device=mem.device)[:n_sample]
            sample = mem[idx]
            sim = sample @ mem.t()
            topk_sim = sim.topk(k=k, dim=1).values
            mean_topk = topk_sim.mean(dim=1)
            s_max = mean_topk.max().clamp(0.0, 1.0 - 1e-4).item()
            beta = -math.log(1.0 - self.calibration_target_score) / max(1.0 - s_max, 1e-6)
            beta = max(0.1, min(20.0, beta))
        return beta

    def _measure_compactness(self, mem: torch.Tensor) -> float:
        """Compute compactness from the bank without touching ``self.temperature``.

        Used for lazy restore after state_dict load and by ``_calibrate_compactness``.
        """
        with torch.no_grad():
            k = min(self.K, mem.shape[0])
            n_sample = min(512, mem.shape[0])
            idx = torch.randperm(mem.shape[0], device=mem.device)[:n_sample]
            sample = mem[idx]
            sim = sample @ mem.t()  # [n_sample, M]
            sim[torch.arange(n_sample, device=mem.device), idx] = -1.0
            topk_sim = sim.topk(k=k, dim=1).values
            local_density = topk_sim.mean(dim=1)
            return local_density.mean().clamp(0.0, 1.0 - 1e-4).item()

    def _calibrate_compactness(self, mem: torch.Tensor) -> None:
        """Measure compactness and calibrate sigmoid threshold in cosine space.

        Compactness = mean local cosine density on the coreset.  The sigmoid
        operates directly on cosine similarity so that the dynamic range is
        never compressed by the bank spread:

            psi = sigmoid(β × (cos − threshold_cos))

        threshold_cos = compactness − logit(1−target)/β.

        A normal query has cos ≈ compactness → psi ≈ 1−target → normal
        anomaly score ≈ target.  β is the user-controlled ``temperature``.
        """
        import math

        compactness = self._measure_compactness(mem)
        self._compactness = compactness
        beta = self.temperature
        t = self.calibration_target_score
        logit = math.log(max((1.0 - t) / max(t, 1e-6), 1e-6))
        self._threshold = compactness - logit / max(beta, 0.1)
        self._calibrated = True
        import logging
        logger = logging.getLogger(__name__)
        logger.debug(
            "BackboneMemoryBank: compactness=%.4f  β=%.3f  threshold_cos=%.4f  target=%.2f",
            compactness, beta, self._threshold, self.calibration_target_score,
        )

    def _score_with_threshold(self, features: torch.Tensor, mem: torch.Tensor,
                              threshold_cos: float, beta: float | None = None) -> torch.Tensor:
        """Compute anomaly scores with specific threshold & β (stateless, for calibration)."""
        if beta is None:
            beta = self.temperature
        q = F.normalize(features.view(-1, self.feature_dim), p=2, dim=1)
        k = min(self.K, mem.shape[0])
        n, m = q.shape[0], mem.shape[0]
        chunk = max(1, int(getattr(self, "score_chunk_elems", 1 << 27)) // max(m, 1))
        out = []
        for i in range(0, n, chunk):
            cos = q[i: i + chunk] @ mem.t()
            psi = torch.sigmoid(beta * (cos - threshold_cos))
            topk = psi.topk(k=k, dim=1).values
            log_prob = torch.log((1.0 - topk).clamp(min=1e-8)).mean(dim=1)
            out.append(torch.exp(log_prob))
        return torch.cat(out).clamp(0, 1)

    def _calibrate_threshold_from_holdout(self, holdout: torch.Tensor,
                                          mem: torch.Tensor) -> None:
        """Calibrate β & threshold so p95 of hold-out normal scores ≈ target.

        The user β is a sensitivity floor — it is never lowered.  For each
        candidate β, binary-searches ``threshold_cos`` to put p95 at
        ``calibration_target_score``, then picks the β that gives the tightest
        normal-score distribution (smallest p95−p5).
        """
        import math
        import logging
        logger = logging.getLogger(__name__)

        n_holdout = holdout.shape[0]
        target = self.calibration_target_score
        target_q = self.calibration_target_quantile  # e.g. 0.95 → 95% of normal scores ≤ target
        c = self._compactness
        beta0 = self.temperature

        # β candidates: user β as floor, log-spaced upward
        beta_candidates = [beta0]
        n_extra = 8
        for i in range(1, n_extra + 1):
            b = beta0 * (10 ** (i / n_extra))
            if b > 100:
                break
            beta_candidates.append(round(b, 4))

        # Pre-compute cosine matrix and top-K once (sigmoid is monotonic,
        # so the top-K indices of cos are identical to top-K of psi).
        q_feats = F.normalize(holdout.view(-1, self.feature_dim), p=2, dim=1)
        cos_mat = q_feats @ mem.t()  # [N_holdout, M]
        k = min(self.K, mem.shape[0])
        topk_cos = cos_mat.topk(k=k, dim=1).values  # [N_holdout, k] — only these matter

        # z-score for the target quantile (√2 · erf⁻¹(2q−1))
        z_q = math.sqrt(2) * torch.erfinv(
            torch.tensor(2.0 * target_q - 1.0)).item()

        def _scores_for(beta, thresh):
            """Compute anomaly scores from pre-computed top-K cos values."""
            psi = torch.sigmoid(beta * (topk_cos - thresh))
            log_prob = torch.log((1.0 - psi).clamp(min=1e-8)).mean(dim=1)
            return torch.exp(log_prob).clamp(0, 1)

        def _tail_stat(scores):
            """Gaussian tail: μ + z·σ."""
            return scores.mean().item() + z_q * scores.std().item()

        def _spread(scores):
            """Score spread: standard deviation (smaller = tighter normal distribution)."""
            return scores.std().item()

        def _find_thresh(beta):
            """Binary-search threshold so tail-stat ≈ target."""
            half_range = 3.0 / max(beta, 0.1)
            lo, hi = c - half_range, c + half_range
            s_lo = _tail_stat(_scores_for(beta, lo))
            s_hi = _tail_stat(_scores_for(beta, hi))
            for _ in range(5):
                if s_lo > target and lo > c - 5.0:
                    lo -= half_range
                    s_lo = _tail_stat(_scores_for(beta, lo))
                if s_hi < target and hi < c + 5.0:
                    hi += half_range
                    s_hi = _tail_stat(_scores_for(beta, hi))
            if not (s_lo <= target <= s_hi):
                return None, float("inf"), float("inf")
            for _ in range(20):
                mid = (lo + hi) / 2
                sm = _tail_stat(_scores_for(beta, mid))
                if sm > target:
                    hi = mid
                else:
                    lo = mid
            thresh = (lo + hi) / 2
            scores = _scores_for(beta, thresh)
            achieved = _tail_stat(scores)
            spread = _spread(scores)
            return thresh, achieved, spread

        best_beta, best_thresh, best_achieved, best_spread = beta0, c, float("inf"), float("inf")
        for beta in beta_candidates:
            thresh, achieved, spread = _find_thresh(beta)
            if thresh is not None and spread < best_spread:
                best_spread, best_achieved, best_beta, best_thresh = spread, achieved, beta, thresh

        old_beta = self.temperature
        self.temperature = best_beta
        self._threshold = best_thresh
        logger.debug(
            "BackboneMemoryBank: gauss calibration  n=%d  q=%.2f  z=%.4f  "
            "β %.3f→%.3f  thresh(formula)=%.4f→(holdout)=%.4f  "
            "target=%.2f→achieved=%.4f  spread=%.4f",
            n_holdout, target_q, z_q, old_beta, best_beta,
            c - math.log(max((1.0 - target) / max(target, 1e-6), 1e-6)) / max(old_beta, 0.1),
            best_thresh, target, best_achieved, best_spread,
        )

    def _anomaly_scores(self, features: torch.Tensor, mem: torch.Tensor | None = None) -> torch.Tensor:
        """Noisy-OR cosine anomaly scores ∈ [0, 1] for every spatial position.

        Args:
            features: [N, C] query features.
            mem: [M, C] memory bank. If None, uses ``self._effective_bank()``.

        Returns:
            [N] tensor of anomaly scores.
        """
        if mem is None:
            mem = self._effective_bank()
        if mem.numel() == 0 or mem.shape[0] == 0:
            return torch.full((features.shape[0],), 0.5, device=features.device)
        # Lazy recompute after state_dict load (compactness/threshold are not persisted)
        if self._compactness is None and self._calibrated:
            self._compactness = self._measure_compactness(mem)
            self._threshold = None  # force recompute below
        if self._threshold is None and self._compactness is not None:
            import math
            t = self.calibration_target_score
            logit = math.log(max((1.0 - t) / max(t, 1e-6), 1e-6))
            self._threshold = self._compactness - logit / max(self.temperature, 0.1)
        q = F.normalize(features.view(-1, self.feature_dim), p=2, dim=1)
        k = min(self.K, mem.shape[0])
        n, m = q.shape[0], mem.shape[0]
        chunk = max(1, int(getattr(self, "score_chunk_elems", 1 << 27)) // max(m, 1))
        use_sigmoid = self._threshold is not None and self._compactness is not None
        beta = self.temperature
        out = []
        for i in range(0, n, chunk):
            cos = q[i : i + chunk] @ mem.t()  # [chunk, M] cosine similarities
            if self.training:
                cos = cos.masked_fill(cos > 0.999, float("-inf"))
            if use_sigmoid:
                psi = torch.sigmoid(beta * (cos - self._threshold))  # cos-space sigmoid
            else:
                psi = torch.exp(-beta * (1.0 - cos))  # legacy exp, un-normalized
            topk = psi.topk(k=k, dim=1).values  # [chunk, k]
            log_prob = torch.log((1.0 - topk).clamp(min=1e-8)).mean(dim=1)
            out.append(torch.exp(log_prob))
        return torch.cat(out).clamp(0, 1)

    @staticmethod
    def _coreset_subsample(mem: torch.Tensor, max_size: int,
                           return_indices: bool = False):
        """Greedy k-center coreset on L2-normalised features using cosine distance.

        Complexity: O(max_size × M). Features must be L2-normalised.

        Batched greedy: selects ``batch_size`` farthest points per iteration and
        computes distances against all of them in one GEMM call, amortizing the
        large bank read over multiple centres and trading a small amount of
        greediness for a large wall-clock speedup on memory-bandwidth-limited
        devices (MPS / CPU).

        Args:
            mem: [M, C] L2-normalised feature bank.
            max_size: Target coreset size.
            return_indices: If True, also return the indices of selected rows.

        Returns:
            Coreset tensor, or (coreset, indices) if ``return_indices``.
        """
        from ultralytics.utils import TQDM

        M = mem.shape[0]
        if M <= max_size:
            idx = torch.arange(M, device=mem.device)
            return (mem, idx) if return_indices else mem
        device = mem.device
        BATCH = 64  # centres per GEMM call — amortizes bank reads
        dist = torch.full((M,), float("inf"), device=device, dtype=torch.float32)
        selected: list[int] = []
        mean = mem.mean(dim=0)
        mean = mean / mean.norm().clamp(min=1e-8)
        seed = int((mem @ mean).argmax().item())
        selected.append(seed)
        # seed distance so the first topk isn't random (all-inf)
        centre = mem[seed].unsqueeze(0)
        seed_cos = (mem @ centre.t()).squeeze(1)
        dist = (1.0 - seed_cos).clamp(min=0.0)
        n_needed = max_size - len(selected)
        pbar = TQDM(total=n_needed, desc="Coreset subsample", leave=False)
        while n_needed > 0:
            k = min(BATCH, max_size - len(selected))
            _, top_idx = dist.topk(k)
            selected.extend(top_idx.tolist())
            centres = mem[top_idx]                                      # [k, C]
            cos_sim = mem @ centres.t()                                 # [M, k]
            new_dist = (1.0 - cos_sim).clamp(min=0.0).min(dim=1).values  # [M]
            dist = torch.minimum(dist, new_dist)
            n_needed = max_size - len(selected)
            pbar.update(k)
        pbar.close()
        sel = torch.tensor(selected, device=device)
        return (mem[sel], sel) if return_indices else mem[sel]


class LearnedScorer(nn.Module):
    """Learned drop-in replacement for ``BackboneMemoryBank._anomaly_scores``.

    Maps each query patch's top-K cosine similarities against the normal-feature bank to an
    anomaly score in [0, 1] via a small MLP, instead of the fixed Noisy-OR formula
    ``exp(mean log(1 - exp(-beta(1-cos))))``. The bank enters only through top-K similarities,
    so the score is invariant to bank size (any ``M >= 1`` works; ``M < K`` is padded). Trained
    on GT defect masks with the backbone frozen/detached, so only this head learns to tell
    "normal variation" from "real defect". Same call signature as ``_anomaly_scores``
    (``features[N, C]``, ``mem[M, C]`` -> ``scores[N]``) so Phase B can swap it in directly.
    """

    def __init__(
        self,
        k: int = 9,
        hidden: int = 32,
        feature_dim: int | None = None,
        proj_dim: int = 0,
        train_self_match_mask: bool = True,
        self_match_thresh: float = 0.999,
        score_chunk_elems: int = 1 << 27,
    ):
        super().__init__()
        self.k = int(k)
        self.train_self_match_mask = bool(train_self_match_mask)
        self.self_match_thresh = float(self_match_thresh)
        self.score_chunk_elems = int(score_chunk_elems)
        # v2: learned projection g reshapes the metric (similarities are computed in g-space).
        # proj_dim=0 -> v1 (raw-feature cosine readout). proj_dim>0 needs feature_dim; when it equals
        # feature_dim the projection is IDENTITY-INITIALISED so v2 starts == v1 (raw cosine) and learns
        # a refinement (random init would scramble the geometry -> near-chance cosine).
        self.proj = nn.Linear(int(feature_dim), int(proj_dim)) if proj_dim > 0 else None
        if self.proj is not None and int(proj_dim) == int(feature_dim):
            nn.init.eye_(self.proj.weight)
            nn.init.zeros_(self.proj.bias)
        self.mlp = nn.Sequential(nn.Linear(self.k, hidden), nn.GELU(), nn.Linear(hidden, 1))

    def _embed(self, x: torch.Tensor) -> torch.Tensor:
        """Optional learned projection (v2), then L2-normalise -> unit vectors for cosine."""
        if self.proj is not None:
            x = self.proj(x)
        return F.normalize(x, p=2, dim=1)

    def topk_sims(self, features: torch.Tensor, mem: torch.Tensor) -> torch.Tensor:
        """Query patches -> sorted top-K cosine sims ``[N, k]`` in the (optionally projected) space.

        Both query and bank are embedded (v2: projected) and L2-normalised here, so the bank may be
        passed raw. For v1 (no projection) this is plain cosine; v1 may also cache the result.
        """
        q = self._embed(features)
        m = self._embed(mem)
        k = min(self.k, m.shape[0])
        n, mm = q.shape[0], m.shape[0]
        chunk = max(1, int(self.score_chunk_elems) // max(mm, 1))  # bound the [chunk, M] sim slice
        out = []
        for i in range(0, n, chunk):
            sim = q[i : i + chunk] @ m.t()  # [chunk, M]
            if self.training and self.train_self_match_mask:
                # A query finding its own copy in the bank (cos ~ 1) would leak a too-clean score.
                sim = sim.masked_fill(sim > self.self_match_thresh, float("-inf"))
            out.append(sim.topk(k=k, dim=1).values)  # [chunk, k] sorted descending
        return torch.cat(out)

    def score_from_topk(self, topk: torch.Tensor) -> torch.Tensor:
        """Sorted top-K sims ``[N, k']`` -> anomaly scores ``[N]`` in [0, 1] (pads/trims to k)."""
        k = topk.shape[1]
        if k < self.k:  # tiny bank: pad to fixed input width with the lowest available sim
            topk = torch.cat([topk, topk[:, -1:].expand(-1, self.k - k)], dim=1)
        elif k > self.k:
            topk = topk[:, : self.k]
        topk = topk.nan_to_num(neginf=-1.0).clamp(-1.0, 1.0)
        return torch.sigmoid(self.mlp(topk).squeeze(1))

    def forward(self, features: torch.Tensor, mem: torch.Tensor) -> torch.Tensor:
        """Score query patches against the bank: ``features[N, C]``, ``mem[M, C]`` -> ``scores[N]``."""
        if mem.numel() == 0 or mem.shape[0] == 0:
            return torch.full((features.shape[0],), 0.5, device=features.device, dtype=features.dtype)
        return self.score_from_topk(self.topk_sims(features, mem))


class FeatureDiscriminatorScorer(nn.Module):
    """SuperSimpleNet-style learned anomaly scorer — a normal-only complement to the
    ``BackboneMemoryBank`` Noisy-OR readout.

    Fits a small per-patch discriminator on the normal backbone features collected at support
    time: real normal patches are the negative class (label 0); synthetic anomalies (normal
    features + Gaussian noise in feature space, a la SimpleNet/SuperSimpleNet) are the positive
    class (label 1). ``forward(feat_dict)`` returns a ``(B, 1, H, W)`` heatmap of sigmoid
    discriminator scores in [0, 1] — same call site / output scale as ``BackboneMemoryBank.forward``,
    so the two can be fused (``prior_mode="heatmap_fused"``) or swapped (``"heatmap_learned"``).

    No nearest-neighbour at inference: the bank's stored normal features are consumed once to
    fit the discriminator, then only this small MLP runs. Built lazily on the first ``fit`` once
    the feature dimension is known.
    """

    def __init__(self, noise_std: float = 0.015, n_noise: int = 1, hidden: int = 64,
                 steps: int = 400, lr: float = 1e-3, batch: int = 4096,
                 adaptor: bool = True, seed: int = 0, noise_mode: str = "gaussian"):
        super().__init__()
        self.noise_std = float(noise_std)
        self.n_noise = int(n_noise)
        self.hidden = int(hidden)
        self.steps = int(steps)
        self.lr = float(lr)
        self.batch = int(batch)
        self.use_adaptor = bool(adaptor)
        self.seed = int(seed)
        self.noise_mode = str(noise_mode)  # "gaussian" or "mixup"
        self.feature_dim: int | None = None
        self.adaptor: nn.Linear | None = None
        self.mlp: nn.Sequential | None = None
        self._bb_layer_indices: list[int] = []
        self._fitted = False

    @property
    def fitted(self) -> bool:
        return self._fitted

    def _build(self, c: int, device) -> None:
        """Lazily create the (identity-init) adaptor + discriminator MLP for feature dim ``c``."""
        self.feature_dim = int(c)
        if self.use_adaptor:
            self.adaptor = nn.Linear(c, c)
            nn.init.eye_(self.adaptor.weight)
            nn.init.zeros_(self.adaptor.bias)
        self.mlp = nn.Sequential(nn.Linear(c, self.hidden), nn.GELU(), nn.Linear(self.hidden, 1))
        self.to(device)

    def _logits(self, x: torch.Tensor) -> torch.Tensor:
        """Per-patch discriminator logits ``[N]`` from L2-normalised features ``[N, C]``."""
        if self.adaptor is not None:
            x = self.adaptor(x)
        return self.mlp(x).squeeze(-1)

    def fit(self, normal_feats: torch.Tensor) -> None:
        """Train the discriminator on normal patch features ``[N, C]`` (any scale; re-normalised).

        Negative = normal unit features; positive depends on ``noise_mode``:

        * ``"gaussian"`` — normal + per-component Gaussian noise (SimpleNet-style).
        * ``"mixup"`` — convex blend of two *different* normal features, then L2-normalised.
          Creates synthetic anomalies that lie *between* normal clusters on the unit sphere,
          which is more effective when the normal manifold is sparse (e.g. objects with
          diverse poses, where Gaussian noise around a single point stays too close to it).

        BCE-with-logits, ``steps`` minibatches. Backbone is untouched.
        """
        if normal_feats.shape[0] < 2:
            return
        dev = normal_feats.device
        g = torch.Generator(device="cpu").manual_seed(self.seed)
        bce = nn.BCEWithLogitsLoss()
        with torch.inference_mode(False), torch.enable_grad():
            feats = F.normalize(normal_feats.clone().float(), p=2, dim=1)
            n, c = feats.shape
            self._build(c, dev)
            opt = torch.optim.Adam((p for p in self.parameters() if p.requires_grad), lr=self.lr)
            self.train()
            bs = min(self.batch, n)
            from ultralytics.utils.tqdm import TQDM
            pbar = TQDM(range(self.steps), desc=f"Training scorer ({n} feats, bs={bs})", unit="step")
            for _ in pbar:
                idx = torch.randint(0, n, (bs,), generator=g).to(dev)
                x = feats[idx]  # normal (unit)
                if self.noise_mode == "mixup":
                    # Blend each normal with a DIFFERENT randomly chosen normal,
                    # then L2-normalise → synthetic anomaly ON the unit sphere
                    # that sits between two normal clusters.
                    idx2 = torch.randint(0, n, (bs * self.n_noise,), generator=g).to(dev)
                    alpha = torch.rand(bs * self.n_noise, 1, generator=g).to(dev)
                    xa_raw = alpha * x.repeat(self.n_noise, 1) + (1 - alpha) * feats[idx2]
                    xa = F.normalize(xa_raw, p=2, dim=1)
                else:
                    noise = (torch.randn(bs * self.n_noise, c, generator=g) * self.noise_std).to(dev)
                    xa_raw = x.repeat(self.n_noise, 1) + noise
                    xa = F.normalize(xa_raw, p=2, dim=1)
                ln, la = self._logits(x), self._logits(xa)
                loss = bce(ln, torch.zeros_like(ln)) + bce(la, torch.ones_like(la))
                opt.zero_grad()
                loss.backward()
                opt.step()
                pbar.set_postfix(loss=f"{loss.item():.4f}")
        self.eval()
        self._fitted = True

    def _build_fused_feature(self, feat_dict: dict[int, torch.Tensor]) -> torch.Tensor:
        """Gather configured backbone layers, upsample to the first scale, concat (matches the bank)."""
        indices = self._bb_layer_indices
        if not indices:
            return feat_dict[list(feat_dict.keys())[0]]
        feats = [feat_dict[i] for i in indices if i in feat_dict]
        if not feats:
            return feat_dict[list(feat_dict.keys())[0]]
        target = feats[0].shape[-2:]
        aligned = [F.interpolate(f, size=target, mode="nearest") if f.shape[-2:] != target else f for f in feats]
        return torch.cat(aligned, dim=1) if len(aligned) > 1 else aligned[0]

    @torch.no_grad()
    def forward(self, feat_dict: dict[int, torch.Tensor]) -> torch.Tensor:
        """Produce a ``(B, 1, H, W)`` anomaly heatmap in [0, 1] (zeros if unfitted/empty)."""
        if not feat_dict:
            return torch.zeros(1, 1, 80, 80)
        fused = self._build_fused_feature(feat_dict)
        b, c, h, w = fused.shape
        if not self._fitted or c != self.feature_dim:
            return torch.zeros(b, 1, h, w, device=fused.device)
        flat = F.normalize(fused.permute(0, 2, 3, 1).reshape(-1, c).float(), p=2, dim=1)
        score = torch.sigmoid(self._logits(flat))
        return score.view(b, 1, h, w)


if __name__ == "__main__":
    # Smoke test: render, fusion init=0 passthrough, fusion with learned beta bounded,
    # broadcast onto a C-channel PAN feature.
    B, H = 4, 80
    renderer = BboxMaskRenderer(mask_size=H, mode="rect")
    bboxes = torch.tensor(
        [
            [0.25, 0.25, 0.20, 0.20],
            [0.75, 0.75, 0.30, 0.10],
            [0.50, 0.50, 0.10, 0.10],
        ]
    )
    batch_idx = torch.tensor([0, 0, 2], dtype=torch.long)
    mask = renderer(bboxes, batch_idx, B)
    assert mask.shape == (B, 1, H, H)
    assert mask[1].sum().item() == 0.0  # image with no bboxes

    # Gauss with a [lo, hi] range: peak ~1, random per-bbox while training, deterministic at eval.
    grender = BboxMaskRenderer(mask_size=H, mode="gauss", sigma_factor=[0.25, 0.75])
    assert (grender.sigma_lo, grender.sigma_hi) == (0.25, 0.75)
    grender.train()
    torch.manual_seed(0)
    g1 = grender(bboxes, batch_idx, B)
    torch.manual_seed(1)
    g2 = grender(bboxes, batch_idx, B)
    assert 0.9 < g1[0].max().item() <= 1.0 + 1e-6, "gauss peak should approach 1.0 at a box center"
    assert not torch.allclose(g1, g2), "training renders should vary with random sigma factor"
    grender.eval()
    e1 = grender(bboxes, batch_idx, B)
    e2 = grender(bboxes, batch_idx, B)
    assert torch.allclose(e1, e2), "eval renders should be deterministic (midpoint sigma)"
    print("BboxMaskRenderer gauss range [0.25,0.75] OK (train random, eval deterministic).")

    fusion = HeatmapBiasFusion(num_scales=3)
    # beta init=0 -> bias is exactly zero for every scale
    for s in range(3):
        bias = fusion(mask, s)
        assert bias.shape == (B, 1, H, H)
        assert bias.abs().max().item() == 0.0, f"scale {s} bias not zero at init"
    print("HeatmapBiasFusion init=0 passthrough OK.")

    # With beta set non-zero, output is bounded in [-beta, +beta]
    with torch.no_grad():
        fusion.beta.fill_(1.5)
    bias = fusion(mask, 0)
    assert bias.abs().max().item() <= 1.5 + 1e-6, "bias exceeded beta after tanh"
    print(f"HeatmapBiasFusion beta=1.5 bounded OK (max abs = {bias.abs().max().item():.4f}).")

    # Broadcast onto a C-channel PAN feature: (B, 1, H, W) + (B, C, H, W) -> (B, C, H, W).
    p = torch.randn(B, 256, H, H)
    p_fused = p + fusion(mask, 0)
    assert p_fused.shape == p.shape
    print(f"Broadcast OK: P (B,256,H,W) + bias (B,1,H,W) -> {tuple(p_fused.shape)}.")

    # Resize-per-scale smoke
    mask_p4 = F.interpolate(mask, size=(40, 40), mode="bilinear", align_corners=False)
    mask_p5 = F.interpolate(mask, size=(20, 20), mode="bilinear", align_corners=False)
    assert fusion(mask_p4, 1).shape == (B, 1, 40, 40)
    assert fusion(mask_p5, 2).shape == (B, 1, 20, 20)
    print("HeatmapBiasFusion multi-scale OK.")

    print("\n--- HeatmapSoftFusion ---")
    fusion_soft = HeatmapSoftFusion(num_scales=3)
    # beta init=0 -> bias zero at init
    for s in range(3):
        bias = fusion_soft(mask, s)
        assert bias.shape == (B, 1, H, H)
        assert bias.abs().max().item() == 0.0, f"scale {s} soft bias not zero at init"
    print("HeatmapSoftFusion init=0 passthrough OK.")

    with torch.no_grad():
        fusion_soft.beta.fill_(1.5)
    bias = fusion_soft(mask, 0)
    assert bias.abs().max().item() <= 1.5 + 1e-6, "soft bias exceeded beta after tanh"
    print(f"HeatmapSoftFusion beta=1.5 bounded OK (max abs = {bias.abs().max().item():.4f}).")

    assert fusion_soft(mask_p4, 1).shape == (B, 1, 40, 40)
    assert fusion_soft(mask_p5, 2).shape == (B, 1, 20, 20)
    print("HeatmapSoftFusion multi-scale OK.")

    print("\n--- HeatmapFiLMFusion ---")
    pan_ch = [256, 512, 512]  # yolo26m PAN channel counts (differ per scale)
    film = HeatmapFiLMFusion(pan_channels=pan_ch, num_groups=16, group_dim=16)
    feats = [torch.randn(B, c, s, s) for c, s in zip(pan_ch, (80, 40, 20))]
    masks = [
        mask,
        F.interpolate(mask, size=(40, 40), mode="bilinear", align_corners=False),
        F.interpolate(mask, size=(20, 20), mode="bilinear", align_corners=False),
    ]
    # alpha init ~0 -> increment is negligible (near-vanilla start).
    for i, (f, msk) in enumerate(zip(feats, masks)):
        delta = film(f, msk, i)
        assert delta.shape == f.shape, f"scale {i}: {tuple(delta.shape)} != {tuple(f.shape)}"
        assert delta.abs().max().item() < 1e-2, f"scale {i} increment not ~0 at init"
    print("HeatmapFiLMFusion init near-identity + per-scale shapes OK.")

    # With alpha grown, the increment is non-trivial and still shaped like the PAN feature.
    with torch.no_grad():
        for a in film.alpha:
            a.fill_(1.0)
    delta = film(feats[0], masks[0], 0)
    assert delta.shape == feats[0].shape and delta.abs().max().item() > 0.0
    print(f"HeatmapFiLMFusion alpha=1 active OK (max abs = {delta.abs().max().item():.4f}).")

    # gamma_bound: bounded 1+tanh(gamma) gate; still near-identity at init via alpha.
    film_b = HeatmapFiLMFusion(pan_channels=pan_ch, gamma_bound=True)
    db = film_b(feats[0], masks[0], 0)
    assert db.shape == feats[0].shape and db.abs().max().item() < 1e-2, "gamma_bound init not ~0"
    print("HeatmapFiLMFusion gamma_bound init near-identity OK.")

    print("\n--- BackboneMemoryBank ---")
    mb = BackboneMemoryBank(temperature=3.0, K=5)
    assert not mb.built, "fresh bank should not be built (update=True)"
    feat_dict = {4: torch.randn(2, 512, 80, 80), 6: torch.randn(2, 512, 40, 40), 10: torch.randn(2, 1024, 20, 20)}
    mb._bb_layer_indices = [4, 6, 10]
    # Build mode: returns zeros
    out = mb(feat_dict)
    assert out.shape == (2, 1, 80, 80), f"expected (2,1,80,80) got {tuple(out.shape)}"
    assert out.abs().max().item() == 0.0, "build mode should return zeros"
    # Load bank and freeze
    bank_features = torch.randn(100, 512)
    mb.load_bank(bank_features)
    mb.freeze_memory_bank()
    assert mb.built, "should be built after freeze"
    # Inference mode: non-zero heatmap
    out = mb(feat_dict)
    assert out.shape == (2, 1, 80, 80)
    assert 0.0 <= out.min().item() <= out.max().item() <= 1.0, "scores should be in [0,1]"
    # Reset
    mb.reset_memory_bank()
    assert not mb.built
    # Coreset smoke
    big_bank = torch.randn(200, 512)
    mb.load_bank(big_bank)
    mb.freeze_memory_bank()  # max_bank_size=None → no compression
    mb2 = BackboneMemoryBank(max_bank_size=50)
    mb2._bb_layer_indices = [4]
    mb2.load_bank(big_bank)
    mb2.freeze_memory_bank()
    assert mb2.memory_bank.shape[0] == 50, f"coreset should keep 50, got {mb2.memory_bank.shape[0]}"
    print("BackboneMemoryBank smoke OK.")

    print("\n--- QueryFiLMFusion ---")
    qf = QueryFiLMFusion(p3_channels=256, num_queries_k=16, query_dim=128, num_groups=16, alpha_init=0.0)
    p3 = torch.randn(B, 256, H, H)
    heat = torch.rand(B, 1, H, H)
    # Identity at init: alpha=0 + zero-init FiLM head -> increment is bit-exact zero.
    delta, aux = qf(p3, heat, return_aux=True)
    assert delta.shape == p3.shape, f"{tuple(delta.shape)} != {tuple(p3.shape)}"
    assert delta.abs().max().item() == 0.0, "QueryFiLM increment not exactly zero at init"
    assert qf.film_mlp[-1].weight.abs().sum().item() == 0.0, "FiLM head weight not zero-init"
    assert qf.film_mlp[-1].bias.abs().sum().item() == 0.0, "FiLM head bias not zero-init"
    assert aux["A"].shape == (B, 16, H, H) and aux["obj_logits"].shape == (B, 16)
    print("QueryFiLMFusion identity-at-init OK (delta == 0, FiLM head zero-init).")

    # With alpha grown, increment is non-trivial and correctly shaped.
    with torch.no_grad():
        qf.alpha.fill_(1.0)
        nn.init.normal_(qf.film_mlp[-1].weight, std=0.1)
    delta = qf(p3, heat)
    assert delta.shape == p3.shape and delta.abs().max().item() > 0.0
    print(f"QueryFiLMFusion alpha=1 active OK (max abs = {delta.abs().max().item():.4f}).")

    # Per-instance render + query loss (training-only Hungarian matching).
    grenderer = BboxMaskRenderer(mask_size=H, mode="gauss", sigma_factor=0.15)
    gt_per_inst = grenderer.render_per_instance(bboxes, batch_idx, B)
    assert len(gt_per_inst) == B
    assert gt_per_inst[0].shape[0] == 2 and gt_per_inst[1].shape[0] == 0  # img0: 2 boxes, img1: none
    _, aux = qf(p3, heat, return_aux=True)
    ql = query_film_loss(aux["A"], aux["attn_logits"], aux["obj_logits"], gt_per_inst)
    assert set(ql) == {"mask", "obj", "overlap"}
    for kk, vv in ql.items():
        assert vv.ndim == 0 and torch.isfinite(vv), f"query loss {kk} not a finite scalar"
    print(f"query_film_loss OK (mask={ql['mask']:.4f}, obj={ql['obj']:.4f}, overlap={ql['overlap']:.4f}).")

    print("\nAll smoke tests passed.")
