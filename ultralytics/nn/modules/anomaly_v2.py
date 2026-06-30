# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""YOLOA v2 modules: bbox-mask renderer, heatmap fusion, and memory-bank scorer.

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

__all__ = (
    "BboxMaskRenderer",
    "HeatmapBiasFusion",
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
            inside = torch.exp(-(dx**2 / (2 * sigma_x[:, None, None] ** 2) + dy**2 / (2 * sigma_y[:, None, None] ** 2)))

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
    pe = torch.cat(
        [pe_y[:, None, :].expand(h, w, dim // 2), pe_x[None, :, :].expand(h, w, dim // 2)], dim=2
    )  # (h, w, dim)
    return pe.permute(2, 0, 1).unsqueeze(0).to(dtype)  # (1, dim, h, w)


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
        self._threshold: float | None = None  # sigmoid threshold in d_norm space
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
        """Keep a plain-attribute memory bank in sync with ``.to()``/``.float()``/``.half()``.

        If ``memory_bank`` is ever held as a plain attribute (not a registered buffer), it is
        skipped by ``nn.Module._apply`` and would stay on the old device/dtype after
        ``model.to(device)``; this defensively moves it. A no-op for the registered-buffer case.
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
            self.memory_bank, coreset_idx = self._coreset_subsample(mem, self.max_bank_size, return_indices=True)
            holdout_mask = torch.ones(mem.shape[0], dtype=torch.bool, device=mem.device)
            holdout_mask[coreset_idx] = False
            holdout = mem[holdout_mask]
        mem = self._effective_bank()

        if mem.shape[0] > 0:
            self._calibrate_compactness(mem)
            if holdout is not None and holdout.shape[0] > 0:
                if holdout.shape[0] > self.holdout_max:
                    idx = torch.randperm(holdout.shape[0], device=holdout.device)[: self.holdout_max]
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
            F.interpolate(f, size=target_size, mode="nearest") if f.shape[-2:] != target_size else f for f in feats
        ]
        fused = torch.cat(aligned, dim=1) if len(aligned) > 1 else aligned[0]
        # Random projection (Johnson-Lindenstrauss): fixed random matrix, ~preserves cosine structure.
        if self.proj_dim > 0 and fused.shape[1] > self.proj_dim:
            if self._proj_weight.numel() == 0:
                in_dim = fused.shape[1]
                # Random Gaussian projection: E[||Wx||^2] = ||x||^2 (Johnson-Lindenstrauss)
                w = torch.randn(in_dim, self.proj_dim, device=fused.device, dtype=torch.float32)
                self._proj_weight = w / (self.proj_dim**0.5)
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
            compactness,
            beta,
            self._threshold,
            self.calibration_target_score,
        )

    def _score_with_threshold(
        self, features: torch.Tensor, mem: torch.Tensor, threshold_cos: float, beta: float | None = None
    ) -> torch.Tensor:
        """Compute anomaly scores with specific threshold & β (stateless, for calibration)."""
        if beta is None:
            beta = self.temperature
        q = F.normalize(features.view(-1, self.feature_dim), p=2, dim=1)
        k = min(self.K, mem.shape[0])
        n, m = q.shape[0], mem.shape[0]
        chunk = max(1, int(getattr(self, "score_chunk_elems", 1 << 27)) // max(m, 1))
        out = []
        for i in range(0, n, chunk):
            cos = q[i : i + chunk] @ mem.t()
            psi = torch.sigmoid(beta * (cos - threshold_cos))
            topk = psi.topk(k=k, dim=1).values
            log_prob = torch.log((1.0 - topk).clamp(min=1e-8)).mean(dim=1)
            out.append(torch.exp(log_prob))
        return torch.cat(out).clamp(0, 1)

    def _calibrate_threshold_from_holdout(self, holdout: torch.Tensor, mem: torch.Tensor) -> None:
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
        z_q = math.sqrt(2) * torch.erfinv(torch.tensor(2.0 * target_q - 1.0)).item()

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
            n_holdout,
            target_q,
            z_q,
            old_beta,
            best_beta,
            c - math.log(max((1.0 - target) / max(target, 1e-6), 1e-6)) / max(old_beta, 0.1),
            best_thresh,
            target,
            best_achieved,
            best_spread,
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
    def _coreset_subsample(mem: torch.Tensor, max_size: int, return_indices: bool = False):
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
            centres = mem[top_idx]  # [k, C]
            cos_sim = mem @ centres.t()  # [M, k]
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

    def __init__(
        self,
        noise_std: float = 0.015,
        n_noise: int = 1,
        hidden: int = 64,
        steps: int = 400,
        lr: float = 1e-3,
        batch: int = 4096,
        adaptor: bool = True,
        seed: int = 0,
        noise_mode: str = "gaussian",
    ):
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


def heatmap_local_contrast(h: torch.Tensor, k: int = 9, eps: float = 1e-3) -> torch.Tensor:
    """Local z-score of a [B,1,H,W] map in [0,1]: (h - local_mean) / (local_std + eps), ~[-1,1]."""
    mean = F.avg_pool2d(h, k, stride=1, padding=k // 2)
    sq = F.avg_pool2d(h * h, k, stride=1, padding=k // 2)
    std = (sq - mean * mean).clamp(min=0).sqrt()
    z = (h - mean) / (std + eps)
    return (z / 3.0).clamp(-1, 1)


class _RefinerDoubleConv(nn.Module):
    """(Conv -> BN -> ReLU) x 2."""

    def __init__(self, c_in: int, c_out: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(c_in, c_out, 3, padding=1, bias=False),
            nn.BatchNorm2d(c_out),
            nn.ReLU(inplace=True),
            nn.Conv2d(c_out, c_out, 3, padding=1, bias=False),
            nn.BatchNorm2d(c_out),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)


class HeatmapRefiner(nn.Module):
    """Standalone, offline-trained heatmap->mask refiner (tiny 3-level U-Net, ~117k params).

    External interface: 1-channel heatmap in -> 1-channel logits out. Internally appends a
    local-contrast channel -> 2ch. Fully-convolutional (trains at 80x80, runs at any res).
    Trained decoupled from detection (see tools/yoloa_refiner/); grafted into the YOLOA prior
    path via ``YOLOAnomalyV2Model.set_heatmap_refiner``. Deploy = gate (``raw*sigmoid(R)``,
    suppress-only; never injects signal where the bank heatmap is zero).
    """

    def __init__(self, base: int = 16, contrast_k: int = 9):
        super().__init__()
        self.contrast_k = contrast_k
        self.pool = nn.MaxPool2d(2)
        self.e1 = _RefinerDoubleConv(2, base)
        self.e2 = _RefinerDoubleConv(base, base * 2)
        self.e3 = _RefinerDoubleConv(base * 2, base * 4)
        self.up2 = nn.ConvTranspose2d(base * 4, base * 2, 2, stride=2)
        self.d2 = _RefinerDoubleConv(base * 4, base * 2)
        self.up1 = nn.ConvTranspose2d(base * 2, base, 2, stride=2)
        self.d1 = _RefinerDoubleConv(base * 2, base)
        self.outc = nn.Conv2d(base, 1, 1)
        nn.init.zeros_(self.outc.weight)  # identity-at-init: sigmoid(R)~=0
        nn.init.constant_(self.outc.bias, -4.0)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        """h: [B,1,H,W] heatmap in [0,1] (H,W divisible by 4). Returns [B,1,H,W] logits."""
        x = torch.cat([h, heatmap_local_contrast(h, self.contrast_k)], dim=1)
        x1 = self.e1(x)
        x2 = self.e2(self.pool(x1))
        x3 = self.e3(self.pool(x2))
        y = self.d2(torch.cat([self.up2(x3), x2], dim=1))
        y = self.d1(torch.cat([self.up1(y), x1], dim=1))
        return self.outc(y)

    @torch.no_grad()
    def refine_gated(self, h: torch.Tensor) -> torch.Tensor:
        """Gate: refined = raw * sigmoid(R) (suppress-only; never injects signal where raw=0)."""
        return h * self.forward(h).sigmoid()
