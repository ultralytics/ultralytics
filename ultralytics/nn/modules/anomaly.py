# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""YOLOA v2 modules: heatmap fusion and memory-bank scorer.

Soft-hint fusion: a 1-channel memory-bank heatmap is turned into a bounded per-pixel bias
added (broadcast over channels) to PAN features before the Detect head.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .head import HeatmapBiasFusion, HeatmapProcessor

__all__ = (
    "AnomalyMemoryBank",
    "BboxMaskRenderer",
    "HeatmapBiasFusion",
    "HeatmapProcessor",
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


class AnomalyMemoryBank(nn.Module):
    """Cosine-similarity memory bank that produces an anomaly heatmap.

    The bank stores L2-normalised backbone feature vectors from normal images.
    At inference each spatial query looks up its ``K`` nearest bank vectors and
    returns a noisy-OR anomaly score.  The score is calibrated so that a typical
    normal query maps to ``target_score``.

    Args:
        bank_size: Maximum number of vectors kept after coreset compression.
        K: Number of nearest neighbours used per query.
        temperature: β sharpness of the sigmoid over cosine similarity.
        target_score: Desired anomaly score for normal queries.
        stretch: Optional quadratic stretch applied to the output heatmap.
        score_chunk: Element budget for chunked scoring (device memory knob).
    """

    def __init__(
        self,
        bank_size: int = 10000,
        K: int = 5,
        temperature: float = 5.0,
        target_score: float = 0.4,
        stretch: float = 0.0,
        score_chunk: int = 1 << 27,
    ):
        super().__init__()
        self.bank_size = bank_size
        self.K = K
        self.temperature = float(temperature)
        self.target_score = float(target_score)
        self.stretch = float(stretch)
        self.score_chunk = int(score_chunk)

        self.dim: int | None = None
        self._compactness: float | None = None
        self._threshold: float | None = None
        self.building = True
        self._chunks: list[torch.Tensor] = []

        self.register_buffer("bank", torch.empty(0, 0))

    @property
    def is_ready(self) -> bool:
        """True when a non-empty, calibrated bank is available."""
        return self.bank.shape[0] > 0 and self._threshold is not None and not self.building

    def reset(self) -> None:
        """Clear the bank and return to build mode."""
        self.bank = torch.empty(0, 0, device=self.bank.device)
        self.dim = None
        self._compactness = None
        self._threshold = None
        self.building = True
        self._chunks = []

    def load_bank(self, features: torch.Tensor) -> None:
        """Direct-load a pre-built bank from L2-normalised feature vectors [M, C]."""
        if features.numel() == 0:
            return
        if not torch.isfinite(features).all():
            raise ValueError(f"AnomalyMemoryBank.load_bank: features contain NaN/Inf ({features.shape})")
        self.dim = features.shape[1]
        self.bank = F.normalize(features.to(self.bank.device), p=2, dim=1).float()
        self.building = False
        self._calibrate(self._active_bank())

    def add_features(self, feats: list[torch.Tensor]) -> None:
        """Extract and accumulate backbone features into the bank (build phase)."""
        if not feats:
            return
        fused = self._fuse_features(feats)
        C = fused.shape[1]
        if self.dim is None:
            self.dim = C
        flat = fused.permute(0, 2, 3, 1).reshape(-1, C)
        self._chunks.append(F.normalize(flat, p=2, dim=1).float())

    def freeze(self) -> None:
        """Materialise the bank, optionally coreset-compress it, then calibrate and freeze."""
        if self._chunks:
            self.bank = torch.cat(self._chunks, dim=0)
            self._chunks = []

        mem = self._active_bank()
        holdout = None
        if self.bank_size is not None and mem.shape[0] > self.bank_size:
            self.bank, selected = self._coreset(mem, self.bank_size, return_indices=True)
            mask = torch.zeros(mem.shape[0], dtype=torch.bool, device=mem.device)
            mask[selected] = True
            holdout = mem[~mask]
            mem = self._active_bank()

        if mem.shape[0] > 0:
            self._calibrate(mem, holdout=holdout)

        self.building = False

    def forward(self, feats: list[torch.Tensor]) -> torch.Tensor:
        """Return an anomaly heatmap (B, 1, H, W) from backbone features."""
        first = feats[0]
        b, device, h, w = first.shape[0], first.device, first.shape[2], first.shape[3]
        mem = self._active_bank()
        if self.building or mem.shape[0] == 0:
            return torch.zeros(b, 1, h, w, device=device)

        fused = self._fuse_features(feats)
        flat = fused.permute(0, 2, 3, 1).reshape(-1, fused.shape[1])
        scores = self._score(flat, mem)
        hmap = scores.view(b, 1, fused.shape[2], fused.shape[3])
        s = self.stretch
        if s:
            hmap = (hmap + s * hmap * hmap).clamp(0, 1)
        return hmap

    def _active_bank(self) -> torch.Tensor:
        """Return the current bank tensor, or an empty tensor if none exists."""
        if self.building and self._chunks:
            return torch.cat(self._chunks, dim=0)
        if self.dim is None or self.bank.shape[0] == 0:
            return self.bank[:0]
        return self.bank[self.bank.norm(dim=1) > 0]

    @staticmethod
    def _fuse_features(feats: list[torch.Tensor]) -> torch.Tensor:
        """Concatenate tapped backbone features after resizing to the first map size."""
        if len(feats) == 1:
            return feats[0]
        target_size = feats[0].shape[-2:]
        aligned = [
            F.interpolate(f, size=target_size, mode="nearest") if f.shape[-2:] != target_size else f for f in feats
        ]
        return torch.cat(aligned, dim=1) if len(aligned) > 1 else aligned[0]

    def _estimate_compactness(self, mem: torch.Tensor) -> float:
        """Mean local cosine density of the bank."""
        with torch.no_grad():
            k = min(self.K, mem.shape[0])
            n_sample = min(512, mem.shape[0])
            idx = torch.randperm(mem.shape[0], device=mem.device)[:n_sample]
            sample = mem[idx]
            sim = sample @ mem.t()
            sim[torch.arange(n_sample, device=mem.device), idx] = -1.0
            topk_sim = sim.topk(k=k, dim=1).values
            return topk_sim.mean().clamp(0.0, 1.0 - 1e-4).item()

    def _calibrate(self, mem: torch.Tensor, holdout: torch.Tensor | None = None) -> None:
        """Set the threshold so the upper tail of normal-query scores ≈ target_score.

        Normal queries come from the coreset hold-out (queries not present in the
        final bank).  If no hold-out is available, a sample of the bank itself is
        used with its own match masked out.  This is a single threshold search at
        the user β; the previous multi-β hold-out sweep has been removed.
        """
        compactness = self._estimate_compactness(mem)
        self._compactness = compactness

        beta = max(self.temperature, 0.1)
        target = self.target_score
        target_quantile = 0.95
        z_q = math.sqrt(2) * torch.erfinv(torch.tensor(2.0 * target_quantile - 1.0)).item()
        k = min(self.K, mem.shape[0])

        if holdout is not None and holdout.shape[0] > 0:
            holdout_max = 5000
            if holdout.shape[0] > holdout_max:
                holdout = holdout[torch.randperm(holdout.shape[0], device=holdout.device)[:holdout_max]]
            n_query = min(1024, holdout.shape[0])
            idx = torch.randperm(holdout.shape[0], device=holdout.device)[:n_query]
            queries = holdout[idx]
            topk_cos = (queries @ mem.t()).topk(k=k, dim=1).values
        else:
            n_query = min(1024, mem.shape[0])
            idx = torch.randperm(mem.shape[0], device=mem.device)[:n_query]
            queries = mem[idx]
            cos_mat = queries @ mem.t()
            cos_mat[torch.arange(n_query, device=mem.device), idx] = -1.0
            topk_cos = cos_mat.topk(k=k, dim=1).values

        def _tail_stat(thresh: float) -> float:
            psi = torch.sigmoid(beta * (topk_cos - thresh))
            log_prob = torch.log((1.0 - psi).clamp(min=1e-8)).mean(dim=1)
            scores = torch.exp(log_prob).clamp(0, 1)
            return scores.mean().item() + z_q * scores.std().item()

        half_range = 5.0 / beta
        lo, hi = compactness - half_range, compactness + half_range
        s_lo = _tail_stat(lo)
        s_hi = _tail_stat(hi)
        # Expand the bracket until the target tail-stat is inside it.
        for _ in range(10):
            if s_lo <= target <= s_hi:
                break
            if s_lo > target:
                lo -= half_range
                s_lo = _tail_stat(lo)
            if s_hi < target:
                hi += half_range
                s_hi = _tail_stat(hi)

        if not (s_lo <= target <= s_hi):
            # Fallback: keep the compactness-based closed-form threshold.
            logit = math.log(max((1.0 - target) / max(target, 1e-6), 1e-6))
            self._threshold = compactness - logit / beta
            return

        for _ in range(30):
            mid = (lo + hi) / 2.0
            s_mid = _tail_stat(mid)
            if s_mid > target:
                hi = mid
            else:
                lo = mid
        self._threshold = (lo + hi) / 2.0

    def _score(self, features: torch.Tensor, mem: torch.Tensor | None = None) -> torch.Tensor:
        """Noisy-OR anomaly scores in [0, 1] for each query feature."""
        if mem is None:
            mem = self._active_bank()
        if mem.shape[0] == 0 or self._threshold is None:
            return torch.full((features.shape[0],), 0.5, device=features.device)
        q = F.normalize(features.view(-1, self.dim), p=2, dim=1).to(mem.dtype)
        k = min(self.K, mem.shape[0])
        n, m = q.shape[0], mem.shape[0]
        chunk = max(1, int(self.score_chunk) // max(m, 1))
        beta, thresh = self.temperature, self._threshold
        out = []
        for i in range(0, n, chunk):
            cos = q[i : i + chunk] @ mem.t()
            psi = torch.sigmoid(beta * (cos - thresh))
            topk = psi.topk(k=k, dim=1).values
            log_prob = torch.log((1.0 - topk).clamp(min=1e-8)).mean(dim=1)
            out.append(torch.exp(log_prob))
        return torch.cat(out).clamp(0, 1).to(features.dtype)

    @staticmethod
    def _coreset(mem: torch.Tensor, max_size: int, return_indices: bool = False):
        """Greedy k-center coreset on L2-normalised features using cosine distance."""
        from ultralytics.utils import TQDM

        M = mem.shape[0]
        if M <= max_size:
            idx = torch.arange(M, device=mem.device)
            return (mem, idx) if return_indices else mem
        device = mem.device
        BATCH = 64  # centres per GEMM call — amortises bank reads
        dist = torch.full((M,), float("inf"), device=device, dtype=torch.float32)
        selected: list[int] = []
        mean = mem.mean(dim=0)
        mean = mean / mean.norm().clamp(min=1e-8)
        seed = int((mem @ mean).argmax().item())
        selected.append(seed)
        centre = mem[seed].unsqueeze(0)
        seed_cos = (mem @ centre.t()).squeeze(1)
        dist = (1.0 - seed_cos).clamp(min=0.0)
        n_needed = max_size - len(selected)
        pbar = TQDM(total=n_needed, desc="Coreset subsample", leave=False)
        while n_needed > 0:
            k = min(BATCH, max_size - len(selected))
            _, top_idx = dist.topk(k)
            selected.extend(top_idx.tolist())
            centres = mem[top_idx]
            cos_sim = mem @ centres.t()
            new_dist = (1.0 - cos_sim).clamp(min=0.0).min(dim=1).values
            dist = torch.minimum(dist, new_dist)
            n_needed = max_size - len(selected)
            pbar.update(k)
        pbar.close()
        sel = torch.tensor(selected, device=device)
        return (mem[sel], sel) if return_indices else mem[sel]


