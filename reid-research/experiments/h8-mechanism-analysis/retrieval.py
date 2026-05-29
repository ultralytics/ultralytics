"""Retrieval math for h8: junk-filtered ranking, CMC/mAP, bootstrap CIs.

All embeddings are assumed L2-normalised on input (the validator's contract).
Distance = 1 - cosine = 1 - q @ g.T (cheap, equivalent to L2 squared / 2).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class Ranks:
    """Pre-junk-filtered top-K rankings for downstream metric computation.

    top_gids[q, k] is the *gallery index* of the k-th retrieved gallery item for query q,
    AFTER same-pid-same-cam junk has been removed (junk positions get distance=+inf so
    they sort to the back).

    q_camids and g_camids are stored so that metric functions can correctly exclude
    same-pid-same-cam gallery items from the relevant-count denominator (Market-1501 protocol).
    """

    top_gids: np.ndarray  # (Nq, K) int64
    top_dists: np.ndarray  # (Nq, K) float64
    matches: np.ndarray  # (Nq, K) bool — top_gid_pid == query_pid
    q_camids: np.ndarray  # (Nq,) int — query camera ids
    g_camids: np.ndarray  # (Ng,) int — gallery camera ids


def rank_with_junk(
    q_feats: np.ndarray,
    q_pids: np.ndarray,
    q_camids: np.ndarray,
    g_feats: np.ndarray,
    g_pids: np.ndarray,
    g_camids: np.ndarray,
    top_k: int = 50,
) -> Ranks:
    """Compute top-K gallery rankings per query with Market-1501 junk filter.

    Same-pid-same-camid gallery items are "junk" and pushed to the back of the ranking.
    """
    # Distance = 1 - cosine. Assumes inputs are L2-normed.
    sim = q_feats @ g_feats.T  # (Nq, Ng)
    dist = 1.0 - sim
    # Junk mask: same pid AND same camid.
    junk = (q_pids[:, None] == g_pids[None, :]) & (q_camids[:, None] == g_camids[None, :])
    dist = np.where(junk, np.inf, dist)
    order = np.argsort(dist, axis=1, kind="stable")
    top_gids = order[:, :top_k]
    nq = q_feats.shape[0]
    top_dists = dist[np.arange(nq)[:, None], top_gids]
    top_pids = g_pids[top_gids]
    top_gcams = g_camids[top_gids]
    # A position is a true match: same pid AND not junk (not same cam as query).
    matches = (top_pids == q_pids[:, None]) & (top_gcams != q_camids[:, None])
    return Ranks(top_gids=top_gids, top_dists=top_dists, matches=matches, q_camids=q_camids, g_camids=g_camids)


def compute_cmc_map(ranks: Ranks, q_pids: np.ndarray, g_pids: np.ndarray) -> dict[str, float]:
    """Compute aggregate R1/R5/R10 and mAP from pre-computed Ranks."""
    nq = ranks.matches.shape[0]
    # A query is "valid" if it has at least one non-junk relevant item in the gallery.
    has_any_match = np.array([
        int(((g_pids == q_pids[i]) & ~((g_pids == q_pids[i]) & (ranks.g_camids == ranks.q_camids[i]))).sum()) > 0
        for i in range(nq)
    ])
    valid = has_any_match.sum()
    r1 = ranks.matches[:, 0].sum() / max(1, valid)
    r5 = (ranks.matches[:, :5].any(axis=1)).sum() / max(1, valid)
    r10 = (ranks.matches[:, :10].any(axis=1)).sum() / max(1, valid)
    aps = per_query_ap(ranks, q_pids, g_pids)
    mAP = aps[has_any_match].mean() if valid > 0 else 0.0
    return {"r1": float(r1), "r5": float(r5), "r10": float(r10), "mAP": float(mAP)}


def per_query_ap(ranks: Ranks, q_pids: np.ndarray, g_pids: np.ndarray) -> np.ndarray:
    """Per-query Average Precision over the top-K returned rankings.

    Note: AP is computed using only the top-K returned matches (not the full gallery).
    For K=50 this is effectively exact on Market-1501 since most queries have <20
    correct matches after junk-filter.
    """
    nq = ranks.matches.shape[0]
    aps = np.zeros(nq, dtype=np.float64)
    for i in range(nq):
        # Total relevant in gallery (for normalising AP), excluding junk (same pid+cam).
        same_pid = g_pids == q_pids[i]
        junk = same_pid & (ranks.g_camids == ranks.q_camids[i])
        n_rel = int((same_pid & ~junk).sum())
        if n_rel == 0:
            aps[i] = 0.0
            continue
        hits = ranks.matches[i].astype(np.float64)
        if hits.sum() == 0:
            aps[i] = 0.0
            continue
        cumhits = np.cumsum(hits)
        ranks_idx = np.arange(1, hits.shape[0] + 1)
        precision_at_k = cumhits / ranks_idx
        aps[i] = (precision_at_k * hits).sum() / n_rel
    return aps


def bootstrap_mean_ci(
    values: np.ndarray,
    n_resamples: int = 1000,
    alpha: float = 0.05,
    seed: Optional[int] = None,
) -> tuple[float, float]:
    """Bootstrap CI of the mean of `values` (e.g. per-query hit bits for R1, or per-query AP)."""
    rng = np.random.default_rng(seed)
    n = values.shape[0]
    means = np.empty(n_resamples, dtype=np.float64)
    for b in range(n_resamples):
        idx = rng.integers(0, n, size=n)
        means[b] = values[idx].mean()
    lo = float(np.quantile(means, alpha / 2))
    hi = float(np.quantile(means, 1 - alpha / 2))
    return lo, hi
