"""Batch 5: ReidMetrics state safety.

Before the fix:
  * process() didn't reset self.mAP/rank1/5/10 at the top, so an empty-eval epoch
    (zero positive matches) silently kept the previous epoch's values — wrong best-ckpt selection.
  * process() called self.update_gallery() in the no-gallery fallback, which flipped the
    internal state and silenced the warning on subsequent epochs. From epoch 2 onward, new
    queries were silently scored against epoch 1's stale query embeddings.
"""
from __future__ import annotations

import numpy as np
import pytest

from ultralytics.utils.metrics import ReidMetrics


def _rng_features(n: int, d: int = 32, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    x = rng.standard_normal((n, d)).astype(np.float32)
    x /= np.linalg.norm(x, axis=1, keepdims=True)
    return x


def test_process_resets_stale_metrics_on_empty_eval():
    """If process() finds zero positive matches, it must reset mAP/rank1/5/10 to 0.0 —
    not keep the previous epoch's values."""
    m = ReidMetrics()
    # Seed stale values to simulate the prior epoch
    m.mAP, m.rank1, m.rank5, m.rank10 = 0.87, 0.93, 0.97, 0.99
    # Now run a degenerate eval: query/gallery have disjoint pids → zero positive matches
    q_feats = _rng_features(4)
    q_pids = np.array([100, 101, 102, 103], dtype=np.int64)
    q_camids = np.array([1, 2, 3, 4], dtype=np.int64)
    g_feats = _rng_features(4, seed=1)
    g_pids = np.array([200, 201, 202, 203], dtype=np.int64)  # disjoint from queries
    g_camids = np.array([5, 6, 7, 8], dtype=np.int64)
    m.update_gallery(g_feats, g_pids, g_camids)
    m.process(q_feats, q_pids, q_camids)
    assert m.mAP == 0.0, f"mAP must reset to 0 on empty eval, got {m.mAP}"
    assert m.rank1 == 0.0
    assert m.rank5 == 0.0
    assert m.rank10 == 0.0


def test_fallback_does_not_persist_gallery_across_epochs():
    """When no gallery is provided, process() falls back to using queries as gallery. On
    the NEXT epoch with fresh queries (and still no gallery), the metric must score the
    new queries against THEMSELVES — not against epoch 1's stale embeddings."""
    m = ReidMetrics()
    # Epoch 1: no gallery, use queries
    q1 = _rng_features(8, seed=0)
    p1 = np.array([1, 1, 2, 2, 3, 3, 4, 4], dtype=np.int64)
    c1 = np.array([1, 2, 1, 2, 1, 2, 1, 2], dtype=np.int64)
    m.process(q1, p1, c1)
    assert m.gallery_feats is q1, "epoch 1 fallback should make queries the gallery"
    # Epoch 2: still no gallery (e.g., misconfigured YAML), fresh queries
    q2 = _rng_features(8, seed=1)
    p2 = np.array([10, 10, 20, 20, 30, 30, 40, 40], dtype=np.int64)
    c2 = np.array([3, 4, 3, 4, 3, 4, 3, 4], dtype=np.int64)
    m.process(q2, p2, c2)
    # The gallery for epoch 2 must be epoch 2's queries, NOT epoch 1's
    assert m.gallery_feats is q2, "epoch 2 fallback must rebind gallery to NEW queries, not reuse stale"
    assert np.array_equal(m.gallery_pids, p2)


def test_update_gallery_flips_provided_flag():
    """update_gallery() must mark the gallery as 'provided' so process() does NOT enter
    the no-gallery fallback path."""
    m = ReidMetrics()
    assert m._gallery_provided is False
    m.update_gallery(_rng_features(4), np.zeros(4, dtype=np.int64), np.zeros(4, dtype=np.int64))
    assert m._gallery_provided is True


def test_fallback_warns_every_epoch():
    """When the gallery is never provided, the no-gallery warning should fire EVERY epoch —
    not just the first — so the misconfiguration stays visible in the log."""
    import logging

    m = ReidMetrics()
    q = _rng_features(4)
    p = np.array([1, 1, 2, 2], dtype=np.int64)
    c = np.array([1, 2, 1, 2], dtype=np.int64)

    # Monkey-patch the module logger to capture warning calls
    from ultralytics.utils import metrics as metrics_mod

    calls: list[str] = []
    original = metrics_mod.LOGGER.warning
    metrics_mod.LOGGER.warning = lambda msg, *a, **kw: calls.append(str(msg))
    try:
        m.process(q, p, c)
        m.process(q, p, c)
    finally:
        metrics_mod.LOGGER.warning = original
    matching = [c for c in calls if "no gallery path" in c.lower() or "gallery" in c.lower()]
    assert len(matching) >= 2, f"expected the no-gallery warning to fire on both epochs, got {calls}"
