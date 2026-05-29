"""Unit tests for h8 retrieval math. All synthetic — no weights, no GPU."""
import numpy as np
import pytest

import retrieval as R


def _toy_setup():
    """3 queries × 5 gallery. Construct so we know the correct rankings."""
    # Embeddings: column 0 = "id signal", column 1 = "noise".
    # q0 (pid=1,cam=A) closest to g0 (pid=1,cam=A junk), g1 (pid=1,cam=B correct), g3 (pid=2)...
    q_feats = np.array([[1.0, 0.0], [0.0, 1.0], [0.5, 0.5]])
    g_feats = np.array(
        [
            [1.0, 0.0],  # g0: same pid+cam as q0 -> junk
            [0.99, 0.05],  # g1: same pid, different cam -> correct match for q0
            [0.0, 1.0],  # g2: q1 correct
            [-1.0, 0.0],  # g3: distractor
            [0.5, 0.6],  # g4: q2 correct-ish
        ]
    )
    # Normalise to mimic L2-normed embeddings.
    q_feats = q_feats / np.linalg.norm(q_feats, axis=1, keepdims=True)
    g_feats = g_feats / np.linalg.norm(g_feats, axis=1, keepdims=True)
    q_pids = np.array([1, 2, 3])
    q_camids = np.array([0, 0, 0])
    g_pids = np.array([1, 1, 2, 4, 3])
    g_camids = np.array([0, 1, 1, 1, 1])
    return q_feats, q_pids, q_camids, g_feats, g_pids, g_camids


def test_rank_with_junk_filter_excludes_same_pid_same_cam():
    q_feats, q_pids, q_camids, g_feats, g_pids, g_camids = _toy_setup()
    ranks = R.rank_with_junk(q_feats, q_pids, q_camids, g_feats, g_pids, g_camids)
    # q0's top match must NOT be g0 (junk); should be g1.
    assert ranks.top_gids[0, 0] == 1, "junk gallery item leaked into top-1"


def test_compute_r1_matches_expected():
    q_feats, q_pids, q_camids, g_feats, g_pids, g_camids = _toy_setup()
    ranks = R.rank_with_junk(q_feats, q_pids, q_camids, g_feats, g_pids, g_camids)
    metrics = R.compute_cmc_map(ranks, q_pids, g_pids)
    assert metrics["r1"] == pytest.approx(1.0)
    assert metrics["mAP"] == pytest.approx(1.0)


def test_compute_per_query_ap_returns_one_value_per_query():
    q_feats, q_pids, q_camids, g_feats, g_pids, g_camids = _toy_setup()
    ranks = R.rank_with_junk(q_feats, q_pids, q_camids, g_feats, g_pids, g_camids)
    aps = R.per_query_ap(ranks, q_pids, g_pids)
    assert aps.shape == (3,)
    assert np.all((aps >= 0) & (aps <= 1))


def test_bootstrap_ci_returns_lo_hi_for_r1():
    rng = np.random.default_rng(0)
    # 1000 queries, 70% correct.
    hits = (rng.random(1000) < 0.7).astype(np.float64)
    lo, hi = R.bootstrap_mean_ci(hits, n_resamples=200, alpha=0.05, seed=0)
    assert lo < hi
    assert lo < 0.7 < hi
    assert hi - lo < 0.1  # 1000 samples gives a tight CI


def test_no_gallery_match_yields_zero_ap():
    """Query whose true pid is absent from gallery should get AP=0."""
    q_feats = np.array([[1.0, 0.0]])
    q_feats = q_feats / np.linalg.norm(q_feats, axis=1, keepdims=True)
    g_feats = np.array([[0.0, 1.0]])
    g_feats = g_feats / np.linalg.norm(g_feats, axis=1, keepdims=True)
    ranks = R.rank_with_junk(q_feats, np.array([1]), np.array([0]), g_feats, np.array([2]), np.array([1]))
    aps = R.per_query_ap(ranks, np.array([1]), np.array([2]))
    assert aps[0] == 0.0
