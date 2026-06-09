"""Batch 11: gallery retrieval engine + Results.matches + predictor ranking."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch


# ---------- engine: cosine_topk / scan_gallery / l2_normalize ----------------


def test_l2_normalize_rows_unit_norm():
    from ultralytics.models.yolo.reid.retrieval import l2_normalize

    x = np.array([[3.0, 4.0], [0.0, 2.0]], dtype=np.float32)
    out = l2_normalize(x)
    norms = np.linalg.norm(out, axis=1)
    assert np.allclose(norms, 1.0, atol=1e-5)


def test_l2_normalize_zero_row_safe():
    from ultralytics.models.yolo.reid.retrieval import l2_normalize

    x = np.zeros((1, 4), dtype=np.float32)
    out = l2_normalize(x)  # must not divide-by-zero -> NaN
    assert np.isfinite(out).all()


def test_cosine_topk_orders_by_similarity():
    from ultralytics.models.yolo.reid.retrieval import cosine_topk, l2_normalize

    gallery = l2_normalize(np.array([[1.0, 0.0], [0.9, 0.1], [0.0, 1.0]], dtype=np.float32))
    query = l2_normalize(np.array([[1.0, 0.0]], dtype=np.float32))
    idx, scores = cosine_topk(query, gallery, topk=2)
    assert idx.shape == (1, 2) and scores.shape == (1, 2)
    assert idx[0, 0] == 0  # identical vector ranks first
    assert idx[0, 1] == 1  # near vector second
    assert scores[0, 0] >= scores[0, 1]


def test_cosine_topk_clamps_to_gallery_size():
    from ultralytics.models.yolo.reid.retrieval import cosine_topk, l2_normalize

    gallery = l2_normalize(np.random.rand(2, 8).astype(np.float32))
    query = l2_normalize(np.random.rand(1, 8).astype(np.float32))
    idx, scores = cosine_topk(query, gallery, topk=5)  # topk > N
    assert idx.shape[1] == 2 and scores.shape[1] == 2


def test_scan_gallery_finds_images_recursively(tmp_path):
    from ultralytics.models.yolo.reid.retrieval import scan_gallery

    (tmp_path / "sub").mkdir()
    for name in ["a.jpg", "sub/b.png", "c.txt"]:
        (tmp_path / name).write_bytes(b"x")
    found = scan_gallery(tmp_path)
    names = sorted(p.name for p in found)
    assert names == ["a.jpg", "b.png"]  # .txt excluded


def test_scan_gallery_empty_raises(tmp_path):
    from ultralytics.models.yolo.reid.retrieval import scan_gallery

    with pytest.raises((FileNotFoundError, RuntimeError)):
        scan_gallery(tmp_path)  # no images


# ---------- engine: cache + build_gallery -----------------------------------


def _const_embedder(dim=4):
    """Return an embed_fn that maps each path to a deterministic vector by filename order."""

    def embed(paths):
        return np.stack([np.full(dim, float(i), dtype=np.float32) for i, _ in enumerate(paths)], axis=0)

    return embed


def test_build_gallery_no_cache(tmp_path):
    from ultralytics.models.yolo.reid.retrieval import build_gallery

    for name in ["a.jpg", "b.jpg", "c.jpg"]:
        (tmp_path / name).write_bytes(b"x")
    paths, embs = build_gallery(_const_embedder(), tmp_path, cache=None, model_id="m", imgsz=64)
    assert len(paths) == 3
    assert embs.shape == (3, 4)
    # rows are L2-normalized
    assert np.allclose(np.linalg.norm(embs[1:], axis=1), 1.0, atol=1e-5)


def test_build_gallery_writes_and_reuses_cache(tmp_path):
    from ultralytics.models.yolo.reid import retrieval

    gdir = tmp_path / "g"
    gdir.mkdir()
    for name in ["a.jpg", "b.jpg"]:
        (gdir / name).write_bytes(b"x")
    cache = tmp_path / "cache.pt"

    calls = {"n": 0}

    def counting_embed(paths):
        calls["n"] += 1
        return _const_embedder()(paths)

    # First call builds + writes cache
    p1, e1 = retrieval.build_gallery(counting_embed, gdir, cache=cache, model_id="m", imgsz=64)
    assert cache.exists()
    assert calls["n"] == 1
    # Second call loads from cache, does NOT re-embed
    p2, e2 = retrieval.build_gallery(counting_embed, gdir, cache=cache, model_id="m", imgsz=64)
    assert calls["n"] == 1
    assert [str(p) for p in p1] == [str(p) for p in p2]
    assert np.allclose(e1, e2)


def test_build_gallery_rebuilds_on_stale_cache(tmp_path):
    from ultralytics.models.yolo.reid import retrieval

    gdir = tmp_path / "g"
    gdir.mkdir()
    for name in ["a.jpg", "b.jpg"]:
        (gdir / name).write_bytes(b"x")
    cache = tmp_path / "cache.pt"

    calls = {"n": 0}

    def counting_embed(paths):
        calls["n"] += 1
        return _const_embedder()(paths)

    retrieval.build_gallery(counting_embed, gdir, cache=cache, model_id="m", imgsz=64)
    assert calls["n"] == 1
    # Different imgsz -> stale -> rebuild
    retrieval.build_gallery(counting_embed, gdir, cache=cache, model_id="m", imgsz=128)
    assert calls["n"] == 2


# ---------- ReIDVisualizer reuses the engine --------------------------------


def test_visualizer_rank_uses_engine(monkeypatch, tmp_path):
    """ReIDVisualizer.rank ranks generically (no Market PID logic) via the shared engine."""
    from ultralytics.solutions import reid_visualizer as rv

    gdir = tmp_path / "g"
    gdir.mkdir()
    for name in ["m1.jpg", "m2.jpg", "m3.jpg"]:
        (gdir / name).write_bytes(b"x")
    query = tmp_path / "q.jpg"
    query.write_bytes(b"x")

    # Avoid constructing a real YOLO model
    viz = rv.ReIDVisualizer.__new__(rv.ReIDVisualizer)
    viz.imgsz = 64
    viz.device = None
    viz.model = None

    vecs = {
        str(query): np.array([1.0, 0.0], dtype=np.float32),
        str(gdir / "m1.jpg"): np.array([1.0, 0.0], dtype=np.float32),  # identical → top-1
        str(gdir / "m2.jpg"): np.array([0.2, 0.9], dtype=np.float32),
        str(gdir / "m3.jpg"): np.array([0.0, 1.0], dtype=np.float32),
    }
    monkeypatch.setattr(viz, "_embed_paths", lambda paths: np.stack([vecs[str(p)] for p in paths], axis=0))

    items = viz.rank(query, gdir, k=2)
    assert len(items) == 2
    assert Path(items[0].path).name == "m1.jpg"
    assert items[0].score >= items[1].score


def test_visualizer_has_no_market_pid_helpers():
    """The Market-1501 filename parsing helpers are removed from the CLI-facing visualizer."""
    from ultralytics.solutions.reid_visualizer import ReIDVisualizer

    assert not hasattr(ReIDVisualizer, "_pid_from_name")
    assert not hasattr(ReIDVisualizer, "_cam_from_name")
