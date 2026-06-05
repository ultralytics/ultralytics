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
