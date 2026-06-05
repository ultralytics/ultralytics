# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""Model-agnostic gallery retrieval helpers for ReID predict.

Pure NumPy ranking + cache I/O, decoupled from any model. Callers supply an
``embed_fn(list[Path]) -> np.ndarray (N, D)`` so the same engine serves both the
``ReidPredictor`` CLI path and the ``ReIDVisualizer`` solution.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable

import numpy as np

from ultralytics.data.utils import IMG_FORMATS
from ultralytics.utils import LOGGER


def scan_gallery(root: str | Path) -> list[Path]:
    """Recursively collect image paths under a directory (sorted)."""
    root = Path(root)
    if not root.exists():
        raise FileNotFoundError(f"gallery '{root}' does not exist")
    paths = [p for p in sorted(root.rglob("*")) if p.is_file() and p.suffix.lower().lstrip(".") in IMG_FORMATS]
    if not paths:
        raise RuntimeError(f"no image files found under gallery '{root}'")
    return paths


def l2_normalize(x: np.ndarray) -> np.ndarray:
    """Row-wise L2 normalization, safe for zero rows."""
    x = np.asarray(x, dtype=np.float32)
    return x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-12)


def cosine_topk(query_embs: np.ndarray, gallery_embs: np.ndarray, topk: int) -> tuple[np.ndarray, np.ndarray]:
    """Return (indices, scores) of the top-k gallery rows per query by cosine similarity.

    Inputs are assumed L2-normalized, so cosine == dot product. ``topk`` is clamped to
    the gallery size. Shapes: query (Q, D), gallery (N, D) -> indices (Q, k), scores (Q, k).
    """
    k = min(topk, gallery_embs.shape[0])
    sims = query_embs @ gallery_embs.T  # (Q, N)
    idx = np.argsort(-sims, axis=1)[:, :k]
    scores = np.take_along_axis(sims, idx, axis=1)
    return idx, scores
