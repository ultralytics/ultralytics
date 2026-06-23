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

    Inputs are assumed L2-normalized, so cosine == dot product. ``topk`` is clamped to the gallery size. Shapes: query
    (Q, D), gallery (N, D) -> indices (Q, k), scores (Q, k).
    """
    k = min(topk, gallery_embs.shape[0])
    sims = query_embs @ gallery_embs.T  # (Q, N)
    idx = np.argsort(-sims, axis=1)[:, :k]
    scores = np.take_along_axis(sims, idx, axis=1)
    return idx, scores


def _signature(paths: list[Path], model_id: str, imgsz) -> dict:
    """Cache key: gallery file list (as strings) + model id + imgsz."""
    return {"paths": [str(p) for p in paths], "model_id": str(model_id), "imgsz": list(np.atleast_1d(imgsz))}


def build_gallery(
    embed_fn: Callable[[list[Path]], np.ndarray],
    gallery: str | Path,
    cache: str | Path | None,
    model_id: str,
    imgsz,
) -> tuple[list[Path], np.ndarray]:
    """Scan the gallery, embed (or load from cache), and return (paths, L2-normalized embeddings).

    The cache (a ``.pt`` file) is reused only when its recorded gallery file list, model id, and imgsz match the current
    request; otherwise it is rebuilt and rewritten.
    """
    import torch

    paths = scan_gallery(gallery)
    sig = _signature(paths, model_id, imgsz)

    if cache is not None and Path(cache).exists():
        blob = torch.load(str(cache), weights_only=False)
        if blob.get("signature") == sig:
            return paths, np.asarray(blob["embs"], dtype=np.float32)
        LOGGER.warning(f"reid_cache '{cache}' is stale (model/imgsz/gallery changed); rebuilding.")

    embs = l2_normalize(np.asarray(embed_fn(paths), dtype=np.float32))
    if cache is not None:
        Path(cache).parent.mkdir(parents=True, exist_ok=True)
        torch.save({"signature": sig, "embs": embs}, str(cache))
    return paths, embs
