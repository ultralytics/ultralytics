# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from ultralytics.models.yolo.reid import retrieval
from ultralytics.utils.plotting import plot_reid_retrieval


@dataclass(slots=True)
class RetrievalItem:
    """A single ranked gallery match for a query image."""

    path: Path
    score: float


class ReIDVisualizer:
    """Rank gallery images for a query person and render the top matches.

    Uses any Ultralytics ReID model (PyTorch or exported ONNX) to extract embeddings, ranks gallery images by cosine
    similarity via the shared retrieval engine, and writes a comparison montage. Generic: it does not parse Market-1501
    filenames — tiles are labeled by rank and similarity.

    Args:
        model: Path or name of a ReID model, e.g. ``best.pt`` or ``best.onnx``.
        imgsz: Inference image size.
        device: Optional inference device.

    Examples:
        >>> from ultralytics.solutions import ReIDVisualizer
        >>> viz = ReIDVisualizer("best.onnx", imgsz=448)
        >>> viz.visualize("query.jpg", "gallery/", k=5)
    """

    def __init__(self, model: str | Path, imgsz: int = 448, device: str | None = None) -> None:
        """Initialize the ReID visualizer.

        Args:
            model (str | Path): Path or name of a ReID model, e.g. ``best.pt`` or ``best.onnx``.
            imgsz (int): Inference image size.
            device (str | None): Optional inference device.
        """
        from ultralytics import YOLO

        self.model = YOLO(model, task="reid")
        self.imgsz = imgsz
        self.device = device

    def _embed_paths(self, paths: list) -> np.ndarray:
        """Embed image paths via the model's predict() in one batched call (N, D)."""
        results = self.model.predict(
            [str(p) for p in paths], imgsz=self.imgsz, task="reid", device=self.device, verbose=False
        )
        embs = []
        for r in results:
            if r.embeddings is None:
                raise RuntimeError("model produced no embedding")
            data = r.embeddings.data
            data = data.detach().cpu().numpy() if isinstance(data, torch.Tensor) else np.asarray(data)
            embs.append(data.reshape(-1).astype(np.float32))
        return np.stack(embs, axis=0)

    def rank(self, query: str | Path, gallery: str | Path, k: int = 5) -> list[RetrievalItem]:
        """Rank gallery images for a query image by cosine similarity (top-k)."""
        query = Path(query)
        gallery_paths, gallery_embs = retrieval.build_gallery(
            self._embed_paths,
            gallery,
            cache=None,
            model_id=str(getattr(self.model, "model_name", "")),
            imgsz=self.imgsz,
        )
        query_emb = retrieval.l2_normalize(self._embed_paths([query]))
        idx, scores = retrieval.cosine_topk(query_emb, gallery_embs, k)
        return [RetrievalItem(path=gallery_paths[j], score=float(s)) for j, s in zip(idx[0], scores[0])]

    def visualize(self, query: str | Path, gallery: str | Path, k: int = 5, out_path: str | Path | None = None) -> Path:
        """Rank the gallery and save a comparison strip with the top-k matches."""
        query = Path(query)
        matches = self.rank(query, gallery, k=k)
        q_tile = (query, "QUERY", (80, 170, 255))
        match_tiles = [
            (item.path, f"#{rank}  sim={item.score:.4f}", (200, 200, 200)) for rank, item in enumerate(matches, start=1)
        ]
        out_path = Path(out_path) if out_path is not None else query.with_name(f"{query.stem}_reid_top{k}.jpg")
        return plot_reid_retrieval([[q_tile, *match_tiles]], out_path)

    def __call__(self, query: str | Path, gallery: str | Path, k: int = 5, out_path: str | Path | None = None) -> Path:
        """Shortcut for ``visualize()``."""
        return self.visualize(query, gallery, k=k, out_path=out_path)
