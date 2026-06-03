# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from ultralytics.data.utils import IMG_FORMATS
from ultralytics.utils import LOGGER
from ultralytics.utils.plotting import plot_reid_retrieval


@dataclass(slots=True)
class RetrievalItem:
    """A single ranked gallery match for a query image."""

    path: Path
    score: float


class ReIDVisualizer:
    """Rank gallery images for a query person and render the top matches.

    This helper uses any Ultralytics ReID model (PyTorch or exported ONNX) to extract embeddings,
    ranks gallery images by cosine similarity, and writes a simple comparison montage.

    Args:
        model: Path or name of a ReID model, e.g. ``best.pt`` or ``best.onnx``.
        imgsz: Inference image size.
        device: Optional inference device.

    Examples:
        >>> from ultralytics.solutions import ReIDVisualizer
        >>> viz = ReIDVisualizer("best.onnx", imgsz=448)
        >>> viz.visualize("query.jpg", "bounding_box_test", k=5)
    """

    def __init__(self, model: str | Path, imgsz: int = 448, device: str | None = None) -> None:
        from ultralytics import YOLO

        self.model = YOLO(model, task="reid")
        self.imgsz = imgsz
        self.device = device

    def _embed(self, image: str | Path) -> np.ndarray:
        """Extract a single L2-normalized embedding."""
        result = self.model.predict(image, imgsz=self.imgsz, task="reid", device=self.device, verbose=False)[0]
        if result.embeddings is None:
            raise RuntimeError(f"No embedding produced for {image}")
        embeddings = result.embeddings.data
        if isinstance(embeddings, torch.Tensor):
            embeddings = embeddings.detach().cpu().numpy()
        else:
            embeddings = np.asarray(embeddings)
        return embeddings.reshape(-1).astype(np.float32)

    @staticmethod
    def _pid_from_name(path: Path) -> str:
        """Extract Market-style PID from filename, robust to renamed files.

        Accepted patterns include e.g. ``0001_c3s1_...jpg`` or ``query_0001.jpg``.
        """
        stem = path.stem
        m = re.search(r"(^|_)(-?\d{1,4})(_|$)", stem)
        if m:
            return m.group(2).zfill(4) if not m.group(2).startswith("-") else m.group(2)
        return "na"

    @staticmethod
    def _cam_from_name(path: Path) -> str:
        """Extract Market-style camera token from filename (e.g. 'c3')."""
        m = re.search(r"(c\d)", path.stem)
        return m.group(1) if m else "na"

    def _iter_images(self, root: str | Path) -> list[Path]:
        """Collect image paths under a directory or from a single image path."""
        root = Path(root)
        if root.is_file():
            return [root]
        if not root.exists():
            raise FileNotFoundError(f"{root} does not exist")
        paths = [p for p in sorted(root.rglob("*")) if p.is_file() and p.suffix.lower().lstrip(".") in IMG_FORMATS]
        if not paths:
            raise RuntimeError(f"No image files found under {root}")
        return paths

    @staticmethod
    def _cosine_similarity(query: np.ndarray, gallery: np.ndarray) -> np.ndarray:
        """Compute cosine similarity for already normalized embeddings."""
        query = query / (np.linalg.norm(query) + 1e-12)
        gallery = gallery / (np.linalg.norm(gallery, axis=1, keepdims=True) + 1e-12)
        return gallery @ query

    def rank(
        self,
        query: str | Path,
        gallery: str | Path,
        k: int = 5,
        exclude_query: bool = True,
        ignore_junk: bool = True,
    ) -> list[RetrievalItem]:
        """Rank gallery images for a query image."""
        query = Path(query)
        query_pid = self._pid_from_name(query)
        gallery_paths = self._iter_images(gallery)
        query_emb = self._embed(query)

        gallery_embs = []
        filtered_paths = []
        for path in gallery_paths:
            if exclude_query and path.resolve() == query.resolve():
                continue
            # Market-1501 junk id uses pid=-1; filter by default to avoid meaningless retrievals.
            if ignore_junk and self._pid_from_name(path) == "-1":
                continue
            try:
                gallery_embs.append(self._embed(path))
                filtered_paths.append(path)
            except Exception as exc:  # noqa: BLE001
                LOGGER.warning(f"Skipping {path}: {exc}")

        if not gallery_embs:
            raise RuntimeError("No gallery embeddings could be generated.")

        gallery_embs = np.stack(gallery_embs, axis=0)
        sims = self._cosine_similarity(query_emb, gallery_embs)
        order = np.argsort(-sims)[:k]
        return [RetrievalItem(path=filtered_paths[i], score=float(sims[i])) for i in order]

    def visualize(
        self,
        query: str | Path,
        gallery: str | Path,
        k: int = 5,
        out_path: str | Path | None = None,
        exclude_query: bool = True,
        ignore_junk: bool = True,
    ) -> Path:
        """Rank the gallery and save a comparison strip with the top-k matches."""
        query = Path(query)
        matches = self.rank(query, gallery, k=k, exclude_query=exclude_query, ignore_junk=ignore_junk)
        q_pid = self._pid_from_name(query)
        q_cam = self._cam_from_name(query)

        q_tile = (query, f"QUERY  pid={q_pid} {q_cam}", (80, 170, 255))
        match_tiles = []
        for rank, item in enumerate(matches, start=1):
            pid = self._pid_from_name(item.path)
            cam = self._cam_from_name(item.path)
            is_match = pid == q_pid and pid not in {"na", "-1"}
            color = (70, 200, 120) if is_match else (215, 95, 95)
            match_tiles.append((item.path, f"#{rank}  pid={pid} {cam}  sim={item.score:.4f}", color))

        out_path = Path(out_path) if out_path is not None else query.with_name(f"{query.stem}_reid_top{k}.jpg")
        return plot_reid_retrieval([[q_tile, *match_tiles]], out_path)

    def __call__(self, query: str | Path, gallery: str | Path, k: int = 5, out_path: str | Path | None = None) -> Path:
        """Shortcut for ``visualize()``."""
        return self.visualize(query, gallery, k=k, out_path=out_path)
