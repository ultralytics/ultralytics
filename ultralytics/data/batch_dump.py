# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""Copy training batches to disk for sampler / dataloader inspection.

Enable during training::

    yolo detect train ... dump_batches=True

Edit ``OUT_DIR``, ``BATCHES_PER_EPOCH``, and ``COPY_LABELS`` below (do not pass paths on the CLI).
"""

from __future__ import annotations

import shutil
from pathlib import Path

from ultralytics.data.utils import img2label_paths

# ---------------------------------------------------------------------------
# Edit these when using dump_batches=True
# ---------------------------------------------------------------------------
OUT_DIR = "runs/sampler_preview"  # output root
BATCHES_PER_EPOCH = 1  # batches copied per epoch; 0 = every batch in the epoch
COPY_LABELS = True  # also copy YOLO label .txt (labels/... mirror layout)


def log_message() -> str:
    """One-line settings summary for trainer startup log."""
    limit = "all batches/epoch" if BATCHES_PER_EPOCH <= 0 else f"{BATCHES_PER_EPOCH} batch(es)/epoch"
    return f"OUT_DIR={Path(OUT_DIR).resolve()} {limit} COPY_LABELS={COPY_LABELS}"


def should_dump(batch_in_epoch: int) -> bool:
    """Return True if this in-epoch batch index should be copied (counter resets each epoch)."""
    return BATCHES_PER_EPOCH <= 0 or batch_in_epoch < BATCHES_PER_EPOCH


def rel_to_dataset_root(path: Path, data_root: Path) -> Path:
    """Path relative to dataset root (preserves images/labels directory layout)."""
    path, data_root = path.resolve(), data_root.resolve()
    try:
        return path.relative_to(data_root)
    except ValueError:
        return Path(path.name)


def copy_keep_layout(src: Path, out_dir: Path, data_root: Path) -> Path:
    """Copy file under ``out_dir`` keeping the same relative path as under ``data_root``."""
    dest = out_dir / rel_to_dataset_root(src, data_root)
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dest)
    return dest


def _image_paths_from_batch(batch: dict) -> list[Path]:
    """Extract image paths from a training batch dict."""
    im_file = batch.get("im_file")
    if im_file is None:
        return []
    if isinstance(im_file, (list, tuple)):
        return [Path(p) for p in im_file]
    return [Path(im_file)]


def dump_training_batch(
    batch: dict,
    out_root: str | Path | None = None,
    data_root: str | Path = "",
    batch_in_epoch: int = 0,
    epoch: int = 0,
    copy_labels: bool | None = None,
) -> Path:
    """Copy all images (and labels) from one training batch, preserving dataset folder layout.

    Args:
        batch (dict): Training batch after ``preprocess_batch`` (must include ``im_file``).
        out_root (str | Path, optional): Output root; defaults to ``OUT_DIR``.
        data_root (str | Path): Dataset ``path`` from data YAML.
        batch_in_epoch (int): Batch index within the current epoch (0-based).
        epoch (int): Current epoch index.
        copy_labels (bool, optional): Copy label files; defaults to ``COPY_LABELS``.

    Returns:
        (Path): Directory containing this batch copy.
    """
    out_root = Path(out_root or OUT_DIR).resolve()
    data_root = Path(data_root).resolve()
    copy_labels = COPY_LABELS if copy_labels is None else copy_labels
    batch_dir = out_root / f"epoch_{epoch:03d}" / f"batch_{batch_in_epoch:04d}"
    batch_dir.mkdir(parents=True, exist_ok=True)

    for src_path in _image_paths_from_batch(batch):
        copy_keep_layout(src_path, batch_dir, data_root)
        if copy_labels:
            label_path = Path(img2label_paths([str(src_path)])[0])
            if label_path.is_file():
                copy_keep_layout(label_path, batch_dir, data_root)

    return batch_dir
