# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""Copy training batches to disk for sampler / dataloader inspection.

Enable during training (omit ``dump_batches_dir`` to disable)::

    yolo detect train ... dump_batches_dir=runs/sampler_preview
    # dump_batches_per_epoch defaults to 1; set e.g. dump_batches_per_epoch=3 to override (0 = all batches)
"""

from __future__ import annotations

import shutil
from pathlib import Path

from ultralytics.data.utils import img2label_paths

DEFAULT_BATCHES_PER_EPOCH = 1  # 0 = every batch in the epoch


def is_enabled(out_dir: str | Path | None) -> bool:
    """Return True if batch dumping is enabled (non-empty output directory)."""
    return bool(out_dir and str(out_dir).strip())


def log_message(
    out_dir: str | Path,
    batches_per_epoch: int = DEFAULT_BATCHES_PER_EPOCH,
) -> str:
    """One-line settings summary for trainer startup log."""
    limit = "all batches/epoch" if batches_per_epoch <= 0 else f"{batches_per_epoch} batch(es)/epoch"
    return f"OUT_DIR={Path(out_dir).resolve()} {limit}"


def should_dump(batch_in_epoch: int, batches_per_epoch: int = DEFAULT_BATCHES_PER_EPOCH) -> bool:
    """Return True if this in-epoch batch index should be copied (counter resets each epoch)."""
    return batches_per_epoch <= 0 or batch_in_epoch < batches_per_epoch


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
    out_root: str | Path,
    data_root: str | Path = "",
    batch_in_epoch: int = 0,
    epoch: int = 0,
) -> Path:
    """Copy all images and labels from one training batch, preserving dataset folder layout.

    Args:
        batch (dict): Training batch after ``preprocess_batch`` (must include ``im_file``).
        out_root (str | Path): Output root (must be non-empty; use ``is_enabled`` to check first).
        data_root (str | Path): Dataset ``path`` from data YAML.
        batch_in_epoch (int): Batch index within the current epoch (0-based).
        epoch (int): Current epoch index.

    Returns:
        (Path): Directory containing this batch copy.
    """
    out_root = Path(out_root).resolve()
    data_root = Path(data_root).resolve()
    batch_dir = out_root / f"epoch_{epoch:03d}" / f"batch_{batch_in_epoch:04d}"
    batch_dir.mkdir(parents=True, exist_ok=True)

    for src_path in _image_paths_from_batch(batch):
        copy_keep_layout(src_path, batch_dir, data_root)
        label_path = Path(img2label_paths([str(src_path)])[0])
        if label_path.is_file():
            copy_keep_layout(label_path, batch_dir, data_root)

    return batch_dir
