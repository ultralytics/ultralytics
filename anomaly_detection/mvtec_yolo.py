"""MVTec-YOLO dataset reader."""

from pathlib import Path
from typing import Any

ROOT = Path("/Users/louis/workspace/ultra_louis_work/buffer/AnomalyData/MVTEC/MVTec-YOLO")

CATEGORIES: list[str] = [
    "leather", "grid", "tile", "wood", "carpet",
    "cable", "hazelnut", "pill", "screw", "toothbrush",
    "metal_nut", "capsule", "bottle", "transistor", "zipper",
]


def collect_images(directory: Path, exts: tuple[str, ...] = (".png", ".jpg", ".jpeg", ".bmp")) -> list[str]:
    """Return sorted image paths under *directory*."""
    return sorted(str(p) for p in directory.rglob("*") if p.suffix.lower() in exts)


def get_mvtec_yolo_data(category: str) -> dict[str, Any]:
    """Return train/test image paths for one MVTec category."""
    if category not in CATEGORIES:
        raise ValueError(f"Unknown category {category!r}. Valid: {CATEGORIES}")

    root = Path(ROOT) / category
    train_im_dir = root / "train" / "good"
    test_im_dir = root / "test"
    test_good_im_dir = test_im_dir / "good"

    train_im_list = collect_images(train_im_dir)
    test_im_list = collect_images(test_im_dir)
    test_good_im_list = collect_images(test_good_im_dir)

    return dict(
        train_im_dir=str(train_im_dir),
        test_im_dir=str(test_im_dir),
        train_im_list=train_im_list,
        test_im_list=test_im_list,
        test_good_im_list=test_good_im_list,
        test_anomaly_im_list=[im for im in test_im_list if im not in test_good_im_list],
        data_yaml=str(root / "data.yaml"),
    )
