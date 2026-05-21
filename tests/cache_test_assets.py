# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""Pre-download shared test assets to avoid race conditions under pytest-xdist.

Run this script once before `pytest -n auto` to ensure all model weights,
datasets, and solution assets are already cached locally. Each xdist worker
will then only create symlinks / read existing files instead of competing
to download the same remote resources.
"""

from ultralytics.cfg import TASK2MODEL
from ultralytics.data.utils import check_cls_dataset, check_det_dataset
from ultralytics.utils import ASSETS_URL, WEIGHTS_DIR
from ultralytics.utils.downloads import attempt_download_asset, safe_download

# ---------------------------------------------------------------------------
# 1. Model weights referenced by the test suite
# ---------------------------------------------------------------------------
WEIGHTS = [
    # Core task models
    *TASK2MODEL.values(),
    # Extra models used in parametrization / specific tests
    "yolo11n-grayscale.pt",
    "rtdetr-l.pt",
    "FastSAM-s.pt",
    "mobile_sam.pt",
    "sam2.1_b.pt",
    "yolov8s-world.pt",
    "yolov8s-worldv2.pt",
    "yoloe-11s-seg.pt",
    "yoloe-11s-seg-pf.pt",
    "solutions_ci_parking_model.pt",
]

# ---------------------------------------------------------------------------
# 2. Datasets referenced by the test suite
# ---------------------------------------------------------------------------
DATASETS = [
    "coco8.yaml",
    "coco8-seg.yaml",
    "coco8-pose.yaml",
    "coco8-grayscale.yaml",
    "coco8-multispectral.yaml",
    "coco12-formats.yaml",
    "coco128.yaml",
    "coco128-seg.yaml",
    "dota8.yaml",
    "dota128.yaml",
    "cityscapes8.yaml",
    "imagenet10",
    "imagenet100",
]

# ---------------------------------------------------------------------------
# 3. Solution assets (videos, JSONs, etc.)
# ---------------------------------------------------------------------------
SOLUTION_ASSETS = [
    "solutions_ci_demo.mp4",
    "decelera_landscape_min.mov",
    "solution_ci_pose_demo.mp4",
    "solution_ci_parking_demo.mp4",
    "solution_vertical_demo.mp4",
    "solution_ci_parking_areas.json",
    "solutions_ci_parking_model.pt",
]


def cache_weights() -> None:
    """Download all model weights used by tests."""
    print("[cache] Downloading model weights ...")
    for w in WEIGHTS:
        attempt_download_asset(w)
    print("[cache] Weights done.")


def cache_datasets() -> None:
    """Download / extract all datasets used by tests."""
    print("[cache] Downloading datasets ...")
    for ds in DATASETS:
        if ds.startswith("imagenet"):
            check_cls_dataset(ds, autodownload=True)
        else:
            check_det_dataset(ds, autodownload=True)
    print("[cache] Datasets done.")


def cache_solution_assets() -> None:
    """Download solution test assets (videos, parking json, etc.)."""
    print("[cache] Downloading solution assets ...")
    cache_dir = WEIGHTS_DIR / "solution_assets"
    cache_dir.mkdir(parents=True, exist_ok=True)
    for asset in SOLUTION_ASSETS:
        dst = cache_dir / asset
        if not dst.exists():
            safe_download(url=f"{ASSETS_URL}/{asset}", dir=cache_dir)
    print("[cache] Solution assets done.")


def main() -> None:
    cache_weights()
    cache_datasets()
    cache_solution_assets()
    print("[cache] All test assets are ready.")


if __name__ == "__main__":
    main()
