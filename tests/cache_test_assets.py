# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""Pre-download shared test assets to avoid race conditions under pytest-xdist.

Run this script once before `pytest -n auto` to ensure all model weights,
datasets, and solution assets are already cached locally. Each xdist worker
can then reuse existing files instead of competing to download the same remote resources.
"""

import shutil
import sys
from argparse import ArgumentParser
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tests import MODEL, SOLUTION_ASSETS
from ultralytics.cfg import TASK2CALIBRATIONDATA, TASK2DATA, TASK2MODEL
from ultralytics.data.utils import check_cls_dataset, check_det_dataset
from ultralytics.utils import ARM64, ASSETS_URL, DATASETS_DIR, IS_RASPBERRYPI, LINUX, LOGGER, WEIGHTS_DIR, checks
from ultralytics.utils.downloads import attempt_download_asset, safe_download

COMMON_WEIGHTS = [
    *TASK2MODEL.values(),
    "yolo11n-grayscale.pt",
    "rtdetr-l.pt",
    "FastSAM-s.pt",
    "mobile_sam.pt",
    "mobileclip_blt.ts",
    "yolov8s-world.pt",
    "yolov8s-worldv2.pt",
    "yoloe-11s-seg.pt",
    "yoloe-11s-seg-pf.pt",
]

SLOW_WEIGHTS = [
    "sam2.1_b.pt",
]

DATASETS = [
    *TASK2DATA.values(),
    *TASK2CALIBRATIONDATA.values(),
    "coco8-grayscale.yaml",
    "coco8-multispectral.yaml",
    "coco12-formats.yaml",
]


def cache_weights(slow: bool = False) -> None:
    """Download all model weights used by tests."""
    LOGGER.info("[cache] Downloading model weights ...")
    weights = COMMON_WEIGHTS + (SLOW_WEIGHTS if slow else [])
    for w in weights:
        attempt_download_asset(WEIGHTS_DIR / w)
    if not MODEL.exists():
        MODEL.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(WEIGHTS_DIR / "yolo26n.pt", MODEL)
    LOGGER.info("[cache] Weights done.")


def cache_datasets() -> None:
    """Download / extract all datasets used by tests."""
    LOGGER.info("[cache] Downloading datasets ...")
    for ds in DATASETS:
        if ds.startswith("imagenet"):
            check_cls_dataset(ds)
        else:
            check_det_dataset(ds, autodownload=True)
    safe_download(f"{ASSETS_URL}/instances_val2017.json", dir=DATASETS_DIR / "annotations")
    LOGGER.info("[cache] Datasets done.")


def cache_solution_assets() -> None:
    """Download solution test assets (videos, parking json, etc.)."""
    LOGGER.info("[cache] Downloading solution assets ...")
    cache_dir = WEIGHTS_DIR / "solution_assets"
    cache_dir.mkdir(parents=True, exist_ok=True)
    for asset in SOLUTION_ASSETS.values():
        dst = cache_dir / asset
        if not dst.exists():
            safe_download(url=f"{ASSETS_URL}/{asset}", dir=cache_dir)
    LOGGER.info("[cache] Solution assets done.")


def cache_clip_model() -> None:
    """Download the CLIP text encoder before xdist workers can race on the shared cache file."""
    if IS_RASPBERRYPI or checks.IS_PYTHON_3_12 or (checks.IS_PYTHON_3_8 and LINUX and ARM64):
        return

    LOGGER.info("[cache] Downloading CLIP text encoder ...")
    from ultralytics.nn.text_model import CLIP

    model = CLIP("ViT-B/32", device=torch.device("cpu"))
    del model
    LOGGER.info("[cache] CLIP text encoder done.")


def parse_args() -> bool:
    """Parse command-line arguments."""
    parser = ArgumentParser(description="Pre-download test assets.")
    parser.add_argument("--slow", action="store_true", help="Include assets used only by slow tests.")
    return parser.parse_args().slow


def main() -> None:
    """Main function to orchestrate caching of all test assets."""
    slow = parse_args()
    cache_weights(slow=slow)
    cache_datasets()
    cache_solution_assets()
    cache_clip_model()
    LOGGER.info("[cache] All test assets are ready.")


if __name__ == "__main__":
    main()
