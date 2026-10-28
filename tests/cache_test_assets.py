# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""Pre-download every asset CI tests and benchmarks use.

CI's Assets job runs this once per manifest change and saves the result as one cross-OS cache that every other job
restores, so no job downloads from GitHub Releases at run time. Locally, run it once before `pytest -n auto` so xdist
workers reuse existing files instead of racing to download them.
"""

import shutil
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tests import MODEL, SOLUTION_ASSETS
from ultralytics.cfg import TASK2CALIBRATIONDATA, TASK2DATA, TASK2MODEL
from ultralytics.data.utils import check_cls_dataset, check_det_dataset
from ultralytics.utils import ARM64, ASSETS_URL, DATASETS_DIR, IS_RASPBERRYPI, LINUX, LOGGER, WEIGHTS_DIR, checks
from ultralytics.utils.downloads import attempt_download_asset, safe_download

WEIGHTS = [
    *TASK2MODEL.values(),
    "yolo11n-grayscale.pt",
    "rtdetr-l.pt",
    "FastSAM-s.pt",
    "mobile_sam.pt",
    "mobileclip2_b.ts",
    "yoloe-26n-seg.pt",
    "yoloe-26n-seg-pf.pt",
    "yolo26s.pt",
    "yolo26s-seg.pt",
    "yolo26s-pose.pt",
    "yolo26s-obb.pt",
]

DATASETS = [
    *TASK2DATA.values(),
    *TASK2CALIBRATIONDATA.values(),
    "coco8-grayscale.yaml",
    "coco8-multispectral.yaml",
    "coco12-formats.yaml",
]


def cache_weights() -> None:
    """Download all model weights, copying task weights to the 'path with spaces' folder tests and benchmarks load."""
    LOGGER.info("[cache] Downloading model weights ...")
    for w in WEIGHTS:
        attempt_download_asset(WEIGHTS_DIR / w)
    MODEL.parent.mkdir(parents=True, exist_ok=True)
    for w in TASK2MODEL.values():
        shutil.copy2(WEIGHTS_DIR / w, MODEL.parent / w)
    LOGGER.info("[cache] Weights done.")


def cache_datasets() -> None:
    """Download / extract all datasets used by tests."""
    LOGGER.info("[cache] Downloading datasets ...")
    for ds in DATASETS:
        if ds.startswith("imagenet"):
            check_cls_dataset(ds)
        else:
            check_det_dataset(ds, autodownload=True)
    for name in "instances_val2017.json", "person_keypoints_val2017.json":
        safe_download(f"{ASSETS_URL}/{name}", dir=DATASETS_DIR / "annotations")
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
    if IS_RASPBERRYPI or (checks.IS_PYTHON_3_8 and LINUX and ARM64):
        return

    LOGGER.info("[cache] Downloading CLIP text encoder ...")
    from ultralytics.nn.text_model import CLIP

    model = CLIP("ViT-B/32", device=torch.device("cpu"))
    del model
    LOGGER.info("[cache] CLIP text encoder done.")


def main() -> None:
    """Main function to orchestrate caching of all test assets."""
    cache_weights()
    cache_datasets()
    cache_solution_assets()
    cache_clip_model()
    LOGGER.info("[cache] All test assets are ready.")


if __name__ == "__main__":
    main()
