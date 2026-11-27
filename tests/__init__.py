# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from pathlib import Path

from ultralytics.cfg import TASK2DATA, TASK2MODEL, TASKS
from ultralytics.utils import ASSETS, ASSETS_URL, DATASETS_DIR, WEIGHTS_DIR, checks
from ultralytics.utils.downloads import safe_download

# Shared test constants for model, config, data source, and environment info
MODEL = WEIGHTS_DIR / "path with spaces" / "yolo26n.pt"  # path with spaces to test path handling
CFG = "yolo26n.yaml"
SOURCE = ASSETS / "bus.jpg"
SOURCES_LIST = [ASSETS / "bus.jpg", ASSETS, ASSETS / "*", ASSETS / "**/*.jpg"]  # file, dir, and glob patterns
CUDA_IS_AVAILABLE = checks.cuda_is_available()
CUDA_DEVICE_COUNT = checks.cuda_device_count()
TASK_MODEL_DATA = sorted(
    [(task, WEIGHTS_DIR / TASK2MODEL[task], TASK2DATA[task]) for task in TASKS]
)  # (task, model, data) tuples
MODELS = sorted([*list(TASK2MODEL.values()), "yolo11n-grayscale.pt"])  # task models plus grayscale variant
SOLUTION_ASSETS = {
    "boats": "boats.jpg",
    "demo_video": "solutions_ci_demo.mp4",
    "crop_video": "decelera_landscape_min.mov",
    "pose_video": "solution_ci_pose_demo.mp4",
    "parking_video": "solution_ci_parking_demo.mp4",
    "vertical_video": "solution_vertical_demo.mp4",
    "track_video": "decelera_portrait_min.mov",
    "parking_areas": "solution_ci_parking_areas.json",
    "parking_model": "solutions_ci_parking_model.pt",
    "similarity_images": "4-imgs-similaritysearch.zip",
}


def coco_eval_image() -> Path:
    """Cache official detection and pose annotations and a val2017 image containing a person crowd."""
    for name in "instances", "person_keypoints":
        safe_download(f"{ASSETS_URL}/{name}_val2017.json", dir=DATASETS_DIR / "annotations")
    return safe_download(
        "https://s3.amazonaws.com/images.cocodataset.org/val2017/000000007511.jpg", dir=DATASETS_DIR / "coco-eval"
    )


__all__ = (
    "CFG",
    "CUDA_DEVICE_COUNT",
    "CUDA_IS_AVAILABLE",
    "MODEL",
    "SOLUTION_ASSETS",
    "SOURCE",
    "SOURCES_LIST",
)
