# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

import shutil
from pathlib import Path

from ultralytics.cfg import TASK2DATA, TASK2MODEL, TASKS
from ultralytics.utils import ASSETS, WEIGHTS_DIR, checks


def isolated_model_path(tmp_path, model):
    """Copy a model to a per-test path so exported artifacts cannot collide under pytest-xdist.

    Args:
        tmp_path (pathlib.Path): Temporary directory provided by pytest.
        model (str | pathlib.Path): Path to the source model file.

    Returns:
        str: Path to the isolated copy in the temporary directory.
    """
    model = Path(model)
    if not model.exists():
        from ultralytics.utils.downloads import attempt_download_asset

        model.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(attempt_download_asset(model.name), model)

    dst = tmp_path / model.name
    shutil.copy(model, dst)
    return str(dst)


# Constants used in tests
MODEL = WEIGHTS_DIR / "path with spaces" / "yolo26n.pt"  # test spaces in path
CFG = "yolo26n.yaml"
SOURCE = ASSETS / "bus.jpg"
SOURCES_LIST = [ASSETS / "bus.jpg", ASSETS, ASSETS / "*", ASSETS / "**/*.jpg"]
CUDA_IS_AVAILABLE = checks.cuda_is_available()
CUDA_DEVICE_COUNT = checks.cuda_device_count()
TASK_MODEL_DATA = sorted([(task, WEIGHTS_DIR / TASK2MODEL[task], TASK2DATA[task]) for task in TASKS])
MODELS = sorted([*list(TASK2MODEL.values()), "yolo11n-grayscale.pt"])
SOLUTION_ASSETS = {
    "demo_video": "solutions_ci_demo.mp4",
    "crop_video": "decelera_landscape_min.mov",
    "pose_video": "solution_ci_pose_demo.mp4",
    "parking_video": "solution_ci_parking_demo.mp4",
    "vertical_video": "solution_vertical_demo.mp4",
    "parking_areas": "solution_ci_parking_areas.json",
    "parking_model": "solutions_ci_parking_model.pt",
}

__all__ = (
    "CFG",
    "CUDA_DEVICE_COUNT",
    "CUDA_IS_AVAILABLE",
    "MODEL",
    "SOLUTION_ASSETS",
    "SOURCE",
    "SOURCES_LIST",
)
