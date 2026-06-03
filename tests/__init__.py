# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from ultralytics.cfg import TASK2DATA, TASK2MODEL, TASKS
from ultralytics.utils import ASSETS, WEIGHTS_DIR, checks

# Constants used in tests
MODEL = WEIGHTS_DIR / "path with spaces" / "yolo26n.pt"  # test spaces in path
CFG = "yolo26n.yaml"
SOURCE = ASSETS / "bus.jpg"
SOURCES_LIST = [ASSETS / "bus.jpg", ASSETS, ASSETS / "*", ASSETS / "**/*.jpg"]
CUDA_IS_AVAILABLE = checks.cuda_is_available()
CUDA_DEVICE_COUNT = checks.cuda_device_count()
ROCM_IS_AVAILABLE = checks.rocm_is_available()
ROCM_DEVICE_COUNT = checks.rocm_device_count()
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
    "ROCM_DEVICE_COUNT",
    "ROCM_IS_AVAILABLE",
    "SOLUTION_ASSETS",
    "SOURCE",
    "SOURCES_LIST",
)
