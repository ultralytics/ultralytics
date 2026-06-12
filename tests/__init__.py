# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from ultralytics.cfg import TASK2DATA, TASK2MODEL, TASKS
from ultralytics.utils import ASSETS, WEIGHTS_DIR, checks

# Shared test constants for model, config, data source, and environment info
MODEL = WEIGHTS_DIR / "path with spaces" / "yolo26n.pt"  # model path with spaces to verify path handling
CFG = "yolo26n.yaml"  # model configuration file for testing
SOURCE = ASSETS / "bus.jpg"  # default single-image test source
SOURCES_LIST = [ASSETS / "bus.jpg", ASSETS, ASSETS / "*", ASSETS / "**/*.jpg"]  # various source patterns for glob/path tests
CUDA_IS_AVAILABLE = checks.cuda_is_available()  # whether a CUDA GPU is accessible
CUDA_DEVICE_COUNT = checks.cuda_device_count()  # number of available CUDA devices
TASK_MODEL_DATA = sorted([(task, WEIGHTS_DIR / TASK2MODEL[task], TASK2DATA[task]) for task in TASKS])  # (task, model, dataset) tuples for parametrized tests
MODELS = sorted([*list(TASK2MODEL.values()), "yolo11n-grayscale.pt"])  # all default task models plus grayscale variant
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
