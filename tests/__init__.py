# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from ultralytics.cfg import TASK2DATA, TASK2MODEL, TASKS
from ultralytics.utils import ASSETS, WEIGHTS_DIR, checks

# Shared test constants for model, config, data source, and environment info
MODEL = WEIGHTS_DIR / "path with spaces" / "yolo26n.pt"  # path with spaces to test path handling
CFG = "yolo26n.yaml"
SOURCE = ASSETS / "bus.jpg"
SOURCES_LIST = [ASSETS / "bus.jpg", ASSETS, ASSETS / "*", ASSETS / "**/*.jpg"]  # file, dir, and glob patterns
CUDA_IS_AVAILABLE = checks.cuda_is_available()
CUDA_DEVICE_COUNT = checks.cuda_device_count()
# Tasks whose predictor requires a PAIRED source and so cannot take the mono images in ASSETS.
# `s3d` raises "requires both left and right images" on a single .jpg, by design — it needs a
# (left, right) tuple plus calibration. These tasks are excluded from the shared mono-source
# predict/track/export/results tests below and keep their own end-to-end coverage in
# tests/test_s3d.py (train, val, predict, ONNX and TensorRT export). Dataset-driven tests are
# unaffected: they read a stereo dataset YAML, not ASSETS.
PAIRED_SOURCE_TASKS = {"s3d"}

TASK_MODEL_DATA = sorted(
    [(task, WEIGHTS_DIR / TASK2MODEL[task], TASK2DATA[task]) for task in TASKS]
)  # (task, model, data) tuples — train/val are dataset-driven, so paired-source tasks belong here
MONO_TASK_MODEL_DATA = [t for t in TASK_MODEL_DATA if t[0] not in PAIRED_SOURCE_TASKS]
MODELS = sorted(
    [*(m for task, m in TASK2MODEL.items() if task not in PAIRED_SOURCE_TASKS), "yolo11n-grayscale.pt"]
)  # task models plus grayscale variant; consumed only by mono-source tests
SOLUTION_ASSETS = {
    "demo_video": "solutions_ci_demo.mp4",
    "crop_video": "decelera_landscape_min.mov",
    "pose_video": "solution_ci_pose_demo.mp4",
    "parking_video": "solution_ci_parking_demo.mp4",
    "vertical_video": "solution_vertical_demo.mp4",
    "track_video": "decelera_portrait_min.mov",
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
