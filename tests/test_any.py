# Ultralytics YOLO 🚀, AGPL-3.0 license

from ultralytics.utils import ASSETS, ROOT, WEIGHTS_DIR, checks, is_dir_writeable

# Constants used in tests
MODEL = WEIGHTS_DIR / "path with spaces" / "yolov8n.pt"  # test spaces in path
CFG = "yolov8n.yaml"
SOURCE = ASSETS / "bus.jpg"
TMP = (ROOT / "../tests/tmp").resolve()  # temp directory for test files
IS_TMP_WRITEABLE = is_dir_writeable(TMP)
CUDA_IS_AVAILABLE = checks.cuda_is_available()
CUDA_DEVICE_COUNT = checks.cuda_device_count()

__all__ = (
    "MODEL",
    "CFG",
    "SOURCE",
    "TMP",
    "IS_TMP_WRITEABLE",
    "CUDA_IS_AVAILABLE",
    "CUDA_DEVICE_COUNT",
)
