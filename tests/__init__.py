# Ultralytics YOLO 🚀, AGPL-3.0 license

from ultralytics.utils import ASSETS, ROOT, WEIGHTS_DIR, checks

# Constants used in tests
MODEL = WEIGHTS_DIR / "path with spaces" / "yolo11n.pt"  # test spaces in path
CFG = "yolo11n.yaml"
SOURCE = ASSETS / "bus.jpg"
SOURCES_LIST = [ASSETS / "bus.jpg", ASSETS, ASSETS / "*", ASSETS / "**/*.jpg"]
TMP = (ROOT / "../tests/tmp").resolve()  # temp directory for test files
CUDA_IS_AVAILABLE = checks.cuda_is_available()
CUDA_DEVICE_COUNT = checks.cuda_device_count()

__all__ = (
    "MODEL",
    "CFG",
    "SOURCE",
    "SOURCES_LIST",
    "TMP",
    "IS_TMP_WRITEABLE",
    "CUDA_IS_AVAILABLE",
    "CUDA_DEVICE_COUNT",
)
