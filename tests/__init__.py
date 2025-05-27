# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from ultralytics.utils import ASSETS, ROOT, WEIGHTS_DIR, checks
from ultralytics.cfg import TASK2DATA, TASK2MODEL, TASKS

# Constants used in tests
MODEL = WEIGHTS_DIR / "path with spaces" / "yolo11n.pt"  # test spaces in path
CFG = "yolo11n.yaml"
SOURCE = ASSETS / "bus.jpg"
SOURCES_LIST = [ASSETS / "bus.jpg", ASSETS, ASSETS / "*", ASSETS / "**/*.jpg"]
TMP = (ROOT / "../tests/tmp").resolve()  # temp directory for test files
CUDA_IS_AVAILABLE = checks.cuda_is_available()
CUDA_DEVICE_COUNT = checks.cuda_device_count()
TASK_MODEL_DATA = [(task, WEIGHTS_DIR / TASK2MODEL[task], TASK2DATA[task]) for task in TASKS]

__all__ = (
    "MODEL",
    "CFG",
    "SOURCE",
    "SOURCES_LIST",
    "TMP",
    "CUDA_IS_AVAILABLE",
    "CUDA_DEVICE_COUNT",
)
