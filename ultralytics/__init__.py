# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

__version__ = "8.3.195"

import importlib
import os

# Set ENV variables (place before imports)
if not os.environ.get("OMP_NUM_THREADS"):
    os.environ["OMP_NUM_THREADS"] = "1"  # default for reduced CPU utilization during training

from ultralytics.utils import ASSETS, SETTINGS
from ultralytics.utils.checks import check_yolo as checks
from ultralytics.utils.downloads import download

settings = SETTINGS
__all__ = (
    "__version__",
    "ASSETS",
    "YOLO",
    "YOLOWorld",
    "YOLOE",
    "NAS",
    "SAM",
    "FastSAM",
    "RTDETR",
    "checks",
    "download",
    "settings",
)

# Lazy import (~3% speedup package import time)
def __getattr__(name: str):
    if name in {"NAS", "RTDETR", "SAM", "YOLO", "YOLOE", "FastSAM", "YOLOWorld"}:
        module = importlib.import_module("ultralytics.models")
        return getattr(module, name)
    raise AttributeError(f"module {__name__} has no attribute {name}")
