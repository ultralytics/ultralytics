# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

__version__ = "8.3.196"

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


def __getattr__(name: str):
    """
    Dynamically import Ultralytics model classes on first access. This function implements lazy imports for selected
    model classes (e.g., YOLO, NAS, RTDETR, SAM). Instead of loading all models when the Ultralytics package is
    imported, the required class is imported from `ultralytics.models` only when first accessed. This reduces initial
    package load time by ~3%.

    Args:
        name (str): The attribute name being accessed.

    Raises:
        AttributeError: If the requested attribute is not a known model.
    """
    if name in {"NAS", "RTDETR", "SAM", "YOLO", "YOLOE", "FastSAM", "YOLOWorld"}:
        module = importlib.import_module("ultralytics.models")
        return getattr(module, name)
    raise AttributeError(f"module {__name__} has no attribute {name}")
