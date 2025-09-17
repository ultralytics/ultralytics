# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

__version__ = "8.3.201"

import importlib
import os
from pathlib import Path

# Set ENV variables (place before imports)
if not os.environ.get("OMP_NUM_THREADS"):
    os.environ["OMP_NUM_THREADS"] = "1"  # default for reduced CPU utilization during training

ASSETS = Path(__file__).resolve().parents[0] / "assets"

MODELS = ("YOLO", "YOLOWorld", "YOLOE", "NAS", "SAM", "FastSAM", "RTDETR")

__all__ = (
    "__version__",
    "ASSETS",
    *MODELS,
    "checks",
    "download",
    "SETTINGS",
)


def __getattr__(name: str):
    """Lazy-import model classes and utilities on first access."""
    if name in MODELS:
        return getattr(importlib.import_module("ultralytics.models"), name)
    elif name == "SETTINGS":
        from ultralytics.utils import SETTINGS
        return SETTINGS
    elif name == "checks":
        from ultralytics.utils.checks import check_yolo
        return check_yolo
    elif name == "download":
        from ultralytics.utils.downloads import download
        return download
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")


def __dir__():
    """Extend dir() to include lazily available model names for IDE autocompletion."""
    return sorted(set(globals().keys()) | set(MODELS) | {"settings", "checks", "download"})


if __name__ == "__main__":
    print(__version__)
