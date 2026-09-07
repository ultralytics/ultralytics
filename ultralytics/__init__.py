# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

__version__ = "8.4.142"

import importlib
import os
import sys
from typing import TYPE_CHECKING

# Set ENV variables (place before imports)
if not os.environ.get("OMP_NUM_THREADS"):
    os.environ["OMP_NUM_THREADS"] = "1"  # default for reduced CPU utilization during training

from ultralytics.utils import ASSETS, SETTINGS
from ultralytics.utils.checks import check_yolo as checks
from ultralytics.utils.downloads import download

settings = SETTINGS

MODELS = ("YOLO", "YOLOWorld", "YOLOE", "NAS", "SAM", "FastSAM", "RTDETR", "LLM")
PLATFORM_EXPORTS = ("Platform", "AsyncPlatform", "APIError", "APIConnectionError")

__all__ = (  # noqa: PLE0604
    "__version__",
    "ASSETS",
    *MODELS,
    *(PLATFORM_EXPORTS if sys.version_info >= (3, 11) else ()),
    "checks",
    "download",
    "settings",
)

if TYPE_CHECKING:
    # Enable hints for type checkers
    from ultralytics.models import LLM, YOLO, YOLOWorld, YOLOE, NAS, SAM, FastSAM, RTDETR  # noqa
    from ultralytics_platform import APIConnectionError, APIError, AsyncPlatform, Platform  # noqa: F401


def __getattr__(name: str):
    """Lazy-import public classes on first access."""
    if name in MODELS:
        return getattr(importlib.import_module("ultralytics.models"), name)
    if name in PLATFORM_EXPORTS:
        if sys.version_info < (3, 11):
            raise ImportError("Ultralytics Platform requires Python 3.11 or newer.")
        return getattr(importlib.import_module("ultralytics_platform"), name)
    raise AttributeError(f"module {__name__} has no attribute {name}")


def __dir__():
    """Extend dir() to include lazily available public names for IDE autocompletion."""
    return sorted(set(globals()) | set(MODELS) | set(PLATFORM_EXPORTS))


if __name__ == "__main__":
    print(__version__)
