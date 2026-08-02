# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

__version__ = "8.4.115"

import importlib
import os
from typing import TYPE_CHECKING

from autoimport import lazy_import

# Set ENV variables (place before imports)
if not os.environ.get("OMP_NUM_THREADS"):
    os.environ["OMP_NUM_THREADS"] = "1"  # default for reduced CPU utilization during training

_utils = lazy_import("ultralytics.utils")

MODELS = ("YOLO", "YOLOWorld", "YOLOE", "NAS", "SAM", "FastSAM", "RTDETR")

__all__ = (  # noqa: PLE0604
    "__version__",
    "ASSETS",
    *MODELS,
    "checks",
    "download",
    "settings",
)

if TYPE_CHECKING:
    # Enable hints for type checkers
    from ultralytics.models import YOLO, YOLOWorld, YOLOE, NAS, SAM, FastSAM, RTDETR  # noqa
    from ultralytics.utils import ASSETS, SETTINGS
    from ultralytics.utils.checks import check_yolo as checks
    from ultralytics.utils.downloads import download

    settings = SETTINGS


def __getattr__(name: str):
    """Lazy-import public attributes on first access."""
    if name in MODELS:
        return getattr(importlib.import_module("ultralytics.models"), name)
    if name in {"ASSETS", "SETTINGS"}:
        return getattr(_utils, name)
    if name == "settings":
        return _utils.SETTINGS
    if name == "checks":
        return importlib.import_module("ultralytics.utils.checks").check_yolo
    if name == "download":
        return importlib.import_module("ultralytics.utils.downloads").download
    raise AttributeError(f"module {__name__} has no attribute {name}")


def __dir__():
    """Extend dir() with lazily available public names for IDE autocompletion."""
    return sorted(set(globals()) | set(__all__) | {"SETTINGS"})


if __name__ == "__main__":
    print(__version__)
