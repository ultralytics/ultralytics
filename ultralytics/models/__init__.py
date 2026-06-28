# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from .fastsam import FastSAM
from .nas import NAS
from .rtdetr import RTDETR
from .yolo import YOLO, YOLOE, YOLOWorld
from .yolodetr import YOLODETR

__all__ = "NAS", "RTDETR", "SAM", "YOLO", "YOLODETR", "YOLOE", "FastSAM", "YOLOWorld"  # allow simpler import


def __getattr__(name):
    """Lazy-import SAM so standard YOLO imports don't load optional torchvision internals."""
    if name == "SAM":
        # Scoped for import ultralytics speed: SAM pulls optional torchvision-heavy modules.
        from .sam import SAM

        return SAM
    raise AttributeError(f"module {__name__} has no attribute {name}")
