__version__ = "8.0.0.dev0"

from ultralytics.yolo.engine.model import YOLO
from ultralytics.yolo.utils import ops

__all__ = ["__version__", "YOLO", "hub"]  # allow simpler import
