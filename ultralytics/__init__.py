__version__ = "8.0.0"

from ultralytics.hub import checks
from ultralytics.yolo.engine.model import YOLO
from ultralytics.yolo.utils import ops

__all__ = ["__version__", "YOLO", "hub", "checks"]  # allow simpler import
