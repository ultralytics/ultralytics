# Ultralytics YOLO 🚀, AGPL-3.0 license

from ultralytics.models.yolo import classify, detect, obb, pose, segment
from ultralytics.models.yolo import regress

from .model import YOLO, YOLOWorld

__all__ = "classify", "segment", "detect", "pose", "obb", "regress", "YOLO", "YOLOWorld"
