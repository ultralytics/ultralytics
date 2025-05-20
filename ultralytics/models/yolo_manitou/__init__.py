from ultralytics.models.yolo_manitou import detect
from ultralytics.models.yolo_manitou import detect_multiCam
from .model import YOLOManitou


__all__ = ["detect", "detect_multiCam", "YOLOManitou"]  # allow simpler import