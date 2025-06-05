from ultralytics.models.yolo_manitou import detect
from ultralytics.models.yolo_manitou import detect_multiCam
from ultralytics.models.yolo_manitou import segment
from .model import YOLOManitou, YOLOManitou_MultiCam


__all__ = ["detect", "detect_multiCam", "segment", "YOLOManitou", "YOLOManitou_MultiCam"]  # allow simpler import