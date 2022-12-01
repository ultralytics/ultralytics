from pathlib import Path

from ultralytics.yolo.v8 import classify,segment
from ultralytics.yolo.v8 import detect

ROOT = Path(__file__).parents[0]  # yolov8 ROOT

__all__ = ["classify", "segment", "detect"]
