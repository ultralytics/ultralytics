from pathlib import Path

from ultralytics.yolo.v8 import classify

ROOT = Path(__file__).parents[0]  # yolov8 ROOT

__all__ = ["classify"]
