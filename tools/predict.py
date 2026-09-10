"""Predict images, videos, or camera streams with any supported YOLO task."""

from ultralytics import YOLO
from ultralytics.cfg import TASK2MODEL
from ultralytics.utils import ASSETS

TASK = "detect"
MODEL = None  # Set to the actual best.pt path to predict with your trained model.
SOURCE = ASSETS  # Image, directory, video path, stream URL, or 0 for a webcam.
# device=None: framework default; "cpu", 0 (CUDA), or "mps" (Apple Silicon).
ARGS = {"imgsz": 640, "device": None, "save": True, "show": False, "project": None, "name": "predict"}

if __name__ == "__main__":
    model = YOLO(MODEL or TASK2MODEL[TASK])
    for result in model.predict(source=SOURCE, stream=True, **ARGS):
        pass  # Consume every result to run inference and save visualizations.
