"""Validate any supported YOLO task by editing the configuration below."""

from ultralytics import YOLO
from ultralytics.cfg import TASK2DATA, TASK2MODEL

TASK = "detect"
MODEL = None  # Set to the actual best.pt path to validate your trained model.
DATA = None  # Set to your training dataset YAML or classification directory.
# device=None: framework default; "cpu", 0 (CUDA), or "mps" (Apple Silicon).
ARGS = {"split": "val", "imgsz": 640, "batch": 8, "workers": 0, "device": None, "project": None, "name": "val"}

if __name__ == "__main__":
    model = YOLO(MODEL or TASK2MODEL[TASK])
    metrics = model.val(data=DATA or TASK2DATA[model.task], **ARGS)
    print(metrics.results_dict)
