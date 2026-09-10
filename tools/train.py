"""Train any supported YOLO task by editing the configuration below."""

from ultralytics import YOLO
from ultralytics.cfg import TASK2DATA, TASK2MODEL

TASK = "detect"
MODEL = None  # None: task default; otherwise a .pt checkpoint or model .yaml path.
DATA = None  # None: task demo data; otherwise dataset YAML or classification directory.
# device=None: framework default; "cpu", 0 (CUDA), or "mps" (Apple Silicon).
ARGS = {"epochs": 100, "imgsz": 640, "batch": 8, "workers": 0, "device": None, "project": None, "name": "train"}

if __name__ == "__main__":
    model = YOLO(MODEL or TASK2MODEL[TASK])
    model.train(data=DATA or TASK2DATA[model.task], **ARGS)
    print(f"Training output: {model.trainer.save_dir}")
    print(f"Best weights: {model.trainer.best}")
