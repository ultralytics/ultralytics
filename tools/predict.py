# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""Predict images, videos, or camera streams with any supported YOLO task."""

import argparse

from ultralytics import YOLO
from ultralytics.cfg import TASK2MODEL, TASKS
from ultralytics.utils import ASSETS

ARGS = {"imgsz": 640, "device": None, "project": None, "name": "predict"}

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=TASKS, default="detect", help="Default model task (default: detect)")
    parser.add_argument("--model", help="Model checkpoint, e.g. the actual best.pt path; defaults to the task model")
    parser.add_argument("--source", default=str(ASSETS), help="Image, directory, video, stream URL, or 0 for a webcam")
    parser.add_argument("--no-save", dest="save", action="store_false", help="Disable saving visualizations")
    parser.add_argument("--show", action="store_true", help="Display predictions in a window")
    for key, default in ARGS.items():
        parser.add_argument(
            f"--{key}", type=type(default) if default is not None else str, default=default, help="Default: %(default)s"
        )
    args = vars(parser.parse_args())
    task = args.pop("task")
    model = YOLO(args.pop("model") or TASK2MODEL[task])
    for result in model.predict(stream=True, **args):
        pass  # Consume every result to run inference and save visualizations.
