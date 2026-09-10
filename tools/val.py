"""Validate any supported YOLO task with command-line arguments."""

import argparse

from ultralytics import YOLO
from ultralytics.cfg import TASK2DATA, TASK2MODEL, TASKS

ARGS = {"imgsz": 640, "batch": 8, "workers": 0, "device": None, "project": None, "name": "val"}

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=TASKS, default="detect", help="Default model task (default: detect)")
    parser.add_argument("--model", help="Model checkpoint, e.g. the actual best.pt path; defaults to the task model")
    parser.add_argument("--data", help="Dataset YAML or classification directory (default: model task demo data)")
    parser.add_argument("--split", choices=("train", "val", "test"), default="val", help="Dataset split (default: val)")
    for key, default in ARGS.items():
        parser.add_argument(
            f"--{key}", type=type(default) if default is not None else str, default=default, help="Default: %(default)s"
        )
    args = vars(parser.parse_args())
    task = args.pop("task")
    model = YOLO(args.pop("model") or TASK2MODEL[task])
    metrics = model.val(data=args.pop("data") or TASK2DATA[model.task], **args)
    print(metrics.results_dict)
