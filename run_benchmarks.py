#!/usr/bin/env python

from ultralytics.utils.benchmarks import benchmark

models = [
    # "yolov5su.pt",

    # "yolov8m.pt",
    # "yolov8n.pt",
    # "yolov8s.pt",

    # "yolov8m.pt",
    # "yolov8l.pt",

    # "yolov9t.pt",
    # "yolov10n.pt",

    # "yolov10s.pt",

    # "yolov10m.pt",

    # "yolov10l.pt",

    "yolo11n.pt",
    "yolo11s.pt",
]

precision=[
    False,       # float32
    # True,       # float16
]

batchs = [
    # 1,
    # 8,
    16
]

data = "coco128.yaml"
imgsz = 640
device = "0"

for model in models:
    for half in precision:
        for batch in batchs:
            print(f"Benchmarking model: {model} with half:{half} batch: {batch}")
            benchmark(model=model, data=data, imgsz=imgsz, batch=batch, half=half, device=device)