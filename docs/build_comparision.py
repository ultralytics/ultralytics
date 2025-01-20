# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""Effortlessly generate model comparison pages with this script."""

import os
from itertools import combinations

# Model's details i.e Ultralytics YOLO11, YOLOv10, YOLOv9
data = {
    "YOLO11": {
        "n": {"speed": 1.55, "mAP": 39.5},
        "s": {"speed": 2.63, "mAP": 47.0},
        "m": {"speed": 5.27, "mAP": 51.4},
        "l": {"speed": 6.84, "mAP": 53.2},
        "x": {"speed": 12.49, "mAP": 54.7},
    },
    "YOLOv10": {
        "n": {"speed": 1.56, "mAP": 39.5},
        "s": {"speed": 2.66, "mAP": 46.7},
        "m": {"speed": 5.48, "mAP": 51.3},
        "b": {"speed": 6.54, "mAP": 52.7},
        "l": {"speed": 8.33, "mAP": 53.3},
        "x": {"speed": 12.2, "mAP": 54.4},
    },
    "YOLOv9": {
        "t": {"speed": 2.3, "mAP": 37.8},
        "s": {"speed": 3.54, "mAP": 46.5},
        "m": {"speed": 6.43, "mAP": 51.5},
        "c": {"speed": 7.16, "mAP": 52.8},
        "e": {"speed": 16.77, "mAP": 55.1},
    },
    "YOLOv8": {
        "n": {"speed": 1.47, "mAP": 37.3},
        "s": {"speed": 2.66, "mAP": 44.9},
        "m": {"speed": 5.86, "mAP": 50.2},
        "l": {"speed": 9.06, "mAP": 52.9},
        "x": {"speed": 14.37, "mAP": 53.9},
    },
    "YOLOv7": {"l": {"speed": 6.84, "mAP": 51.4}, "x": {"speed": 11.57, "mAP": 53.1}},
    "YOLOv6-3.0": {
        "n": {"speed": 1.17, "mAP": 37.5},
        "s": {"speed": 2.66, "mAP": 45.0},
        "m": {"speed": 5.28, "mAP": 50.0},
        "l": {"speed": 8.95, "mAP": 52.8},
    },
    "YOLOv5": {
        "s": {"speed": 1.92, "mAP": 37.4},
        "m": {"speed": 4.03, "mAP": 45.4},
        "l": {"speed": 6.61, "mAP": 49.0},
        "x": {"speed": 11.89, "mAP": 50.7},
    },
    "PP-YOLOE+": {
        "t": {"speed": 2.84, "mAP": 39.9},
        "s": {"speed": 2.62, "mAP": 43.7},
        "m": {"speed": 5.56, "mAP": 49.8},
        "l": {"speed": 8.36, "mAP": 52.9},
        "x": {"speed": 14.3, "mAP": 54.7},
    },
    "DAMO-YOLO": {
        "t": {"speed": 2.32, "mAP": 42.0},
        "s": {"speed": 3.45, "mAP": 46.0},
        "m": {"speed": 5.09, "mAP": 49.2},
        "l": {"speed": 7.18, "mAP": 50.8},
    },
    "YOLOX": {
        "s": {"speed": 2.56, "mAP": 40.5},
        "m": {"speed": 5.43, "mAP": 46.9},
        "l": {"speed": 9.04, "mAP": 49.7},
        "x": {"speed": 16.1, "mAP": 51.1},
    },
    "RTDETRv2": {
        "s": {"speed": 5.03, "mAP": 48.1},
        "m": {"speed": 7.51, "mAP": 51.9},
        "l": {"speed": 9.76, "mAP": 53.4},
        "x": {"speed": 15.03, "mAP": 54.3},
    },
}

# Directory for the docs
DOCS_DIR = "/en/comparisons"

# Ensure the directory exists
os.makedirs(DOCS_DIR, exist_ok=True)

# Generate all combinations of models
model_pairs = list(combinations(data.keys(), 2))

# Create documentation pages
for model1, model2 in model_pairs:
    filename = f"{model1.lower()}-vs-{model2.lower()}.md"
    filepath = os.path.join(DOCS_DIR, filename)

    with open(filepath, "w") as f:
        f.write(f"# {model1} vs {model2}\n\n")

        # Accuracy Table
        f.write("## Accuracy (mAP) Comparison\n\n")
        f.write("| Model | Variant | mAP (%) |\n")
        f.write("|-------|---------|---------|\n")
        for model, details in [(model1, data[model1]), (model2, data[model2])]:
            for variant, stats in details.items():
                f.write(f"| {model} | {variant} | {stats['mAP']} |\n")

        # Speed Table
        f.write("\n## Speed Comparison\n\n")
        f.write("| Model | Variant | Speed (ms) |\n")
        f.write("|-------|---------|------------|\n")
        for model, details in [(model1, data[model1]), (model2, data[model2])]:
            for variant, stats in details.items():
                f.write(f"| {model} | {variant} | {stats['speed']} |\n")
