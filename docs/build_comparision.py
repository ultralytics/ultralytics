# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""Effortlessly generate model comparison pages with this script."""

import os
from itertools import permutations

MKDOCS_FILE = "mkdocs.yml"

# Model's details i.e. Ultralytics YOLO11, YOLOv10, YOLOv9, Ultralytics YOLOv8 and so on.
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
        "n": {"speed": 2.3, "mAP": 37.8},  # it's official t variant for YOLOv9
        "s": {"speed": 3.54, "mAP": 46.5},
        "m": {"speed": 6.43, "mAP": 51.5},
        "l": {"speed": 7.16, "mAP": 52.8},  # it's official c variant for YOLOv9
        "x": {"speed": 16.77, "mAP": 55.1},  # it's official e variant for YOLOv9
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
        "n": {"speed": 2.84, "mAP": 39.9},  # it's official t variant for PP-YOLOE+
        "s": {"speed": 2.62, "mAP": 43.7},
        "m": {"speed": 5.56, "mAP": 49.8},
        "l": {"speed": 8.36, "mAP": 52.9},
        "x": {"speed": 14.3, "mAP": 54.7},
    },
    "DAMO-YOLO": {
        "n": {"speed": 2.32, "mAP": 42.0},  # it's official t variant for DAMO-YOLO
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


def main():
    """
    Generates model comparison markdown files for documenting the performance of various models.

    The comparisons include metrics such as mAP (mean Average Precision) and inference speed for different variants of
    the models.
    """
    DOCS_DIR = os.path.join(os.getcwd(), "docs/en/compare")  # Model's comparison directory of the docs
    os.makedirs(DOCS_DIR, exist_ok=True)  # Ensure the directory exists

    model_pairs = list(permutations(data.keys(), 2))  # Generate all combinations of models
    variant_order = ["n", "s", "m", "b", "l", "x"]  # Define variant order preference

    # Create model comparison pages
    for model1, model2 in model_pairs:
        filename = f"{model1.lower()}-vs-{model2.lower()}.md"
        filepath = os.path.join(DOCS_DIR, filename)
        with open(filepath, "w") as f:
            # Metadata Section
            f.write("---\n")
            f.write("---\n")

            f.write(f"# {model1} VS {model2}\n\n")  # Page Title

            # mAP Comparison Table
            f.write("## mAP Comparison\n\n")
            f.write(
                f"| **Variant** | <center><span style='width: 400px;'>**mAP<sup>val<br>50**<br>**{model1}**</span></center> | <center><span style='width: 400px;'>**mAP<sup>val<br>50**<br>**{model2}**</span></center> |\n")
            f.write("|----|----------------------------------|------------------------------------|\n")

            variants = set(data[model1].keys()).union(set(data[model2].keys()))  # Add rows for mAP comparison

            for variant in sorted(variants, key=lambda v: variant_order.index(v)):
                m1_map = data[model1][variant]["mAP"] if variant in data[model1] else "N/A"
                m2_map = data[model2][variant]["mAP"] if variant in data[model2] else "N/A"
                f.write(f"| {variant} | {m1_map} | {m2_map} |\n")

            # Speed Comparison Table
            f.write("\n## Speed Comparison\n\n")
            f.write(
                f"| **Variant** | <center><span style='width: 200px;'>**Speed**<br><sup>T4 TensorRT10<br>(ms)</sup><br>**{model1}**</span></center> | <center><span style='width: 200px;'>**Speed**<br><sup>T4 TensorRT10<br>(ms)</sup><br>**{model2}**</span></center> |\n")
            f.write("|---------|-----------------------|-----------------------|\n")

            for variant in sorted(variants, key=lambda v: variant_order.index(v)):  # Add rows for speed comparison
                m1_speed = data[model1][variant]["speed"] if variant in data[model1] else "N/A"
                m2_speed = data[model2][variant]["speed"] if variant in data[model2] else "N/A"
                f.write(f"| {variant} | {m1_speed} | {m2_speed} |\n")


if __name__ == "__main__":
    main()
