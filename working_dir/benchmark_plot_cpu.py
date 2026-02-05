from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt


# =============================================================================
# SELECT BENCHMARK HERE: "m5", "xeon", "t4", "jetson-agx-thor-gpu",
# "jetson-agx-thor-cpu", "jetson-agx-orin-gpu", "jetson-agx-orin-cpu",
# "jetson-orin-nano-super-gpu", or "jetson-orin-nano-super-cpu"
# =============================================================================
BENCHMARK = "t4"


# =============================================================================
# BENCHMARK DATA
# =============================================================================
# Format: (size_label, latency_ms, mAP50-95)

BENCHMARKS = {
    "m5": {
        "title": "Object Detection Models: Latency vs mAP (Apple M5 CPU, ONNX)",
        "models": {
            "YOLO26": [
                ("n", 21.0, 40.9),
                ("s", 61.7, 48.6),
                ("m", 168.6, 53.1),
                ("l", 219.1, 55.0),
                ("x", 439.3, 57.5),
            ],
            "RF-DETR": [
                ("n", 79.6, 48.4),
                ("s", 145.5, 53.0),
                ("m", 192.0, 54.7),
                ("l", 307.2, 56.5),
                ("x", 688.3, 58.6),
                ("xxl", 956.2, 60.1),
            ],
            "LW-DETR": [
                ("t", 73.2, 42.9),
                ("s", 110.9, 48.1),
                ("m", 226.7, 52.6),
                ("l", 353.3, 56.1),
                ("x", 765.1, 58.3),
            ],
            "DEIM D-FINE": [
                ("n", 31.4, 43.0),
                ("s", 78.7, 49.0),
                ("m", 158.4, 52.7),
                ("l", 244.5, 54.7),
                ("x", 486.8, 56.5),
            ],
            "DEIM RT-DETRv2": [
                ("r18", 157.8, 49.0),
                ("r34", 233.6, 50.9),
                ("r50m", 254.3, 53.2),
                ("r50", 335.4, 54.3),
                ("r101", 584.6, 55.5),
            ],
            "DEIMv2": [
                ("pico", 22.4, 38.5),
                ("n", 30.5, 43.0),
                ("s", 157.2, 50.9),
                ("m", 240.9, 53.0),
                ("l", 386.2, 56.0),
                ("x", 525.0, 57.8),
            ],
        },
    },
    "xeon": {
        "title": "Object Detection Models: Latency vs mAP (Intel Xeon CPU @ 2.00GHz, ONNX)",
        "models": {
            "YOLO26": [
                ("n", 38.4, 40.9),
                ("s", 84.6, 48.6),
                ("m", 220.0, 53.1),
                ("l", 284.2, 55.0),
                ("x", 568.6, 57.5),
            ],
            "RF-DETR": [
                ("n", 117.2, 48.4),
                ("s", 211.1, 53.0),
                ("m", 270.7, 54.7),
                ("l", 427.3, 56.5),
                ("x", 977.2, 58.6),
                ("xxl", 1334.4, 60.1),
            ],
        },
    },
    "t4": {
        "title": "Object Detection Models: Latency vs mAP (Tesla T4 GPU, TensorRT)",
        "models": {
            "YOLO26": [
                ("n", 1.8, 40.9),
                ("s", 2.7, 48.6),
                ("m", 5.2, 53.1),
                ("l", 7.0, 55.0),
                ("x", 13.3, 57.5),
            ],
            "YOLO26-reported": [
                ("n", 1.7, 40.9),
                ("s", 2.5, 48.6),
                ("m", 4.7, 53.1),
                ("l", 6.2, 55.0),
                ("x", 11.8, 57.5),
            ],
            "YOLO26_RTDETR": [
                ("n", 1.7, 41.2),
                ("ns", 2.4, 47.4),
                ("s", 3.1, 49.5),
                ("sm", 3.8, 50.8),
                ("m", 5.6, 53.5),
                ("l", 7.0, 55.2),
                ("x", 11.6, 56.6),
            ],
            "RF-DETR": [
                ("n", 3.0, 48.4),
                ("s", 4.4, 53.0),
                ("m", 5.6, 54.7),
                ("l", 9.0, 56.5),
                ("x", 18.1, 58.6),
                ("xxl", 25.9, 60.1),
            ],
            "RF-DETR-reported": [
                ("n", 2.3, 48.4),
                ("s", 3.5, 53.0),
                ("m", 4.4, 54.7),
                ("l", 6.8, 56.5),
                ("x", 11.5, 58.6),
                ("xxl", 17.2, 60.1),
            ],
        },
    },
    "jetson-agx-thor-gpu": {
        "title": "Object Detection Models: Latency vs mAP (Jetson AGX Thor GPU, TensorRT)",
        "models": {
            "YOLO26": [
                ("n", 1.286, 40.9),
                ("s", 1.607, 48.6),
                ("m", 2.431, 53.1),
                ("l", 3.132, 55.0),
                ("x", 4.98, 57.5),
            ],
            "RF-DETR": [
                ("n", 1.727, 48.4),
                ("s", 2.285, 53.0),
                ("m", 2.808, 54.7),
                ("l", 3.717, 56.5),
                ("x", 4.948, 58.6),
                ("xxl", 7.634, 60.1),
            ],
        },
    },
    "jetson-agx-thor-cpu": {
        "title": "Object Detection Models: Latency vs mAP (Jetson AGX Thor CPU, ONNX)",
        "models": {
            "YOLO26": [
                ("n", 33.89, 40.9),
                ("s", 95.413, 48.6),
                ("m", 273.25, 53.1),
                ("l", 343.833, 55.0),
                ("x", 733.873, 57.5),
            ],
            "RF-DETR": [
                ("n", 131.548, 48.4),
                ("s", 241.317, 53.0),
                ("m", 323.0, 54.7),
                ("l", 520.615, 56.5),
                ("x", 1147.943, 58.6),
                ("xxl", 1615.155, 60.1),
            ],
        },
    },
    "jetson-agx-orin-gpu": {
        "title": "Object Detection Models: Latency vs mAP (Jetson AGX Orin GPU, TensorRT)",
        "models": {
            "YOLO26": [
                ("n", 2.518, 40.9),
                ("s", 3.629, 48.6),
                ("m", 6.116, 53.1),
                ("l", 7.814, 55.0),
                ("x", 13.317, 57.5),
            ],
            "RF-DETR": [
                ("n", 3.628, 48.4),
                ("s", 5.765, 53.0),
                ("m", 7.158, 54.7),
                ("l", 9.662, 56.5),
                ("x", 16.203, 58.6),
                ("xxl", 22.234, 60.1),
            ],
        },
    },
    "jetson-agx-orin-cpu": {
        "title": "Object Detection Models: Latency vs mAP (Jetson AGX Orin CPU, ONNX)",
        "models": {
            "YOLO26": [
                ("n", 51.03, 40.9),
                ("s", 134.092, 48.6),
                ("m", 360.06, 53.1),
                ("l", 455.814, 55.0),
                ("x", 927.723, 57.5),
            ],
            "RF-DETR": [
                ("n", 162.998, 48.4),
                ("s", 300.142, 53.0),
                ("m", 398.034, 54.7),
                ("l", 629.981, 56.5),
                ("x", 1413.559, 58.6),
                ("xxl", 1988.597, 60.1),
            ],
        },
    },
    "jetson-orin-nano-super-gpu": {
        "title": "Object Detection Models: Latency vs mAP (Jetson Orin Nano Super GPU, TensorRT)",
        "models": {
            "YOLO26": [
                ("n", 4.385, 40.9),
                ("s", 6.915, 48.6),
                ("m", 13.399, 53.1),
                ("l", 17.147, 55.0),
                ("x", 32.072, 57.5),
            ],
            "RF-DETR": [
                ("n", 7.223, 48.4),
                ("s", 12.027, 53.0),
                ("m", 15.543, 54.7),
                ("l", 22.32, 56.5),
                ("x", 42.316, 58.6),
                ("xxl", 62.461, 60.1),
            ],
        },
    },
    "jetson-orin-nano-super-cpu": {
        "title": "Object Detection Models: Latency vs mAP (Jetson Orin Nano Super CPU, ONNX)",
        "models": {
            "YOLO26": [
                ("n", 183.055, 40.9),
                ("s", 475.926, 48.6),
                ("m", 1105.241, 53.1),
                ("l", 1460.185, 55.0),
                ("x", 2689.455, 57.5),
            ],
            "RF-DETR": [
                ("n", 651.97, 48.4),
                ("s", 1083.453, 53.0),
                ("m", 1368.739, 54.7),
                ("l", 1913.115, 56.5),
                ("x", 3890.967, 58.6),
                ("xxl", 4939.052, 60.1),
            ],
        },
    },
}

# Marker and label offset config for each model
MODEL_STYLES = {
    "YOLO26": ("o", 8),
    "YOLO26-reported": ("o", -12),
    "YOLO26_RTDETR": ("^", -12),
    "RF-DETR": ("s", -12),
    "RF-DETR-reported": ("s", 8),
    "LW-DETR": ("^", 8),
    "DEIM D-FINE": ("D", -12),
    "DEIM RT-DETRv2": ("v", 8),
    "DEIMv2": ("p", -12),
}


def plot_series(ax, points, label, color, marker, label_offset):
    xs = [point[1] for point in points]
    ys = [point[2] for point in points]
    ax.plot(
        xs,
        ys,
        label=label,
        color=color,
        marker=marker,
        linewidth=2,
        markersize=7,
    )
    for size, x_value, y_value in points:
        ax.annotate(
            size,
            (x_value, y_value),
            textcoords="offset points",
            xytext=(0, label_offset),
            ha="center",
            fontsize=9,
            color=color,
        )


def build_plot(output_path: Path, show: bool) -> None:
    benchmark_data = BENCHMARKS[BENCHMARK]
    models = benchmark_data["models"]
    title = benchmark_data["title"]

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=(10, 6), dpi=120)
    ax.set_axisbelow(True)
    ax.grid(True, which="major", linestyle="--", linewidth=0.6, alpha=0.6)

    palette = plt.get_cmap("tab10")
    for i, (model_name, data) in enumerate(models.items()):
        marker, label_offset = MODEL_STYLES.get(model_name, ("o", 8))
        plot_series(ax, data, model_name, palette(i), marker, label_offset)

    ax.set_title(title)
    ax.set_xlabel("Latency (ms)")
    ax.set_ylabel("mAP50-95 (COCO)")
    ax.legend(frameon=True, loc="lower right", fontsize=9)
    ax.margins(x=0.05, y=0.08)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    if show:
        plt.show()


def parse_args() -> argparse.Namespace:
    default_output = Path(__file__).with_name(f"benchmark_plot_{BENCHMARK}.png")
    parser = argparse.ArgumentParser(
        description="Plot object detection model latency vs mAP benchmarks."
    )
    parser.add_argument("--output", type=Path, default=default_output)
    parser.add_argument("--show", action="store_true", help="Display the plot window.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    build_plot(args.output, args.show)


if __name__ == "__main__":
    main()
