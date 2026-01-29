from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt


# Apple M5 CPU ONNX benchmarks
# Format: (size_label, latency_ms, mAP50-95)

YOLO26 = [
    ("n", 21.0, 40.9),
    ("s", 61.7, 48.6),
    ("m", 168.6, 53.1),
    ("l", 219.1, 55.0),
    ("x", 439.3, 57.5),
]

RF_DETR = [
    ("n", 79.6, 48.4),
    ("s", 145.5, 53.0),
    ("m", 192.0, 54.7),
    ("l", 307.2, 56.5),
    ("x", 688.3, 58.6),
    ("xxl", 956.2, 60.1),
]

LWDETR = [
    ("t", 73.2, 42.9),
    ("s", 110.9, 48.1),
    ("m", 226.7, 52.6),
    ("l", 353.3, 56.1),
    ("x", 765.1, 58.3),
]

DEIM_DFINE = [
    ("n", 31.4, 43.0),
    ("s", 78.7, 49.0),
    ("m", 158.4, 52.7),
    ("l", 244.5, 54.7),
    ("x", 486.8, 56.5),
]

DEIM_RTDETRV2 = [
    ("r18", 157.8, 49.0),
    ("r34", 233.6, 50.9),
    ("r50m", 254.3, 53.2),
    ("r50", 335.4, 54.3),
    ("r101", 584.6, 55.5),
]

DEIMV2 = [
    ("pico", 22.4, 38.5),
    ("n", 30.5, 43.0),
    ("s", 157.2, 50.9),
    ("m", 240.9, 53.0),
    ("l", 386.2, 56.0),
    ("x", 525.0, 57.8),
]

# Free points without connecting lines: (label, latency_ms, mAP, color_index, marker, label_offset)
FREE_POINTS = []


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
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=(10, 6), dpi=120)
    ax.set_axisbelow(True)
    ax.grid(True, which="major", linestyle="--", linewidth=0.6, alpha=0.6)

    palette = plt.get_cmap("tab10")
    plot_series(ax, YOLO26, "YOLO26", palette(0), "o", 8)
    plot_series(ax, RF_DETR, "RF-DETR", palette(1), "s", -12)
    plot_series(ax, LWDETR, "LW-DETR", palette(2), "^", 8)
    plot_series(ax, DEIM_DFINE, "DEIM D-FINE", palette(3), "D", -12)
    plot_series(ax, DEIM_RTDETRV2, "DEIM RT-DETRv2", palette(4), "v", 8)
    plot_series(ax, DEIMV2, "DEIMv2", palette(5), "p", -12)

    # Add free points without connecting to a line
    for label, latency, mAP, color_idx, marker, label_offset in FREE_POINTS:
        ax.scatter(latency, mAP, color=palette(color_idx), marker=marker, s=70, zorder=5)
        ax.annotate(
            label,
            (latency, mAP),
            textcoords="offset points",
            xytext=(0, label_offset),
            ha="center",
            fontsize=9,
            color=palette(color_idx),
        )

    ax.set_title("Object Detection Models: Latency vs mAP (Apple M5 CPU, ONNX)")
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
    default_output = Path(__file__).with_name("benchmark_plot_cpu.png")
    parser = argparse.ArgumentParser(
        description="Plot object detection model latency vs mAP benchmarks (Apple M5 CPU, ONNX)."
    )
    parser.add_argument("--output", type=Path, default=default_output)
    parser.add_argument("--show", action="store_true", help="Display the plot window.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    build_plot(args.output, args.show)


if __name__ == "__main__":
    main()
