from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt


YOLO26 = [
    ("n", 1.7, 40.9),
    ("s", 2.8, 48.6),
    ("m", 5.1, 53.1),
    ("l", 7.1, 55.0),
    ("x", 13.2, 57.5),
]

YOLO26_RTDETR = [
    ("n", 1.7, 41.2),
    ("s", 3.8, 50.8),
    ("m", 5.6, 53.5),
    ("l", 7.0, 55.2),
    ("x", 11.6, 56.6),
]

RTDETR_V4 = [
    ("s", 3.66, 49.8),
    ("m", 5.91, 53.7),
    ("l", 8.07, 55.4),
    ("x", 12.90, 57.0),
]

LW_DETR = [
    ("n", 1.9, 42.9),
    ("s", 2.6, 48.0),
    ("m", 4.4, 52.6),
    ("l", 6.9, 56.1),
    ("x", 13.0, 58.3),
]

RF_DETR = [
    ("n", 2.3, 48.0),
    ("s", 3.5, 52.9),
    ("m", 4.4, 54.7),
    ("l", 6.8, 56.5),
    ("xl", 11.5, 58.6),
    ("2xl", 17.2, 60.1),
]


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
    fig, ax = plt.subplots(figsize=(7.5, 4.5), dpi=120)
    ax.set_axisbelow(True)
    ax.grid(True, which="major", linestyle="--", linewidth=0.6, alpha=0.6)

    palette = plt.get_cmap("tab10")
    plot_series(ax, YOLO26, "YOLO26", palette(0), "o", 8)
    plot_series(ax, YOLO26_RTDETR, "YOLO26-RTDETR", palette(1), "s", -14)
    plot_series(ax, RTDETR_V4, "RT-DETRv4", palette(2), "D", 10)
    plot_series(ax, LW_DETR, "LW-DETR", palette(3), "^", 14)
    plot_series(ax, RF_DETR, "RF-DETR", palette(4), "v", -12)

    ax.set_title(
        "Latency vs mAP: YOLO26, YOLO26-RTDETR, RT-DETRv4, LW-DETR, RF-DETR"
    )
    ax.set_xlabel("Latency (ms)")
    ax.set_ylabel("mAP (COCO)")
    ax.legend(frameon=False, loc="lower right")
    ax.margins(x=0.08, y=0.08)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    if show:
        plt.show()


def parse_args() -> argparse.Namespace:
    default_output = Path(__file__).with_name("benchmark_plot.png")
    parser = argparse.ArgumentParser(
        description=(
            "Plot YOLO26, YOLO26-RTDETR, RT-DETRv4, LW-DETR, and RF-DETR latency vs mAP benchmarks."
        )
    )
    parser.add_argument("--output", type=Path, default=default_output)
    parser.add_argument("--show", action="store_true", help="Display the plot window.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    build_plot(args.output, args.show)


if __name__ == "__main__":
    main()
