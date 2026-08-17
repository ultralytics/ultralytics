#!/usr/bin/env python3
"""Render the speed comparison charts for the ggml integration report.

Inputs:  benchmarks/bench.jsonl (yolo-cli bench lines, any backend)
         benchmarks/pytorch.jsonl (PyTorch reference lines)
Output:  benchmarks/speed_by_model.png, latency_by_backend.png,
         speed_by_dtype.png, depth_latency.png, and speedup_table.md

De-duplicates bench.jsonl by (backend, model, dtype), keeping the last run.
"""

import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

RES = Path(__file__).resolve().parent.parent / "benchmarks"
DETECT_MODELS = [
    "yolov8n",
    "yolov8s",
    "yolov8m",
    "yolov8l",
    "yolov8x",
    "yolo26n",
    "yolo26s",
    "yolo26m",
    "yolo26l",
    "yolo26x",
]
DEPTH_MODELS = ["yolo26n-depth"]
MODELS = DETECT_MODELS + DEPTH_MODELS
SERIES = [  # (label, backend tag, color)
    ("PyTorch CUDA (f32)", "pytorch-cuda", "#d62728"),
    ("ggml CUDA", "cuda", "#1f77b4"),
    ("ggml Vulkan", "vulkan", "#2ca02c"),
    ("ggml CPU 8T", "cpu", "#7f7f7f"),
]


def present_series(bench):
    """Return only backend series represented in the current benchmark snapshot."""
    tags = {key[0] for key in bench}
    return [series for series in SERIES if series[1] == "pytorch-cuda" or series[1] in tags]


def plot_values(values):
    """Use NaN for missing samples so charts never imply a measured zero latency."""
    return [value if value is not None else math.nan for value in values]


def load(path, key):
    out = {}
    if not path.exists():
        return out
    for line in path.read_text().splitlines():
        if not line.startswith("{"):
            continue
        d = json.loads(line)
        ms = d.get("e2e_ms", {}).get("mean")
        if ms:
            out[key(d)] = (ms, d.get("backend", ""))
    return out


def main():
    bench = load(RES / "bench.jsonl", lambda d: (d["backend"], d["model"], d["dtype"]))
    torchref = load(RES / "pytorch.jsonl", lambda d: (d["model"],))
    series = present_series(bench)

    # ---- Chart 1: e2e by model scale (f16 where available, else first dtype) ----
    fig, axes = plt.subplots(1, 2, figsize=(15, 5.5))
    for ax, family in zip(axes, ["yolov8", "yolo26"]):
        models = [m for m in DETECT_MODELS if m.startswith(family)]
        x = range(len(models))
        width = 0.8 / len(series)
        center = (len(series) - 1) / 2
        for i, (label, tag, color) in enumerate(series):
            vals = []
            for m in models:
                if tag == "pytorch-cuda":
                    vals.append(torchref.get((m,), (None,))[0])
                else:
                    dt = (
                        "f16"
                        if (tag, m, "f16") in bench
                        else next((dt for dt in ("f32", "q8_0") if (tag, m, dt) in bench), None)
                    )
                    vals.append(bench.get((tag, m, dt), (None,))[0])
            bars = ax.bar([xi + (i - center) * width for xi in x], plot_values(vals), width, label=label, color=color)
            for b, v in zip(bars, vals):
                if v:
                    ax.text(b.get_x() + b.get_width() / 2, v + 0.3, f"{v:.1f}", ha="center", va="bottom", fontsize=7)
        ax.set_xticks(list(x))
        ax.set_xticklabels([m.replace(family, "") for m in models])
        ax.set_yscale("log")
        ax.set_title(f"{family} — end-to-end latency on bus.jpg (lower is better)")
        ax.set_ylabel("ms / frame (log scale)")
        ax.legend(fontsize=8)
        ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(RES / "speed_by_model.png", dpi=150)

    # ---- Chart 2: direct latency comparison across all detection models ----
    fig, ax = plt.subplots(figsize=(15, 6))
    x = range(len(DETECT_MODELS))
    width = 0.8 / len(series)
    center = (len(series) - 1) / 2
    for i, (label, tag, color) in enumerate(series):
        vals = []
        for model in DETECT_MODELS:
            if tag == "pytorch-cuda":
                vals.append(torchref.get((model,), (None,))[0])
            else:
                vals.append(bench.get((tag, model, "f16"), (None,))[0])
        ax.bar([p + (i - center) * width for p in x], plot_values(vals), width, label=label, color=color)
    ax.set_xticks(list(x))
    ax.set_xticklabels(DETECT_MODELS, rotation=35, ha="right")
    ax.set_yscale("log")
    ax.set_title("End-to-end latency by backend (ggml F16, lower is better)")
    ax.set_ylabel("ms / frame (log scale)")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(RES / "latency_by_backend.png", dpi=150)

    # ---- Chart 3: dtype sweep at n/s scale, CUDA + Vulkan ----
    small = ["yolov8n", "yolov8s", "yolo26n", "yolo26s"]
    dtypes = ["f32", "f16", "q8_0"]
    fig, ax = plt.subplots(figsize=(12, 5))
    width = 0.13
    colors = {"f32": "#aec7e8", "f16": "#1f77b4", "q8_0": "#ff9896", "pytorch": "#d62728"}
    x = range(len(small))
    col_i = 0
    # PyTorch baseline
    vals = [torchref.get((m,), (None,))[0] for m in small]
    bars = ax.bar(
        [xi - 3.5 * width for xi in x], plot_values(vals), width, label="PyTorch CUDA f32", color=colors["pytorch"]
    )
    for b, v in zip(bars, vals):
        if v:
            ax.text(b.get_x() + b.get_width() / 2, v + 0.2, f"{v:.1f}", ha="center", fontsize=7)
    for tag, taglabel in [("cuda", "CUDA"), ("vulkan", "Vulkan")]:
        for j, dt in enumerate(dtypes):
            vals = [bench.get((tag, m, dt), (None,))[0] for m in small]
            pos = [xi + (col_i - 1.5) * width for xi in x]
            bars = ax.bar(
                pos,
                plot_values(vals),
                width,
                label=f"ggml {taglabel} {dt}",
                color=colors[dt],
                alpha=0.85 if tag == "cuda" else 0.5,
            )
            for b, v in zip(bars, vals):
                if v:
                    ax.text(b.get_x() + b.get_width() / 2, v + 0.2, f"{v:.1f}", ha="center", fontsize=6)
            col_i += 1
    ax.set_xticks(list(x))
    ax.set_xticklabels(small)
    ax.set_title("dtype sweep — end-to-end latency (lower is better)")
    ax.set_ylabel("ms / frame")
    ax.legend(fontsize=7, ncol=4)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(RES / "speed_by_dtype.png", dpi=150)

    # ---- Chart 4: depth backend and dtype latency ----
    model = DEPTH_MODELS[0]
    depth_rows = [("PyTorch CUDA F32", torchref.get((model,), (None,))[0], "#d62728")]
    for tag, color in (("cuda", "#1f77b4"), ("vulkan", "#2ca02c"), ("cpu", "#7f7f7f")):
        for dt in dtypes:
            value = bench.get((tag, model, dt), (None,))[0]
            if value is not None:
                depth_rows.append((f"ggml {tag} {dt}", value, color))
    labels, vals, colors = zip(*depth_rows)
    fig, ax = plt.subplots(figsize=(13, 5.5))
    bars = ax.bar(range(len(labels)), vals, color=colors)
    for bar, value in zip(bars, vals):
        if value:
            ax.text(bar.get_x() + bar.get_width() / 2, value, f"{value:.1f}", ha="center", va="bottom", fontsize=8)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=35, ha="right")
    ax.set_title("YOLO26n absolute-depth end-to-end latency at 768 input")
    ax.set_ylabel("ms / frame")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(RES / "depth_latency.png", dpi=150)

    # ---- Speedup table (markdown) ----
    lines = [
        "# ggml vs PyTorch speedup (e2e mean, bus.jpg)",
        "",
        "| model | dtype | PyTorch | CUDA | speedup | Vulkan | speedup | CPU 8T | speedup |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for m in MODELS:
        pt = torchref.get((m,), (None,))[0]
        for dt in dtypes:
            row = [m, dt, f"{pt:.2f}" if pt else "-"]
            for tag in ("cuda", "vulkan", "cpu"):
                v = bench.get((tag, m, dt), (None,))[0]
                row += [f"{v:.2f}" if v else "-", f"x{pt / v:.2f}" if (v and pt) else "-"]
            lines.append("| " + " | ".join(row) + " |")
    (RES / "speedup_table.md").write_text("\n".join(lines) + "\n")
    print("wrote speed_by_model.png, latency_by_backend.png, speed_by_dtype.png, depth_latency.png, speedup_table.md")


if __name__ == "__main__":
    main()
