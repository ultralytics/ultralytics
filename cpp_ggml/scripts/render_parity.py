#!/usr/bin/env python3
"""Render detection parity grids: PyTorch vs ggml (CUDA/Vulkan/CPU).

Runs yolov8n + yolo26n (f16) on the two upstream asset images through every
backend, then tiles the annotated frames into one grid per image:

    benchmarks/parity_grid_bus.png
    benchmarks/parity_grid_zidane.png

Row = model, column = backend. Boxes/scores come from each pipeline's own
renderer, so visual drift is detection drift.
"""

import os
import struct
import subprocess
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT.parent))  # local ultralytics checkout
from ultralytics import YOLO

IMAGES = ["bus", "zidane"]
MODELS = ["yolov8n", "yolo26n"]
BACKENDS = [  # (label, build dir, threads)
    ("ggml CUDA", os.getenv("YOLO_CUDA_BUILD", "build-cuda"), None),
    ("ggml Vulkan", os.getenv("YOLO_VULKAN_BUILD", "build-vulkan"), None),
    ("ggml CPU 8T", os.getenv("YOLO_CPU_BUILD", "build-cpu"), "8"),
]


def torch_frame(model: str, img: str):
    r = YOLO(ROOT / "models" / "pytorch" / f"{model}.pt").predict(
        ROOT.parent / f"ultralytics/assets/{img}.jpg", imgsz=640, conf=0.25, device="cuda:0", verbose=False
    )[0]
    return r.plot()[:, :, ::-1]  # BGR -> RGB


def ggml_frame(model: str, img: str, build: str, threads):
    out = ROOT / "benchmarks" / f"_{model}_{img}_{build}.png"
    cmd = [
        str(ROOT / build / "bin" / "yolo-cli"),
        "detect",
        "--model",
        str(ROOT / f"models/gguf/{model}-f16.gguf"),
        "--source",
        str(ROOT.parent / f"ultralytics/assets/{img}.jpg"),
        "--out",
        str(out),
    ]
    if threads:
        cmd += ["--threads", threads]
    subprocess.run(cmd, check=True, capture_output=True)
    import matplotlib.image as mpimg

    return mpimg.imread(out)


def torch_depth(img: str):
    result = YOLO(ROOT / "models/pytorch/yolo26n-depth.pt").predict(
        ROOT.parent / f"ultralytics/assets/{img}.jpg", imgsz=768, device="cuda:0", verbose=False
    )[0]
    return result.depth.data.float().cpu().numpy()


def ggml_depth(img: str, build: str, threads):
    raw = ROOT / "benchmarks" / f"_yolo26n-depth_{img}_{build}.bin"
    cmd = [
        str(ROOT / build / "bin/yolo-cli"),
        "depth",
        "--model",
        str(ROOT / "models/gguf/yolo26n-depth-f16.gguf"),
        "--source",
        str(ROOT.parent / f"ultralytics/assets/{img}.jpg"),
        "--raw",
        str(raw),
    ]
    if threads:
        cmd += ["--threads", threads]
    subprocess.run(cmd, check=True, capture_output=True)
    with raw.open("rb") as f:
        if f.read(8) != b"YDEP0001":
            raise ValueError(f"invalid depth output: {raw}")
        height, width = struct.unpack("<2i", f.read(8))
        return np.fromfile(f, dtype=np.float32).reshape(height, width)


def main():
    (ROOT / "benchmarks").mkdir(exist_ok=True)
    for img in IMAGES:
        fig, axes = plt.subplots(len(MODELS), len(BACKENDS) + 1, figsize=(20, 8))
        axes = axes.reshape(len(MODELS), -1)
        for r, model in enumerate(MODELS):
            axes[r, 0].imshow(torch_frame(model, img))
            axes[r, 0].set_ylabel(model, fontsize=13)
            for c, (label, build, threads) in enumerate(BACKENDS, start=1):
                axes[r, c].imshow(ggml_frame(model, img, build, threads))
            axes[r, 0].set_title("PyTorch CUDA (f32)" if r == 0 else "", fontsize=11)
        for c, (label, _, _) in enumerate(BACKENDS, start=1):
            axes[0, c].set_title(f"{label} (f16)", fontsize=11)
        for ax in axes.flat:
            ax.set_xticks([])
            ax.set_yticks([])
        fig.suptitle(f"Detection parity — {img}.jpg (conf 0.25, imgsz 640)", fontsize=14)
        fig.tight_layout()
        out = ROOT / "benchmarks" / f"parity_grid_{img}.png"
        fig.savefig(out, dpi=130)
        plt.close(fig)
        print(f"wrote {out}")

    img = "bus"
    depth_maps = [("PyTorch CUDA (F32)", torch_depth(img))]
    depth_maps += [(f"{label} (F16)", ggml_depth(img, build, threads)) for label, build, threads in BACKENDS]
    lo, hi = np.quantile(depth_maps[0][1], [0.02, 0.98])
    fig, axes = plt.subplots(1, len(depth_maps), figsize=(20, 5))
    for ax, (label, depth) in zip(axes, depth_maps):
        view = ax.imshow(depth, cmap="magma_r", vmin=lo, vmax=hi)
        ax.set_title(label)
        ax.set_xticks([])
        ax.set_yticks([])
    fig.colorbar(view, ax=axes, label="meters", shrink=0.8)
    fig.suptitle("YOLO26n absolute-depth parity - bus.jpg")
    out = ROOT / "benchmarks/depth_parity_bus.png"
    fig.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
