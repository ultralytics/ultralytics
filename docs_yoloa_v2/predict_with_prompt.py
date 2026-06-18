#!/usr/bin/env python
"""Predict with the YOLOAnomalyV2 model, optionally injecting a mask/bbox prompt.

Three modes, all selected at runtime:

  1. mask-off (default)        — equivalent to vanilla YOLO inference.
                                  Use when you have no prior signal.
  2. bbox prompt               — "I think there's an anomaly around here":
                                  list of normalized [cx, cy, w, h] bboxes.
                                  The model's configured mask_mode (rect/gauss
                                  from the YAML) renders them to a mask.
  3. external mask             — hand-painted or computed (e.g. MemoryBank).
                                  Grayscale image, resized to ``mask_size``,
                                  fed directly to the heatmap encoder.

The script always runs the mask-on variant you ask for, and with --compare
also runs the mask-off counterpart so you can eyeball the two side by side.

Run from the v2 worktree so the local ``ultralytics`` package is picked up:

    cd /Users/louis/workspace/ultra_louis_work/ultralytics/.claude/worktrees/yoloa_v2
    python docs_yoloa_v2/predict_with_prompt.py --model ... --image ... --bbox 0.3 0.4 0.2 0.2 --compare

Examples
--------
# Default (mask-off, == vanilla):
    python docs_yoloa_v2/predict_with_prompt.py \\
        --model ~/workspace/ultra_louis_work/ultra6/runs/yoloa_v2/26m_yoloav2_v5_binary_cm20_gauss_pd50_v1/weights/best.pt \\
        --image some.jpg

# Bbox prompt (single):
    python docs_yoloa_v2/predict_with_prompt.py \\
        --model .../best.pt --image some.jpg \\
        --bbox 0.45 0.50 0.20 0.20 --compare

# Bbox prompt (multiple):
    python docs_yoloa_v2/predict_with_prompt.py \\
        --model .../best.pt --image some.jpg \\
        --bbox 0.30 0.40 0.15 0.15 \\
        --bbox 0.70 0.60 0.10 0.10 --compare

# External mask file (grayscale PNG, foreground = anomaly):
    python docs_yoloa_v2/predict_with_prompt.py \\
        --model .../best.pt --image some.jpg \\
        --mask my_paint.png --compare
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import torch

from ultralytics import YOLO


def load_external_mask(path: Path, size: int) -> torch.Tensor:
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Could not read mask: {path}")
    img = cv2.resize(img, (size, size), interpolation=cv2.INTER_LINEAR)
    t = torch.from_numpy(img.astype("float32") / 255.0)
    return t.unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)


def reset_prompts(y: YOLO) -> None:
    if y.predictor is not None:
        y.predictor.bbox_prompt = None
        y.predictor.external_mask = None


def run_one(
    y: YOLO,
    image,
    *,
    bbox_prompt=None,
    external_mask=None,
    save_dir: Path,
    name: str,
    conf: float,
):
    reset_prompts(y)
    if y.predictor is None:
        # First call initializes the predictor; do a no-op predict if needed.
        y.predict(image, save=False, verbose=False, conf=conf)
        reset_prompts(y)

    if external_mask is not None:
        y.predictor.external_mask = external_mask
    elif bbox_prompt is not None:
        bb = torch.tensor(bbox_prompt, dtype=torch.float32)
        bi = torch.zeros(bb.shape[0], dtype=torch.long)
        y.predictor.bbox_prompt = (bb, bi)

    results = y.predict(
        image,
        save=True,
        verbose=False,
        conf=conf,
        project=str(save_dir),
        name=name,
        exist_ok=True,
    )
    reset_prompts(y)
    return results


def main():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--model", required=True, help="Path to v2 best.pt")
    p.add_argument("--image", required=True, help="Image path or directory")
    p.add_argument(
        "--bbox",
        nargs=4,
        type=float,
        action="append",
        metavar=("CX", "CY", "W", "H"),
        help="Normalized bbox prompt; repeat for multiple. Mutually exclusive with --mask.",
    )
    p.add_argument("--mask", type=Path, help="Grayscale mask PNG. Mutually exclusive with --bbox.")
    p.add_argument("--mask-size", type=int, default=80, help="Resize external mask to this HxW (default 80=P3 scale)")
    p.add_argument("--compare", action="store_true", help="Also run a mask-off prediction for side-by-side comparison")
    p.add_argument("--save-dir", type=Path, default=Path("./runs/predict_v2"), help="Project dir for outputs")
    p.add_argument("--conf", type=float, default=0.25)
    args = p.parse_args()

    if args.bbox and args.mask:
        sys.exit("ERROR: --bbox and --mask are mutually exclusive.")

    external_mask = load_external_mask(args.mask, args.mask_size) if args.mask else None

    y = YOLO(args.model)
    print(f"Loaded model: task={y.task} class={type(y.model).__name__}")
    print(f"  mask_renderer: mode={y.model.mask_renderer.mode}, size={y.model.mask_renderer.mask_size}")

    # Primary run (with whatever prompt was provided, or mask-off if neither)
    if external_mask is not None:
        mode = "ext_mask"
    elif args.bbox:
        mode = "bbox_prompt"
    else:
        mode = "mask_off"
    print(f"\n=== Primary run: {mode} ===")
    r_on = run_one(
        y,
        args.image,
        bbox_prompt=args.bbox,
        external_mask=external_mask,
        save_dir=args.save_dir,
        name=mode,
        conf=args.conf,
    )
    for r in r_on:
        print(f"  {Path(r.path).name}: {len(r.boxes)} detections | saved to {r.save_dir}")

    # Optional mask-off comparison
    if args.compare and mode != "mask_off":
        print("\n=== Comparison run: mask_off ===")
        r_off = run_one(
            y,
            args.image,
            save_dir=args.save_dir,
            name="mask_off",
            conf=args.conf,
        )
        for r in r_off:
            print(f"  {Path(r.path).name}: {len(r.boxes)} detections | saved to {r.save_dir}")

    print(f"\nAll outputs under: {args.save_dir.resolve()}")


if __name__ == "__main__":
    main()
