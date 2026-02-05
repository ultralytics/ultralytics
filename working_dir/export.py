#!/usr/bin/env python3
"""
YOLO-DETR Export and Profile Script

ProfileModels handles export automatically when given .pt files.

Usage examples:
    python export.py --model large --imgsz 640
    python export.py --model /path/to/model.pt --imgsz 640
    python export.py --model all --imgsz 640
"""

import argparse
from pathlib import Path

from ultralytics.utils.benchmarks import ProfileModels


# Friendly name to model file pattern mapping
MODEL_NAME_TO_FILE = {
    "nano": "yolo26_detr_n_obj_480.pt",
    "nano-small": "yolo26_detr_ns_coco_512.pt",
    "small": "yolo26_detr_s_coco_640.pt",
    "medium": "yolo26_detr_m_coco_640.pt",
    "large": "yolo26_detr_l_obj_640.pt",
    "xlarge": "yolo26_detr_x_coco_640.pt",
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Export YOLO-DETR models to ONNX and run profiling",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Model: friendly name (nano, small, medium, large, xlarge), "
             "path to .pt file, or 'all'",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=640,
        help="Export image size (pixels)",
    )
    return parser.parse_args()


def get_model_path(model_arg: str) -> Path:
    """Resolve model path from argument."""
    weights_dir = Path(__file__).parent / "yolodetr_weights"

    # Check if it's a friendly name
    if model_arg.lower() in MODEL_NAME_TO_FILE:
        model_file = MODEL_NAME_TO_FILE[model_arg.lower()]
        model_path = weights_dir / model_file
        if model_path.exists():
            return model_path
        raise FileNotFoundError(f"Model file not found: {model_path}")

    # Check if it's a direct path
    model_path = Path(model_arg)
    if model_path.exists():
        return model_path

    # Check in weights directory by filename
    model_path = weights_dir / model_arg
    if model_path.exists():
        return model_path

    # Not found
    friendly_names = ", ".join(MODEL_NAME_TO_FILE.keys())
    raise FileNotFoundError(
        f"Model '{model_arg}' not found.\n"
        f"Use friendly names: {friendly_names}\n"
        f"Or provide a path to a .pt file"
    )


def main():
    args = parse_args()

    # Determine which models to profile
    if args.model.lower() == "all":
        model_paths = [
            str(get_model_path(name))
            for name in MODEL_NAME_TO_FILE.keys()
        ]
    else:
        model_paths = [str(get_model_path(args.model))]

    # Run ProfileModels (handles export + profiling)
    print(f"Models: {model_paths}")
    print(f"Image size: {args.imgsz}")
    ProfileModels(paths=model_paths, imgsz=args.imgsz).run()


if __name__ == "__main__":
    main()
