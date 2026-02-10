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

from ultralytics import YOLO
from ultralytics.utils.benchmarks import ProfileModels


# Friendly name to (weights_subfolder, filename) mapping
MODEL_NAME_TO_FILE = {
    "nano": ("yolodetr_weights", "yolo26_detr_n_obj_480.pt"),
    "nano-small": ("yolodetr_weights", "yolo26_detr_ns_coco_512.pt"),
    "small": ("yolodetr_weights", "yolo26_detr_s_coco_640.pt"),
    "medium": ("yolodetr_weights", "yolo26_detr_m_coco_640.pt"),
    "large": ("yolodetr_weights", "yolo26_detr_l_obj_640.pt"),
    "xlarge": ("yolodetr_weights", "yolo26_detr_x_coco_640.pt"),
    "nano-nms": ("yolo26nms_weights", "yolo26n_nms.pt"),
    "small-nms": ("yolo26nms_weights", "yolo26s_nms.pt"),
    "medium-nms": ("yolo26nms_weights", "yolo26m_nms.pt"),
    "large-nms": ("yolo26nms_weights", "yolo26l_nms.pt"),
    "xlarge-nms": ("yolo26nms_weights", "yolo26x_nms.pt"),
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Export YOLO-DETR models to ONNX and run profiling",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Model: friendly name (nano, small, medium, large, xlarge), "
             "path to .pt file, or 'all'",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=640,
        help="Export image size (pixels)",
    )
    parser.add_argument(
        "--opset",
        type=int,
        default=None,
        help="ONNX opset version (default: auto-select best for current torch)",
    )
    parser.add_argument(
        "--nms",
        action="store_true",
        help="Bake NMS into the ONNX graph during export",
    )
    parser.add_argument(
        "--check",
        type=str,
        nargs="+",
        metavar="ONNX_FILE",
        help="Check opset version of one or more .onnx files and exit",
    )
    return parser.parse_args()


def get_model_path(model_arg: str) -> Path:
    """Resolve model path from argument."""
    base_dir = Path(__file__).parent

    # Check if it's a friendly name
    if model_arg.lower() in MODEL_NAME_TO_FILE:
        subfolder, model_file = MODEL_NAME_TO_FILE[model_arg.lower()]
        model_path = base_dir / subfolder / model_file
        if model_path.exists():
            return model_path
        raise FileNotFoundError(f"Model file not found: {model_path}")

    # Check if it's a direct path
    model_path = Path(model_arg)
    if model_path.exists():
        return model_path

    # Check in both weight directories by filename
    for subfolder in ("yolodetr_weights", "yolo26nms_weights"):
        model_path = base_dir / subfolder / model_arg
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

    # Check opset version of existing ONNX files
    if args.check:
        import onnx

        for path in args.check:
            model = onnx.load(path, load_external_data=False)
            ops = ", ".join(f"{o.domain or 'ai.onnx'}: opset {o.version}" for o in model.opset_import)
            print(f"{Path(path).name} → {ops}")
        return

    if args.model is None:
        raise SystemExit("error: --model is required when not using --check")

    # Determine which models to profile
    if args.model.lower() == "all":
        model_paths = [
            str(get_model_path(name))
            for name in MODEL_NAME_TO_FILE.keys()
        ]
    else:
        model_paths = [str(get_model_path(args.model))]

    # Pre-export with custom opset / nms if specified, then profile the .onnx files directly
    if args.opset is not None or args.nms:
        export_kwargs = dict(format="onnx", imgsz=args.imgsz)
        if args.opset is not None:
            export_kwargs["opset"] = args.opset
        if args.nms:
            export_kwargs["nms"] = True

        onnx_paths = []
        for pt_path in model_paths:
            print(f"Exporting {pt_path} (opset={args.opset}, nms={args.nms})...")
            onnx_file = Path(YOLO(pt_path).export(**export_kwargs))
            if args.opset is not None:
                renamed = onnx_file.with_stem(f"{onnx_file.stem}_opset{args.opset}")
                onnx_file.rename(renamed)
                onnx_file = renamed
            onnx_paths.append(str(onnx_file))
        model_paths = onnx_paths

    # Run ProfileModels (handles export + profiling)
    print(f"Models: {model_paths}")
    print(f"Image size: {args.imgsz}")
    ProfileModels(paths=model_paths, imgsz=args.imgsz).run()


if __name__ == "__main__":
    main()
