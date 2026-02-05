#!/usr/bin/env python3
"""
YOLO-DETR Prediction Script

Usage examples:
    python predict.py --model yolo26_detr_l_obj_640.pt --source image.jpg
    python predict.py --model yolo26_detr_s_coco_640.pt --source bus.jpg --conf 0.5 --imgsz 640
    python predict.py --model yolo26_detr_n_obj_480.pt --source ./images/ --conf 0.25 --imgsz 480
"""

import argparse
from pathlib import Path

from ultralytics import RTDETR
from ultralytics.utils import ROOT, YAML


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run YOLO-DETR model predictions",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Model filename (e.g., yolo26_detr_l_obj_640.pt) or full path to weights",
    )
    parser.add_argument(
        "--source",
        type=str,
        default="https://ultralytics.com/images/bus.jpg",
        help="Image source: file path, directory, URL, or '0' for webcam",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.25,
        help="Confidence threshold for detections",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=640,
        help="Inference image size (pixels)",
    )
    parser.add_argument(
        "--save-txt",
        action="store_true",
        help="Save detection results to text file",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Display results in a window",
    )
    return parser.parse_args()


# Friendly name to model file pattern mapping
MODEL_NAME_TO_FILE = {
    "nano": "yolo26_detr_n_obj_480.pt",
    "nano-small": "yolo26_detr_ns_coco_512.pt",
    "small": "yolo26_detr_s_coco_640.pt",
    "medium": "yolo26_detr_m_coco_640.pt",
    "large": "yolo26_detr_l_obj_640.pt",
    "xlarge": "yolo26_detr_x_coco_640.pt",
}


def get_model_path(model_arg: str) -> Path:
    """Resolve model path from argument (accepts friendly names like 'large')."""
    weights_dir = Path(__file__).parent / "yolodetr_weights"

    # Check if it's a friendly name (nano, small, medium, large, xlarge)
    if model_arg.lower() in MODEL_NAME_TO_FILE:
        model_file = MODEL_NAME_TO_FILE[model_arg.lower()]
        model_path = weights_dir / model_file
        if model_path.exists():
            return model_path

    # Check if it's already a full path
    if Path(model_arg).exists():
        return Path(model_arg)

    # Check in weights directory
    model_path = weights_dir / model_arg
    if model_path.exists():
        return model_path

    # List available models if not found
    available = list(weights_dir.glob("*.pt"))
    available_names = [m.name for m in available]
    friendly_names = ", ".join(MODEL_NAME_TO_FILE.keys())
    raise FileNotFoundError(
        f"Model '{model_arg}' not found.\n"
        f"Use friendly names: {friendly_names}\n"
        f"Or full filenames:\n  " + "\n  ".join(available_names)
    )


def load_coco_names() -> dict:
    """Load COCO class names."""
    return YAML.load(ROOT / "cfg/datasets/coco.yaml").get("names")


# Model size mapping: short code -> friendly name
MODEL_SIZE_MAP = {
    "n": "nano",
    "ns": "nano-small",
    "s": "small",
    "m": "medium",
    "l": "large",
    "x": "xlarge",
}


def get_model_size(model_name: str) -> str:
    """Extract and map model size from model filename."""
    base_name = Path(model_name).stem.lower()  # e.g., yolo26_detr_l_obj_640
    # Try to find size code in the model name (after 'detr_')
    for code, name in sorted(MODEL_SIZE_MAP.items(), key=lambda x: -len(x[0])):
        # Check for patterns like _l_, _ns_, _s_ etc.
        if f"_detr_{code}_" in base_name or f"detr_{code}_" in base_name:
            return name
        if base_name.endswith(f"_{code}") or f"_{code}_" in base_name:
            return name
    return "unknown"


def get_results_dir() -> Path:
    """Get or create the results directory."""
    results_dir = Path(__file__).parent / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    return results_dir


def get_image_name(source: str, result_path: str = None) -> str:
    """Extract image name from source or result path."""
    # Try to get from result path first (more reliable for URLs)
    if result_path:
        return Path(result_path).stem
    # Fallback to source
    if source.startswith("http"):
        return Path(source.split("?")[0]).stem
    return Path(source).stem


def get_output_filename(model_name: str, conf: float, image_name: str) -> str:
    """Generate output filename with model size, confidence, and image name."""
    model_size = get_model_size(model_name)
    conf_str = f"conf{conf:.2f}".replace(".", "")  # e.g., conf025 for 0.25
    return f"{model_size}_{conf_str}_{image_name}"


def main():
    args = parse_args()

    # Resolve model path
    model_path = get_model_path(args.model)
    print(f"Loading model: {model_path}")

    # Load model
    model = RTDETR(model_path)

    # Load COCO class names (all models use COCO labels for prediction)
    coco_names = load_coco_names()
    model.model.names = coco_names

    # Setup results directory
    results_dir = get_results_dir()
    model_size = get_model_size(model_path.name)
    print(f"Model size: {model_size}")

    # Run inference
    print(f"\nRunning inference with conf={args.conf}, imgsz={args.imgsz}")
    print(f"Source: {args.source}\n")

    results = model(args.source, conf=args.conf, imgsz=args.imgsz)

    # Process and save results
    for i, r in enumerate(results):
        print(f"--- Image {i + 1} ---")
        print(f"Total detections: {len(r.boxes)}")

        # Print detection details
        for box in r.boxes:
            coords = box.xyxy[0].cpu().numpy()
            conf = float(box.conf[0])
            cls_id = int(box.cls[0])
            cls_name = r.names[cls_id]
            box_str = f"[{coords[0]:.1f}, {coords[1]:.1f}, {coords[2]:.1f}, {coords[3]:.1f}]"
            print(f"  {cls_name}: {conf:.2f} | Box: {box_str}")

        # Get image name and generate output filename
        image_name = get_image_name(args.source, r.path)
        output_name = get_output_filename(model_path.name, args.conf, image_name)

        # Save visualization
        save_path = results_dir / f"{output_name}.jpg"
        r.save(filename=str(save_path))
        print(f"  Saved: {save_path}")

        # Save detections to text file if requested
        if args.save_txt:
            txt_path = results_dir / f"{output_name}.txt"
            with open(txt_path, "w") as f:
                for box in r.boxes:
                    cls_id = int(box.cls[0])
                    conf = float(box.conf[0])
                    coords = box.xyxy[0].cpu().numpy()
                    x1, y1, x2, y2 = coords
                    f.write(f"{cls_id} {r.names[cls_id]} {conf:.4f} {x1:.1f} {y1:.1f} {x2:.1f} {y2:.1f}\n")
            print(f"  Saved labels: {txt_path}")

        # Show if requested
        if args.show:
            r.show()

        print()

    print(f"Done! Results saved to: {results_dir}")


if __name__ == "__main__":
    main()
