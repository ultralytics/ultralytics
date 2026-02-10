"""Profile ONNX models using Ultralytics ProfileModels.

Usage:
    python profile_onnx.py model.onnx --imgsz 640
    python profile_onnx.py model1.onnx model2.onnx --imgsz 512
    python profile_onnx.py onnx_exports/*.onnx --imgsz 640
"""

import argparse
from ultralytics.utils.benchmarks import ProfileModels


def main():
    parser = argparse.ArgumentParser(description="Profile ONNX models with TensorRT + ONNX Runtime")
    parser.add_argument("models", nargs="+", help="Path(s) to .onnx file(s)")
    parser.add_argument("--imgsz", type=int, default=640, help="Input image size (default: 640)")
    args = parser.parse_args()

    for model_path in args.models:
        print(f"\n{'='*60}")
        print(f"Profiling: {model_path} @ imgsz={args.imgsz}")
        print(f"{'='*60}\n")
        ProfileModels(paths=[model_path], imgsz=args.imgsz).run()


if __name__ == "__main__":
    main()
