from ultralytics import YOLO
from training_utils import (
    GetModelYaml,
    training_task,
    GiveModel
)
import os
import argparse


def Export(checkpoint_file_path):
    model = YOLO(GetModelYaml(training_task))  # Initialize model
    if os.path.exists(args.model):
        model = GiveModel(args.load)
    else:
        print(f"[ERROR] : Model {args.load} does not exists")
        exit(1)

    path = model.export(format="onnx", imgsz=[2144, 768], opset=12)
    base_path, name = os.path.split(args.model)
    os.system(f"mv {path} {base_path}/best_full_height.onnx")

    path = model.export(format="onnx", imgsz=[2144, 4096], opset=12)
    base_path, name = os.path.split(args.model)
    os.system(f"mv {path} {base_path}/best_full_frame.onnx")

    path = model.export(format="onnx", imgsz=[768, 768], opset=12)
    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export the model")
    parser.add_argument(
        "-m",
        "--model",
        type=str,
        required=True,
        help="Path to the model weights to load. This model will be exported")

    args = parser.parse_args()
    Export(args.model)
