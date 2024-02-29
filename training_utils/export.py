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
    if os.path.exists(checkpoint_file_path):
        model = GiveModel(checkpoint_file_path)
    else:
        print(f"[ERROR] : Model {checkpoint_file_path} does not exists")
        exit(1)

    base_path, _ = os.path.split(checkpoint_file_path)
    path = model.export(format="onnx", imgsz=[2144, 768], opset=12)
    os.system(f"mv {path} {base_path}/best_full_height.onnx")

    path = model.export(format="onnx", imgsz=[2144, 4096], opset=12)
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
