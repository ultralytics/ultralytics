import glob
import subprocess
import argparse
import os
from ultralytics.config import DATASET_DESCRIPTION, TEST_DATA_PATH
from ultralytics.utils.custom_utils.helpers import get_fiftyone_dataset
from predict import run_prediction

def train(epochs, model, imgsz=640, device="0", batch_size=16, patience=30, save_dir="train"):

    model_path = f'ultralytics/cfg/models/v8/{model}'

    command = [
        "yolo",
        "detect",
        "train",
        f"data={DATASET_DESCRIPTION}",
        f"model={model_path}",
        f"epochs={epochs}",
        f"imgsz={imgsz}",
        f"device={device}",
        "save_txt=True",
        f"batch={batch_size}",
        f"patience={patience}",
        "nms=True", 
        f"name={save_dir}",
    ]

    try:
        subprocess.run(command, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error: {e}")


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="training parameters")

    parser.add_argument(
        "--model",
        # nargs="+",  # '+' allows one or more values
        type=str,   # specify the type of elements in the list
        help="List of all models",
    )
    parser.add_argument(
        "--device",
        # nargs="+",  # '+' allows one or more values
        type=str,   # specify the type of elements in the list
        help="List of predictions to perform",
    )
    parser.add_argument(
        "--epochs",
        # nargs="+",  # '+' allows one or more values
        type=int,   # specify the type of elements in the list
        help="List of predictions to perform",
    )
    parser.add_argument(
        "--batch_size",
        # nargs="+",  # '+' allows one or more values
        type=int,   # specify the type of elements in the list
        help="List of predictions to perform",
    )
    args = parser.parse_args()
    parser.add_argument(
        "--save_dir",
        # nargs="+",  # '+' allows one or more values
        default=args.model[:-5],
        type=str,   # specify the type of elements in the list
        help="List of predictions to perform",
    )

    args = parser.parse_args()
    model = args.model
    device = args.device
    epochs = args.epochs
    batch_size = args.batch_size
    save_dir = args.save_dir
    train(epochs=epochs, model=model, device=device, batch_size=batch_size, save_dir=save_dir)

    dataset, classes = get_fiftyone_dataset(0)

    run_prediction(TEST_DATA_PATH, dataset, save_dir, model[:6])
