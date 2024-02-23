import subprocess
import argparse
import os
import glob
import shutil
import warnings
from ultralytics import YOLO
from ultralytics.config import ROOT_DIR
from ultralytics.utils.custom_utils.helpers import copy_model_config

def train(epochs, model, imgsz=640, device="0", batch_size=16, patience=30, save_dir="train", data="data"):
    import re

    model_prefixes = {
        "yolov8": "ultralytics/cfg/models/v8/",
        "rtdetr": "ultralytics/cfg/models/rt-detr/"
    }

    pattern = re.compile(r'^(yolov8|rtdetr)(.*)(\.yaml|\.pt)$')
    match = pattern.match(model)
    if match:
        prefix = match.group(1)
        extension = match.group(3)
        model_path = f"{model_prefixes[prefix]}{'weights/' if extension == '.pt' else ''}{model}"

    try:
        test_model = YOLO(model_path, "detect")
    except FileNotFoundError:
        raise FileNotFoundError(f"{model_path} does not exist")

    command = [
        "yolo",
        "detect",
        "train",
        # f"cfg=./model_args.yaml",
        f"data={ROOT_DIR}/data/{data}/data.yaml",
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
    
    if data != "data" and match:
        results_path = max(glob.glob(os.path.join(f"{ROOT_DIR}/runs/detect/", '*/')), key=os.path.getmtime)
        weights_path = f"{results_path}/weights/last.pt"
        shutil.copy(weights_path, f"{ROOT_DIR}/{model_prefixes[prefix]}weights/{model.split('.')[0]}.pt")
    # return model_path

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="training parameters")

    parser.add_argument(
        "--model",
        type=str,   # specify the type of elements in the list
        help="Name of model, matching file name",
    )
    parser.add_argument(
        "--device",
        default="0",
        type=str,   # specify the type of elements in the list
        help="Device to run training on",
    )
    parser.add_argument(
        "--epochs",
        # nargs="+",  # '+' allows one or more values
        type=int,   # specify the type of elements in the list
        help="Number of epochs for the model",
    )
    parser.add_argument(
        "--batch_size",
        # nargs="+",  # '+' allows one or more values
        type=int,   # specify the type of elements in the list
        help="Batch size of the model",
    )
    parser.add_argument(
        "--save_dir",
        # nargs="+",  # '+' allows one or more values
        # default=args.model[:-5],
        type=str,   # specify the type of elements in the list
        help="Directory to save the results",
    )
    parser.add_argument(
        "--data_dir",
        # nargs="+",  # '+' allows one or more values
        default="data",
        type=str,   # specify the type of elements in the list
        help="Path to data.yaml file for dataset",
    )

    args = parser.parse_args()
    save_dir = args.save_dir

    if os.path.exists(save_dir):
        for i in range(2, 100):
            if os.path.exists(f'{save_dir}{i}'):
                continue
            else:
                save_dir = f'{save_dir}{i}'
                break
    
    warnings.filterwarnings("ignore", message="Variable._execution_engine.run_backward")
    warnings.filterwarnings("ignore", message="(Triggered internally at ../aten/src/ATen/Context.cpp:79.)")

    train(epochs=args.epochs, model=args.model, device=args.device, batch_size=args.batch_size, save_dir=save_dir, data=args.data_dir)

    copy_model_config(args.model, save_dir)
