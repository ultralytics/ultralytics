import subprocess
import argparse
import os
import re
import shutil
from ultralytics import YOLO
from ultralytics.config import DATASET_DESCRIPTION, TEST_DATA_PATH, ROOT_DIR
# from ultralytics.utils.custom_utils.helpers import get_fiftyone_dataset
# from predict import run_prediction

def train(epochs, model, imgsz=640, device="0", batch_size=16, patience=30, save_dir="train", data=DATASET_DESCRIPTION):
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

    print(model_path)
    try:
        test_model = YOLO(model_path, "detect")
    except FileNotFoundError:
        raise FileNotFoundError(f"{model_path} does not exist")

    command = [
        "yolo",
        "detect",
        "train",
        f"data={data}",
        f"model={model_path}",
        f"epochs={epochs}",
        f"imgsz={imgsz}",
        f"device={device}",
        "save_txt=True",
        f"batch={batch_size}",
        f"patience={patience}",
        "nms=True",
        f"name={save_dir}",
        f"cfg=./model_args.yaml"
    ]

    try:
        subprocess.run(command, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error: {e}")
    
    # return model_path

def copy_model_config(model, save_dir):
    if model[:6] != "rtdetr":
        model = model.split('-', 1)
        if len(model) > 1:
            path_to_yaml = model[0][:-1]+"-"+model[1]
        else:
            temp_path_to_yaml = model[0].split("yolov8")
            path_to_yaml = "yolov8"+temp_path_to_yaml[1][1:]
    else:
        path_to_yaml = model
    
    source_dir= f'ultralytics/cfg/models/v8/{path_to_yaml}'
    dest_dir = f'./runs/detect/{save_dir}/{save_dir}'

    if os.path.exists(f'{dest_dir}.yaml'):
        for i in range(2, 100):
            if os.path.exists(f'{dest_dir}_doesnt_belong_here_{i}.yaml'):
                continue
            else:
                dest_dir = f'{dest_dir}_doesnt_belong_here_{i}.yaml'
                break
    else:
        dest_dir = f'{dest_dir}.yaml'
    print(source_dir)
    print(dest_dir)
    shutil.copy(source_dir, dest_dir)

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
        default=DATASET_DESCRIPTION,
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

    train(epochs=args.epochs, model=args.model, device=args.device, batch_size=args.batch_size, save_dir=save_dir, data=args.data_dir)

    # copy_model_config(args.model, save_dir)

    # dataset, classes = get_fiftyone_dataset(0)

    # run_prediction(TEST_DATA_PATH, dataset, save_dir, model[:6])
