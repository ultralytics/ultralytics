import os
import shutil
import fiftyone as fo
import torch
from fiftyone import ViewField as F

from ultralytics.utils.custom_utils.init_setup import setup
from ultralytics.config import DATASET_NAME, ORIGINAL_CLASSES, CLASSES_TO_KEEP


def check_cuda():
    print(torch.cuda.is_available())
    print(torch.cuda.device_count())
    print(torch.cuda._parse_visible_devices())

    os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
    torch.cuda.empty_cache()
    torch.cuda.memory_summary(device=None, abbreviated=False)


def get_fiftyone_dataset(rank):
    dataset_name = f"{DATASET_NAME}{rank}"
    if dataset_name in fo.list_datasets():
        print("Loading dataset")
        dataset: fo.Dataset = fo.load_dataset(dataset_name)
    else:
        setup(rank)
        dataset: fo.Dataset = fo.load_dataset(dataset_name)

    dataset.persistent = True
    dataset = dataset.match(F("detections.detections").length() > 0)
    classes = lambda dataset: sorted(list(dataset.count_values("detections.detections.label").keys()))
    # classes = dataset.default_classes

    dataset.persistent = True

    return dataset, classes(dataset)


def copy_model_config(model, save_dir):
    if model[:6] != "rtdetr":
        model = model.split("-", 1)
        if len(model) > 1:
            path_to_yaml = model[0][:-1] + "-" + model[1]
        else:
            temp_path_to_yaml = model[0].split("yolov8")
            path_to_yaml = "yolov8" + temp_path_to_yaml[1][1:]

        source_dir = f"ultralytics/cfg/models/v8/{path_to_yaml}"
    else:
        path_to_yaml = model
        source_dir = f"ultralytics/cfg/models/rt-detr/{path_to_yaml}"

    dest_dir = f"./runs/detect/{save_dir}/{save_dir}"

    if os.path.exists(f"{dest_dir}.yaml"):
        for i in range(2, 100):
            if os.path.exists(f"{dest_dir}_doesnt_belong_here_{i}.yaml"):
                continue
            else:
                dest_dir = f"{dest_dir}_doesnt_belong_here_{i}.yaml"
                break
    else:
        dest_dir = f"{dest_dir}.yaml"

    shutil.copy(source_dir, dest_dir)

def get_yolo_classes():
    indexes = []
    for i in range(len(ORIGINAL_CLASSES)):
        if ORIGINAL_CLASSES[i] in CLASSES_TO_KEEP:
            indexes.append(i)
    return indexes

def SPONGEBOB(model_name, word="TRAINING MODEL"):
    return rf"""
            {word} 
        .--..--..--..--..--..--.
        .' \  (`._   (_)     _   \
    .'    |  '._)         (_)  |
    \ _.')\      .----..---.   /
    |(_.'  |    /    .-\-.  \  |
    \     0|    |   ( O| O) | o|
    |  _  |  .--.____.'._.-.  |
    \ (_) | o         -` .-`  |
        |    \   |`-._ _ _ _ _\ /
        \    |   |  `. |_||_|   |    {model_name.upper()}
        | o  |    \_      \     |     -.   .-.
        |.-.  \     `--..-'   O |     `.`-' .'
    _.'  .' |     `-.-'      /-.__   ' .-'
    .' `-.` '.|='=.='=.='=.='=|._/_ `-'.'
    `-._  `.  |________/\_____|    `-.'
    .'   ).| '=' '='\/ '=' |
    `._.`  '---------------'
            //___\   //___\
                ||       ||
                ||_.-.   ||_.-.
                (_.--__) (_.--__)
    """
