import glob
import os
import shutil

import fiftyone as fo
import torch
from fiftyone import ViewField as F

from ultralytics.utils.custom_utils.init_setup import setup
from ultralytics.config import DATASET_NAME

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

    dataset.persistent = True
    return dataset, classes(dataset)
