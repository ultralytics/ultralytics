import glob
import os

import fiftyone as fo
import torch
from fiftyone import ViewField as F

from ultralytics.utils import custom_utils


def check_cuda():
    print(torch.cuda.is_available())
    print(torch.cuda.device_count())
    print(torch.cuda._parse_visible_devices())

    os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
    torch.cuda.empty_cache()
    torch.cuda.memory_summary(device=None, abbreviated=False)

def get_fiftyone_dataset(rank):
    dataset_name = f"My_dataset{rank}"
    if dataset_name in fo.list_datasets():
        print("Loading dataset")
        dataset: fo.Dataset = fo.load_dataset(dataset_name)
    else:
        custom_utils.setup(rank)
        dataset: fo.Dataset = fo.load_dataset(dataset_name)
    
    dataset.persistent = True
    dataset = dataset.match(F("detections.detections").length() > 0)
    classes = lambda dataset: sorted(list(dataset.count_values("detections.detections.label").keys()))

    dataset.persistent = True
    return dataset, classes(dataset)
    
def get_run_number(num=None):
    if num:
        return f"./runs/detect/train{num}"
    else:
        return max(glob.glob(os.path.join("./runs/detect/", 'train*/')), key=os.path.getmtime)
    

if __name__ == "__main__":
    print("hei")
