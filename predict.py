import fiftyone as fo
from fiftyone import ViewField as F
import glob
import os
import argparse

from ultralytics.config import ROOT_DIR, TEST_DATA_PATH, get_yolo_classes
from ultralytics import YOLO, RTDETR

from ultralytics.utils.custom_utils.predictor import add_detections_to_fiftyone
from ultralytics.utils.custom_utils.helpers import get_fiftyone_dataset


def run_prediction(export_dir_test, dataset, model_root_path, model_name='yolov8'):

    if model_name == "yolov8":
        weights_path = f'{ROOT_DIR}/runs/detect/{model_root_path}/'
        model = YOLO(f'{weights_path}weights/best.pt')
    else:
        weights_path = f'{ROOT_DIR}/runs/detect/{model_root_path}/'
        model = RTDETR(f'{weights_path}weights/best.pt')

    results = model.predict(f'{export_dir_test}/images/val', save_txt=True, imgsz=640, conf=0.1, save_conf=True, classes=get_yolo_classes(), name=f"{model_root_path}_predict")

    # Extracts the most recently added/changed folder
    run_number = max(glob.glob(os.path.join("./runs/detect/", '*/')), key=os.path.getmtime)
    
    dataset_test = dataset.match_tags("test")
    add_detections_to_fiftyone(dataset_test, model_name, run_number)

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Prediction parameters")

    parser.add_argument(
        "--model_root_path",
        # nargs="+",  # '+' allows one or more values
        type=str,   # specify the type of elements in the list
        help="List of all models",
    )

    args = parser.parse_args()
    model_root_path = args.model_root_path

    dataset = get_fiftyone_dataset(0)
    test_set_path = TEST_DATA_PATH
    
    run_prediction(test_set_path, dataset, model_root_path)