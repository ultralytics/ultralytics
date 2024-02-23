import argparse
import glob
import os

from ultralytics.config import ROOT_DIR, TEST_DATA_PATH, get_yolo_classes, SPONGEBOB
from ultralytics import YOLO, RTDETR

from ultralytics.utils.custom_utils.plots import create_result_plots
from ultralytics.utils.custom_utils.helpers import get_fiftyone_dataset

def run_prediction(export_dir_test, model_root_path, new_predict, model_type='yolov8'):
    if model_type == "yolov8":
        weights_path = f'{ROOT_DIR}/runs/detect/{model_root_path}/'
        model = YOLO(f'{weights_path}weights/best.pt')
    else:
        weights_path = f'{ROOT_DIR}/runs/detect/{model_root_path}/'
        model = RTDETR(f'{weights_path}weights/best.pt')

    if new_predict:
        results = model.predict(f'{export_dir_test}/images', save_txt=True, imgsz=640, conf=0.1, save_conf=True, classes=get_yolo_classes(), name=f"{model_root_path}/test_predictions")

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Prediction parameters")

    parser.add_argument(
        "--model_root_path",
        # nargs="+",  # '+' allows one or more values
        type=str,   # specify the type of elements in the list
        help="List of all models",
    )
    parser.add_argument(
        "--model_type",
        # nargs="+",  # '+' allows one or more values
        type=str,   # specify the type of elements in the list
        help="The type of model, either YOLOv8 or RTDETR",
    )
    parser.add_argument(
        "--run_new_prediction",
        # nargs="+",  # '+' allows one or more values
        type=str,   # specify the type of elements in the list
        default=False,
        help="If you want to run new predictions or use the ones from right after training",
    )

    args = parser.parse_args()
    model_root_path = args.model_root_path
    model_type = args.model_type
    new_predict = args.run_new_prediction

    print(SPONGEBOB(model_root_path, "RUNNING PREDICTION"))

    dataset, classes = get_fiftyone_dataset(0)

    test_set_path = TEST_DATA_PATH
    run_prediction(test_set_path, model_root_path, new_predict, model_type)
    
    run_prediction(test_set_path, dataset, model_root_path, model_type)
    create_result_plots(model_root_path, dataset)
