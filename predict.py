import argparse
import glob
import os

from ultralytics.config import ROOT_DIR, TEST_DATA_PATH
from ultralytics import YOLO, RTDETR

from ultralytics.utils.custom_utils.predictor_helpers import add_detections_to_fiftyone
from ultralytics.utils.custom_utils.plots import create_result_plots
from ultralytics.utils.custom_utils.helpers import get_fiftyone_dataset, get_yolo_classes, SPONGEBOB

def run_prediction(export_dir_test, dataset, model_root_path, model_type='yolov8'):
    if model_type == "yolov8":
        weights_path = f'{ROOT_DIR}/runs/detect/{model_root_path}/'
        model = YOLO(f'{weights_path}weights/best.pt')
    else:
        weights_path = f'{ROOT_DIR}/runs/detect/{model_root_path}/'
        model = RTDETR(f'{weights_path}weights/best.pt')

    results = model.predict(f'{export_dir_test}/images', save_txt=True, imgsz=640, conf=0.1, save_conf=True, classes=get_yolo_classes(), name=f"{model_root_path}/test_predictions")

    run_number = max(glob.glob(os.path.join(f"runs/detect/{model_root_path}/", '*/')), key=os.path.getmtime)
    dataset_test = dataset.match_tags("test")
    add_detections_to_fiftyone(dataset_test, model_root_path, run_number)

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

    args = parser.parse_args()
    model_root_path = args.model_root_path
    model_type = args.model_type

    print(SPONGEBOB(model_root_path, "RUNNING PREDICTION"))

    dataset, classes = get_fiftyone_dataset(0)

    test_set_path = TEST_DATA_PATH
    
    run_prediction(test_set_path, dataset, model_root_path, model_type)
    create_result_plots(model_root_path, dataset)
