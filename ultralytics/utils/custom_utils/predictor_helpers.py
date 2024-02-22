import os

import fiftyone as fo
import numpy as np
from tqdm import tqdm

from ultralytics.config import CLASSES_TO_KEEP

def read_yolo_detections_file(filepath):
    detections = []
    if not os.path.exists(filepath):
        return np.array([])

    with open(filepath) as f:
        lines = [line.rstrip('\n').split(' ') for line in f]

    for line in lines:
        detection = [float(l) for l in line]
        detections.append(detection)
    return np.array(detections)

def _uncenter_boxes(boxes):
    '''convert from center coords to corner coords'''
    boxes[:, 0] -= boxes[:, 2]/2.
    boxes[:, 1] -= boxes[:, 3]/2.

def _get_class_labels(predicted_classes, class_list):
    labels = (predicted_classes).astype(int)
    labels = [class_list[l] for l in labels]
    return labels

def convert_yolo_detections_to_fiftyone(
    yolo_detections,
    class_list
    ):

    detections = []
    if yolo_detections.size == 0:
        return fo.Detections(detections=detections)

    boxes = yolo_detections[:, 1:-1]
    # boxes = yolo_detections[:, 1:]
    _uncenter_boxes(boxes)

    confs = yolo_detections[:, -1]

    labels = _get_class_labels(yolo_detections[:, 0], class_list)
    for label, conf, box in zip(labels, confs, boxes):
        detections.append(
            fo.Detection(
                label=label,
                # label=CLASSES_MAPPING[label],
                bounding_box=box.tolist(),
                confidence=conf
            )
        )

    return fo.Detections(detections=detections)

def get_prediction_filepath(filepath, run_number):
    filename = filepath.split("/")[-1].split(".")[0]
    return f"{run_number}labels/{filename}.txt"

def add_yolo_detections(
    samples,
    prediction_field,
    prediction_filepath,
    class_list
    ):

    prediction_filepaths = samples.values(prediction_filepath)
    print("Read yolo detection")
    yolo_detections = [read_yolo_detections_file(pf) for pf in prediction_filepaths]

    print("Converting yolo detection to fiftyone")
    detections = [convert_yolo_detections_to_fiftyone(yolo_detections[i], class_list) for i in tqdm(range(len(yolo_detections)))]

    print("Setting prediction field")
    samples.set_values(prediction_field, detections)

def add_detections_to_fiftyone(dataset, model_name, run_number):
    filepaths = dataset.values("filepath")
    detection_filepath = "detection_filepath"
    classes = dataset.default_classes
    classes = CLASSES_TO_KEEP

    print("Adding prediction file paths")
    prediction_filepaths = [get_prediction_filepath(fp, run_number=run_number) for fp in filepaths]

    print("Setting values ...")
    dataset.set_values(detection_filepath, prediction_filepaths)

    print("Adding detections ...")
    add_yolo_detections(dataset, model_name, detection_filepath, classes)
    print("Finished adding detections")
