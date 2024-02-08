import os

import fiftyone as fo
import numpy as np
from ultralytics.config import CLASSES_MAPPING, original_classes
from fiftyone.utils.eval.coco import DetectionResults
from tqdm import tqdm

import matplotlib.pyplot as plt
from tqdm import tqdm
import seaborn as sns
import json

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
                label=CLASSES_MAPPING[label],
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
    # classes = dataset.default_classes[1:]
    classes = original_classes

    print("Adding prediction file paths")
    prediction_filepaths = [get_prediction_filepath(fp, run_number=run_number) for fp in filepaths]

    print("Setting values ...")
    dataset.set_values(detection_filepath, prediction_filepaths)

    print("Adding detections ...")
    add_yolo_detections(dataset, model_name, detection_filepath, classes)
    print("Finished adding detections")

def plots_and_matrixes(model_name, dataset, classes, save_path, evaluators_path, pred=(0,100), pred_name="all"):
    _classes = lambda dataset: list(dataset.count_values(f'{model_name}.detections.label'))

    lower_thresh = pred[0]
    upper_thresh = pred[1]
    
    view = dataset.match_tags("test").clone()
    clone = view

    print("BEFORE:",len(view))

    counts = dataset.count_values("detections.detections.label")
    # classes_top10 = sorted(counts, key=counts.get, reverse=True)[:10]

    
    for sample in tqdm(view):
        detections = sample.detections.detections
        filtered_detections = []
        for detection in detections:
            if (lower_thresh < detection["bbox_area_percentage"] <= upper_thresh):
                filtered_detections.append(detection)
        sample.detections.detections = filtered_detections
        sample.save()
        predictions = sample[model_name].detections
        filtered_predictions = []
        for prediction in predictions:
            bounding_box = prediction["bounding_box"]
            bbox_area_percentage = bounding_box[2] * bounding_box[3] * 100
            if prediction["confidence"] > 0.2 and (lower_thresh < bbox_area_percentage <= upper_thresh):
                filtered_predictions.append(prediction)
        sample[model_name].detections = filtered_predictions
        sample.save()

    if "TVT_svamp" not in _classes(dataset):
        view = view.map_labels(model_name, CLASSES_MAPPING)

    eval_results: DetectionResults = fo.evaluate_detections(
        view,
        model_name,
        classes=classes,
        compute_mAP=True,
        classwise=False,
        gt_field="detections",
        eval_key=model_name,
        method="coco",
    )

    eval_results_2 = fo.evaluate_detections(
        view,
        model_name,
        classes=classes,
        compute_mAP=True,
        classwise=False,
        gt_field="detections",
        eval_key=model_name,
        method="coco",
        iou_threshs=[0.5]
    )

    eval_results_3 = fo.evaluate_detections(
        view,
        model_name,
        classes=classes,
        compute_mAP=True,
        classwise=False,
        gt_field="detections",
        eval_key=model_name,
        method="coco",
        iou_threshs=[0.75]
    )
    map_50_95 = eval_results.mAP()
    map_50 = eval_results_2.mAP()
    map_75 = eval_results_3.mAP()
    print("mAP@0.5:0.95", eval_results.mAP())
    print("mAP@0.5", eval_results_2.mAP())
    print("mAP@0.75", eval_results_3.mAP())

    report = eval_results.report()
    weighted_avg = report['weighted avg']

    obj = {
        "mAP@0.5:0.95": map_50_95,
        "mAP@0.5": map_50,
        "mAP@0.75": map_75,
        "weighted_avg": weighted_avg
    }

    # test_stats_path = f"{evaluators_path}/results/{model_type}/test_set_stats_{pred_name}.json"
    test_stats_path = f"{evaluators_path}/test_set_stats_{pred_name}.json"

    with open(test_stats_path, "w") as file:
        json.dump(obj, file, indent=2)

    eval_results.print_report(classes=classes)

    cm, labels, _ = eval_results._confusion_matrix(
        classes=classes,
        include_other=None,
        include_missing=None,
        other_label="background",
        tabulate_ids=True,
        )
    
    cmn = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    fig, ax = plt.subplots(figsize=(11,9))
    sns.heatmap(cmn, annot=False, cmap="Blues", xticklabels=labels, yticklabels=labels)
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.tight_layout()
    plt.savefig(f"{save_path}confusion_matrix_{pred_name}_conf02.png")

    plot_PR = eval_results.plot_pr_curves(classes=classes, title=f"{model_name} {pred_name} objects")
    plot_PR.write_image(f"{save_path}pr_curve_{pred_name}_conf02.png")

    clone.delete()