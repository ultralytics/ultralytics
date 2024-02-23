import json
import os
import re
import glob

import fiftyone as fo
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from ultralytics.config import PLOTS_PATH, CLASSES_TO_KEEP, ROOT_DIR
from fiftyone import ViewField as F
from fiftyone.utils.eval.coco import DetectionResults
from tqdm import tqdm
from ultralytics.utils.custom_utils.helpers import get_fiftyone_dataset
from ultralytics.utils.custom_utils.predictor_helpers import add_detections_to_fiftyone


def get_order_func(dataset, split):
    # Count values of full dataset to get order to sort in
    label_counts = dataset.count_values('detections.detections.label')
    sorted_data = sorted(label_counts.items(), key=lambda x: x[1], reverse=True)
    sorted_keys = [item[0] for item in sorted_data]
    
    # Count values of split dataset and sort after full dataset
    dataset = dataset.match_tags(split)
    label_counts = dataset.count_values('detections.detections.label')
    sorted_data = sorted(label_counts.items(), key=lambda x: sorted_keys.index(x[0]))

    def custom_order(item):
        return sorted_data.index(item)
    return custom_order

def save_distribution_plot(dataset, split="full"):
    for split in ["full", "train", "test", "val"]:
        if split == "full":
            init_view = dataset
            plot = fo.CategoricalHistogram("detections.detections.label", order="frequency", init_view=init_view)
            plot.save(f"{PLOTS_PATH}/class_distribution/{split}_distribution.png", scale=3.0, height=720)
        else:
            order_func = get_order_func(dataset, split)
            init_view = dataset.match_tags(split)
            plot = fo.CategoricalHistogram("detections.detections.label", order=order_func, init_view=init_view)
            plot.save(f"{PLOTS_PATH}/class_distribution/{split}_distribution.png", scale=3.0, height=720)


def save_bounding_box_plots(dataset, split="full"):
    for split in ["full", "train", "test", "val"]:
        if split == "full":
            init_view = dataset
        else:
            init_view = dataset.match_tags(split)
            
        plot = fo.NumericalHistogram("detections.detections.bbox_area_percentage", bins=100, range=[0, 5.1], init_view=init_view)
        plot.save(f"{PLOTS_PATH}/bounding_box/{split}_bbox_area_percentage.png", scale=3.0, height=720)

        plot = fo.NumericalHistogram("detections.detections.bbox_aspect_ratio", bins=100, range=[0, 2.1], init_view=init_view)
        plot.save(f"{PLOTS_PATH}/bounding_box/{split}_bbox_aspect_ratio.png", scale=3.0, height=720)


def create_color_distribution_plot(init_view):
    # plot_red = fo.NumericalHistogram("mean_red", bins=200, range=[0.1, 0.85])
    # plot_green = fo.NumericalHistogram("mean_green", bins=200, range=[0.1, 0.85])
    SAVE_DIR = f"{PLOTS_PATH}/color_distribution"
    if not os.path.isdir(SAVE_DIR):
        os.mkdir(SAVE_DIR)
    
    for color in ["red", "green", "blue"]:
        plot = fo.NumericalHistogram(f"mean_{color}", bins=200, range=[0.1, 0.85], init_view=init_view)
        # plot = fo.ViewGrid([hist], init_view=init_view)
        # plot.show()
        plot.save(f"{SAVE_DIR}/full_distribution_{color}.png", scale=2.0, height=360)

def sort_key(filename):
    match = re.search(r'evaluator_epoch(\d+)', filename)
    return int(match.group(1))

def plots_and_matrixes(model_name, dataset, classes, save_path, evaluators_path, pred=(0,100), pred_name="all"):
    lower_thresh = pred[0]
    upper_thresh = pred[1]
    
    view = dataset.match_tags("test").clone()
    clone = view

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
            prediction["confidence"]
            if prediction["confidence"] > 0.2 and (lower_thresh < bbox_area_percentage <= upper_thresh):
                filtered_predictions.append(prediction)
        sample[model_name].detections = filtered_predictions
        sample.save()

    eval_key = model_name.replace("-", "_")

    eval_results: DetectionResults = fo.evaluate_detections(
        view,
        model_name,
        classes=classes,
        compute_mAP=True,
        classwise=False,
        gt_field="detections",
        eval_key=eval_key,
        method="coco",
    )

    eval_results_2 = fo.evaluate_detections(
        view,
        model_name,
        classes=classes,
        compute_mAP=True,
        classwise=False,
        gt_field="detections",
        eval_key=eval_key,
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
        eval_key=eval_key,
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
        # other_label="background",
        tabulate_ids=True,
        )
    
    cmn = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    cmn[cmn < 0.005] = np.nan

    fig, ax = plt.subplots(figsize=(11,9))
    sns.heatmap(cmn, annot=False, cmap="Blues", xticklabels=labels, square=True, vmin=0.0, yticklabels=labels).set_facecolor((1, 1, 1))
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.tight_layout()
    plt.savefig(f"{save_path}confusion_matrix_{pred_name}_conf02.png")

    plot_PR = eval_results.plot_pr_curves(classes=classes, title=f"{model_name} {pred_name} objects")
    plot_PR.write_image(f"{save_path}pr_curve_{pred_name}_conf02.png")

    clone.delete()

def create_result_plots(model_root_path, dataset):
    run_number = max(glob.glob(os.path.join(f"{ROOT_DIR}/runs/detect/{model_root_path}/", '*/')), key=os.path.getmtime)
    dataset_test = dataset.match_tags("test")
    add_detections_to_fiftyone(dataset_test, model_root_path, run_number)
    
    classes = CLASSES_TO_KEEP

    filter_threshold = {
        "all": (0, 100),
        "small": (0, 0.33),
        "medium": (0.33, 3),
        "large": (3, 100)
    }

    model_results_path = f"runs/detect/{model_root_path}/results"

    if not os.path.isdir(f"{model_results_path}"):
        os.mkdir(f"{model_results_path}")
        os.mkdir(f"{model_results_path}/plots")
    model_path_save_plots = f"{model_results_path}/plots/"

    for pred in ["all", "small", "medium", "large"]:
        print("------------------------------------------")
        print(pred)
        print("------------------------------------------")
        plots_and_matrixes(
            model_name=model_root_path,
            dataset=dataset,
            classes=classes,
            save_path=model_path_save_plots,
            pred=filter_threshold[pred],
            pred_name=pred,
            evaluators_path=model_results_path
        )

if __name__ == "__main__":
    dataset, classes = get_fiftyone_dataset(0)
    for split in ["full", "train", "test", "val"]:
        save_distribution_plot(dataset, split)
        save_bounding_box_plots(dataset, split)
    # create_color_distribution_plot(dataset)
    # session = fo.launch_app(dataset)
    # session.wait()