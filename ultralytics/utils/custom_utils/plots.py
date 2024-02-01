import json
import os
import pickle
import re

import fiftyone as fo
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from config import CLASSES_MAPPING, PLOTS_PATH
from custom_utils.helpers import get_fiftyone_dataset
from fiftyone import ViewField as F
from fiftyone.utils.eval.coco import DetectionResults
from tqdm import tqdm


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

def create_result_plots(model_type, model_path):
    evaluators_path = f"{model_path}/evaluators"
    evaluators = []
    evaluator_paths = sorted(os.listdir(evaluators_path), key=sort_key)

    for eval_path in evaluator_paths:
        with open(f"{evaluators_path}/{eval_path}", "rb") as file:
            evaluator = pickle.load(file)
            evaluators.append(evaluator)

    metrics = {"AP_0.5_0.95": [], "AP_0.5": [], "AP_0.75": [], "loss_box_reg": []}

    for i, e in enumerate(evaluators):
        AP_05_95, AP_05, AP_075 = e["stats"][0], e["stats"][1], e["stats"][2]
        metrics["AP_0.5_0.95"].append(AP_05_95)
        metrics["AP_0.5"].append(AP_05)
        metrics["AP_0.75"].append(AP_075)

        # sums = {key: 0.0 for key in e[0]}

        # for entry in e:
        #     for key, value in entry.items():
        #         sums[key] += value

        # num_entries = len(e)
        # e = {key: total / num_entries for key, total in sums.items()}
        
        for key, value in e["loss_dict_reduced"].items():
            if key in metrics.keys():
                metrics[key].append(value.item())
            else:
                metrics[key] = [value.item()]

    for l in metrics: 
        epochs = list(range(0, len(metrics[l])))
        plt.figure()
        plt.plot(epochs, metrics[l], marker='o', linestyle='-', color='tab:blue')
        plt.xlabel("Epochs")
        plt.ylabel(l)
        plt.title(f"{l} for {model_type}")
        plt.savefig(f"{model_path}/plots/{l}.png")




def plots_and_matrixes(model_name, dataset, classes, save_path, evaluators_path, pred=(0,100), pred_name="all"):
    if model_name.split("_", 1)[0] != "ultra":
        create_result_plots(model_type=model_name, model_path=evaluators_path) 
    else:
        model_name = model_name.split("_", 1)[1]

    _classes = lambda dataset: list(dataset.count_values(f'{model_name}.detections.label'))
 
    label = F("bbox_area_percentage")
    conf = F("confidence")
    lower_thresh = pred[0]
    upper_thresh = pred[1]

    print(dataset)
    
    expr = F("is_yellow") == True
    view = dataset.match_tags("test").match(expr).clone()
    clone = view

    print("BEFORE:",len(view))

    # counts = dataset.count_values("detections.detections.label")
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

    print("AFTER:",len(view))

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

    if pred_name == "all":
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
        map_50 = eval_results_2.mAP()
        map_75 = eval_results_3.mAP()
        print("mAP@0.5", eval_results_2.mAP())
        print("mAP@0.75", eval_results_3.mAP())
    else:
        map_50 = 0
        map_75 = 0


    print("mAP@0.5:0.95", eval_results.mAP())
    
    report = eval_results.report()
    weighted_avg = report['weighted avg']
    map_50_95 = eval_results.mAP()

    obj = {
        "mAP@0.5:0.95": map_50_95,
        "mAP@0.5": map_50,
        "mAP@0.75": map_75,
        "weighted_avg": weighted_avg
    }

    # test_stats_path = f"{evaluators_path}/results/{model_type}/test_set_stats_{pred_name}.json"
    test_stats_path = f"{evaluators_path}/test_set_stats_{pred_name}_gul.json"

    with open(test_stats_path, "w") as file:
        json.dump(obj, file, indent=2)

    eval_results.print_report(classes=classes)


    # if model_type == "retinanet" or model_cat == "ultra":
    if model_name.split("_", 1)[0] == "ultra":
        title = f"{model_name.split('_', 1)[0]} {pred_name} objects"
    else:
        title = f"{model_name} {pred_name} objects"

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
    plt.savefig(f"{save_path}confusion_matrix_{pred_name}_conf02_norm_gul.png")

    # plot_matrix = eval_results.plot_confusion_matrix(classes=classes, title=title, log_colorscale=True, colorscale="matter")
    # plot_matrix.update_layout(
    #     xaxis_title=dict(text='Predicted'),
    #     yaxis_title=dict(text='Truth'),
    # )
    # plot_matrix.save(f"{save_path}confusion_matrix_{pred_name}_conf02_testing_again.png")
    # plot_matrix.show()
    # plot_matrix.savefig(f"{save_path}confusion_matrix.png")
    
    plot_PR = eval_results.plot_pr_curves(classes=classes, title=f"{model_name} {pred_name} objects")
    # plot_PR.show()
    plot_PR.write_image(f"{save_path}pr_curve_{pred_name}_conf02_gul.png")
    # plot_PR.save(f"{save_path}pr_curve.png")

    clone.delete()





if __name__ == "__main__":
    dataset, classes = get_fiftyone_dataset(0)
    # for split in ["full", "train", "test", "val"]:
    #     # save_distribution_plot(dataset, split)
    #     save_bounding_box_plots(dataset, split)
    create_color_distribution_plot(dataset)
    # session = fo.launch_app(dataset)
    # session.wait()