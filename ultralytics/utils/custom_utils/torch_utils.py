import json

import fiftyone as fo
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from config import CLASSES_MAPPING
from custom_utils.plots import create_result_plots
from fiftyone import ViewField as F
from fiftyone.utils.eval.coco import DetectionResults
from tqdm import tqdm


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

