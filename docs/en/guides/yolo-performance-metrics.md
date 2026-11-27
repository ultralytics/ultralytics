---
title: YOLO Performance Metrics - mAP50 vs mAP50-95, Precision, Recall
comments: true
description: "mAP50 scores YOLO26 detections at an IoU of 0.50, while mAP50-95 averages AP over ten IoU thresholds from 0.50 to 0.95. Learn what these, precision, recall, IoU and F1 mean and how to compute them with val mode."
keywords: YOLO26 performance metrics, mAP50, mAP50-95, mAP50 vs mAP50-95, mAP, IoU, F1 Score, Precision, Recall, object detection, Ultralytics
---

# YOLO Performance Metrics: mAP50, mAP50-95, Precision and Recall

## Introduction

Performance metrics are key tools to evaluate the [accuracy](https://www.ultralytics.com/glossary/accuracy) and efficiency of [object detection](https://www.ultralytics.com/glossary/object-detection) models. They shed light on how effectively a model can identify and localize objects within images. Additionally, they help in understanding the model's handling of false positives and false negatives. These insights are crucial for evaluating and enhancing the model's performance. This guide explores the performance metrics associated with YOLO26, their significance, and [how to interpret them](https://academy.ultralytics.com/courses/computer-vision-foundations/reading-detection-metrics).

<p align="center">
  <br>
  <iframe loading="lazy" width="720" height="405" src="https://www.youtube.com/embed/q7LwPoM7tSQ"
    title="YouTube video player" frameborder="0"
    allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share"
    allowfullscreen>
  </iframe>
  <br>
  <strong>Watch:</strong> Ultralytics YOLO26 Performance Metrics | mAP, F1 Score, Precision, IoU & Accuracy
</p>

## Object Detection Metrics

The following metrics are important for YOLO26 and broadly applicable across different object detection models.

- **[Intersection over Union](https://www.ultralytics.com/glossary/intersection-over-union-iou) (IoU):** IoU is a measure that quantifies the overlap between a predicted [bounding box](https://www.ultralytics.com/glossary/bounding-box) and a ground truth bounding box. It plays a fundamental role in evaluating the accuracy of object localization.

- **Average Precision (AP):** AP computes the area under the precision-recall curve, providing a single value that encapsulates the model's precision and recall performance. Ultralytics samples the curve at COCO's 101 recall thresholds (0.00 to 1.00), taking at each one the highest precision reached at that recall or beyond, so a perfect detector scores 1.0. Since `ultralytics 8.4.219` uses this rule, the same weights can score slightly higher or lower than with earlier versions, so compare runs made with the same package version.

- **[Mean Average Precision](https://www.ultralytics.com/glossary/mean-average-precision-map) (mAP):** mAP extends the concept of AP by calculating the average AP values across multiple object classes. This is useful in multi-class object detection scenarios to provide a comprehensive evaluation of the model's performance.

- **Precision and Recall:** [Precision](https://www.ultralytics.com/glossary/precision) quantifies the proportion of true positives among all positive predictions, assessing the model's capability to avoid false positives. [Recall](https://www.ultralytics.com/glossary/recall) calculates the proportion of true positives among all actual positives, measuring the model's ability to detect all instances of a class.

- **F1 Score:** The [F1 Score](https://www.ultralytics.com/glossary/f1-score) is the harmonic mean of precision and recall, providing a balanced assessment of a model's performance while considering both false positives and false negatives.

## How to Calculate Metrics for YOLO26 Model

[YOLO26's Validation mode](../modes/val.md) computes the evaluation metrics described above.

Using the validation mode is simple. Once you have a trained model, call the `model.val()` function. This function will then process the validation dataset and return a variety of performance metrics. But what do these metrics mean? And how should you interpret them?

### Interpreting the Output

The sections below break down each segment of the `model.val()` output.

#### Class-wise Metrics

One of the sections of the output is the class-wise breakdown of performance metrics. This granular information is useful when you are trying to understand how well the model is doing for each specific class, especially in datasets with a diverse range of object categories. For each class in the dataset, the following is provided:

- **Class**: This denotes the name of the object class, such as "person", "car", or "dog".

- **Images**: This metric tells you the number of images in the validation set that contain the object class. In the `all` row, it is the total number of images validated.

- **Instances**: This provides the count of how many times the class appears across all images in the validation set.

- **Box(P, R, mAP<sub>50</sub>, mAP<sub>50-95</sub>)**: This metric provides insights into the model's performance in detecting objects:
    - **P (Precision)**: The accuracy of the detected objects, indicating how many detections were correct.

    - **R (Recall)**: The ability of the model to identify all instances of objects in the images.

    - **mAP<sub>50</sub>**: Mean average precision calculated at an intersection over union (IoU) threshold of 0.50. A detection counts as correct when it matches an object of the same class with an IoU of at least 0.50, which makes it the most lenient of the reported scores.

    - **mAP<sub>50-95</sub>**: The average of the mean average precision calculated at varying IoU thresholds, ranging from 0.50 to 0.95 in steps of 0.05. It gives a comprehensive view of the model's performance across different levels of detection difficulty.

P and R are reported at one confidence threshold: the one that maximizes the F1 curve averaged over classes and smoothed (about 0.20 for YOLO26n on COCO val2017), with detections matched at an IoU of 0.50. They are not measured at the `conf` value passed to `model.val()`, although a `conf` close to or above that threshold shifts them. In class rows, the mAP<sub>50</sub> and mAP<sub>50-95</sub> columns hold the AP of that class, and the `all` row is the mean over the classes that have labels.

#### Speed Metrics

The speed of inference can be as critical as accuracy, especially in real-time object detection scenarios. This section breaks down the time taken for various stages of the validation process, from preprocessing to post-processing. The `Speed` line and `metrics.speed` report the `preprocess`, `inference`, `loss`, and `postprocess` stages in milliseconds per image.

#### COCO Metrics Evaluation

For users validating on the [COCO dataset](../datasets/detect/coco.md) with `coco.yaml`, the predictions are also saved to `predictions.json` and additional metrics are calculated with [`faster-coco-eval`](https://github.com/MiXaiLL76/faster_coco_eval). These metrics give insights into precision and recall at different IoU thresholds and for objects of different sizes. On other detection datasets, pass `save_json=True` to get them.

On COCO val2017, native box, mask, and keypoint AP use the official annotations when available. Ultralytics evaluates the original-image predictions at the same precision as the JSON export. It matches ordinary objects first and can ignore multiple detections in a crowd region independently at each IoU threshold. COCO keeps up to 100 detections per image and category for boxes and masks and 20 for keypoints. People without labeled keypoints count for box AP and are ignored for keypoint AP. Custom YOLO text labels contain no crowd annotations.

Instance-mask evaluation uses the existing `faster-coco-eval` decoder for the official polygons and run-length encoded masks. Install it before training to get official COCO box and mask AP during each segmentation epoch. Scoring masks at original image resolution increases both validation time and peak GPU memory during training. Without the decoder, segmentation validation logs a warning and evaluates the YOLO text labels. The small-, medium-, and large-object mAP are added to `metrics.results_dict` as `metrics/mAP_small(B)`, `metrics/mAP_medium(B)`, and `metrics/mAP_large(B)`.

#### Visual Outputs

The `model.val()` function, apart from producing numeric metrics, also yields visual outputs that can provide a more intuitive understanding of the model's performance. Here's a breakdown of the visual outputs you can expect:

- **F1 Score Curve (`BoxF1_curve.png`)**: This curve represents the F1 score across various thresholds. Interpreting this curve can offer insights into the model's balance between false positives and false negatives over different thresholds.

- **Precision-Recall Curve (`BoxPR_curve.png`)**: An integral visualization for any classification problem, this curve showcases the trade-offs between precision and recall at varied thresholds. It becomes especially significant when dealing with imbalanced classes.

- **Precision Curve (`BoxP_curve.png`)**: A graphical representation of precision values at different thresholds. This curve helps in understanding how precision varies as the threshold changes.

- **Recall Curve (`BoxR_curve.png`)**: Correspondingly, this graph illustrates how the recall values change across different thresholds.

- **[Confusion Matrix](https://www.ultralytics.com/glossary/confusion-matrix) (`confusion_matrix.png`)**: The confusion matrix provides a detailed view of the outcomes, showcasing correct detections on the diagonal and confusions between classes off the diagonal. The `background` column counts detections that match no object and the `background` row counts missed objects; object detection has no true negatives. The matrix counts postprocessed detections above the validation `conf`, subject to the `max_det` limit (default 300).

- **Normalized Confusion Matrix (`confusion_matrix_normalized.png`)**: This visualization is a normalized version of the confusion matrix. It represents the data in proportions rather than raw counts, with each column (true class) divided by its total. This format makes it simpler to compare the performance across classes.

- **Validation Batch Labels (`val_batchX_labels.jpg`)**: These images depict the ground truth labels for distinct batches from the validation dataset. They provide a clear picture of what the objects are and their respective locations as per the dataset.

- **Validation Batch Predictions (`val_batchX_pred.jpg`)**: Contrasting the label images, these visuals display the predictions made by the YOLO26 model for the respective batches. By comparing these to the label images, you can easily assess how well the model detects and classifies objects visually.

For the [detection](../tasks/detect.md), [segmentation](../tasks/segment.md), [pose](../tasks/pose.md), and [OBB](../tasks/obb.md) tasks, the curve plots are prefixed by metric type: detection writes `Box*` curves, segmentation writes both `Box*` and `Mask*` curves, pose writes both `Box*` and `Pose*` curves, and OBB writes `Box*` curves.

#### Results Storage

For future reference, the results are saved to a directory, typically named `runs/detect/val`.

## Choosing the Right Metrics

[Choosing the right metrics](https://www.ultralytics.com/blog/accuracy-precision-recall) to evaluate often depends on the specific application.

- **mAP:** Suitable for a broad assessment of model performance.

- **IoU:** Essential when precise object location is crucial.

- **Precision:** Important when minimizing false detections is a priority.

- **Recall:** Vital when it's important to detect every instance of an object.

- **F1 Score:** Useful when a balance between precision and recall is needed.

For real-time applications, speed metrics like [FPS (Frames Per Second)](https://www.ultralytics.com/blog/understanding-the-role-of-fps-in-computer-vision) and [latency](https://www.ultralytics.com/glossary/inference-latency) are crucial to ensure timely results.

## Interpretation of Results

Here's what some commonly observed low scores might suggest:

- **Low mAP:** Indicates the model may need general refinements.

- **Low IoU:** The model might be struggling to pinpoint objects accurately. Validation does not print an IoU score: loose boxes can show up as mAP<sub>50-95</sub> falling well below mAP<sub>50</sub>.

- **Low Precision:** The model may be detecting too many non-existent objects. Adjusting the confidence threshold used for prediction might reduce this.

- **Low Recall:** The model could be missing real objects. Improving [feature extraction](https://www.ultralytics.com/glossary/feature-extraction) or using more data might help.

- **Imbalanced F1 Score:** There's a disparity between precision and recall.

- **Class-specific AP:** Low scores here can highlight classes the model struggles with.

## Case Studies

Real-world examples can help clarify how these metrics work in practice.

### Case 1

- **Situation:** mAP and F1 Score are suboptimal; Recall is good, but Precision isn't.

- **Interpretation & Action:** There might be too many incorrect detections. Tightening the confidence threshold used for prediction could reduce these, though it might also slightly decrease recall.

### Case 2

- **Situation:** mAP<sub>50</sub> and Recall are acceptable, but mAP<sub>50-95</sub> is much lower.

- **Interpretation & Action:** The model detects objects well but might not be localizing them precisely. Refining bounding box predictions might help.

### Case 3

- **Situation:** Some classes have a much lower AP than others, even with a decent overall mAP.

- **Interpretation & Action:** These classes might be more challenging for the model. Using more data for these classes or [adjusting class weights](../modes/train.md#how-do-i-train-a-model-on-an-imbalanced-dataset) with the `cls_pw` training argument could be beneficial.

## Connect and Collaborate

The YOLO26 community can help with learning, troubleshooting, and networking through the following channels.

### Engage with the Broader Community

- **GitHub Issues:** The Ultralytics repository on GitHub has an [Issues tab](https://github.com/ultralytics/ultralytics/issues) where you can ask questions, report bugs, and suggest new features. The community and maintainers are active here, and it's a great place to get help with specific problems.

- **Ultralytics Discord Server:** Ultralytics has a [Discord server](https://discord.com/invite/ultralytics) where you can interact with other users and the developers.

### Official Documentation and Resources

- **Ultralytics YOLO26 Docs:** The [official documentation](../index.md) provides a comprehensive overview of YOLO26, along with guides on installation, usage, and troubleshooting.

These resources can help you resolve problems and stay current with best practices in the YOLO26 community.

## Conclusion

In this guide, we've taken a close look at the essential performance metrics for YOLO26. These metrics are key to understanding how well a model is performing and are vital for anyone aiming to fine-tune their models. They offer the necessary insights for improvements and to make sure the model works effectively in real-life situations.

The YOLO26 and Ultralytics community is also a useful resource: engaging with other developers and experts can surface solutions not covered in the documentation, and sharing your findings helps others working on object detection.

## FAQ

### What is the significance of Mean Average Precision (mAP) in evaluating YOLO26 model performance?

Mean Average Precision (mAP) is crucial for evaluating YOLO26 models as it provides a single metric encapsulating precision and recall across multiple classes. mAP<sub>50</sub> measures average precision at an IoU threshold of 0.50, focusing on the model's ability to detect objects correctly. mAP<sub>50-95</sub> averages it across a range of IoU thresholds, offering a comprehensive assessment of detection performance. High mAP scores indicate that the model effectively balances precision and recall, essential for applications like [autonomous driving](https://www.ultralytics.com/solutions/computer-vision-in-automotive) and surveillance systems where both accurate detection and minimal false alarms are critical.

### Why do precision and recall from the confusion matrix differ from the validation results?

Precision and recall in the Ultralytics results table and in the confusion matrix are computed differently. The results table averages per-class values at the confidence selected from the smoothed mean F1 curve, while the confusion matrix counts the postprocessed detections above `conf`, which defaults to 0.001, and matches boxes regardless of class. Pooling its counts over classes for YOLO26n on COCO gives a precision of 0.039 at the default `conf`, against 0.66 in the table; at `conf=0.25` the table reports 0.698 and the pooled matrix 0.733.

### Why does mAP drop when I raise `conf` in validation?

mAP averages the area under the precision-recall curve of each class, and a higher `conf` removes the low-confidence predictions that form the high-recall end of those curves. For YOLO26n on COCO, mAP<sub>50-95</sub> went from 0.402 at the default `conf=0.001` to 0.336 at `conf=0.25`. The `conf` used for [prediction](../modes/predict.md) is a separate setting.
