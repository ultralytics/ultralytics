---
title: YOLO Performance Metrics
comments: true
description: Read YOLO26 validation metrics: mAP50, mAP50-95, precision, recall, F1 and the confusion matrix, with measured COCO results for YOLO26n to YOLO26x.
keywords: YOLO26 performance metrics, mAP, IoU, F1 Score, Precision, Recall, object detection, Ultralytics
---

# Performance Metrics Deep Dive

## Introduction

Performance metrics are key tools to evaluate the [accuracy](https://www.ultralytics.com/glossary/accuracy) and efficiency of [object detection](https://www.ultralytics.com/glossary/object-detection) models. They shed light on how effectively a model can identify and localize objects within images. Additionally, they help in understanding the model's handling of false positives and false negatives. These insights are crucial for evaluating and enhancing the model's performance. This guide explains what the precision, recall, mAP<sub>50</sub>, and mAP<sub>50-95</sub> that Ultralytics YOLO26 validation reports mean, at which confidence they are measured, and [how to interpret them](https://academy.ultralytics.com/courses/computer-vision-foundations/reading-detection-metrics), with measured results for the five YOLO26 detection models on COCO val2017.

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

- **Average Precision (AP):** AP computes the area under the precision-recall curve, providing a single value that encapsulates the model's precision and recall performance.

- **[Mean Average Precision](https://www.ultralytics.com/glossary/mean-average-precision-map) (mAP):** mAP extends the concept of AP by calculating the average AP values across multiple object classes. This is useful in multi-class object detection scenarios to provide a comprehensive evaluation of the model's performance.

- **Precision and Recall:** [Precision](https://www.ultralytics.com/glossary/precision) quantifies the proportion of true positives among all positive predictions, assessing the model's capability to avoid false positives. [Recall](https://www.ultralytics.com/glossary/recall) calculates the proportion of true positives among all actual positives, measuring the model's ability to detect all instances of a class.

- **F1 Score:** The [F1 Score](https://www.ultralytics.com/glossary/f1-score) is the harmonic mean of precision and recall, providing a balanced assessment of a model's performance while considering both false positives and false negatives.

## How to Calculate Metrics for YOLO26 Model

[YOLO26's Validation mode](../modes/val.md) computes the evaluation metrics described above.

Using the validation mode is simple. Once you have a trained model, call the `model.val()` function. This function will then process the validation dataset and return a variety of performance metrics. But what do these metrics mean? And how should you interpret them?

!!! example "Validate YOLO26n on COCO and read the metrics"

    === "Python"

        ```python
        from ultralytics import YOLO

        model = YOLO("yolo26n.pt")
        metrics = model.val(data="coco.yaml")

        print(metrics.box.map)  # mAP50-95
        print(metrics.box.map50)  # mAP50
        print(metrics.box.map75)  # mAP75
        print(metrics.box.mp, metrics.box.mr)  # mean precision and recall
        print(metrics.speed)  # milliseconds per image for each stage
        ```

    === "CLI"

        ```bash
        yolo val model=yolo26n.pt data=coco.yaml
        ```

`coco.yaml` downloads the [COCO dataset](../datasets/detect/coco.md) (20.3 GB) when it is not already present; `data="coco8.yaml"` runs the same code on 4 images. Pass `split="test"` to validate the test split when the dataset YAML defines one. The Python run prints the output below, shortened here to the `all` row, the first three and last two of the 80 class rows, and the first three lines of the COCO-style summary:

```text
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95)
                   all       5000      36335      0.658      0.516      0.563      0.402
                person       2693      10777       0.76      0.688      0.756      0.517
               bicycle        149        314      0.664      0.427      0.506      0.291
                   car        535       1918      0.644      0.534      0.579       0.37
...
            hair drier          9         11          1          0     0.0113    0.00428
            toothbrush         34         57      0.534      0.263      0.295      0.195
Speed: 0.2ms preprocess, 0.4ms inference, 0.0ms loss, 0.3ms postprocess per image
...
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.408
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.569
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.443
...
0.4023844784908059
0.5626923432513534
0.43753269617463014
0.6580044618349068 0.5160882784840829
{'preprocess': 0.1911066472530365, 'inference': 0.3786215854808688, 'loss': 0.00022933557629585264, 'postprocess': 0.3305732322856784}
```

The per-class values are in `metrics.box.p`, `metrics.box.r`, `metrics.box.f1`, `metrics.box.ap50`, and `metrics.box.ap`, ordered by `metrics.box.ap_class_index`. `metrics.box.all_ap` holds the AP of each class at each of the 10 IoU thresholds, and [`metrics.summary()`](../modes/val.md#example-validation-with-arguments) returns the class rows as a list.

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

P and R are reported at one confidence threshold: the one that maximizes the F1 curve averaged over classes and smoothed, with detections matched at an IoU of 0.50. They are not measured at the `conf` value passed to `model.val()`, although a `conf` close to or above that threshold shifts them. In class rows, the mAP<sub>50</sub> and mAP<sub>50-95</sub> columns hold the AP of that class, and the `all` row is the mean over the classes that have labels. The [benchmarks](#confidence-threshold) show the confidence selected for each YOLO26 model.

#### Speed Metrics

The speed of inference can be as critical as accuracy, especially in real-time object detection scenarios. This section breaks down the time taken for various stages of the validation process, from preprocessing to post-processing. The `Speed` line and `metrics.speed` report the `preprocess`, `inference`, `loss`, and `postprocess` stages in milliseconds per image. The [benchmarks](#speed) list these times for the five YOLO26 models.

#### COCO Metrics Evaluation

For users validating on the [COCO dataset](../datasets/detect/coco.md) with `coco.yaml`, the predictions are also saved to `predictions.json` and additional metrics are calculated with [`faster-coco-eval`](https://github.com/MiXaiLL76/faster_coco_eval). These metrics give insights into precision and recall at different IoU thresholds and for objects of different sizes. On other detection datasets, pass `save_json=True` to get them. The object that `model.val()` returns keeps the values Ultralytics computes itself, so `metrics.box.map` can differ slightly from the printed COCO-style AP, as the [benchmarks](#validation-results) show.

#### Visual Outputs

The `model.val()` function, apart from producing numeric metrics, also yields visual outputs that can provide a more intuitive understanding of the model's performance. Here's a breakdown of the visual outputs you can expect:

- **F1 Score Curve (`BoxF1_curve.png`)**: This curve represents the F1 score across various thresholds. Interpreting this curve can offer insights into the model's balance between false positives and false negatives over different thresholds.

- **Precision-Recall Curve (`BoxPR_curve.png`)**: An integral visualization for any classification problem, this curve showcases the trade-offs between precision and recall at varied thresholds. It becomes especially significant when dealing with imbalanced classes.

- **Precision Curve (`BoxP_curve.png`)**: A graphical representation of precision values at different thresholds. This curve helps in understanding how precision varies as the threshold changes.

- **Recall Curve (`BoxR_curve.png`)**: Correspondingly, this graph illustrates how the recall values change across different thresholds.

- **[Confusion Matrix](https://www.ultralytics.com/glossary/confusion-matrix) (`confusion_matrix.png`)**: The confusion matrix provides a detailed view of the outcomes, showcasing correct detections on the diagonal and confusions between classes off the diagonal. The `background` column counts detections that match no object and the `background` row counts missed objects; object detection has no true negatives. The matrix counts every detection above the validation `conf`, which defaults to 0.001.

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

## Benchmarks

The benchmarks ran Ultralytics 8.4.216 in PyTorch FP32 on one **NVIDIA RTX PRO 6000 Blackwell Server Edition** GPU. Each of the five pretrained [YOLO26](../models/yolo26.md) detection models was validated on the 5,000 images of the [COCO](../datasets/detect/coco.md) val2017 split with `model.val(data="coco.yaml")`. Unless a column names another setting, the runs use the validation defaults `imgsz=640`, `batch=16`, `conf=0.001`, and `iou=0.7`. Scores are on the 0 to 1 scale that validation prints.

### Validation Results

By default, validation scores the YOLO26 one-to-many head with [NMS](https://www.ultralytics.com/glossary/non-maximum-suppression-nms); `nms=False` scores the [NMS-free head](end2end-detection.md) instead.

=== "Default"

    The `all` row of each run, and the COCO-style AP at IoU 0.50 to 0.95 that `faster-coco-eval` gives for the predictions saved by the same run. Higher is better.

    | Model   | P     | R     | mAP<sub>50</sub> | mAP<sub>50-95</sub> | COCO-style AP |
    | ------- | ----- | ----- | ---------------- | ------------------- | ------------- |
    | YOLO26n | 0.658 | 0.516 | 0.563            | 0.402               | 0.408         |
    | YOLO26s | 0.699 | 0.598 | 0.651            | 0.480               | 0.486         |
    | YOLO26m | 0.738 | 0.637 | 0.701            | 0.524               | 0.531         |
    | YOLO26l | 0.754 | 0.653 | 0.721            | 0.545               | 0.551         |
    | YOLO26x | 0.756 | 0.680 | 0.745            | 0.568               | 0.575         |

=== "`nms=False`"

    The NMS-free head. Higher is better.

    | Model   | P     | R     | mAP<sub>50</sub> | mAP<sub>50-95</sub> | COCO-style AP |
    | ------- | ----- | ----- | ---------------- | ------------------- | ------------- |
    | YOLO26n | 0.656 | 0.505 | 0.551            | 0.395               | 0.401         |
    | YOLO26s | 0.693 | 0.583 | 0.639            | 0.472               | 0.478         |
    | YOLO26m | 0.734 | 0.629 | 0.691            | 0.518               | 0.525         |
    | YOLO26l | 0.739 | 0.642 | 0.710            | 0.538               | 0.543         |
    | YOLO26x | 0.744 | 0.673 | 0.736            | 0.563               | 0.569         |

- **COCO-style AP**, multiplied by 100, is within 0.1 of the [published table](../tasks/detect.md#models) for all five models and both heads.

### IoU Threshold and Object Size

=== "IoU threshold"

    Mean AP at one IoU threshold and averaged over the 10 thresholds, from `metrics.box.map50`, `metrics.box.map75`, and `metrics.box.map`. Higher is better.

    | Model   | mAP<sub>50</sub> | mAP<sub>75</sub> | mAP<sub>50-95</sub> |
    | ------- | ---------------- | ---------------- | ------------------- |
    | YOLO26n | 0.563            | 0.438            | 0.402               |
    | YOLO26s | 0.651            | 0.522            | 0.480               |
    | YOLO26m | 0.701            | 0.571            | 0.524               |
    | YOLO26l | 0.721            | 0.596            | 0.545               |
    | YOLO26x | 0.745            | 0.622            | 0.568               |

=== "Object size"

    COCO-style AP at IoU 0.50 to 0.95 for small, medium, and large objects and for all objects, from the summary that `faster-coco-eval` prints. Higher is better.

    | Model   | AP<sub>S</sub> | AP<sub>M</sub> | AP<sub>L</sub> | AP    |
    | ------- | -------------- | -------------- | -------------- | ----- |
    | YOLO26n | 0.212          | 0.448          | 0.589          | 0.408 |
    | YOLO26s | 0.296          | 0.532          | 0.659          | 0.486 |
    | YOLO26m | 0.367          | 0.577          | 0.690          | 0.531 |
    | YOLO26l | 0.385          | 0.596          | 0.710          | 0.551 |
    | YOLO26x | 0.419          | 0.621          | 0.733          | 0.575 |

- **Small objects** scored a COCO-style AP of 0.212 (YOLO26n) to 0.419 (YOLO26x), against 0.589 to 0.733 for large objects.

### Confidence Threshold

The first tab is the default run: it shows the confidence that validation selected from the smoothed mean F1 curve, with the P, R, and mean F1 at that confidence. The other three tabs pass a different `conf` to `model.val()`.

=== "Selected confidence"

    Default run on COCO val2017, `conf=0.001`. F1 is the mean of the per-class F1 values.

    | Model   | Confidence | P     | R     | F1    |
    | ------- | ---------- | ----- | ----- | ----- |
    | YOLO26n | 0.201      | 0.658 | 0.516 | 0.565 |
    | YOLO26s | 0.261      | 0.699 | 0.598 | 0.638 |
    | YOLO26m | 0.296      | 0.738 | 0.637 | 0.679 |
    | YOLO26l | 0.297      | 0.754 | 0.653 | 0.694 |
    | YOLO26x | 0.300      | 0.756 | 0.680 | 0.711 |

=== "Mean AP by `conf`"

    mAP<sub>50-95</sub> from `metrics.box.map`, higher is better.

    | Model   | `conf=0.001` | `conf=0.1` | `conf=0.25` | `conf=0.5` |
    | ------- | ------------ | ---------- | ----------- | ---------- |
    | YOLO26n | 0.402        | 0.374      | 0.336       | 0.276      |
    | YOLO26s | 0.480        | 0.453      | 0.421       | 0.364      |
    | YOLO26m | 0.524        | 0.497      | 0.466       | 0.411      |
    | YOLO26l | 0.545        | 0.517      | 0.485       | 0.427      |
    | YOLO26x | 0.568        | 0.541      | 0.509       | 0.454      |

=== "Precision by `conf`"

    The P column of the `all` row.

    | Model   | `conf=0.001` | `conf=0.1` | `conf=0.25` | `conf=0.5` |
    | ------- | ------------ | ---------- | ----------- | ---------- |
    | YOLO26n | 0.658        | 0.658      | 0.698       | 0.860      |
    | YOLO26s | 0.699        | 0.699      | 0.690       | 0.839      |
    | YOLO26m | 0.738        | 0.738      | 0.729       | 0.843      |
    | YOLO26l | 0.754        | 0.754      | 0.742       | 0.863      |
    | YOLO26x | 0.756        | 0.756      | 0.752       | 0.854      |

=== "Recall by `conf`"

    The R column of the `all` row.

    | Model   | `conf=0.001` | `conf=0.1` | `conf=0.25` | `conf=0.5` |
    | ------- | ------------ | ---------- | ----------- | ---------- |
    | YOLO26n | 0.516        | 0.516      | 0.487       | 0.363      |
    | YOLO26s | 0.598        | 0.598      | 0.602       | 0.480      |
    | YOLO26m | 0.637        | 0.637      | 0.644       | 0.543      |
    | YOLO26l | 0.653        | 0.653      | 0.659       | 0.556      |
    | YOLO26x | 0.680        | 0.680      | 0.683       | 0.590      |

- **`conf=0.25`** lowered mAP<sub>50-95</sub> by 0.058 to 0.066 and **`conf=0.5`** by 0.113 to 0.126, because predictions below the threshold are removed before the precision-recall curve is built.
- **`conf=0.1`** left the P and R columns unchanged for every model. **`conf=0.25`** changed them for all five, including the four whose selected confidence in the default run is above 0.25.

### NMS IoU Threshold

The `iou` argument is the NMS threshold of the default head. It is not the IoU used to match predictions to labels, which is fixed at 0.50 to 0.95.

mAP<sub>50-95</sub> from `metrics.box.map`, higher is better.

| Model   | `iou=0.5` | `iou=0.6` | `iou=0.7` |
| ------- | --------- | --------- | --------- |
| YOLO26n | 0.398     | 0.402     | 0.402     |
| YOLO26s | 0.474     | 0.479     | 0.480     |
| YOLO26m | 0.517     | 0.522     | 0.524     |
| YOLO26l | 0.537     | 0.543     | 0.545     |
| YOLO26x | 0.561     | 0.566     | 0.568     |

### Confusion Matrix

Precision and recall can also be derived from `metrics.confusion_matrix.matrix`, which is filled when `plots=True` (the default), pooled over all classes: correct detections divided by all predictions in the matrix, and correct detections divided by all labeled objects.

=== "Precision"

    Results table is the P column of the `all` row of the default run.

    | Model   | Results table | Matrix at `conf=0.001` | Matrix at `conf=0.25` |
    | ------- | ------------- | ---------------------- | --------------------- |
    | YOLO26n | 0.658         | 0.039                  | 0.733                 |
    | YOLO26s | 0.699         | 0.049                  | 0.700                 |
    | YOLO26m | 0.738         | 0.061                  | 0.709                 |
    | YOLO26l | 0.754         | 0.065                  | 0.720                 |
    | YOLO26x | 0.756         | 0.074                  | 0.720                 |

=== "Recall"

    Results table is the R column of the `all` row of the default run.

    | Model   | Results table | Matrix at `conf=0.001` | Matrix at `conf=0.25` |
    | ------- | ------------- | ---------------------- | --------------------- |
    | YOLO26n | 0.516         | 0.780                  | 0.496                 |
    | YOLO26s | 0.598         | 0.845                  | 0.613                 |
    | YOLO26m | 0.637         | 0.876                  | 0.667                 |
    | YOLO26l | 0.653         | 0.884                  | 0.675                 |
    | YOLO26x | 0.680         | 0.898                  | 0.702                 |

- **At the default `conf=0.001`** the matrix held 437,988 to 732,586 predictions for the 36,335 labeled objects.

### Per-Class AP

AP<sub>50-95</sub> from `metrics.box.ap` for the four classes that were the highest or the lowest of the 80 for at least one model, next to the mean over all classes.

| Model   | bear  | cat   | book  | hair drier | All classes |
| ------- | ----- | ----- | ----- | ---------- | ----------- |
| YOLO26n | 0.726 | 0.719 | 0.107 | 0.004      | 0.402       |
| YOLO26s | 0.766 | 0.748 | 0.159 | 0.068      | 0.480       |
| YOLO26m | 0.793 | 0.798 | 0.206 | 0.141      | 0.524       |
| YOLO26l | 0.801 | 0.804 | 0.212 | 0.284      | 0.545       |
| YOLO26x | 0.818 | 0.817 | 0.239 | 0.364      | 0.568       |

### Speed

Milliseconds per image from `metrics.speed`, median of three runs, lower is better. `model.val()` uses `rect=True`, so `batch` also changes the padded input shapes.

=== "Inference"

    | Model   | Default, `batch=16` | `batch=1` | `nms=False` |
    | ------- | ------------------- | --------- | ----------- |
    | YOLO26n | 0.37                | 2.69      | 0.38        |
    | YOLO26s | 0.67                | 2.94      | 0.68        |
    | YOLO26m | 1.45                | 3.27      | 1.46        |
    | YOLO26l | 1.75                | 4.68      | 1.78        |
    | YOLO26x | 3.01                | 5.89      | 2.98        |

=== "Postprocess"

    | Model   | Default, `batch=16` | `batch=1` | `nms=False` |
    | ------- | ------------------- | --------- | ----------- |
    | YOLO26n | 0.33                | 0.46      | 0.04        |
    | YOLO26s | 0.30                | 0.45      | 0.04        |
    | YOLO26m | 0.30                | 0.43      | 0.04        |
    | YOLO26l | 0.28                | 0.44      | 0.04        |
    | YOLO26x | 0.28                | 0.42      | 0.04        |

- **`batch=1`** took 2.0 to 7.3 times longer per image in inference than `batch=16`.

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

### How do I interpret the Intersection over Union (IoU) value for YOLO26 object detection?

Intersection over Union (IoU) measures the overlap between the predicted and ground truth bounding boxes. IoU values range from 0 to 1, where higher values indicate better localization accuracy. An IoU of 1.0 means perfect alignment. Typically, an IoU threshold of 0.50 is used to define true positives in metrics like mAP. Validation does not print an IoU score: lower IoU can show up as mAP<sub>50-95</sub> falling well below mAP<sub>50</sub>, and suggests that the model struggles with precise object localization, which can be improved by refining bounding box regression or increasing annotation accuracy in your [training dataset](../datasets/index.md).

### Why is the F1 Score important for evaluating YOLO26 models in object detection?

The F1 Score is important for evaluating YOLO26 models because it provides a harmonic mean of precision and recall, balancing both false positives and false negatives. It is particularly valuable when dealing with imbalanced datasets or applications where either precision or recall alone is insufficient. A high F1 Score indicates that the model effectively detects objects while minimizing both missed detections and false alarms, making it suitable for critical applications like [security systems](security-alarm-system.md) and [medical imaging](https://www.ultralytics.com/solutions/computer-vision-in-healthcare).

### What are the key advantages of using Ultralytics YOLO26 for real-time object detection?

Ultralytics YOLO26 offers multiple advantages for real-time object detection:

- **Speed and Efficiency**: Optimized for high-speed inference, suitable for applications requiring low latency.
- **High Accuracy**: Achieves high mAP across model sizes while balancing precision and recall.
- **Flexibility**: Supports various tasks including [object detection](../tasks/detect.md), [instance segmentation](../tasks/segment.md), [semantic segmentation](../tasks/semantic.md), and [classification](../tasks/classify.md).
- **Ease of Use**: User-friendly interfaces, extensive documentation, and integration with tools like [Ultralytics Platform](../platform/quickstart.md).

This makes YOLO26 well suited to diverse applications, from autonomous vehicles to [smart city solutions](queue-management.md).

### How can validation metrics from YOLO26 help improve model performance?

Validation metrics from YOLO26 like precision, recall, and mAP help diagnose and improve model performance by providing insights into different aspects of detection:

- **Precision**: Helps identify and minimize false positives.
- **Recall**: Ensures all relevant objects are detected.
- **mAP**: Offers an overall performance snapshot, guiding general improvements.
- **mAP<sub>50-95</sub>**: Helps fine-tune object localization accuracy, because it includes tighter IoU thresholds than mAP<sub>50</sub>.

By analyzing these metrics, specific weaknesses can be targeted, such as adjusting the confidence threshold used for prediction to improve precision or gathering more diverse data to enhance recall. For detailed explanations of these metrics and how to interpret them, check [Object Detection Metrics](#object-detection-metrics) and consider implementing [hyperparameter tuning](hyperparameter-tuning.md) to optimize your model.

### At what confidence threshold are precision and recall calculated in YOLO validation?

Ultralytics YOLO validation reports precision and recall at the confidence that maximizes the smoothed F1 curve averaged over classes, not at the `conf` argument. A detection counts as correct at that point when it matches a label of the same class with an IoU of at least 0.50. For the pretrained YOLO26 models on COCO, that confidence was 0.201 to 0.300. See [Confidence Threshold](#confidence-threshold).

### Why do precision and recall from the confusion matrix differ from the validation results?

Precision and recall in the Ultralytics results table and in the confusion matrix are computed differently. The results table averages per-class values at the confidence selected from the smoothed mean F1 curve, while the confusion matrix counts every prediction above `conf`, which defaults to 0.001, matches boxes regardless of class, and pools the counts over classes. For YOLO26n on COCO, the table reported a precision of 0.658 and the matrix gave 0.039 at the default `conf`; at `conf=0.25` the table reported 0.698 and the matrix gave 0.733. See [Confusion Matrix](#confusion-matrix).

### Why does mAP drop when I raise `conf` in validation?

mAP averages the area under the precision-recall curve of each class, and a higher `conf` removes the low-confidence predictions that form the high-recall end of those curves. For YOLO26n on COCO, mAP<sub>50-95</sub> went from 0.402 at the default `conf=0.001` to 0.336 at `conf=0.25`. The `conf` used for [prediction](../modes/predict.md) is a separate setting. See [Confidence Threshold](#confidence-threshold).

### Why is `metrics.box.map` different from the published COCO mAP?

The published values are COCO-style AP, which validation computes with `faster-coco-eval` and prints for `coco.yaml`, while `metrics.box.map` keeps the value Ultralytics computes itself. On COCO val2017, the five YOLO26 models returned 0.402 to 0.568 in `metrics.box.map` against a printed COCO-style AP of 0.408 to 0.575. See [Validation Results](#validation-results).

### How is `best.pt` chosen during training?

Ultralytics saves `best.pt` at the epoch with the highest validation fitness; for detection, fitness is mAP<sub>50-95</sub>, also available as `metrics.fitness`. To save the best model by another metric, see [Saving the Best Model by Custom Metric](custom-trainer.md#saving-the-best-model-by-custom-metric).
