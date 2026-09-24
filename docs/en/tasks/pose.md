---
comments: true
description: Discover how to use YOLO26 for pose estimation tasks. Learn about model training, validation, prediction, and exporting in various formats.
keywords: pose estimation, YOLO26, Ultralytics, keypoints, model training, image recognition, deep learning, human pose detection, computer vision, real-time tracking
model_name: yolo26n-pose
---

# Pose Estimation with Ultralytics YOLO {#pose-estimation}

<img width="1024" src="https://cdn.ul.run/i/78b97a2089bd03e3a54f0ee693be727f.avif" alt="Ultralytics YOLO pose estimation with human body keypoint detection">

Pose estimation is a task that involves identifying the location of specific points in an image, usually referred to as keypoints. The keypoints can represent various parts of the object such as joints, landmarks, or other distinctive features. The locations of the keypoints are usually represented as a set of 2D `[x, y]` coordinates, optionally with a visibility flag `[x, y, visible]`.

The output of a pose estimation model is a set of points that represent the keypoints on an object in the image, usually along with the confidence scores for each point. Pose estimation is a good choice when you need to identify specific parts of an object in a scene, and their location in relation to each other.

<p align="center">
  <br>
  <iframe loading="lazy" width="720" height="405" src="https://www.youtube.com/embed/4VTuqfrOIws"
    title="YouTube video player" frameborder="0"
    allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share"
    allowfullscreen>
  </iframe>
  <br>
  <strong>Watch:</strong> How to Run Real-Time Pose Estimation with Ultralytics YOLO26 | Tracking & Keypoints Extraction 🕺
</p>

!!! tip

    YOLO26 Pose models use the `-pose` suffix, i.e., `yolo26n-pose.pt`, and are pretrained on [COCO keypoints](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/datasets/coco-pose.yaml).

## [Models](https://github.com/ultralytics/ultralytics/tree/main/ultralytics/cfg/models/26)

YOLO26 Pose models pretrained on the [COCO keypoints](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/datasets/coco-pose.yaml) dataset are shown below.

[Models](https://github.com/ultralytics/ultralytics/tree/main/ultralytics/cfg/models) download automatically from the latest Ultralytics [release](https://github.com/ultralytics/assets/releases) on first use.

{% include "macros/yolo-pose-perf.md" %}

- **mAP<sup>val</sup>** values are for single-model single-scale on [COCO Keypoints val2017](https://cocodataset.org/) dataset. <br>Reproduce with `yolo pose val data=coco-pose.yaml device=0 nms=False`
- **Speed** averaged over COCO val images using an [Amazon EC2 P4d](https://aws.amazon.com/ec2/instance-types/p4/) instance. <br>Reproduce with `yolo pose val data=coco-pose.yaml batch=1 device=0|cpu nms=False`
- **Params** and **FLOPs** values are for fused models after Conv/BatchNorm folding and removal of the unused detection branch. Pretrained checkpoints retain the full training architecture and may show higher counts.

See the [unreleased YOLO27 preview](../models/yolo27.md#performance-metrics) for preliminary COCO keypoint results.

## Train

Train YOLO26n-pose on the [COCO8-pose](../datasets/pose/coco8-pose.md) dataset for 100 [epochs](https://www.ultralytics.com/glossary/epoch) at image size 640. For a full list of available arguments see the [Configuration](../usage/cfg.md) page.

!!! example

    === "Python"

        ```python
        from ultralytics import YOLO

        # Load a model
        model = YOLO("yolo26n-pose.yaml")  # build a new model from YAML
        model = YOLO("yolo26n-pose.pt")  # load a pretrained model (recommended for training)
        model = YOLO("yolo26n-pose.yaml").load("yolo26n-pose.pt")  # build from YAML and transfer weights

        # Train the model
        results = model.train(data="coco8-pose.yaml", epochs=100, imgsz=640)
        ```

    === "CLI"

        ```bash
        # Build a new model from YAML and start training from scratch
        yolo pose train data=coco8-pose.yaml model=yolo26n-pose.yaml epochs=100 imgsz=640

        # Start training from a pretrained *.pt model
        yolo pose train data=coco8-pose.yaml model=yolo26n-pose.pt epochs=100 imgsz=640

        # Build a new model from YAML, transfer pretrained weights to it and start training
        yolo pose train data=coco8-pose.yaml model=yolo26n-pose.yaml pretrained=yolo26n-pose.pt epochs=100 imgsz=640
        ```

See full `train` mode details in the [Train](../modes/train.md) page. Pose models can also be trained with [Ultralytics Platform cloud training](../platform/train/cloud-training.md).

### Dataset format

YOLO pose dataset format can be found in detail in the [Dataset Guide](../datasets/pose/index.md). To convert an existing [COCO keypoints](../datasets/pose/coco.md) JSON dataset to YOLO format, use the built-in `convert_coco` function with `use_keypoints=True`, as described in the [COCO to YOLO guide](../guides/coco-to-yolo.md). [Ultralytics Platform annotation](../platform/data/annotation.md) also supports pose labels with built-in skeleton templates for person, hand, face, and custom keypoint layouts.

For custom pose estimation tasks, you can also explore specialized datasets like [Tiger-Pose](../datasets/pose/tiger-pose.md) for animal pose estimation, [Hand Keypoints](../datasets/pose/hand-keypoints.md) for hand tracking, or [Dog-Pose](../datasets/pose/dog-pose.md) for canine pose analysis.

## Val

Validate trained YOLO26n-pose model [accuracy](https://www.ultralytics.com/glossary/accuracy). No arguments are needed, as the `model` retains its training `data` and arguments as model attributes: `path/to/best.pt` from the [Train](#train) example validates on COCO8-pose. Official weights record a training dataset path that doesn't exist on your machine, so they fall back to the task default `coco8-pose.yaml` with a warning. Pass `data` to validate on another dataset.

!!! example

    === "Python"

        ```python
        from ultralytics import YOLO

        # Load a model
        model = YOLO("yolo26n-pose.pt")  # load an official model
        model = YOLO("path/to/best.pt")  # load a custom model

        # Validate the model
        metrics = model.val()  # no arguments needed, dataset and settings remembered
        metrics.box.map  # map50-95
        metrics.box.map50  # map50
        metrics.box.map75  # map75
        metrics.box.maps  # a list containing mAP50-95 for each category
        metrics.box.image_metrics  # per-image metrics dictionary for box with precision, recall, F1, TP, FP, and FN
        metrics.pose.map  # map50-95(P)
        metrics.pose.map50  # map50(P)
        metrics.pose.map75  # map75(P)
        metrics.pose.maps  # a list containing mAP50-95(P) for each category
        metrics.pose.image_metrics  # per-image metrics dictionary for pose with precision, recall, F1, TP, FP, and FN
        ```

    === "CLI"

        ```bash
        yolo pose val model=yolo26n-pose.pt data=coco8-pose.yaml   # val official model
        yolo pose val model=path/to/best.pt data=path/to/data.yaml # val custom model
        ```

## Predict

Use a trained YOLO26n-pose model to run predictions on images.

!!! example

    === "Python"

        ```python
        from ultralytics import YOLO

        # Load a model
        model = YOLO("yolo26n-pose.pt")  # load an official model
        model = YOLO("path/to/best.pt")  # load a custom model

        # Predict with the model
        results = model("https://ultralytics.com/images/bus.jpg")  # predict on an image

        # Access the results
        for result in results:
            xy = result.keypoints.xy  # x and y coordinates
            xyn = result.keypoints.xyn  # normalized
            kpts = result.keypoints.data  # x, y, visibility (if available)
        ```

    === "CLI"

        ```bash
        yolo pose predict model=yolo26n-pose.pt source='https://ultralytics.com/images/bus.jpg' # predict with official model
        yolo pose predict model=path/to/best.pt source='https://ultralytics.com/images/bus.jpg' # predict with custom model
        ```

See full `predict` mode details in the [Predict](../modes/predict.md) page.

### Results Output

Pose estimation returns one `Results` object per image. The primary prediction fields are `result.keypoints` for pose
coordinates and `result.boxes` for the detected instances that those keypoints belong to.

| Attribute               | Type            | Shape       | Description                                |
| ----------------------- | --------------- | ----------- | ------------------------------------------ |
| `result.keypoints`      | `Keypoints`     | `(N)`       | Keypoints.                                 |
| `result.keypoints.data` | `torch.float32` | `(N,K,2/3)` | `x,y` plus optional visibility/confidence. |
| `result.keypoints.xy`   | `torch.float32` | `(N,K,2)`   | Pixel keypoints.                           |
| `result.keypoints.xyn`  | `torch.float32` | `(N,K,2)`   | Normalized keypoints.                      |
| `result.boxes`          | `Boxes`         | `(N)`       | Instance boxes.                            |

For task-specific `Results` fields across every task, see the [Predict Results by Task](../modes/predict.md#results-by-task) section.

### Keypoint index map

The default YOLO26 Pose models predict the 17 COCO keypoints, so index `k` in `result.keypoints.xy[:, k]` is:

0. Nose
1. Left Eye
2. Right Eye
3. Left Ear
4. Right Ear
5. Left Shoulder
6. Right Shoulder
7. Left Elbow
8. Right Elbow
9. Left Wrist
10. Right Wrist
11. Left Hip
12. Right Hip
13. Left Knee
14. Right Knee
15. Left Ankle
16. Right Ankle

## Export

Export a YOLO26n-pose model to a different format like ONNX, CoreML, etc.

!!! example

    === "Python"

        ```python
        from ultralytics import YOLO

        # Load a model
        model = YOLO("yolo26n-pose.pt")  # load an official model
        model = YOLO("path/to/best.pt")  # load a custom model

        # Export the model
        model.export(format="onnx")
        ```

    === "CLI"

        ```bash
        yolo export model=yolo26n-pose.pt format=onnx # export official model
        yolo export model=path/to/best.pt format=onnx # export custom model
        ```

Available YOLO26-pose export formats are in the table below. You can export to any format using the `format` argument, i.e., `format='onnx'` or `format='engine'`. You can predict or validate directly on exported models, i.e., `yolo predict model=yolo26n-pose.onnx`. Usage examples are shown for your model after export completes.

{% include "macros/export-table.md" %}

See full `export` details in the [Export](../modes/export.md) page.

## FAQ

### What is Pose Estimation with Ultralytics YOLO26 and how does it work?

Pose estimation with Ultralytics YOLO26 involves identifying specific points, known as keypoints, in an image. These keypoints typically represent joints or other important features of the object. The output includes the `[x, y]` coordinates and confidence scores for each point. YOLO26-pose models are specifically designed for this task and use the `-pose` suffix, such as `yolo26n-pose.pt`. These models are pretrained on datasets like [COCO keypoints](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/datasets/coco-pose.yaml) and can be used for various pose estimation tasks. For more information, visit the [Pose Estimation Page](#pose-estimation).

### How can I train a YOLO26-pose model on a custom dataset?

Training a YOLO26-pose model on a custom dataset involves loading a model, either a new model defined by a YAML file or a pretrained model. You can then start the training process using your specified dataset and parameters.

!!! example

    === "Python"

        ```python
        from ultralytics import YOLO

        # Load a model
        model = YOLO("yolo26n-pose.yaml")  # build a new model from YAML
        model = YOLO("yolo26n-pose.pt")  # load a pretrained model (recommended for training)

        # Train the model
        results = model.train(data="your-dataset.yaml", epochs=100, imgsz=640)
        ```

    === "CLI"

        ```bash
        yolo pose train data=your-dataset.yaml model=yolo26n-pose.pt epochs=100 imgsz=640
        ```

For comprehensive details on training, refer to the [Train Section](#train). You can also use [Ultralytics Platform cloud training](../platform/train/cloud-training.md) for a no-code approach to training custom pose estimation models.

### How do I validate a trained YOLO26-pose model?

Validation of a YOLO26-pose model involves assessing its accuracy using the same dataset parameters retained during training. Here's an example:

!!! example

    === "Python"

        ```python
        from ultralytics import YOLO

        # Load a model
        model = YOLO("yolo26n-pose.pt")  # load an official model
        model = YOLO("path/to/best.pt")  # load a custom model

        # Validate the model
        metrics = model.val()  # no arguments needed, dataset and settings remembered
        ```

    === "CLI"

        ```bash
        yolo pose val model=yolo26n-pose.pt data=coco8-pose.yaml
        ```

For more information, visit the [Val Section](#val).

### Can I export a YOLO26-pose model to other formats, and how?

Yes, you can export a YOLO26-pose model to various formats like ONNX, CoreML, TensorRT, and more. This can be done using either Python or the Command Line Interface (CLI).

!!! example

    === "Python"

        ```python
        from ultralytics import YOLO

        # Load a model
        model = YOLO("yolo26n-pose.pt")  # load an official model
        model = YOLO("path/to/best.pt")  # load a custom model

        # Export the model
        model.export(format="onnx")
        ```

    === "CLI"

        ```bash
        yolo export model=yolo26n-pose.pt format=onnx
        ```

Refer to the [Export Section](#export) for more details. Exported models can be deployed on edge devices for [real-time applications](https://www.ultralytics.com/blog/real-time-inferences-in-vision-ai-solutions-are-making-an-impact) like fitness tracking, sports analysis, or [robotics](https://www.ultralytics.com/blog/from-algorithms-to-automation-ais-role-in-robotics).

### What are the available Ultralytics YOLO26-pose models and their performance metrics?

Ultralytics YOLO26 offers various pretrained pose models such as YOLO26n-pose, YOLO26s-pose, YOLO26m-pose, among others. These models differ in size, accuracy (mAP), and speed. For the complete list with COCO keypoint mAP, speed, parameters, and FLOPs per model, see the [Models](#models) section.
