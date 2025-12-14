---
comments: true
description: Discover how to use YOLO11 for pose estimation tasks. Learn about model training, validation, prediction, and exporting in various formats.
keywords: pose estimation, YOLO11, Ultralytics, keypoints, model training, image recognition, deep learning, human pose detection, computer vision, real-time tracking
model_name: yolo11n-pose
---

# Pose Estimation

<img width="1024" src="https://github.com/ultralytics/docs/releases/download/0/pose-estimation-examples.avif" alt="Pose estimation examples">

Pose estimation is a task that involves identifying the location of specific points in an image, usually referred to as keypoints. The keypoints can represent various parts of the object such as joints, landmarks, or other distinctive features. The locations of the keypoints are usually represented as a set of 2D `[x, y]` or 3D `[x, y, visible]` coordinates.

The output of a pose estimation model is a set of points that represent the keypoints on an object in the image, usually along with the confidence scores for each point. Pose estimation is a good choice when you need to identify specific parts of an object in a scene, and their location in relation to each other.

<p align="center">
  <br>
  <iframe loading="lazy" width="720" height="405" src="https://www.youtube.com/embed/AAkfToU3nAc"
    title="YouTube video player" frameborder="0"
    allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share"
    allowfullscreen>
  </iframe>
  <br>
  <strong>Watch:</strong> Ultralytics YOLO11 Pose Estimation Tutorial | Real-Time Object Tracking and Human Pose Detection
</p>

!!! tip

    YOLO11 _pose_ models use the `-pose` suffix, i.e., `yolo11n-pose.pt`. These models are trained on the [COCO keypoints](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/datasets/coco-pose.yaml) dataset and are suitable for a variety of pose estimation tasks.

    In the default YOLO11 pose model, there are 17 keypoints, each representing a different part of the human body. Here is the mapping of each index to its respective body joint:

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

## [Models](https://github.com/ultralytics/ultralytics/tree/main/ultralytics/cfg/models/11)

Ultralytics YOLO11 pretrained Pose models are shown here. Detect, Segment and Pose models are pretrained on the [COCO](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/datasets/coco.yaml) dataset, while Classify models are pretrained on the [ImageNet](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/datasets/ImageNet.yaml) dataset.

[Models](https://github.com/ultralytics/ultralytics/tree/main/ultralytics/cfg/models) download automatically from the latest Ultralytics [release](https://github.com/ultralytics/assets/releases) on first use.

{% include "macros/yolo-pose-perf.md" %}

- **mAP<sup>val</sup>** values are for single-model single-scale on [COCO Keypoints val2017](https://cocodataset.org/) dataset. <br>Reproduce by `yolo val pose data=coco-pose.yaml device=0`
- **Speed** averaged over COCO val images using an [Amazon EC2 P4d](https://aws.amazon.com/ec2/instance-types/p4/) instance. <br>Reproduce by `yolo val pose data=coco-pose.yaml batch=1 device=0|cpu`

## Train

Train a YOLO11-pose model on the COCO8-pose dataset. The [COCO8-pose dataset](https://docs.ultralytics.com/datasets/pose/coco8-pose/) is a small sample dataset that's perfect for testing and debugging your pose estimation models.

!!! example

    === "Python"

        ```python
        from ultralytics import YOLO

        # Load a model
        model = YOLO("yolo11n-pose.yaml")  # build a new model from YAML
        model = YOLO("yolo11n-pose.pt")  # load a pretrained model (recommended for training)
        model = YOLO("yolo11n-pose.yaml").load("yolo11n-pose.pt")  # build from YAML and transfer weights

        # Train the model
        results = model.train(data="coco8-pose.yaml", epochs=100, imgsz=640)
        ```

    === "CLI"

        ```bash
        # Build a new model from YAML and start training from scratch
        yolo pose train data=coco8-pose.yaml model=yolo11n-pose.yaml epochs=100 imgsz=640

        # Start training from a pretrained *.pt model
        yolo pose train data=coco8-pose.yaml model=yolo11n-pose.pt epochs=100 imgsz=640

        # Build a new model from YAML, transfer pretrained weights to it and start training
        yolo pose train data=coco8-pose.yaml model=yolo11n-pose.yaml pretrained=yolo11n-pose.pt epochs=100 imgsz=640
        ```

### Dataset format

YOLO pose dataset format can be found in detail in the [Dataset Guide](../datasets/pose/index.md). To convert your existing dataset from other formats (like [COCO](https://docs.ultralytics.com/datasets/pose/coco/) etc.) to YOLO format, please use the [JSON2YOLO](https://github.com/ultralytics/JSON2YOLO) tool by Ultralytics.

For custom pose estimation tasks, you can also explore specialized datasets like [Tiger-Pose](https://docs.ultralytics.com/datasets/pose/tiger-pose/) for animal pose estimation, [Hand Keypoints](https://docs.ultralytics.com/datasets/pose/hand-keypoints/) for hand tracking, or [Dog-Pose](https://docs.ultralytics.com/datasets/pose/dog-pose/) for canine pose analysis.

## Val

Validate trained YOLO11n-pose model [accuracy](https://www.ultralytics.com/glossary/accuracy) on the COCO8-pose dataset. No arguments are needed as the `model` retains its training `data` and arguments as model attributes.

!!! example

    === "Python"

        ```python
        from ultralytics import YOLO

        # Load a model
        model = YOLO("yolo11n-pose.pt")  # load an official model
        model = YOLO("path/to/best.pt")  # load a custom model

        # Validate the model
        metrics = model.val()  # no arguments needed, dataset and settings remembered
        metrics.box.map  # map50-95
        metrics.box.map50  # map50
        metrics.box.map75  # map75
        metrics.box.maps  # a list containing mAP50-95 for each category
        metrics.pose.map  # map50-95(P)
        metrics.pose.map50  # map50(P)
        metrics.pose.map75  # map75(P)
        metrics.pose.maps  # a list containing mAP50-95(P) for each category
        ```

    === "CLI"

        ```bash
        yolo pose val model=yolo11n-pose.pt # val official model
        yolo pose val model=path/to/best.pt # val custom model
        ```

## Predict

Use a trained YOLO11n-pose model to run predictions on images. The [predict mode](https://docs.ultralytics.com/modes/predict/) allows you to perform inference on images, videos, or real-time streams.

!!! example

    === "Python"

        ```python
        from ultralytics import YOLO

        # Load a model
        model = YOLO("yolo11n-pose.pt")  # load an official model
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
        yolo pose predict model=yolo11n-pose.pt source='https://ultralytics.com/images/bus.jpg' # predict with official model
        yolo pose predict model=path/to/best.pt source='https://ultralytics.com/images/bus.jpg' # predict with custom model
        ```

See full `predict` mode details in the [Predict](../modes/predict.md) page.

## Export

Export a YOLO11n Pose model to a different format like ONNX, CoreML, etc. This allows you to deploy your model on various platforms and devices for [real-time inference](https://www.ultralytics.com/glossary/real-time-inference).

!!! example

    === "Python"

        ```python
        from ultralytics import YOLO

        # Load a model
        model = YOLO("yolo11n-pose.pt")  # load an official model
        model = YOLO("path/to/best.pt")  # load a custom-trained model

        # Export the model
        model.export(format="onnx")
        ```

    === "CLI"

        ```bash
        yolo export model=yolo11n-pose.pt format=onnx # export official model
        yolo export model=path/to/best.pt format=onnx # export custom-trained model
        ```

Available YOLO11-pose export formats are in the table below. You can export to any format using the `format` argument, i.e., `format='onnx'` or `format='engine'`. You can predict or validate directly on exported models, i.e., `yolo predict model=yolo11n-pose.onnx`. Usage examples are shown for your model after export completes.

{% include "macros/export-table.md" %}

See full `export` details in the [Export](../modes/export.md) page.

## FAQ

### What is Pose Estimation with Ultralytics YOLO11 and how does it work?

Pose estimation with Ultralytics YOLO11 involves identifying specific points, known as keypoints, in an image. These keypoints typically represent joints or other important features of the object. The output includes the `[x, y]` coordinates and confidence scores for each point. YOLO11-pose models are specifically designed for this task and use the `-pose` suffix, such as `yolo11n-pose.pt`. These models are pretrained on datasets like [COCO keypoints](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/datasets/coco-pose.yaml) and can be used for various pose estimation tasks. For more information, visit the [Pose Estimation Page](#pose-estimation).

### How can I train a YOLO11-pose model on a custom dataset?

Training a YOLO11-pose model on a custom dataset involves loading a model, either a new model defined by a YAML file or a pretrained model. You can then start the training process using your specified dataset and parameters.

```python
from ultralytics import YOLO

# Load a model
model = YOLO("yolo11n-pose.yaml")  # build a new model from YAML
model = YOLO("yolo11n-pose.pt")  # load a pretrained model (recommended for training)

# Train the model
results = model.train(data="your-dataset.yaml", epochs=100, imgsz=640)
```

For comprehensive details on training, refer to the [Train Section](#train). You can also use [Ultralytics HUB](https://www.ultralytics.com/hub) for a no-code approach to training custom pose estimation models.

### How do I validate a trained YOLO11-pose model?

Validation of a YOLO11-pose model involves assessing its accuracy using the same dataset parameters retained during training. Here's an example:

```python
from ultralytics import YOLO

# Load a model
model = YOLO("yolo11n-pose.pt")  # load an official model
model = YOLO("path/to/best.pt")  # load a custom model

# Validate the model
metrics = model.val()  # no arguments needed, dataset and settings remembered
```

For more information, visit the [Val Section](#val).

### Can I export a YOLO11-pose model to other formats, and how?

Yes, you can export a YOLO11-pose model to various formats like ONNX, CoreML, TensorRT, and more. This can be done using either Python or the Command Line Interface (CLI).

```python
from ultralytics import YOLO

# Load a model
model = YOLO("yolo11n-pose.pt")  # load an official model
model = YOLO("path/to/best.pt")  # load a custom-trained model

# Export the model
model.export(format="onnx")
```

Refer to the [Export Section](#export) for more details. Exported models can be deployed on edge devices for [real-time applications](https://www.ultralytics.com/blog/real-time-inferences-in-vision-ai-solutions-are-making-an-impact) like fitness tracking, sports analysis, or [robotics](https://www.ultralytics.com/blog/from-algorithms-to-automation-ais-role-in-robotics).

### What are the available Ultralytics YOLO11-pose models and their performance metrics?

Ultralytics YOLO11 offers various pretrained pose models such as YOLO11n-pose, YOLO11s-pose, YOLO11m-pose, among others. These models differ in size, accuracy (mAP), and speed. For instance, the YOLO11n-pose model achieves a mAP<sup>pose</sup>50-95 of 50.0 and an mAP<sup>pose</sup>50 of 81.0. For a complete list and performance details, visit the [Models Section](#models).
