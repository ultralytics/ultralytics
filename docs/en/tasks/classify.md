---
comments: true
description: Master image classification using YOLOv8. Learn to train, validate, predict, and export models efficiently.
keywords: YOLOv8, image classification, AI, machine learning, pretrained models, ImageNet, model export, predict, train, validate
model_name: yolov8n-cls
---

# Image Classification

<img width="1024" src="https://github.com/ultralytics/docs/releases/download/0/image-classification-examples.avif" alt="Image classification examples">

Image classification is the simplest of the three tasks and involves classifying an entire image into one of a set of predefined classes.

The output of an image classifier is a single class label and a confidence score. Image classification is useful when you need to know only what class an image belongs to and don't need to know where objects of that class are located or what their exact shape is.

<p align="center">
  <br>
  <iframe loading="lazy" width="720" height="405" src="https://www.youtube.com/embed/5BO0Il_YYAg"
    title="YouTube video player" frameborder="0"
    allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share"
    allowfullscreen>
  </iframe>
  <br>
  <strong>Watch:</strong> Explore Ultralytics YOLO Tasks: Image Classification using Ultralytics HUB
</p>

!!! tip

    YOLOv8 Classify models use the `-cls` suffix, i.e. `yolov8n-cls.pt` and are pretrained on [ImageNet](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/datasets/ImageNet.yaml).

## [Models](https://github.com/ultralytics/ultralytics/tree/main/ultralytics/cfg/models/v8)

YOLOv8 pretrained Classify models are shown here. Detect, Segment and Pose models are pretrained on the [COCO](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/datasets/coco.yaml) dataset, while Classify models are pretrained on the [ImageNet](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/datasets/ImageNet.yaml) dataset.

[Models](https://github.com/ultralytics/ultralytics/tree/main/ultralytics/cfg/models) download automatically from the latest Ultralytics [release](https://github.com/ultralytics/assets/releases) on first use.

| Model                                                                                        | size<br><sup>(pixels) | acc<br><sup>top1 | acc<br><sup>top5 | Speed<br><sup>CPU ONNX<br>(ms) | Speed<br><sup>A100 TensorRT<br>(ms) | params<br><sup>(M) | FLOPs<br><sup>(B) at 640 |
| -------------------------------------------------------------------------------------------- | --------------------- | ---------------- | ---------------- | ------------------------------ | ----------------------------------- | ------------------ | ------------------------ |
| [YOLOv8n-cls](https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov8n-cls.pt) | 224                   | 69.0             | 88.3             | 12.9                           | 0.31                                | 2.7                | 4.3                      |
| [YOLOv8s-cls](https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov8s-cls.pt) | 224                   | 73.8             | 91.7             | 23.4                           | 0.35                                | 6.4                | 13.5                     |
| [YOLOv8m-cls](https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov8m-cls.pt) | 224                   | 76.8             | 93.5             | 85.4                           | 0.62                                | 17.0               | 42.7                     |
| [YOLOv8l-cls](https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov8l-cls.pt) | 224                   | 76.8             | 93.5             | 163.0                          | 0.87                                | 37.5               | 99.7                     |
| [YOLOv8x-cls](https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov8x-cls.pt) | 224                   | 79.0             | 94.6             | 232.0                          | 1.01                                | 57.4               | 154.8                    |

- **acc** values are model accuracies on the [ImageNet](https://www.image-net.org/) dataset validation set. <br>Reproduce by `yolo val classify data=path/to/ImageNet device=0`
- **Speed** averaged over ImageNet val images using an [Amazon EC2 P4d](https://aws.amazon.com/ec2/instance-types/p4/) instance. <br>Reproduce by `yolo val classify data=path/to/ImageNet batch=1 device=0|cpu`

## Train

Train YOLOv8n-cls on the MNIST160 dataset for 100 epochs at image size 64. For a full list of available arguments see the [Configuration](../usage/cfg.md) page.

!!! example

    === "Python"

        ```python
        from ultralytics import YOLO

        # Load a model
        model = YOLO("yolov8n-cls.yaml")  # build a new model from YAML
        model = YOLO("yolov8n-cls.pt")  # load a pretrained model (recommended for training)
        model = YOLO("yolov8n-cls.yaml").load("yolov8n-cls.pt")  # build from YAML and transfer weights

        # Train the model
        results = model.train(data="mnist160", epochs=100, imgsz=64)
        ```

    === "CLI"

        ```bash
        # Build a new model from YAML and start training from scratch
        yolo classify train data=mnist160 model=yolov8n-cls.yaml epochs=100 imgsz=64

        # Start training from a pretrained *.pt model
        yolo classify train data=mnist160 model=yolov8n-cls.pt epochs=100 imgsz=64

        # Build a new model from YAML, transfer pretrained weights to it and start training
        yolo classify train data=mnist160 model=yolov8n-cls.yaml pretrained=yolov8n-cls.pt epochs=100 imgsz=64
        ```

### Dataset format

YOLO classification dataset format can be found in detail in the [Dataset Guide](../datasets/classify/index.md).

## Val

Validate trained YOLOv8n-cls model accuracy on the MNIST160 dataset. No arguments are needed as the `model` retains its training `data` and arguments as model attributes.

!!! example

    === "Python"

        ```python
        from ultralytics import YOLO

        # Load a model
        model = YOLO("yolov8n-cls.pt")  # load an official model
        model = YOLO("path/to/best.pt")  # load a custom model

        # Validate the model
        metrics = model.val()  # no arguments needed, dataset and settings remembered
        metrics.top1  # top1 accuracy
        metrics.top5  # top5 accuracy
        ```

    === "CLI"

        ```bash
        yolo classify val model=yolov8n-cls.pt  # val official model
        yolo classify val model=path/to/best.pt  # val custom model
        ```

## Predict

Use a trained YOLOv8n-cls model to run predictions on images.

!!! example

    === "Python"

        ```python
        from ultralytics import YOLO

        # Load a model
        model = YOLO("yolov8n-cls.pt")  # load an official model
        model = YOLO("path/to/best.pt")  # load a custom model

        # Predict with the model
        results = model("https://ultralytics.com/images/bus.jpg")  # predict on an image
        ```

    === "CLI"

        ```bash
        yolo classify predict model=yolov8n-cls.pt source='https://ultralytics.com/images/bus.jpg'  # predict with official model
        yolo classify predict model=path/to/best.pt source='https://ultralytics.com/images/bus.jpg'  # predict with custom model
        ```

See full `predict` mode details in the [Predict](../modes/predict.md) page.

## Export

Export a YOLOv8n-cls model to a different format like ONNX, CoreML, etc.

!!! example

    === "Python"

        ```python
        from ultralytics import YOLO

        # Load a model
        model = YOLO("yolov8n-cls.pt")  # load an official model
        model = YOLO("path/to/best.pt")  # load a custom trained model

        # Export the model
        model.export(format="onnx")
        ```

    === "CLI"

        ```bash
        yolo export model=yolov8n-cls.pt format=onnx  # export official model
        yolo export model=path/to/best.pt format=onnx  # export custom trained model
        ```

Available YOLOv8-cls export formats are in the table below. You can export to any format using the `format` argument, i.e. `format='onnx'` or `format='engine'`. You can predict or validate directly on exported models, i.e. `yolo predict model=yolov8n-cls.onnx`. Usage examples are shown for your model after export completes.

{% include "macros/export-table.md" %}

See full `export` details in the [Export](../modes/export.md) page.

## FAQ

### What is the purpose of YOLOv8 in image classification?

YOLOv8 models, such as `yolov8n-cls.pt`, are designed for efficient image classification. They assign a single class label to an entire image along with a confidence score. This is particularly useful for applications where knowing the specific class of an image is sufficient, rather than identifying the location or shape of objects within the image.

### How do I train a YOLOv8 model for image classification?

To train a YOLOv8 model, you can use either Python or CLI commands. For example, to train a `yolov8n-cls` model on the MNIST160 dataset for 100 epochs at an image size of 64:

!!! example

    === "Python"

        ```python
        from ultralytics import YOLO

        # Load a model
        model = YOLO("yolov8n-cls.pt")  # load a pretrained model (recommended for training)

        # Train the model
        results = model.train(data="mnist160", epochs=100, imgsz=64)
        ```

    === "CLI"

        ```bash
        yolo classify train data=mnist160 model=yolov8n-cls.pt epochs=100 imgsz=64
        ```

For more configuration options, visit the [Configuration](../usage/cfg.md) page.

### Where can I find pretrained YOLOv8 classification models?

Pretrained YOLOv8 classification models can be found in the [Models](https://github.com/ultralytics/ultralytics/tree/main/ultralytics/cfg/models/v8) section. Models like `yolov8n-cls.pt`, `yolov8s-cls.pt`, `yolov8m-cls.pt`, etc., are pretrained on the [ImageNet](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/datasets/ImageNet.yaml) dataset and can be easily downloaded and used for various image classification tasks.

### How can I export a trained YOLOv8 model to different formats?

You can export a trained YOLOv8 model to various formats using Python or CLI commands. For instance, to export a model to ONNX format:

!!! example

    === "Python"

        ```python
        from ultralytics import YOLO

        # Load a model
        model = YOLO("yolov8n-cls.pt")  # load the trained model

        # Export the model to ONNX
        model.export(format="onnx")
        ```

    === "CLI"

        ```bash
        yolo export model=yolov8n-cls.pt format=onnx  # export the trained model to ONNX format
        ```

For detailed export options, refer to the [Export](../modes/export.md) page.

### How do I validate a trained YOLOv8 classification model?

To validate a trained model's accuracy on a dataset like MNIST160, you can use the following Python or CLI commands:

!!! example

    === "Python"

        ```python
        from ultralytics import YOLO

        # Load a model
        model = YOLO("yolov8n-cls.pt")  # load the trained model

        # Validate the model
        metrics = model.val()  # no arguments needed, uses the dataset and settings from training
        metrics.top1  # top1 accuracy
        metrics.top5  # top5 accuracy
        ```

    === "CLI"

        ```bash
        yolo classify val model=yolov8n-cls.pt  # validate the trained model
        ```

For more information, visit the [Validate](#val) section.
