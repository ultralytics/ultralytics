---
comments: true
description: Master image classification using YOLO11. Learn to train, validate, predict, and export models efficiently.
keywords: YOLO11, image classification, AI, machine learning, pretrained models, ImageNet, model export, predict, train, validate
model_name: yolo11n-cls
---

# Image Classification

<img width="1024" src="https://github.com/ultralytics/docs/releases/download/0/image-classification-examples.avif" alt="Image classification examples">

[Image classification](https://www.ultralytics.com/glossary/image-classification) is the simplest of the three tasks and involves classifying an entire image into one of a set of predefined classes.

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

    YOLO11 Classify models use the `-cls` suffix, i.e. `yolo11n-cls.pt` and are pretrained on [ImageNet](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/datasets/ImageNet.yaml).

## [Models](https://github.com/ultralytics/ultralytics/tree/main/ultralytics/cfg/models/11)

YOLO11 pretrained Classify models are shown here. Detect, Segment and Pose models are pretrained on the [COCO](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/datasets/coco.yaml) dataset, while Classify models are pretrained on the [ImageNet](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/datasets/ImageNet.yaml) dataset.

[Models](https://github.com/ultralytics/ultralytics/tree/main/ultralytics/cfg/models) download automatically from the latest Ultralytics [release](https://github.com/ultralytics/assets/releases) on first use.

{% include "macros/yolo-cls-perf.md" %}

- **acc** values are model accuracies on the [ImageNet](https://www.image-net.org/) dataset validation set. <br>Reproduce by `yolo val classify data=path/to/ImageNet device=0`
- **Speed** averaged over ImageNet val images using an [Amazon EC2 P4d](https://aws.amazon.com/ec2/instance-types/p4/) instance. <br>Reproduce by `yolo val classify data=path/to/ImageNet batch=1 device=0|cpu`

## Train

Train YOLO11n-cls on the MNIST160 dataset for 100 [epochs](https://www.ultralytics.com/glossary/epoch) at image size 64. For a full list of available arguments see the [Configuration](../usage/cfg.md) page.

!!! example

    === "Python"

        ```python
        from ultralytics import YOLO

        # Load a model
        model = YOLO("yolo11n-cls.yaml")  # build a new model from YAML
        model = YOLO("yolo11n-cls.pt")  # load a pretrained model (recommended for training)
        model = YOLO("yolo11n-cls.yaml").load("yolo11n-cls.pt")  # build from YAML and transfer weights

        # Train the model
        results = model.train(data="mnist160", epochs=100, imgsz=64)
        ```

    === "CLI"

        ```bash
        # Build a new model from YAML and start training from scratch
        yolo classify train data=mnist160 model=yolo11n-cls.yaml epochs=100 imgsz=64

        # Start training from a pretrained *.pt model
        yolo classify train data=mnist160 model=yolo11n-cls.pt epochs=100 imgsz=64

        # Build a new model from YAML, transfer pretrained weights to it and start training
        yolo classify train data=mnist160 model=yolo11n-cls.yaml pretrained=yolo11n-cls.pt epochs=100 imgsz=64
        ```

### Dataset format

YOLO classification dataset format can be found in detail in the [Dataset Guide](../datasets/classify/index.md).

## Val

Validate trained YOLO11n-cls model [accuracy](https://www.ultralytics.com/glossary/accuracy) on the MNIST160 dataset. No arguments are needed as the `model` retains its training `data` and arguments as model attributes.

!!! example

    === "Python"

        ```python
        from ultralytics import YOLO

        # Load a model
        model = YOLO("yolo11n-cls.pt")  # load an official model
        model = YOLO("path/to/best.pt")  # load a custom model

        # Validate the model
        metrics = model.val()  # no arguments needed, dataset and settings remembered
        metrics.top1  # top1 accuracy
        metrics.top5  # top5 accuracy
        ```

    === "CLI"

        ```bash
        yolo classify val model=yolo11n-cls.pt  # val official model
        yolo classify val model=path/to/best.pt # val custom model
        ```

## Predict

Use a trained YOLO11n-cls model to run predictions on images.

!!! example

    === "Python"

        ```python
        from ultralytics import YOLO

        # Load a model
        model = YOLO("yolo11n-cls.pt")  # load an official model
        model = YOLO("path/to/best.pt")  # load a custom model

        # Predict with the model
        results = model("https://ultralytics.com/images/bus.jpg")  # predict on an image
        ```

    === "CLI"

        ```bash
        yolo classify predict model=yolo11n-cls.pt source='https://ultralytics.com/images/bus.jpg'  # predict with official model
        yolo classify predict model=path/to/best.pt source='https://ultralytics.com/images/bus.jpg' # predict with custom model
        ```

See full `predict` mode details in the [Predict](../modes/predict.md) page.

## Export

Export a YOLO11n-cls model to a different format like ONNX, CoreML, etc.

!!! example

    === "Python"

        ```python
        from ultralytics import YOLO

        # Load a model
        model = YOLO("yolo11n-cls.pt")  # load an official model
        model = YOLO("path/to/best.pt")  # load a custom trained model

        # Export the model
        model.export(format="onnx")
        ```

    === "CLI"

        ```bash
        yolo export model=yolo11n-cls.pt format=onnx  # export official model
        yolo export model=path/to/best.pt format=onnx # export custom trained model
        ```

Available YOLO11-cls export formats are in the table below. You can export to any format using the `format` argument, i.e. `format='onnx'` or `format='engine'`. You can predict or validate directly on exported models, i.e. `yolo predict model=yolo11n-cls.onnx`. Usage examples are shown for your model after export completes.

{% include "macros/export-table.md" %}

See full `export` details in the [Export](../modes/export.md) page.

## FAQ

### What is the purpose of YOLO11 in image classification?

YOLO11 models, such as `yolo11n-cls.pt`, are designed for efficient image classification. They assign a single class label to an entire image along with a confidence score. This is particularly useful for applications where knowing the specific class of an image is sufficient, rather than identifying the location or shape of objects within the image.

### How do I train a YOLO11 model for image classification?

To train a YOLO11 model, you can use either Python or CLI commands. For example, to train a `yolo11n-cls` model on the MNIST160 dataset for 100 epochs at an image size of 64:

!!! example

    === "Python"

        ```python
        from ultralytics import YOLO

        # Load a model
        model = YOLO("yolo11n-cls.pt")  # load a pretrained model (recommended for training)

        # Train the model
        results = model.train(data="mnist160", epochs=100, imgsz=64)
        ```

    === "CLI"

        ```bash
        yolo classify train data=mnist160 model=yolo11n-cls.pt epochs=100 imgsz=64
        ```

For more configuration options, visit the [Configuration](../usage/cfg.md) page.

### Where can I find pretrained YOLO11 classification models?

Pretrained YOLO11 classification models can be found in the [Models](https://github.com/ultralytics/ultralytics/tree/main/ultralytics/cfg/models/11) section. Models like `yolo11n-cls.pt`, `yolo11s-cls.pt`, `yolo11m-cls.pt`, etc., are pretrained on the [ImageNet](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/datasets/ImageNet.yaml) dataset and can be easily downloaded and used for various image classification tasks.

### How can I export a trained YOLO11 model to different formats?

You can export a trained YOLO11 model to various formats using Python or CLI commands. For instance, to export a model to ONNX format:

!!! example

    === "Python"

        ```python
        from ultralytics import YOLO

        # Load a model
        model = YOLO("yolo11n-cls.pt")  # load the trained model

        # Export the model to ONNX
        model.export(format="onnx")
        ```

    === "CLI"

        ```bash
        yolo export model=yolo11n-cls.pt format=onnx # export the trained model to ONNX format
        ```

For detailed export options, refer to the [Export](../modes/export.md) page.

### How do I validate a trained YOLO11 classification model?

To validate a trained model's accuracy on a dataset like MNIST160, you can use the following Python or CLI commands:

!!! example

    === "Python"

        ```python
        from ultralytics import YOLO

        # Load a model
        model = YOLO("yolo11n-cls.pt")  # load the trained model

        # Validate the model
        metrics = model.val()  # no arguments needed, uses the dataset and settings from training
        metrics.top1  # top1 accuracy
        metrics.top5  # top5 accuracy
        ```

    === "CLI"

        ```bash
        yolo classify val model=yolo11n-cls.pt # validate the trained model
        ```

For more information, visit the [Validate](#val) section.
