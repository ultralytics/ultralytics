---
comments: true
description: Master instance segmentation using YOLO11. Learn how to detect, segment and outline objects in images with detailed guides and examples.
keywords: instance segmentation, YOLO11, object detection, image segmentation, machine learning, deep learning, computer vision, COCO dataset, Ultralytics
model_name: yolo11n-seg
---

# Semantic Segmentation

<img width="1024" src="https://github.com/kuazhangxiaoai/ultralytics-semantic-segment/releases/download/docs/mosaic.png" alt="Semantic segmentation examples">

[Semantic segmentation](https://www.ultralytics.com/glossary/semantic-segmentation) is a fundamental task in the field of computer vision that aims to classify every pixel in an image, thereby achieving a precise partition of different semantic categories within a scene.

Unlike object detection, which only identifies the locations and categories of objects, semantic segmentation provides a finer-grained understanding of visual content — enabling models to know not only what is present, but also where it is.

<p align="center">
  <br>
  <iframe loading="lazy" width="720" height="405" src="https://www.youtube.com/embed/o4Zd-IeMlSY?si=37nusCzDTd74Obsp"
    title="YouTube video player" frameborder="0"
    allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share"
    allowfullscreen>
  </iframe>
  <br>
  <strong>Watch:</strong> Run Segmentation with Pre-Trained Ultralytics YOLO Model in Python.
</p>

!!! tip

    YOLO11 semseg models use the `-semseg` suffix, i.e. `yolo11n-semseg.pt` and are pretrained on [Cityscapes](https://github.com/kuazhangxiaoai/ultralytics-semantic-segment/blob/semseg/ultralytics/cfg/datasets/CityscapesYOLO.yaml).

## [Models](https://github.com/ultralytics/ultralytics/tree/main/ultralytics/cfg/models/11)

YOLO11 pretrained Semantic Segment models are shown here. Detect, Segment and Pose models are pretrained on the [Cityscapes](https://github.com/kuazhangxiaoai/ultralytics-semantic-segment/blob/semseg/ultralytics/cfg/datasets/CityscapesYOLO.yaml) dataset, while Classify models are pretrained on the [ImageNet](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/datasets/ImageNet.yaml) dataset
, And the pretrain Semantic segment model is trained on [Cityscapse](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/datasets/CityscapesYOLO.yaml) dataset.

[Models](https://github.com/ultralytics/ultralytics/tree/main/ultralytics/cfg/models) download automatically from the latest Ultralytics [release](https://github.com/ultralytics/assets/releases) on first use.

{% include "macros/yolo-seg-perf.md" %}

- **IoU<sup>val</sup>** values are for single-model single-scale on [Cityscapes](https://www.cityscapes-dataset.com/) dataset. <br>Reproduce by `yolo val semseg data=CityscapesYOLO.yaml device=0`
- **Speed** averaged over COCO val images using an NVIDIA RTX 4090. <br>Reproduce by `yolo val semseg data=CityscapesYOLO.yaml batch=1 device=0|cpu`

## Train

Train YOLO11n-seg on the COCO8-seg dataset for 100 [epochs](https://www.ultralytics.com/glossary/epoch) at image size 640. For a full list of available arguments see the [Configuration](../usage/cfg.md) page.

!!! example

    === "Python"

        ```python
        from ultralytics import YOLO

        # Load a model
        model = YOLO("yolo11n-semseg.yaml")  # build a new model from YAML
        model = YOLO("yolo11n-semseg.pt")  # load a pretrained model (recommended for training)
        model = YOLO("yolo11n-semseg.yaml").load("yolo11n.pt")  # build from YAML and transfer weights

        # Train the model
        results = model.train(data="CityscapesYOLO.yaml", epochs=100, imgsz=640)
        ```

    === "CLI"

        ```bash
        # Build a new model from YAML and start training from scratch
        yolo semseg train data=CityscapesYOLO.yaml model=yolo11n-semseg.yaml epochs=100 imgsz=640

        # Start training from a pretrained *.pt model
        yolo semseg train data=CityscapesYOLO.yaml model=yolo11n-semseg.pt epochs=100 imgsz=640

        # Build a new model from YAML, transfer pretrained weights to it and start training
        yolo semseg train data=CityscapesYOLO.yaml model=yolo11n-semseg.yaml pretrained=yolo11n-seg.pt epochs=100 imgsz=640
        ```

### Dataset format

YOLO semseg dataset format can be found in detail in the [Dataset Guide](../datasets/semseg/cityscapes.md). To convert your existing dataset from other formats (like Cityscapes etc.) to YOLO format, please use [JSON2YOLO](https://github.com/ultralytics/JSON2YOLO) tool by Ultralytics.

## Val

Validate trained YOLO11n-seg model [accuracy](https://www.ultralytics.com/glossary/accuracy) on the COCO8-seg dataset. No arguments are needed as the `model` retains its training `data` and arguments as model attributes.

!!! example

    === "Python"

        ```python
        from ultralytics import YOLO

        # Load a model
        model = YOLO("yolo11n-semseg.pt")  # load an official model
        model = YOLO("path/to/best.pt")  # load a custom model

        # Validate the model
        metrics = model.val()  # no arguments needed, dataset and settings remembered

        metrics.semseg.precision  # precision
        metrics.semseg.recall  # recall
        metrics.semseg.IoU  # IoU
        metrics.semseg.DiceScore  # DiceScore
        metrics.semseg.MCR  # MCR
        ```

    === "CLI"

        ```bash
        yolo semseg val model=yolo11n-semseg.pt # val official model
        yolo semseg val model=path/to/best.pt   # val custom model
        ```

## Predict

Use a trained YOLO11n-semseg model to run predictions on images.

!!! example

    === "Python"

        ```python
        from ultralytics import YOLO

        # Load a model
        model = YOLO("yolo11n-semseg.pt")  # load an official model
        model = YOLO("path/to/best.pt")  # load a custom model

        # Predict with the model
        results = model("/path/to/image.png")  # predict on an image

        # Access the results
        for result in results:
            masks = result.masks.data  # mask in matrix format (num_objects x H x W)
        ```

    === "CLI"

        ```bash
        yolo semseg predict model=yolo11n-seg.pt source='/path/to/image.jpg'  # predict with official model
        yolo semseg predict model=path/to/best.pt source='/path/to/image.jpg' # predict with custom model
        ```

See full `predict` mode details in the [Predict](../modes/predict.md) page.

## Export

Export a YOLO11n-seg model to a different format like ONNX, CoreML, etc.

!!! example

    === "Python"

        ```python
        from ultralytics import YOLO

        # Load a model
        model = YOLO("yolo11n-semseg.pt")  # load an official model
        model = YOLO("path/to/best.pt")  # load a custom trained model

        # Export the model
        model.export(format="onnx")
        ```

    === "CLI"

        ```bash
        yolo export model=yolo11n-semseg.pt format=onnx # export official model
        yolo export model=path/to/best.pt format=onnx   # export custom trained model
        ```

Available YOLO11-seg export formats are in the table below. You can export to any format using the `format` argument, i.e. `format='onnx'` or `format='engine'`. You can predict or validate directly on exported models, i.e. `yolo predict model=yolo11n-seg.onnx`. Usage examples are shown for your model after export completes.

{% include "macros/export-table.md" %}

See full `export` details in the [Export](../modes/export.md) page.

## FAQ

### How do I train a YOLO11 semantic segmentation model on a custom dataset?

To train a YOLO11 semantic segmentation model on a custom dataset, you first need to prepare your dataset in the YOLO semseg format. You can use tools to convert datasets from other formats. Once your dataset is ready, you can train the model using Python or CLI commands:

!!! example

    === "Python"

        ```python
        from ultralytics import YOLO

        # Load a pretrained YOLO11 segment model
        model = YOLO("yolo11n-semseg.pt")

        # Train the model
        results = model.train(data="path/to/your_dataset.yaml", epochs=100, imgsz=512)
        ```

    === "CLI"

        ```bash
        yolo semseg train data=path/to/your_dataset.yaml model=yolo11n-seg.pt epochs=100 imgsz=512
        ```

Check the [Configuration](../usage/cfg.md) page for more available arguments.

### What is the difference between semantic segmentation and instance segmentation in YOLO11?

Instance segmentation identify the bounding boxes, countours, and categories of objects, whereas semantic segmentation is designed to predict the category of each pixel in the whole image.
For segmentation of some specific category just like sky, vegetation, etc, instance segmentation is not good at detecting its bounding box and coutour. Semantic segmentation algorithm
can easily segment them from the input image.Besides, the semantic segmentation alogrithm usually has fewer costs than instance segmentation.

### Why use YOLO11 for semantic segmentation?

Ultralytics YOLO11 is a state-of-the-art model recognized for its high accuracy and real-time performance, making it ideal for semantic segmentation tasks. YOLO11 Semantic Segment models come pretrained on the [Cityscape dataset](https://github.com/kuazhangxiaoai/ultralytics-semantic-segment/blob/semseg/ultralytics/cfg/datasets/CityscapesYOLO.yaml), ensuring robust performance across a variety of objects. Additionally, YOLO supports training, validation, prediction, and export functionalities with seamless integration, making it highly versatile for both research and industry applications.

### How do I load and validate a pretrained YOLO semantic segmentation model?

Loading and validating a pretrained YOLO semantic segmentation model is straightforward. Here's how you can do it using both Python and CLI:

!!! example

    === "Python"

        ```python
        from ultralytics import YOLO

        # Load a pretrained model
        model = YOLO("yolo11n-semseg.pt")

        # Validate the model
        metrics = model.val()
        print("IoU for masks:", metrics.semseg.IoU)
        ```

    === "CLI"

        ```bash
        yolo semseg val model=yolo11n-semseg.pt
        ```

### How can I export a YOLO semantic segmentation model to ONNX format?

Exporting a YOLO semantic segmentation model to ONNX format is simple and can be done using Python or CLI commands:

!!! example

    === "Python"

        ```python
        from ultralytics import YOLO

        # Load a pretrained model
        model = YOLO("yolo11n-semseg.pt")

        # Export the model to ONNX format
        model.export(format="onnx")
        ```

    === "CLI"

        ```bash
        yolo export model=yolo11n-semseg.pt format=onnx
        ```

For more details on exporting to various formats, refer to the [Export](../modes/export.md) page.
