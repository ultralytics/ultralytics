---
comments: true
description: Discover how to detect objects with rotation for higher precision using YOLO11 OBB models. Learn, train, validate, and export OBB models effortlessly.
keywords: Oriented Bounding Boxes, OBB, Object Detection, YOLO11, Ultralytics, DOTAv1, Model Training, Model Export, AI, Machine Learning
model_name: yolo11n-obb
---

# Oriented Bounding Boxes [Object Detection](https://www.ultralytics.com/glossary/object-detection)

<!-- obb task poster -->

Oriented object detection goes a step further than object detection and introduce an extra angle to locate objects more accurate in an image.

The output of an oriented object detector is a set of rotated bounding boxes that exactly enclose the objects in the image, along with class labels and confidence scores for each box. Object detection is a good choice when you need to identify objects of interest in a scene, but don't need to know exactly where the object is or its exact shape.

<!-- youtube video link for obb task -->

!!! tip

    YOLO11 OBB models use the `-obb` suffix, i.e. `yolo11n-obb.pt` and are pretrained on [DOTAv1](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/datasets/DOTAv1.yaml).

<p align="center">
  <br>
  <iframe loading="lazy" width="720" height="405" src="https://www.youtube.com/embed/Z7Z9pHF8wJc"
    title="YouTube video player" frameborder="0"
    allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share"
    allowfullscreen>
  </iframe>
  <br>
  <strong>Watch:</strong> Object Detection using Ultralytics YOLO Oriented Bounding Boxes (YOLO-OBB)
</p>

## Visual Samples

|                                              Ships Detection using OBB                                               |                                               Vehicle Detection using OBB                                                |
| :------------------------------------------------------------------------------------------------------------------: | :----------------------------------------------------------------------------------------------------------------------: |
| ![Ships Detection using OBB](https://github.com/ultralytics/docs/releases/download/0/ships-detection-using-obb.avif) | ![Vehicle Detection using OBB](https://github.com/ultralytics/docs/releases/download/0/vehicle-detection-using-obb.avif) |

## [Models](https://github.com/ultralytics/ultralytics/tree/main/ultralytics/cfg/models/11)

YOLO11 pretrained OBB models are shown here, which are pretrained on the [DOTAv1](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/datasets/DOTAv1.yaml) dataset.

[Models](https://github.com/ultralytics/ultralytics/tree/main/ultralytics/cfg/models) download automatically from the latest Ultralytics [release](https://github.com/ultralytics/assets/releases) on first use.

| Model                                                                                        | size<br><sup>(pixels) | mAP<sup>test<br>50 | Speed<br><sup>CPU ONNX<br>(ms) | Speed<br><sup>T4 TensorRT10<br>(ms) | params<br><sup>(M) | FLOPs<br><sup>(B) |
| -------------------------------------------------------------------------------------------- | --------------------- | ------------------ | ------------------------------ | ----------------------------------- | ------------------ | ----------------- |
| [YOLO11n-obb](https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11n-obb.pt) | 1024                  | 78.4               | 117.6 ± 0.8                    | 4.4 ± 0.0                           | 2.7                | 17.2              |
| [YOLO11s-obb](https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11s-obb.pt) | 1024                  | 79.5               | 219.4 ± 4.0                    | 5.1 ± 0.0                           | 9.7                | 57.5              |
| [YOLO11m-obb](https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11m-obb.pt) | 1024                  | 80.9               | 562.8 ± 2.9                    | 10.1 ± 0.4                          | 20.9               | 183.5             |
| [YOLO11l-obb](https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11l-obb.pt) | 1024                  | 81.0               | 712.5 ± 5.0                    | 13.5 ± 0.6                          | 26.2               | 232.0             |
| [YOLO11x-obb](https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11x-obb.pt) | 1024                  | 81.3               | 1408.6 ± 7.7                   | 28.6 ± 1.0                          | 58.8               | 520.2             |

- **mAP<sup>test</sup>** values are for single-model multiscale on [DOTAv1](https://captain-whu.github.io/DOTA/index.html) dataset. <br>Reproduce by `yolo val obb data=DOTAv1.yaml device=0 split=test` and submit merged results to [DOTA evaluation](https://captain-whu.github.io/DOTA/evaluation.html).
- **Speed** averaged over DOTAv1 val images using an [Amazon EC2 P4d](https://aws.amazon.com/ec2/instance-types/p4/) instance. <br>Reproduce by `yolo val obb data=DOTAv1.yaml batch=1 device=0|cpu`

## Train

Train YOLO11n-obb on the `dota8.yaml` dataset for 100 [epochs](https://www.ultralytics.com/glossary/epoch) at image size 640. For a full list of available arguments see the [Configuration](../usage/cfg.md) page.

!!! example

    === "Python"

        ```python
        from ultralytics import YOLO

        # Load a model
        model = YOLO("yolo11n-obb.yaml")  # build a new model from YAML
        model = YOLO("yolo11n-obb.pt")  # load a pretrained model (recommended for training)
        model = YOLO("yolo11n-obb.yaml").load("yolo11n.pt")  # build from YAML and transfer weights

        # Train the model
        results = model.train(data="dota8.yaml", epochs=100, imgsz=640)
        ```

    === "CLI"

        ```bash
        # Build a new model from YAML and start training from scratch
        yolo obb train data=dota8.yaml model=yolo11n-obb.yaml epochs=100 imgsz=640

        # Start training from a pretrained *.pt model
        yolo obb train data=dota8.yaml model=yolo11n-obb.pt epochs=100 imgsz=640

        # Build a new model from YAML, transfer pretrained weights to it and start training
        yolo obb train data=dota8.yaml model=yolo11n-obb.yaml pretrained=yolo11n-obb.pt epochs=100 imgsz=640
        ```

<p align="center">
  <br>
  <iframe loading="lazy" width="720" height="405" src="https://www.youtube.com/embed/uZ7SymQfqKI"
    title="YouTube video player" frameborder="0"
    allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share"
    allowfullscreen>
  </iframe>
  <br>
  <strong>Watch:</strong> How to Train Ultralytics YOLO-OBB (Oriented Bounding Boxes) Models on DOTA Dataset using Ultralytics HUB
</p>

### Dataset format

OBB dataset format can be found in detail in the [Dataset Guide](../datasets/obb/index.md).

## Val

Validate trained YOLO11n-obb model [accuracy](https://www.ultralytics.com/glossary/accuracy) on the DOTA8 dataset. No arguments are needed as the `model` retains its training `data` and arguments as model attributes.

!!! example

    === "Python"

        ```python
        from ultralytics import YOLO

        # Load a model
        model = YOLO("yolo11n-obb.pt")  # load an official model
        model = YOLO("path/to/best.pt")  # load a custom model

        # Validate the model
        metrics = model.val(data="dota8.yaml")  # no arguments needed, dataset and settings remembered
        metrics.box.map  # map50-95(B)
        metrics.box.map50  # map50(B)
        metrics.box.map75  # map75(B)
        metrics.box.maps  # a list contains map50-95(B) of each category
        ```

    === "CLI"

        ```bash
        yolo obb val model=yolo11n-obb.pt data=dota8.yaml  # val official model
        yolo obb val model=path/to/best.pt data=path/to/data.yaml  # val custom model
        ```

## Predict

Use a trained YOLO11n-obb model to run predictions on images.

!!! example

    === "Python"

        ```python
        from ultralytics import YOLO

        # Load a model
        model = YOLO("yolo11n-obb.pt")  # load an official model
        model = YOLO("path/to/best.pt")  # load a custom model

        # Predict with the model
        results = model("https://ultralytics.com/images/bus.jpg")  # predict on an image
        ```

    === "CLI"

        ```bash
        yolo obb predict model=yolo11n-obb.pt source='https://ultralytics.com/images/bus.jpg'  # predict with official model
        yolo obb predict model=path/to/best.pt source='https://ultralytics.com/images/bus.jpg'  # predict with custom model
        ```

<p align="center">
  <br>
  <iframe loading="lazy" width="720" height="405" src="https://www.youtube.com/embed/5XYdm5CYODA"
    title="YouTube video player" frameborder="0"
    allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share"
    allowfullscreen>
  </iframe>
  <br>
  <strong>Watch:</strong> How to Detect and Track Storage Tanks using Ultralytics YOLO-OBB | Oriented Bounding Boxes | DOTA
</p>

See full `predict` mode details in the [Predict](../modes/predict.md) page.

## Export

Export a YOLO11n-obb model to a different format like ONNX, CoreML, etc.

!!! example

    === "Python"

        ```python
        from ultralytics import YOLO

        # Load a model
        model = YOLO("yolo11n-obb.pt")  # load an official model
        model = YOLO("path/to/best.pt")  # load a custom trained model

        # Export the model
        model.export(format="onnx")
        ```

    === "CLI"

        ```bash
        yolo export model=yolo11n-obb.pt format=onnx  # export official model
        yolo export model=path/to/best.pt format=onnx  # export custom trained model
        ```

Available YOLO11-obb export formats are in the table below. You can export to any format using the `format` argument, i.e. `format='onnx'` or `format='engine'`. You can predict or validate directly on exported models, i.e. `yolo predict model=yolo11n-obb.onnx`. Usage examples are shown for your model after export completes.

{% include "macros/export-table.md" %}

See full `export` details in the [Export](../modes/export.md) page.

## FAQ

### What are Oriented Bounding Boxes (OBB) and how do they differ from regular bounding boxes?

Oriented Bounding Boxes (OBB) include an additional angle to enhance object localization accuracy in images. Unlike regular bounding boxes, which are axis-aligned rectangles, OBBs can rotate to fit the orientation of the object better. This is particularly useful for applications requiring precise object placement, such as aerial or satellite imagery ([Dataset Guide](../datasets/obb/index.md)).

### How do I train a YOLO11n-obb model using a custom dataset?

To train a YOLO11n-obb model with a custom dataset, follow the example below using Python or CLI:

!!! example

    === "Python"

        ```python
        from ultralytics import YOLO

        # Load a pretrained model
        model = YOLO("yolo11n-obb.pt")

        # Train the model
        results = model.train(data="path/to/custom_dataset.yaml", epochs=100, imgsz=640)
        ```

    === "CLI"

        ```bash
        yolo obb train data=path/to/custom_dataset.yaml model=yolo11n-obb.pt epochs=100 imgsz=640
        ```

For more training arguments, check the [Configuration](../usage/cfg.md) section.

### What datasets can I use for training YOLO11-OBB models?

YOLO11-OBB models are pretrained on datasets like [DOTAv1](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/datasets/DOTAv1.yaml) but you can use any dataset formatted for OBB. Detailed information on OBB dataset formats can be found in the [Dataset Guide](../datasets/obb/index.md).

### How can I export a YOLO11-OBB model to ONNX format?

Exporting a YOLO11-OBB model to ONNX format is straightforward using either Python or CLI:

!!! example

    === "Python"

        ```python
        from ultralytics import YOLO

        # Load a model
        model = YOLO("yolo11n-obb.pt")

        # Export the model
        model.export(format="onnx")
        ```

    === "CLI"

        ```bash
        yolo export model=yolo11n-obb.pt format=onnx
        ```

For more export formats and details, refer to the [Export](../modes/export.md) page.

### How do I validate the accuracy of a YOLO11n-obb model?

To validate a YOLO11n-obb model, you can use Python or CLI commands as shown below:

!!! example

    === "Python"

        ```python
        from ultralytics import YOLO

        # Load a model
        model = YOLO("yolo11n-obb.pt")

        # Validate the model
        metrics = model.val(data="dota8.yaml")
        ```

    === "CLI"

        ```bash
        yolo obb val model=yolo11n-obb.pt data=dota8.yaml
        ```

See full validation details in the [Val](../modes/val.md) section.
