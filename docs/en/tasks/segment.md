---
comments: true
description: Master instance segmentation using YOLO11. Learn how to detect, segment and outline objects in images with detailed guides and examples.
keywords: instance segmentation, YOLO11, object detection, image segmentation, machine learning, deep learning, computer vision, COCO dataset, Ultralytics
model_name: yolo11n-seg
---

# Instance Segmentation

<img width="1024" src="https://github.com/ultralytics/docs/releases/download/0/instance-segmentation-examples.avif" alt="Instance segmentation examples">

[Instance segmentation](https://www.ultralytics.com/glossary/instance-segmentation) goes a step further than object detection and involves identifying individual objects in an image and segmenting them from the rest of the image.

The output of an instance segmentation model is a set of masks or contours that outline each object in the image, along with class labels and confidence scores for each object. Instance segmentation is useful when you need to know not only where objects are in an image, but also what their exact shape is.

<p align="center">
  <br>
  <iframe loading="lazy" width="720" height="405" src="https://www.youtube.com/embed/o4Zd-IeMlSY?si=37nusCzDTd74Obsp"
    title="YouTube video player" frameborder="0"
    allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share"
    allowfullscreen>
  </iframe>
  <br>
  <strong>Watch:</strong> Run Segmentation with Pre-Trained Ultralytics YOLOv8 Model in Python.
</p>

!!! tip

    YOLOv8 Segment models use the `-seg` suffix, i.e. `yolov8n-seg.pt` and are pretrained on [COCO](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/datasets/coco.yaml).

## [Models](https://github.com/ultralytics/ultralytics/tree/main/ultralytics/cfg/models/v8)

YOLOv8 pretrained Segment models are shown here. Detect, Segment and Pose models are pretrained on the [COCO](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/datasets/coco.yaml) dataset, while Classify models are pretrained on the [ImageNet](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/datasets/ImageNet.yaml) dataset.

[Models](https://github.com/ultralytics/ultralytics/tree/main/ultralytics/cfg/models) download automatically from the latest Ultralytics [release](https://github.com/ultralytics/assets/releases) on first use.

| Model                                                                                        | size<br><sup>(pixels) | mAP<sup>box<br>50-95 | mAP<sup>mask<br>50-95 | Speed<br><sup>CPU ONNX<br>(ms) | Speed<br><sup>T4 TensorRT10<br>(ms) | params<br><sup>(M) | FLOPs<br><sup>(B) |
|----------------------------------------------------------------------------------------------|-----------------------|----------------------|-----------------------|--------------------------------|-------------------------------------|--------------------|-------------------|
| [YOLO11n-seg](https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11n-seg.pt) | 640                   | 38.9                 | 32.0                  | 65.9 ± 1.1                     | 1.8 ± 0.0                           | 2.9                | 10.4              |
| [YOLO11s-seg](https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11s-seg.pt) | 640                   | 46.6                 | 37.8                  | 117.6 ± 4.9                    | 2.9 ± 0.0                           | 10.1               | 35.5              |
| [YOLO11m-seg](https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11m-seg.pt) | 640                   | 51.5                 | 41.5                  | 281.6 ± 1.2                    | 6.3 ± 0.1                           | 22.4               | 123.3             |
| [YOLO11l-seg](https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11l-seg.pt) | 640                   | 53.4                 | 42.9                  | 344.2 ± 3.2                    | 7.8 ± 0.2                           | 27.6               | 142.2             |
| [YOLO11x-seg](https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11x-seg.pt) | 640                   | 54.7                 | 43.8                  | 664.5 ± 3.2                    | 15.8 ± 0.7                          | 62.1               | 319.0             |

- **mAP<sup>val</sup>** values are for single-model single-scale on [COCO val2017](https://cocodataset.org/) dataset. <br>Reproduce by `yolo val segment data=coco-seg.yaml device=0`
- **Speed** averaged over COCO val images using an [Amazon EC2 P4d](https://aws.amazon.com/ec2/instance-types/p4/) instance. <br>Reproduce by `yolo val segment data=coco-seg.yaml batch=1 device=0|cpu`

## Train

Train YOLOv8n-seg on the COCO128-seg dataset for 100 [epochs](https://www.ultralytics.com/glossary/epoch) at image size 640. For a full list of available arguments see the [Configuration](../usage/cfg.md) page.

!!! example

    === "Python"

        ```python
        from ultralytics import YOLO

        # Load a model
        model = YOLO("yolov8n-seg.yaml")  # build a new model from YAML
        model = YOLO("yolov8n-seg.pt")  # load a pretrained model (recommended for training)
        model = YOLO("yolov8n-seg.yaml").load("yolov8n.pt")  # build from YAML and transfer weights

        # Train the model
        results = model.train(data="coco8-seg.yaml", epochs=100, imgsz=640)
        ```

    === "CLI"

        ```bash
        # Build a new model from YAML and start training from scratch
        yolo segment train data=coco8-seg.yaml model=yolov8n-seg.yaml epochs=100 imgsz=640

        # Start training from a pretrained *.pt model
        yolo segment train data=coco8-seg.yaml model=yolov8n-seg.pt epochs=100 imgsz=640

        # Build a new model from YAML, transfer pretrained weights to it and start training
        yolo segment train data=coco8-seg.yaml model=yolov8n-seg.yaml pretrained=yolov8n-seg.pt epochs=100 imgsz=640
        ```

### Dataset format

YOLO segmentation dataset format can be found in detail in the [Dataset Guide](../datasets/segment/index.md). To convert your existing dataset from other formats (like COCO etc.) to YOLO format, please use [JSON2YOLO](https://github.com/ultralytics/JSON2YOLO) tool by Ultralytics.

## Val

Validate trained YOLOv8n-seg model [accuracy](https://www.ultralytics.com/glossary/accuracy) on the COCO128-seg dataset. No arguments are needed as the `model` retains its training `data` and arguments as model attributes.

!!! example

    === "Python"

        ```python
        from ultralytics import YOLO

        # Load a model
        model = YOLO("yolov8n-seg.pt")  # load an official model
        model = YOLO("path/to/best.pt")  # load a custom model

        # Validate the model
        metrics = model.val()  # no arguments needed, dataset and settings remembered
        metrics.box.map  # map50-95(B)
        metrics.box.map50  # map50(B)
        metrics.box.map75  # map75(B)
        metrics.box.maps  # a list contains map50-95(B) of each category
        metrics.seg.map  # map50-95(M)
        metrics.seg.map50  # map50(M)
        metrics.seg.map75  # map75(M)
        metrics.seg.maps  # a list contains map50-95(M) of each category
        ```

    === "CLI"

        ```bash
        yolo segment val model=yolov8n-seg.pt  # val official model
        yolo segment val model=path/to/best.pt  # val custom model
        ```

## Predict

Use a trained YOLOv8n-seg model to run predictions on images.

!!! example

    === "Python"

        ```python
        from ultralytics import YOLO

        # Load a model
        model = YOLO("yolov8n-seg.pt")  # load an official model
        model = YOLO("path/to/best.pt")  # load a custom model

        # Predict with the model
        results = model("https://ultralytics.com/images/bus.jpg")  # predict on an image
        ```

    === "CLI"

        ```bash
        yolo segment predict model=yolov8n-seg.pt source='https://ultralytics.com/images/bus.jpg'  # predict with official model
        yolo segment predict model=path/to/best.pt source='https://ultralytics.com/images/bus.jpg'  # predict with custom model
        ```

See full `predict` mode details in the [Predict](../modes/predict.md) page.

## Export

Export a YOLOv8n-seg model to a different format like ONNX, CoreML, etc.

!!! example

    === "Python"

        ```python
        from ultralytics import YOLO

        # Load a model
        model = YOLO("yolov8n-seg.pt")  # load an official model
        model = YOLO("path/to/best.pt")  # load a custom trained model

        # Export the model
        model.export(format="onnx")
        ```

    === "CLI"

        ```bash
        yolo export model=yolov8n-seg.pt format=onnx  # export official model
        yolo export model=path/to/best.pt format=onnx  # export custom trained model
        ```

Available YOLOv8-seg export formats are in the table below. You can export to any format using the `format` argument, i.e. `format='onnx'` or `format='engine'`. You can predict or validate directly on exported models, i.e. `yolo predict model=yolov8n-seg.onnx`. Usage examples are shown for your model after export completes.

{% include "macros/export-table.md" %}

See full `export` details in the [Export](../modes/export.md) page.

## FAQ

### How do I train a YOLOv8 segmentation model on a custom dataset?

To train a YOLOv8 segmentation model on a custom dataset, you first need to prepare your dataset in the YOLO segmentation format. You can use tools like [JSON2YOLO](https://github.com/ultralytics/JSON2YOLO) to convert datasets from other formats. Once your dataset is ready, you can train the model using Python or CLI commands:

!!! example

    === "Python"

        ```python
        from ultralytics import YOLO

        # Load a pretrained YOLOv8 segment model
        model = YOLO("yolov8n-seg.pt")

        # Train the model
        results = model.train(data="path/to/your_dataset.yaml", epochs=100, imgsz=640)
        ```

    === "CLI"

        ```bash
        yolo segment train data=path/to/your_dataset.yaml model=yolov8n-seg.pt epochs=100 imgsz=640
        ```

Check the [Configuration](../usage/cfg.md) page for more available arguments.

### What is the difference between [object detection](https://www.ultralytics.com/glossary/object-detection) and instance segmentation in YOLOv8?

Object detection identifies and localizes objects within an image by drawing bounding boxes around them, whereas instance segmentation not only identifies the bounding boxes but also delineates the exact shape of each object. YOLOv8 instance segmentation models provide masks or contours that outline each detected object, which is particularly useful for tasks where knowing the precise shape of objects is important, such as medical imaging or autonomous driving.

### Why use YOLOv8 for instance segmentation?

Ultralytics YOLOv8 is a state-of-the-art model recognized for its high accuracy and real-time performance, making it ideal for instance segmentation tasks. YOLOv8 Segment models come pretrained on the [COCO dataset](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/datasets/coco.yaml), ensuring robust performance across a variety of objects. Additionally, YOLOv8 supports training, validation, prediction, and export functionalities with seamless integration, making it highly versatile for both research and industry applications.

### How do I load and validate a pretrained YOLOv8 segmentation model?

Loading and validating a pretrained YOLOv8 segmentation model is straightforward. Here's how you can do it using both Python and CLI:

!!! example

    === "Python"

        ```python
        from ultralytics import YOLO

        # Load a pretrained model
        model = YOLO("yolov8n-seg.pt")

        # Validate the model
        metrics = model.val()
        print("Mean Average Precision for boxes:", metrics.box.map)
        print("Mean Average Precision for masks:", metrics.seg.map)
        ```

    === "CLI"

        ```bash
        yolo segment val model=yolov8n-seg.pt
        ```

These steps will provide you with validation metrics like [Mean Average Precision](https://www.ultralytics.com/glossary/mean-average-precision-map) (mAP), crucial for assessing model performance.

### How can I export a YOLOv8 segmentation model to ONNX format?

Exporting a YOLOv8 segmentation model to ONNX format is simple and can be done using Python or CLI commands:

!!! example

    === "Python"

        ```python
        from ultralytics import YOLO

        # Load a pretrained model
        model = YOLO("yolov8n-seg.pt")

        # Export the model to ONNX format
        model.export(format="onnx")
        ```

    === "CLI"

        ```bash
        yolo export model=yolov8n-seg.pt format=onnx
        ```

For more details on exporting to various formats, refer to the [Export](../modes/export.md) page.
