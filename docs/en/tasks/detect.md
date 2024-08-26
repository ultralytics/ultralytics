---
comments: true
description: Learn about object detection with YOLOv8. Explore pretrained models, training, validation, prediction, and export details for efficient object recognition.
keywords: object detection, YOLOv8, pretrained models, training, validation, prediction, export, machine learning, computer vision
---

# Object Detection

<img width="1024" src="https://user-images.githubusercontent.com/26833433/243418624-5785cb93-74c9-4541-9179-d5c6782d491a.png" alt="Object detection examples">

Object detection is a task that involves identifying the location and class of objects in an image or video stream.

The output of an object detector is a set of bounding boxes that enclose the objects in the image, along with class labels and confidence scores for each box. Object detection is a good choice when you need to identify objects of interest in a scene, but don't need to know exactly where the object is or its exact shape.

<p align="center">
  <br>
  <iframe loading="lazy" width="720" height="405" src="https://www.youtube.com/embed/5ku7npMrW40?si=6HQO1dDXunV8gekh"
    title="YouTube video player" frameborder="0"
    allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share"
    allowfullscreen>
  </iframe>
  <br>
  <strong>Watch:</strong> Object Detection with Pre-trained Ultralytics YOLOv8 Model.
</p>

!!! Tip "Tip"

    YOLOv8 Detect models are the default YOLOv8 models, i.e. `yolov8n.pt` and are pretrained on [COCO](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/datasets/coco.yaml).

## [Models](https://github.com/ultralytics/ultralytics/tree/main/ultralytics/cfg/models/v8)

YOLOv8 pretrained Detect models are shown here. Detect, Segment and Pose models are pretrained on the [COCO](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/datasets/coco.yaml) dataset, while Classify models are pretrained on the [ImageNet](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/datasets/ImageNet.yaml) dataset.

[Models](https://github.com/ultralytics/ultralytics/tree/main/ultralytics/cfg/models) download automatically from the latest Ultralytics [release](https://github.com/ultralytics/assets/releases) on first use.

| Model                                                                                | size<br><sup>(pixels) | mAP<sup>val<br>50-95 | Speed<br><sup>CPU ONNX<br>(ms) | Speed<br><sup>A100 TensorRT<br>(ms) | params<br><sup>(M) | FLOPs<br><sup>(B) |
| ------------------------------------------------------------------------------------ | --------------------- | -------------------- | ------------------------------ | ----------------------------------- | ------------------ | ----------------- |
| [YOLOv8n](https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov8n.pt) | 640                   | 37.3                 | 80.4                           | 0.99                                | 3.2                | 8.7               |
| [YOLOv8s](https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov8s.pt) | 640                   | 44.9                 | 128.4                          | 1.20                                | 11.2               | 28.6              |
| [YOLOv8m](https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov8m.pt) | 640                   | 50.2                 | 234.7                          | 1.83                                | 25.9               | 78.9              |
| [YOLOv8l](https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov8l.pt) | 640                   | 52.9                 | 375.2                          | 2.39                                | 43.7               | 165.2             |
| [YOLOv8x](https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov8x.pt) | 640                   | 53.9                 | 479.1                          | 3.53                                | 68.2               | 257.8             |

- **mAP<sup>val</sup>** values are for single-model single-scale on [COCO val2017](https://cocodataset.org) dataset. <br>Reproduce by `yolo val detect data=coco.yaml device=0`
- **Speed** averaged over COCO val images using an [Amazon EC2 P4d](https://aws.amazon.com/ec2/instance-types/p4/) instance. <br>Reproduce by `yolo val detect data=coco8.yaml batch=1 device=0|cpu`

## Train

Train YOLOv8n on the COCO8 dataset for 100 epochs at image size 640. For a full list of available arguments see the [Configuration](../usage/cfg.md) page.

!!! Example

    === "Python"

        ```python
        from ultralytics import YOLO

        # Load a model
        model = YOLO("yolov8n.yaml")  # build a new model from YAML
        model = YOLO("yolov8n.pt")  # load a pretrained model (recommended for training)
        model = YOLO("yolov8n.yaml").load("yolov8n.pt")  # build from YAML and transfer weights

        # Train the model
        results = model.train(data="coco8.yaml", epochs=100, imgsz=640)
        ```

    === "CLI"

        ```bash
        # Build a new model from YAML and start training from scratch
        yolo detect train data=coco8.yaml model=yolov8n.yaml epochs=100 imgsz=640

        # Start training from a pretrained *.pt model
        yolo detect train data=coco8.yaml model=yolov8n.pt epochs=100 imgsz=640

        # Build a new model from YAML, transfer pretrained weights to it and start training
        yolo detect train data=coco8.yaml model=yolov8n.yaml pretrained=yolov8n.pt epochs=100 imgsz=640
        ```

### Dataset format

YOLO detection dataset format can be found in detail in the [Dataset Guide](../datasets/detect/index.md). To convert your existing dataset from other formats (like COCO etc.) to YOLO format, please use [JSON2YOLO](https://github.com/ultralytics/JSON2YOLO) tool by Ultralytics.

## Val

Validate trained YOLOv8n model accuracy on the COCO8 dataset. No argument need to passed as the `model` retains its training `data` and arguments as model attributes.

!!! Example

    === "Python"

        ```python
        from ultralytics import YOLO

        # Load a model
        model = YOLO("yolov8n.pt")  # load an official model
        model = YOLO("path/to/best.pt")  # load a custom model

        # Validate the model
        metrics = model.val()  # no arguments needed, dataset and settings remembered
        metrics.box.map  # map50-95
        metrics.box.map50  # map50
        metrics.box.map75  # map75
        metrics.box.maps  # a list contains map50-95 of each category
        ```

    === "CLI"

        ```bash
        yolo detect val model=yolov8n.pt  # val official model
        yolo detect val model=path/to/best.pt  # val custom model
        ```

## Predict

Use a trained YOLOv8n model to run predictions on images.

!!! Example

    === "Python"

        ```python
        from ultralytics import YOLO

        # Load a model
        model = YOLO("yolov8n.pt")  # load an official model
        model = YOLO("path/to/best.pt")  # load a custom model

        # Predict with the model
        results = model("https://ultralytics.com/images/bus.jpg")  # predict on an image
        ```

    === "CLI"

        ```bash
        yolo detect predict model=yolov8n.pt source='https://ultralytics.com/images/bus.jpg'  # predict with official model
        yolo detect predict model=path/to/best.pt source='https://ultralytics.com/images/bus.jpg'  # predict with custom model
        ```

See full `predict` mode details in the [Predict](../modes/predict.md) page.

## Export

Export a YOLOv8n model to a different format like ONNX, CoreML, etc.

!!! Example

    === "Python"

        ```python
        from ultralytics import YOLO

        # Load a model
        model = YOLO("yolov8n.pt")  # load an official model
        model = YOLO("path/to/best.pt")  # load a custom trained model

        # Export the model
        model.export(format="onnx")
        ```

    === "CLI"

        ```bash
        yolo export model=yolov8n.pt format=onnx  # export official model
        yolo export model=path/to/best.pt format=onnx  # export custom trained model
        ```

Available YOLOv8 export formats are in the table below. You can export to any format using the `format` argument, i.e. `format='onnx'` or `format='engine'`. You can predict or validate directly on exported models, i.e. `yolo predict model=yolov8n.onnx`. Usage examples are shown for your model after export completes.

{% include "macros/export-table.md" %}

See full `export` details in the [Export](../modes/export.md) page.

## FAQ

### How do I train a YOLOv8 model on my custom dataset?

Training a YOLOv8 model on a custom dataset involves a few steps:

1. **Prepare the Dataset**: Ensure your dataset is in the YOLO format. For guidance, refer to our [Dataset Guide](../datasets/detect/index.md).
2. **Load the Model**: Use the Ultralytics YOLO library to load a pre-trained model or create a new model from a YAML file.
3. **Train the Model**: Execute the `train` method in Python or the `yolo detect train` command in CLI.

!!! Example

    === "Python"

        ```python
        from ultralytics import YOLO

        # Load a pretrained model
        model = YOLO("yolov8n.pt")

        # Train the model on your custom dataset
        model.train(data="my_custom_dataset.yaml", epochs=100, imgsz=640)
        ```

    === "CLI"

        ```bash
        yolo detect train data=my_custom_dataset.yaml model=yolov8n.pt epochs=100 imgsz=640
        ```

For detailed configuration options, visit the [Configuration](../usage/cfg.md) page.

### What pretrained models are available in YOLOv8?

Ultralytics YOLOv8 offers various pretrained models for object detection, segmentation, and pose estimation. These models are pretrained on the COCO dataset or ImageNet for classification tasks. Here are some of the available models:

- [YOLOv8n](https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov8n.pt)
- [YOLOv8s](https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov8s.pt)
- [YOLOv8m](https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov8m.pt)
- [YOLOv8l](https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov8l.pt)
- [YOLOv8x](https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov8x.pt)

For a detailed list and performance metrics, refer to the [Models](https://github.com/ultralytics/ultralytics/tree/main/ultralytics/cfg/models/v8) section.

### How can I validate the accuracy of my trained YOLOv8 model?

To validate the accuracy of your trained YOLOv8 model, you can use the `.val()` method in Python or the `yolo detect val` command in CLI. This will provide metrics like mAP50-95, mAP50, and more.

!!! Example

    === "Python"

        ```python
        from ultralytics import YOLO

        # Load the model
        model = YOLO("path/to/best.pt")

        # Validate the model
        metrics = model.val()
        print(metrics.box.map)  # mAP50-95
        ```

    === "CLI"

        ```bash
        yolo detect val model=path/to/best.pt
        ```

For more validation details, visit the [Val](../modes/val.md) page.

### What formats can I export a YOLOv8 model to?

Ultralytics YOLOv8 allows exporting models to various formats such as ONNX, TensorRT, CoreML, and more to ensure compatibility across different platforms and devices.

!!! Example

    === "Python"

        ```python
        from ultralytics import YOLO

        # Load the model
        model = YOLO("yolov8n.pt")

        # Export the model to ONNX format
        model.export(format="onnx")
        ```

    === "CLI"

        ```bash
        yolo export model=yolov8n.pt format=onnx
        ```

Check the full list of supported formats and instructions on the [Export](../modes/export.md) page.

### Why should I use Ultralytics YOLOv8 for object detection?

Ultralytics YOLOv8 is designed to offer state-of-the-art performance for object detection, segmentation, and pose estimation. Here are some key advantages:

1. **Pretrained Models**: Utilize models pretrained on popular datasets like COCO and ImageNet for faster development.
2. **High Accuracy**: Achieves impressive mAP scores, ensuring reliable object detection.
3. **Speed**: Optimized for real-time inference, making it ideal for applications requiring swift processing.
4. **Flexibility**: Export models to various formats like ONNX and TensorRT for deployment across multiple platforms.

Explore our [Blog](https://www.ultralytics.com/blog) for use cases and success stories showcasing YOLOv8 in action.
