---
comments: true
description: Learn about object detection with YOLO26. Explore pretrained models, training, validation, prediction, and export details for efficient object recognition.
keywords: object detection, YOLO26, pretrained models, training, validation, prediction, export, machine learning, computer vision
---

# Object Detection

<img width="1024" src="https://cdn.jsdelivr.net/gh/ultralytics/assets@main/docs/object-detection-examples.avif" alt="YOLO object detection with bounding boxes">

[Object detection](https://www.ultralytics.com/glossary/object-detection) is a task that involves identifying the location and class of objects in an image or video stream.

The output of an object detector is a set of bounding boxes that enclose the objects in the image, along with class labels and confidence scores for each box. Object detection is a good choice when you need to identify objects of interest in a scene, but don't need to know exactly where the object is or its exact shape.

<p align="center">
  <br>
  <iframe loading="lazy" width="720" height="405" src="https://www.youtube.com/embed/5ku7npMrW40?si=6HQO1dDXunV8gekh"
    title="YouTube video player" frameborder="0"
    allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share"
    allowfullscreen>
  </iframe>
  <br>
  <strong>Watch:</strong> Object Detection with Pretrained Ultralytics YOLO Model.
</p>

!!! tip

    YOLO26 Detect models are the default YOLO26 models, i.e., `yolo26n.pt`, and are pretrained on [COCO](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/datasets/coco.yaml).

## [Models](https://github.com/ultralytics/ultralytics/tree/main/ultralytics/cfg/models/26)

YOLO26 pretrained Detect models are shown here. Detect, Segment, and Pose models are pretrained on the [COCO](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/datasets/coco.yaml) dataset, while Classify models are pretrained on the [ImageNet](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/datasets/ImageNet.yaml) dataset.

[Models](https://github.com/ultralytics/ultralytics/tree/main/ultralytics/cfg/models) are downloaded automatically from the latest Ultralytics [release](https://github.com/ultralytics/assets/releases) on first use.

{% include "macros/yolo-det-perf.md" %}

- **mAP<sup>val</sup>** values are for single-model single-scale on [COCO val2017](https://cocodataset.org/) dataset. <br>Reproduce by `yolo val detect data=coco.yaml device=0`
- **Speed** averaged over COCO val images using an [Amazon EC2 P4d](https://aws.amazon.com/ec2/instance-types/p4/) instance. <br>Reproduce by `yolo val detect data=coco.yaml batch=1 device=0|cpu`
- **Params** and **FLOPs** values are for the fused model after `model.fuse()`, which merges Conv and BatchNorm layers and, for end2end models, removes the auxiliary one-to-many detection head. Pretrained checkpoints retain the full training architecture and may show higher counts.

## Train

Train YOLO26n on the COCO8 dataset for 100 [epochs](https://www.ultralytics.com/glossary/epoch) at image size 640. For a full list of available arguments see the [Configuration](../usage/cfg.md) page.

!!! example

    === "Python"

        ```python
        from ultralytics import YOLO

        # Load a model
        model = YOLO("yolo26n.yaml")  # build a new model from YAML
        model = YOLO("yolo26n.pt")  # load a pretrained model (recommended for training)
        model = YOLO("yolo26n.yaml").load("yolo26n.pt")  # build from YAML and transfer weights

        # Train the model
        results = model.train(data="coco8.yaml", epochs=100, imgsz=640)
        ```

    === "CLI"

        ```bash
        # Build a new model from YAML and start training from scratch
        yolo detect train data=coco8.yaml model=yolo26n.yaml epochs=100 imgsz=640

        # Start training from a pretrained *.pt model
        yolo detect train data=coco8.yaml model=yolo26n.pt epochs=100 imgsz=640

        # Build a new model from YAML, transfer pretrained weights to it and start training
        yolo detect train data=coco8.yaml model=yolo26n.yaml pretrained=yolo26n.pt epochs=100 imgsz=640
        ```

### Dataset format

YOLO detection dataset format can be found in detail in the [Dataset Guide](../datasets/detect/index.md). To convert your existing dataset from other formats (like COCO etc.) to YOLO format, please use [JSON2YOLO](https://github.com/ultralytics/JSON2YOLO) tool by Ultralytics.

## Val

Validate trained YOLO26n model [accuracy](https://www.ultralytics.com/glossary/accuracy) on the COCO8 dataset. No arguments are needed as the `model` retains its training `data` and arguments as model attributes.

!!! example

    === "Python"

        ```python
        from ultralytics import YOLO

        # Load a model
        model = YOLO("yolo26n.pt")  # load an official model
        model = YOLO("path/to/best.pt")  # load a custom model

        # Validate the model
        metrics = model.val()  # no arguments needed, dataset and settings remembered
        metrics.box.map  # map50-95
        metrics.box.map50  # map50
        metrics.box.map75  # map75
        metrics.box.maps  # a list containing mAP50-95 for each category
        ```

    === "CLI"

        ```bash
        yolo detect val model=yolo26n.pt      # val official model
        yolo detect val model=path/to/best.pt # val custom model
        ```

## Predict

Use a trained YOLO26n model to run predictions on images.

!!! example

    === "Python"

        ```python
        from ultralytics import YOLO

        # Load a model
        model = YOLO("yolo26n.pt")  # load an official model
        model = YOLO("path/to/best.pt")  # load a custom model

        # Predict with the model
        results = model("https://ultralytics.com/images/bus.jpg")  # predict on an image

        # Access the results
        for result in results:
            xywh = result.boxes.xywh  # center-x, center-y, width, height
            xywhn = result.boxes.xywhn  # normalized
            xyxy = result.boxes.xyxy  # top-left-x, top-left-y, bottom-right-x, bottom-right-y
            xyxyn = result.boxes.xyxyn  # normalized
            names = [result.names[cls.item()] for cls in result.boxes.cls.int()]  # class name of each box
            confs = result.boxes.conf  # confidence score of each box
        ```

    === "CLI"

        ```bash
        yolo detect predict model=yolo26n.pt source='https://ultralytics.com/images/bus.jpg'      # predict with official model
        yolo detect predict model=path/to/best.pt source='https://ultralytics.com/images/bus.jpg' # predict with custom model
        ```

See full `predict` mode details in the [Predict](../modes/predict.md) page.

## Export

Export a YOLO26n model to a different format like ONNX, CoreML, etc.

!!! example

    === "Python"

        ```python
        from ultralytics import YOLO

        # Load a model
        model = YOLO("yolo26n.pt")  # load an official model
        model = YOLO("path/to/best.pt")  # load a custom-trained model

        # Export the model
        model.export(format="onnx")
        ```

    === "CLI"

        ```bash
        yolo export model=yolo26n.pt format=onnx      # export official model
        yolo export model=path/to/best.pt format=onnx # export custom-trained model
        ```

Available YOLO26 export formats are in the table below. You can export to any format using the `format` argument, i.e., `format='onnx'` or `format='engine'`. You can predict or validate directly on exported models, i.e., `yolo predict model=yolo26n.onnx`. Usage examples are shown for your model after export completes.

{% include "macros/export-table.md" %}

See full `export` details in the [Export](../modes/export.md) page.

## FAQ

### How do I train a YOLO26 model on my custom dataset?

Training a YOLO26 model on a custom dataset involves a few steps:

1. **Prepare the Dataset**: Ensure your dataset is in the YOLO format. For guidance, refer to our [Dataset Guide](../datasets/detect/index.md).
2. **Load the Model**: Use the Ultralytics YOLO library to load a pretrained model or create a new model from a YAML file.
3. **Train the Model**: Execute the `train` method in Python or the `yolo detect train` command in CLI.

!!! example

    === "Python"

        ```python
        from ultralytics import YOLO

        # Load a pretrained model
        model = YOLO("yolo26n.pt")

        # Train the model on your custom dataset
        model.train(data="my_custom_dataset.yaml", epochs=100, imgsz=640)
        ```

    === "CLI"

        ```bash
        yolo detect train data=my_custom_dataset.yaml model=yolo26n.pt epochs=100 imgsz=640
        ```

For detailed configuration options, visit the [Configuration](../usage/cfg.md) page.

### What pretrained models are available in YOLO26?

Ultralytics YOLO26 offers various pretrained models for object detection, segmentation, and pose estimation. These models are pretrained on the COCO dataset or ImageNet for classification tasks. Here are some of the available models:

- [YOLO26n](https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26n.pt)
- [YOLO26s](https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26s.pt)
- [YOLO26m](https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26m.pt)
- [YOLO26l](https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26l.pt)
- [YOLO26x](https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26x.pt)

For a detailed list and performance metrics, refer to the [Models](https://github.com/ultralytics/ultralytics/tree/main/ultralytics/cfg/models/26) section.

### How can I validate the accuracy of my trained YOLO model?

To validate the accuracy of your trained YOLO26 model, you can use the `.val()` method in Python or the `yolo detect val` command in CLI. This will provide metrics like mAP50-95, mAP50, and more.

!!! example

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

### What formats can I export a YOLO26 model to?

Ultralytics YOLO26 allows exporting models to various formats such as [ONNX](https://www.ultralytics.com/glossary/onnx-open-neural-network-exchange), [TensorRT](https://www.ultralytics.com/glossary/tensorrt), [CoreML](https://docs.ultralytics.com/integrations/coreml/), and more to ensure compatibility across different platforms and devices.

!!! example

    === "Python"

        ```python
        from ultralytics import YOLO

        # Load the model
        model = YOLO("yolo26n.pt")

        # Export the model to ONNX format
        model.export(format="onnx")
        ```

    === "CLI"

        ```bash
        yolo export model=yolo26n.pt format=onnx
        ```

Check the full list of supported formats and instructions on the [Export](../modes/export.md) page.

### Why should I use Ultralytics YOLO26 for object detection?

Ultralytics YOLO26 is designed to offer state-of-the-art performance for object detection, segmentation, and pose estimation. Here are some key advantages:

1. **Pretrained Models**: Utilize models pretrained on popular datasets like [COCO](https://docs.ultralytics.com/datasets/detect/coco/) and [ImageNet](https://docs.ultralytics.com/datasets/classify/imagenet/) for faster development.
2. **High Accuracy**: Achieves impressive mAP scores, ensuring reliable object detection.
3. **Speed**: Optimized for [real-time inference](https://www.ultralytics.com/glossary/real-time-inference), making it ideal for applications requiring swift processing.
4. **Flexibility**: Export models to various formats like ONNX and TensorRT for deployment across multiple platforms.

Explore our [Blog](https://www.ultralytics.com/blog) for use cases and success stories showcasing YOLO26 in action.
