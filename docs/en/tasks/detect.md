---
comments: true
description: Object detection with Ultralytics YOLO26. Explore COCO-pretrained models and benchmarks, then train, validate, predict, and export detectors in Python or CLI.
keywords: object detection, Ultralytics YOLO, YOLO26, pretrained models, training, validation, prediction, export, machine learning, computer vision
model_name: yolo26n
---

# Object Detection with Ultralytics YOLO {#object-detection}

<img width="1024" src="https://cdn.ul.run/i/e0d2cbeb1a1421b134a38986df7ee7c9.avif" alt="Ultralytics YOLO object detection with bounding boxes">

[Object detection](https://www.ultralytics.com/glossary/object-detection) is a task that involves identifying the location and class of objects in an image or video stream.

The output of an [object detector](https://academy.ultralytics.com/courses/computer-vision-foundations/object-detection) is a set of [bounding boxes](https://www.ultralytics.com/glossary/bounding-box) that enclose the objects in the image, along with class labels and confidence scores for each box. Object detection is a good choice when you need to identify objects of interest in a scene, but don't need to know their exact shape.

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

    YOLO26 Detect models are the default YOLO26 models, e.g., `yolo26n.pt`, and are pretrained on [COCO](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/datasets/coco.yaml).

## Models

YOLO26 Detect models pretrained on the [COCO](../datasets/detect/coco.md) dataset are shown below.

[Models](https://github.com/ultralytics/ultralytics/tree/main/ultralytics/cfg/models) download automatically from the latest Ultralytics [release](https://github.com/ultralytics/assets/releases) on first use.

{% include "macros/yolo-det-perf.md" %}

- **mAP<sup>val</sup>** values are for single-model single-scale on [COCO val2017](https://cocodataset.org/) dataset. <br>Reproduce with `yolo detect val data=coco.yaml device=0`. Add `nms=False` to reproduce the e2e column.
- **Speed** averaged over COCO val images with ONNX on CPU and TensorRT10 on an NVIDIA T4 GPU. <br>Reproduce with `yolo detect val data=coco.yaml batch=1 device=0|cpu nms=False`
- **Params** and **FLOPs** values are for fused models after Conv/BatchNorm folding and removal of the unused detection branch. Pretrained checkpoints retain the full training architecture and may show higher counts.

See the [unreleased YOLO27 preview](../models/yolo27.md#performance-metrics) for preliminary COCO detection results.

## Train

[Train](../usage/cfg.md#train-settings) a [YOLO26n](https://platform.ultralytics.com/ultralytics/yolo26/yolo26n) model on the [COCO8](https://platform.ultralytics.com/ultralytics/datasets/coco8) dataset for 100 [epochs](https://www.ultralytics.com/glossary/epoch) at image size 640.

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

See full `train` mode details in the [Train](../modes/train.md) page. Detection models can also be trained with [Ultralytics Platform cloud training](../platform/train/cloud-training.md).

### Dataset format

YOLO detection dataset format can be found in detail in the [Dataset Guide](../datasets/detect/index.md). To convert an existing COCO JSON dataset to YOLO format, use the built-in `convert_coco` function described in the [COCO to YOLO guide](../guides/coco-to-yolo.md). You can also annotate and manage detection datasets with [Ultralytics Platform's AI-assisted annotation tools](../platform/data/annotation.md).

## Val

Validate trained YOLO26n model accuracy, measured as [mean average precision (mAP)](https://www.ultralytics.com/glossary/mean-average-precision-map). No arguments are needed, as the `model` retains its training `data` and arguments as model attributes: `path/to/best.pt` from the [Train](#train) example validates on COCO8. Official weights record a training dataset path that doesn't exist on your machine, so they fall back to the task default `coco8.yaml` with a warning. Pass `data` to validate on another dataset.

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
        metrics.box.maps  # mAP50-95 per class; classes absent from the val set hold the overall mAP
        metrics.box.ap_class_index  # class indices that were actually evaluated
        metrics.box.image_metrics  # per-image metrics dictionary with precision, recall, F1, TP, FP, and FN
        ```

    === "CLI"

        ```bash
        yolo detect val model=yolo26n.pt data=coco8.yaml             # val official model
        yolo detect val model=path/to/best.pt data=path/to/data.yaml # val custom model
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

### Results Output

Object detection returns one `Results` object per image. The primary prediction field is `result.boxes`, which contains
box coordinates, class IDs, and confidence scores for each detected object.

| Attribute           | Type            | Shape     | Description                                           |
| ------------------- | --------------- | --------- | ----------------------------------------------------- |
| `result.boxes`      | `Boxes`         | `(N)`     | Detection boxes.                                      |
| `result.boxes.data` | `torch.float32` | `(N,6/7)` | Raw `[x1,y1,x2,y2,conf,cls]`, plus optional track ID. |
| `result.boxes.xyxy` | `torch.float32` | `(N,4)`   | `xyxy` pixel boxes.                                   |
| `result.boxes.conf` | `torch.float32` | `(N,)`    | Confidence scores.                                    |
| `result.boxes.cls`  | `torch.float32` | `(N,)`    | Class IDs; cast to `int` for names.                   |

For task-specific `Results` fields across every task, see the [Predict Results by Task](../modes/predict.md#results-by-task) section.

## Export

Export a YOLO26n model to a different format like ONNX, CoreML, etc.

!!! example

    === "Python"

        ```python
        from ultralytics import YOLO

        # Load a model
        model = YOLO("yolo26n.pt")  # load an official model
        model = YOLO("path/to/best.pt")  # load a custom model

        # Export the model
        model.export(format="onnx")
        ```

    === "CLI"

        ```bash
        yolo export model=yolo26n.pt format=onnx      # export official model
        yolo export model=path/to/best.pt format=onnx # export custom model
        ```

Available YOLO26 export formats are in the table below. You can export to any format using the `format` argument, e.g., `format='onnx'` or `format='engine'`. You can predict or validate directly on exported models, e.g., `yolo predict model=yolo26n.onnx`. Usage examples are shown for your model after export completes.

{% include "macros/export-table.md" %}

See full `export` details in the [Export](../modes/export.md) page.

## FAQ

### Can I train and deploy detection models without coding?

Yes. The [Ultralytics Platform quickstart](../platform/quickstart.md) covers a browser-based workflow for annotating datasets, training detection models on cloud GPUs, and deploying them to inference endpoints.

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

Ultralytics YOLO26 offers various pretrained models for object detection, [instance segmentation](segment.md), [semantic segmentation](semantic.md), [depth estimation](depth.md), [classification](classify.md), [pose estimation](pose.md), and [oriented bounding boxes](obb.md). These models are pretrained on the COCO dataset, Cityscapes for semantic segmentation, a multi-dataset mix for depth estimation, ImageNet for classification, or DOTAv1 for OBB. Here are some of the available models:

- [YOLO26n](https://platform.ultralytics.com/ultralytics/yolo26/yolo26n)
- [YOLO26s](https://platform.ultralytics.com/ultralytics/yolo26/yolo26s)
- [YOLO26m](https://platform.ultralytics.com/ultralytics/yolo26/yolo26m)
- [YOLO26l](https://platform.ultralytics.com/ultralytics/yolo26/yolo26l)
- [YOLO26x](https://platform.ultralytics.com/ultralytics/yolo26/yolo26x)

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
        yolo detect val model=path/to/best.pt data=path/to/data.yaml
        ```

For more validation details, visit the [Val](../modes/val.md) page.

### What formats can I export a YOLO26 model to?

Ultralytics YOLO26 allows exporting models to various formats such as [ONNX](https://www.ultralytics.com/glossary/onnx-open-neural-network-exchange), [TensorRT](https://www.ultralytics.com/glossary/tensorrt), [CoreML](../integrations/coreml.md), and more to ensure compatibility across different platforms and devices.

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

Ultralytics YOLO26 provides [fast, accurate models](https://www.ultralytics.com/blog/meet-ultralytics-yolo26-a-better-faster-smaller-yolo-model) for object detection, instance segmentation, semantic segmentation, pose estimation, and the other supported tasks. Key advantages include:

1. **Pretrained Models**: Use models pretrained on popular datasets like [COCO](../datasets/detect/coco.md) and [ImageNet](../datasets/classify/imagenet.md) for faster development.
2. **High Accuracy**: Achieves high [mAP on COCO](#models).
3. **Speed**: Optimized for [real-time inference](https://www.ultralytics.com/glossary/real-time-inference) in latency-sensitive applications.
4. **Flexibility**: Export models to various formats like ONNX and TensorRT for deployment across multiple platforms.
