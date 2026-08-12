---
comments: true
description: Learn about object detection with Ultralytics YOLO26. Explore pretrained models, training, validation, prediction, and export details for efficient object recognition.
keywords: object detection, Ultralytics YOLO, YOLO26, pretrained models, training, validation, prediction, export, machine learning, computer vision
---

# Object Detection with Ultralytics YOLO {#object-detection}

<img width="1024" src="https://cdn.jsdelivr.net/gh/ultralytics/assets@main/docs/object-detection-examples.avif" alt="Ultralytics YOLO object detection with bounding boxes">

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

YOLO26 pretrained Detect models are shown here. Detect, Segment, and Pose models are pretrained on the [COCO](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/datasets/coco.yaml) dataset, [Semantic](semantic.md) models are pretrained on [Cityscapes](../datasets/semantic/cityscapes.md), and Classify models are pretrained on the [ImageNet](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/datasets/ImageNet.yaml) dataset.

[Models](https://github.com/ultralytics/ultralytics/tree/main/ultralytics/cfg/models) are downloaded automatically from the latest Ultralytics [release](https://github.com/ultralytics/assets/releases) on first use.

| Model                                                                  | size<br><sup>(pixels)</sup> | mAP<sup>val<br>50-95</sup> | mAP<sup>val<br>50-95(e2e)</sup> | Speed<br><sup>CPU ONNX<br>(ms)</sup> | Speed<br><sup>T4 TensorRT10<br>(ms)</sup> | params<br><sup>(M)</sup> | FLOPs<br><sup>(B)</sup> |
| ---------------------------------------------------------------------- | --------------------------- | -------------------------- | ------------------------------- | ------------------------------------ | ----------------------------------------- | ------------------------ | ----------------------- |
| [YOLO26n](https://platform.ultralytics.com/ultralytics/yolo26/yolo26n) | 640                         | 40.9                       | 40.1                            | 38.9 ± 0.7                           | 1.7 ± 0.0                                 | 2.4                      | 5.4                     |
| [YOLO26s](https://platform.ultralytics.com/ultralytics/yolo26/yolo26s) | 640                         | 48.6                       | 47.8                            | 87.2 ± 0.9                           | 2.5 ± 0.0                                 | 9.5                      | 20.7                    |
| [YOLO26m](https://platform.ultralytics.com/ultralytics/yolo26/yolo26m) | 640                         | 53.1                       | 52.5                            | 220.0 ± 1.4                          | 4.7 ± 0.1                                 | 20.4                     | 68.2                    |
| [YOLO26l](https://platform.ultralytics.com/ultralytics/yolo26/yolo26l) | 640                         | 55.0                       | 54.4                            | 286.2 ± 2.0                          | 6.2 ± 0.2                                 | 24.8                     | 86.4                    |
| [YOLO26x](https://platform.ultralytics.com/ultralytics/yolo26/yolo26x) | 640                         | 57.5                       | 56.9                            | 525.8 ± 4.0                          | 11.8 ± 0.2                                | 55.7                     | 193.9                   |

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

See full `train` mode details in the [Train](../modes/train.md) page. Detection models can also be trained with [Ultralytics Platform cloud training](../platform/train/cloud-training.md).

### Dataset format

YOLO detection dataset format can be found in detail in the [Dataset Guide](../datasets/detect/index.md). To convert your existing dataset from other formats (like COCO etc.) to YOLO format, please use [JSON2YOLO](https://github.com/ultralytics/JSON2YOLO) tool by Ultralytics. You can also annotate and manage detection datasets with [Ultralytics Platform's AI-assisted annotation tools](../platform/data/annotation.md).

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
        metrics.box.image_metrics  # per-image metrics dictionary with precision, recall, F1, TP, FP, and FN
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

| Format | `format` Argument | Model | Metadata | Arguments |
| ---------------------------------------------------------- | ----------------- | ------------------------------------------------- | -------- | ------------------------------------------------------------------------------------------------------------------------ |
| [PyTorch](https://pytorch.org/) | - | `yolo26n.pt` | ✅ | - |
| [TorchScript](../integrations/torchscript.md) | `torchscript` | `yolo26n.torchscript` | ✅ | `imgsz`, `quantize`, `dynamic`, `nms`:material-information-outline:{ title="conf, iou, agnostic_nms are also available when nms=True" }, `batch`, `device` |
| [ONNX](../integrations/onnx.md) | `onnx` | `yolo26n.onnx` | ✅ | `imgsz`, `quantize`, `dynamic`, `simplify`, `opset`, `nms`:material-information-outline:{ title="conf, iou, agnostic_nms are also available when nms=True" }, `batch`, `data`, `fraction`, `device` |
| [OpenVINO](../integrations/openvino.md) | `openvino` | `yolo26n_openvino_model/` | ✅ | `imgsz`, `quantize`, `dynamic`, `nms`:material-information-outline:{ title="conf, iou, agnostic_nms are also available when nms=True" }, `batch`, `data`, `fraction`, `device` |
| [TensorRT](../integrations/tensorrt.md) | `engine` | `yolo26n.engine` | ✅ | `imgsz`, `quantize`, `dynamic`, `simplify`, `opset`, `workspace`, `nms`:material-information-outline:{ title="conf, iou, agnostic_nms are also available when nms=True" }, `batch`, `data`, `fraction`, `device` |
| [CoreML](../integrations/coreml.md) | `coreml` | `yolo26n.mlpackage` | ✅ | `imgsz`, `dynamic`, `quantize`, `nms`:material-information-outline:{ title="conf, iou, agnostic_nms are also available when nms=True" }, `batch`, `device` |
| [TF SavedModel](../integrations/tf-savedmodel.md) | `saved_model` | `yolo26n_saved_model/` | ✅ | `imgsz`, `keras`, `quantize`, `opset`, `nms`:material-information-outline:{ title="conf, iou, agnostic_nms are also available when nms=True" }, `batch`, `data`, `fraction`, `device` |
| [TF GraphDef](../integrations/tf-graphdef.md) | `pb` | `yolo26n.pb` | ❌ | `imgsz`, `opset`, `batch`, `device` |
| [TF Edge TPU](../integrations/edge-tpu.md) | `edgetpu` | `yolo26n_edgetpu.tflite` | ✅ | `imgsz`, `quantize`, `opset`, `data`, `fraction`, `device` |
| [PaddlePaddle](../integrations/paddlepaddle.md) | `paddle` | `yolo26n_paddle_model/` | ✅ | `imgsz`, `batch`, `device` |
| [MNN](../integrations/mnn.md) | `mnn` | `yolo26n.mnn` | ✅ | `imgsz`, `batch`, `dynamic`, `quantize`, `simplify`, `opset`, `nms`:material-information-outline:{ title="conf, iou, agnostic_nms are also available when nms=True" }, `device` |
| [NCNN](../integrations/ncnn.md) | `ncnn` | `yolo26n_ncnn_model/` | ✅ | `imgsz`, `quantize`, `batch`, `device` |
| [IMX500](../integrations/sony-imx500.md):material-information-outline:{ title="IMX format is currently only supported for YOLOv8n, YOLO11n models" } | `imx` | `yolo26n_imx_model/` | ✅ | `imgsz`, `quantize`, `data`, `fraction`, `nms`:material-information-outline:{ title="conf, iou, agnostic_nms are also available when nms=True" }, `device` |
| [RKNN](../integrations/rockchip-rknn.md) | `rknn` | `yolo26n_rknn_model/` | ✅ | `imgsz`, `batch`, `name`, `quantize`, `simplify`, `opset`, `data`, `fraction`, `device` |
| [ExecuTorch](../integrations/executorch.md) | `executorch` | `yolo26n_executorch_model/` | ✅ | `imgsz`, `batch`, `device` |
| [Axelera](../integrations/axelera.md) | `axelera` | `yolo26n_axelera_model/` | ✅ | `imgsz`, `batch`, `quantize`, `data`, `fraction`, `device` |
| [DEEPX](../integrations/deepx.md) | `deepx` | `yolo26n_deepx_model/` | ✅ | `imgsz`, `quantize`, `simplify`, `opset`, `data`, `optimize`, `device` |
| [Qualcomm QNN](../integrations/qnn.md) | `qnn` | `yolo26n_qnn.onnx` | ✅ | `imgsz`, `batch`, `name`, `quantize`, `simplify`, `opset`, `data`, `fraction`, `device` |
| [LiteRT](../integrations/litert.md) | `litert` | `yolo26n.tflite` | ✅ | `imgsz`, `quantize`, `batch`, `data`, `fraction`, `device` |
| [Hailo](../integrations/hailo.md) | `hailo` | `yolo26n_hailo_model/` | ✅ | `imgsz`, `name`, `quantize`, `data`, `fraction`, `simplify`, `conf`, `iou` |
| [Huawei Ascend](../integrations/ascend.md) | `ascend` | `yolo26n_ascend_model/` | ✅ | `imgsz`, `batch`, `name`, `quantize`, `opset`, `simplify`, `nms`:material-information-outline:{ title="conf, iou, agnostic_nms are also available when nms=True" } |

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

Ultralytics YOLO26 offers various pretrained models for [object detection](detect.md), [instance segmentation](segment.md), [semantic segmentation](semantic.md), and [pose estimation](pose.md). These models are pretrained on the COCO dataset, Cityscapes for semantic segmentation, or ImageNet for classification tasks. Here are some of the available models:

- [YOLO26n](https://platform.ultralytics.com/ultralytics/yolo26/yolo26n)
- [YOLO26s](https://platform.ultralytics.com/ultralytics/yolo26/yolo26s)
- [YOLO26m](https://platform.ultralytics.com/ultralytics/yolo26/yolo26m)
- [YOLO26l](https://platform.ultralytics.com/ultralytics/yolo26/yolo26l)
- [YOLO26x](https://platform.ultralytics.com/ultralytics/yolo26/yolo26x)

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

Ultralytics YOLO26 is designed to offer state-of-the-art performance for [object detection](detect.md), [instance segmentation](segment.md), [semantic segmentation](semantic.md), and [pose estimation](pose.md). Here are some key advantages:

1. **Pretrained Models**: Utilize models pretrained on popular datasets like [COCO](../datasets/detect/coco.md) and [ImageNet](../datasets/classify/imagenet.md) for faster development.
2. **High Accuracy**: Achieves impressive mAP scores, ensuring reliable object detection.
3. **Speed**: Optimized for [real-time inference](https://www.ultralytics.com/glossary/real-time-inference), making it ideal for applications requiring swift processing.
4. **Flexibility**: Export models to various formats like ONNX and TensorRT for deployment across multiple platforms.

Explore our [Blog](https://www.ultralytics.com/blog) for use cases and success stories showcasing YOLO26 in action.
