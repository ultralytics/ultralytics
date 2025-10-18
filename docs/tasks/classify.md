---
comments: true
description: Learn about YOLOv8 Classify models for image classification. Get detailed information on List of Pretrained Models & how to Train, Validate, Predict & Export models.
keywords: Ultralytics, YOLOv8, Image Classification, Pretrained Models, YOLOv8n-cls, Training, Validation, Prediction, Model Export
---

# Image Classification

<img width="1024" src="https://user-images.githubusercontent.com/26833433/243418606-adf35c62-2e11-405d-84c6-b84e7d013804.png">

Image classification is the simplest of the three tasks and involves classifying an entire image into one of a set of predefined classes.

The output of an image classifier is a single class label and a confidence score. Image classification is useful when you need to know only what class an image belongs to and don't need to know where objects of that class are located or what their exact shape is.

!!! tip "Tip"

    YOLOv8 Classify models use the `-cls` suffix, i.e. `yolov8n-cls.pt` and are pretrained on [ImageNet](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/datasets/ImageNet.yaml).

## [Models](https://github.com/ultralytics/ultralytics/tree/main/ultralytics/cfg/models/v8)

YOLOv8 pretrained Classify models are shown here. Detect, Segment and Pose models are pretrained on the [COCO](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/datasets/coco.yaml) dataset, while Classify models are pretrained on the [ImageNet](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/datasets/ImageNet.yaml) dataset.

[Models](https://github.com/ultralytics/ultralytics/tree/main/ultralytics/cfg/models) download automatically from the latest Ultralytics [release](https://github.com/ultralytics/assets/releases) on first use.

| Model                                                                                        | size<br><sup>(pixels) | acc<br><sup>top1 | acc<br><sup>top5 | Speed<br><sup>CPU ONNX<br>(ms) | Speed<br><sup>A100 TensorRT<br>(ms) | params<br><sup>(M) | FLOPs<br><sup>(B) at 640 |
| -------------------------------------------------------------------------------------------- | --------------------- | ---------------- | ---------------- | ------------------------------ | ----------------------------------- | ------------------ | ------------------------ |
| [YOLOv8n-cls](https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n-cls.pt) | 224                   | 66.6             | 87.0             | 12.9                           | 0.31                                | 2.7                | 4.3                      |
| [YOLOv8s-cls](https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8s-cls.pt) | 224                   | 72.3             | 91.1             | 23.4                           | 0.35                                | 6.4                | 13.5                     |
| [YOLOv8m-cls](https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8m-cls.pt) | 224                   | 76.4             | 93.2             | 85.4                           | 0.62                                | 17.0               | 42.7                     |
| [YOLOv8l-cls](https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8l-cls.pt) | 224                   | 78.0             | 94.1             | 163.0                          | 0.87                                | 37.5               | 99.7                     |
| [YOLOv8x-cls](https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8x-cls.pt) | 224                   | 78.4             | 94.3             | 232.0                          | 1.01                                | 57.4               | 154.8                    |

- **acc** values are model accuracies on the [ImageNet](https://www.image-net.org/) dataset validation set.
  <br>Reproduce by `yolo val classify data=path/to/ImageNet device=0`
- **Speed** averaged over ImageNet val images using an [Amazon EC2 P4d](https://aws.amazon.com/ec2/instance-types/p4/)
  instance.
  <br>Reproduce by `yolo val classify data=path/to/ImageNet batch=1 device=0|cpu`

## Train

Train YOLOv8n-cls on the MNIST160 dataset for 100 epochs at image size 64. For a full list of available arguments see the [Configuration](../usage/cfg.md) page.

!!! example ""

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

Validate trained YOLOv8n-cls model accuracy on the MNIST160 dataset. No argument need to passed as the `model` retains it's training `data` and arguments as model attributes.

!!! example ""

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
        yolo classify val model=path/to/best.pt # val custom model
        ```

## Predict

Use a trained YOLOv8n-cls model to run predictions on images.

!!! example ""

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
        yolo classify predict model=path/to/best.pt source='https://ultralytics.com/images/bus.jpg' # predict with custom model
        ```

See full `predict` mode details in the [Predict](https://docs.ultralytics.com/modes/predict/) page.

## Export

Export a YOLOv8n-cls model to a different format like ONNX, CoreML, etc.

!!! example ""

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
        yolo export model=path/to/best.pt format=onnx # export custom trained model
        ```

Available YOLOv8-cls export formats are in the table below. You can predict or validate directly on exported models, i.e. `yolo predict model=yolov8n-cls.onnx`. Usage examples are shown for your model after export completes.

| Format                                                             | `format` Argument | Model                         | Metadata | Arguments                                           |
| ------------------------------------------------------------------ | ----------------- | ----------------------------- | -------- | --------------------------------------------------- |
| [PyTorch](https://pytorch.org/)                                    | -                 | `yolov8n-cls.pt`              | ✅       | -                                                   |
| [TorchScript](https://pytorch.org/docs/stable/jit.html)            | `torchscript`     | `yolov8n-cls.torchscript`     | ✅       | `imgsz`, `optimize`                                 |
| [ONNX](https://onnx.ai/)                                           | `onnx`            | `yolov8n-cls.onnx`            | ✅       | `imgsz`, `half`, `dynamic`, `simplify`, `opset`     |
| [OpenVINO](https://docs.openvino.ai/latest/index.html)             | `openvino`        | `yolov8n-cls_openvino_model/` | ✅       | `imgsz`, `half`                                     |
| [TensorRT](https://developer.nvidia.com/tensorrt)                  | `engine`          | `yolov8n-cls.engine`          | ✅       | `imgsz`, `half`, `dynamic`, `simplify`, `workspace` |
| [CoreML](https://github.com/apple/coremltools)                     | `coreml`          | `yolov8n-cls.mlpackage`       | ✅       | `imgsz`, `half`, `int8`, `nms`                      |
| [TF SavedModel](https://www.tensorflow.org/guide/saved_model)      | `saved_model`     | `yolov8n-cls_saved_model/`    | ✅       | `imgsz`, `keras`                                    |
| [TF GraphDef](https://www.tensorflow.org/api_docs/python/tf/Graph) | `pb`              | `yolov8n-cls.pb`              | ❌       | `imgsz`                                             |
| [TF Lite](https://www.tensorflow.org/lite)                         | `tflite`          | `yolov8n-cls.tflite`          | ✅       | `imgsz`, `half`, `int8`                             |
| [TF Edge TPU](https://coral.ai/docs/edgetpu/models-intro/)         | `edgetpu`         | `yolov8n-cls_edgetpu.tflite`  | ✅       | `imgsz`                                             |
| [TF.js](https://www.tensorflow.org/js)                             | `tfjs`            | `yolov8n-cls_web_model/`      | ✅       | `imgsz`                                             |
| [PaddlePaddle](https://github.com/PaddlePaddle)                    | `paddle`          | `yolov8n-cls_paddle_model/`   | ✅       | `imgsz`                                             |
| [ncnn](https://github.com/Tencent/ncnn)                            | `ncnn`            | `yolov8n-cls_ncnn_model/`     | ✅       | `imgsz`, `half`                                     |

See full `export` details in the [Export](https://docs.ultralytics.com/modes/export/) page.
