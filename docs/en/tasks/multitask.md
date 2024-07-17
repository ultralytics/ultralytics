---
comments: true
description: Learn how to use instance segmentation models with Ultralytics YOLO. Instructions on training, validation, image prediction, and model export.
keywords: yolov8, instance segmentation, pose estimation, keypoints detection, Ultralytics, COCO dataset, image segmentation, object detection, model training, model validation, image prediction, model export
---

# Multitask

<!---
<img width="1024" src="multitask.png" alt="Multitask examples">
-->

The multitask model combines the capabilities of instance segmentation and pose estimation to provide a comprehensive understanding of objects in an image.
This model is designed to not only identify and classify objects within a scene but also to understand their precise shape and the specific parts that make up each object.

The output of the multitask model is a set of masks or contours that outline each object in the image as well as a set of points that represent the keypoints on an object in the image along with class labels and confidence scores for each object.
The keypoints can represent various parts of the object such as joints, landmarks, or other distinctive features. The locations of the keypoints are usually represented as a set of 3D `[x, y, visible]` coordinates.

Multitask is usefull when you need to know the exact shape of objects in an image as well as identify specific parts of some objects, and their location in relation to each other.

<!---
<p align="center">
  <br>
  <iframe width="720" height="405" src=""
    title="YouTube video player" frameborder="0"
    allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share"
    allowfullscreen>
  </iframe>
  <br>
  <strong>Watch:</strong> Run Multitask with Pre-Trained Ultralytics YOLOv8 Model in Python.
</p>
-->

<!--
!!! Tip "Tip"

    YOLOv8 Segment models use the `-multitask` suffix, i.e. `yolov8n-multitask.pt` and are pretrained on [COCO](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/datasets/coco.yaml).

-->

<!--
## [Models](https://github.com/ultralytics/ultralytics/tree/main/ultralytics/cfg/models/v8)

YOLOv8 pretrained Segment models are shown here. Detect, Segment and Pose models are pretrained on the [COCO](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/datasets/coco.yaml) dataset, while Classify models are pretrained on the [ImageNet](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/datasets/ImageNet.yaml) dataset.

[Models](https://github.com/ultralytics/ultralytics/tree/main/ultralytics/cfg/models) download automatically from the latest Ultralytics [release](https://github.com/ultralytics/assets/releases) on first use.

| Model                                                                                        | size<br><sup>(pixels) | mAP<sup>box<br>50-95 | mAP<sup>mask<br>50-95 | Speed<br><sup>CPU ONNX<br>(ms) | Speed<br><sup>A100 TensorRT<br>(ms) | params<br><sup>(M) | FLOPs<br><sup>(B) |
|----------------------------------------------------------------------------------------------|-----------------------|----------------------|-----------------------|--------------------------------|-------------------------------------|--------------------|-------------------|
| [YOLOv8n-multitask](https://github.com/ultralytics/assets/releases/download/v8.1.0/yolov8n-multitask.pt) |            |                      |                      |                                |                                      |                    |                  |
| [YOLOv8s-multitask](https://github.com/ultralytics/assets/releases/download/v8.1.0/yolov8s-multitask.pt) |            |                      |                      |                                |                                      |                    |                  |
| [YOLOv8m-multitask](https://github.com/ultralytics/assets/releases/download/v8.1.0/yolov8m-multitask.pt) |            |                      |                      |                                |                                      |                    |                  |
| [YOLOv8l-multitask](https://github.com/ultralytics/assets/releases/download/v8.1.0/yolov8l-multitask.pt) |            |                      |                      |                                |                                      |                    |                  |
| [YOLOv8x-multitask](https://github.com/ultralytics/assets/releases/download/v8.1.0/yolov8x-multitask.pt) |            |                      |                      |                                |                                      |                    |                  |

- **mAP<sup>val</sup>** values are for single-model single-scale on [COCO val2017](http://cocodataset.org) dataset. <br>Reproduce by `yolo val segment data=coco.yaml device=0`
- **Speed** averaged over COCO val images using an [Amazon EC2 P4d](https://aws.amazon.com/ec2/instance-types/p4/) instance. <br>Reproduce by `yolo val segment data=coco128-seg.yaml batch=1 device=0|cpu`
-->

## Train

Train YOLOv8n-multitask on the COCO8-multitask dataset for 100 epochs at image size 640. For a full list of available arguments see the [Configuration](../usage/cfg.md) page.

!!! Example

    === "Python"

        ```python
        from ultralytics import YOLO

        # Load a model
        model = YOLO("yolov8n-multitask.yaml")  # build a new model from YAML
        model = YOLO("yolov8n-multitask.pt")  # load a pretrained model (recommended for training)
        model = YOLO("yolov8n-multitask.yaml").load("yolov8n.pt")  # build from YAML and transfer weights

        # Train the model
        results = model.train(data="coco8-multitask.yaml", epochs=100, imgsz=640)
        ```
    === "CLI"

        ```bash
        # Build a new model from YAML and start training from scratch
        yolo multitask train data=coco8-multitask.yaml model=yolov8n-multitask.yaml epochs=100 imgsz=640

        # Start training from a pretrained *.pt model
        yolo multitask train data=coco8-multitask.yaml model=yolov8n-multitask.pt epochs=100 imgsz=640

        # Build a new model from YAML, transfer pretrained weights to it and start training
        yolo multitask train data=coco8-multitask.yaml model=yolov8n-multitask.yaml epochs=100 imgsz=640
        ```

### Dataset format

YOLO multitask dataset format can be found in detail in the [Dataset Guide](../datasets/multitask/index.md).
To convert your existing dataset from other formats (like COCO etc.) to YOLO format, please use [JSON2YOLO](https://github.com/ultralytics/JSON2YOLO) tool by Ultralytics.
To combine a keypoint detection and a instance segmentation datasets in YOLO format to a multitask dataset, please use [merge_kpt_seg.py](https://github.com/ultralytics/ultralytics/tree/main/ultralytics/data/scripts/merge_kpt_seg.py) tool.

## Val

Validate trained YOLOv8n-multitask model accuracy on the COCO8-pose dataset. No argument need to passed as the `model`
retains it's training `data` and arguments as model attributes.

!!! Example

    === "Python"

        ```python
        from ultralytics import YOLO

        # Load a model
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
        yolo multitask val model=path/to/best.pt  # val custom model
        ```

## Predict

Use a trained YOLOv8n-seg model to run predictions on images.

!!! Example

    === "Python"

        ```python
        from ultralytics import YOLO

        # Load a model
        model = YOLO("path/to/best.pt")  # load a custom model

        # Predict with the model
        results = model("https://ultralytics.com/images/bus.jpg")  # predict on an image
        ```
    === "CLI"

        ```bash
        yolo multitask predict model=path/to/best.pt source='https://ultralytics.com/images/bus.jpg'  # predict with custom model
        ```

See full `predict` mode details in the [Predict](https://docs.ultralytics.com/modes/predict/) page.

## Export

Export a YOLOv8n-seg model to a different format like ONNX, CoreML, etc.

!!! Example

    === "Python"

        ```python
        from ultralytics import YOLO

        # Load a model
        model = YOLO("path/to/best.pt")  # load a custom trained model

        # Export the model
        model.export(format="onnx")
        ```
    === "CLI"

        ```bash
        yolo export model=path/to/best.pt format=onnx  # export custom trained model
        ```

Tested YOLOv8-multitask export formats are in the table below. You can predict or validate directly on exported models, i.e. `yolo predict model=yolov8n-multitask.onnx`. Usage examples are shown for your model after export completes.

| Format                          | `format` Argument | Model                    | Metadata | Arguments                                       |
| ------------------------------- | ----------------- | ------------------------ | -------- | ----------------------------------------------- |
| [PyTorch](https://pytorch.org/) | -                 | `yolov8n-multitask.pt`   | ✅       | -                                               |
| [ONNX](https://onnx.ai/)        | `onnx`            | `yolov8n-multitask.onnx` | ✅       | `imgsz`, `half`, `dynamic`, `simplify`, `opset` |

<!--
| [TorchScript](https://pytorch.org/docs/stable/jit.html)            | `torchscript`     | `yolov8n-multitask.torchscript`     | ✅        | `imgsz`, `optimize`                                 |
| [OpenVINO](https://docs.openvino.ai/latest/index.html)             | `openvino`        | `yolov8n-multitask/` | ✅        | `imgsz`, `half`                                     |
| [TensorRT](https://developer.nvidia.com/tensorrt)                  | `engine`          | `yolov8n-multitask.engine`          | ✅        | `imgsz`, `half`, `dynamic`, `simplify`, `workspace` |
| [CoreML](https://github.com/apple/coremltools)                     | `coreml`          | `yolov8n-multitask.mlpackage`       | ✅        | `imgsz`, `half`, `int8`, `nms`                      |
| [TF SavedModel](https://www.tensorflow.org/guide/saved_model)      | `saved_model`     | `yolov8n-multitask/`    | ✅        | `imgsz`, `keras`                                    |
| [TF GraphDef](https://www.tensorflow.org/api_docs/python/tf/Graph) | `pb`              | `yolov8n-multitask.pb`              | ❌        | `imgsz`                                             |
| [TF Lite](https://www.tensorflow.org/lite)                         | `tflite`          | `yolov8n-multitask.tflite`          | ✅        | `imgsz`, `half`, `int8`                             |
| [TF Edge TPU](https://coral.ai/docs/edgetpu/models-intro/)         | `edgetpu`         | `yolov8n-multitask_edgetpu.tflite`  | ✅        | `imgsz`                                             |
| [TF.js](https://www.tensorflow.org/js)                             | `tfjs`            | `yolov8n-multitask_web_model/`      | ✅        | `imgsz`, `half`, `int8`                             |
| [PaddlePaddle](https://github.com/PaddlePaddle)                    | `paddle`          | `yolov8n-multitask_paddle_model/`   | ✅        | `imgsz`                                             |
| [ncnn](https://github.com/Tencent/ncnn)                            | `ncnn`            | `yolov8n-multitask_ncnn_model/`     | ✅        | `imgsz`, `half`                                     |
-->

See full `export` details in the [Export](https://docs.ultralytics.com/modes/export/) page.
