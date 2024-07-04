---
comments: true
description: Discover how to use YOLOv8 for pose estimation tasks. Learn about model training, validation, prediction, and exporting in various formats.
keywords: pose estimation, YOLOv8, Ultralytics, keypoints, model training, image recognition, deep learning
---

# Pose Estimation

<img width="1024" src="https://user-images.githubusercontent.com/26833433/243418616-9811ac0b-a4a7-452a-8aba-484ba32bb4a8.png" alt="Pose estimation examples">

Pose estimation is a task that involves identifying the location of specific points in an image, usually referred to as keypoints. The keypoints can represent various parts of the object such as joints, landmarks, or other distinctive features. The locations of the keypoints are usually represented as a set of 2D `[x, y]` or 3D `[x, y, visible]` coordinates.

The output of a pose estimation model is a set of points that represent the keypoints on an object in the image, usually along with the confidence scores for each point. Pose estimation is a good choice when you need to identify specific parts of an object in a scene, and their location in relation to each other.

<table>
  <tr>
    <td align="center">
      <iframe loading="lazy" width="720" height="405" src="https://www.youtube.com/embed/Y28xXQmju64?si=pCY4ZwejZFu6Z4kZ"
        title="YouTube video player" frameborder="0"
        allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share"
        allowfullscreen>
      </iframe>
      <br>
      <strong>Watch:</strong> Pose Estimation with Ultralytics YOLOv8.
    </td>
    <td align="center">
      <iframe loading="lazy" width="720" height="405" src="https://www.youtube.com/embed/aeAX6vWpfR0"
        title="YouTube video player" frameborder="0"
        allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share"
        allowfullscreen>
      </iframe>
      <br>
      <strong>Watch:</strong> Pose Estimation with Ultralytics HUB.
    </td>
  </tr>
</table>

!!! Tip "Tip"

    YOLOv8 _pose_ models use the `-pose` suffix, i.e. `yolov8n-pose.pt`. These models are trained on the [COCO keypoints](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/datasets/coco-pose.yaml) dataset and are suitable for a variety of pose estimation tasks.

## [Models](https://github.com/ultralytics/ultralytics/tree/main/ultralytics/cfg/models/v8)

YOLOv8 pretrained Pose models are shown here. Detect, Segment and Pose models are pretrained on the [COCO](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/datasets/coco.yaml) dataset, while Classify models are pretrained on the [ImageNet](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/datasets/ImageNet.yaml) dataset.

[Models](https://github.com/ultralytics/ultralytics/tree/main/ultralytics/cfg/models) download automatically from the latest Ultralytics [release](https://github.com/ultralytics/assets/releases) on first use.

| Model                                                                                                | size<br><sup>(pixels) | mAP<sup>pose<br>50-95 | mAP<sup>pose<br>50 | Speed<br><sup>CPU ONNX<br>(ms) | Speed<br><sup>A100 TensorRT<br>(ms) | params<br><sup>(M) | FLOPs<br><sup>(B) |
| ---------------------------------------------------------------------------------------------------- | --------------------- | --------------------- | ------------------ | ------------------------------ | ----------------------------------- | ------------------ | ----------------- |
| [YOLOv8n-pose](https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov8n-pose.pt)       | 640                   | 50.4                  | 80.1               | 131.8                          | 1.18                                | 3.3                | 9.2               |
| [YOLOv8s-pose](https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov8s-pose.pt)       | 640                   | 60.0                  | 86.2               | 233.2                          | 1.42                                | 11.6               | 30.2              |
| [YOLOv8m-pose](https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov8m-pose.pt)       | 640                   | 65.0                  | 88.8               | 456.3                          | 2.00                                | 26.4               | 81.0              |
| [YOLOv8l-pose](https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov8l-pose.pt)       | 640                   | 67.6                  | 90.0               | 784.5                          | 2.59                                | 44.4               | 168.6             |
| [YOLOv8x-pose](https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov8x-pose.pt)       | 640                   | 69.2                  | 90.2               | 1607.1                         | 3.73                                | 69.4               | 263.2             |
| [YOLOv8x-pose-p6](https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov8x-pose-p6.pt) | 1280                  | 71.6                  | 91.2               | 4088.7                         | 10.04                               | 99.1               | 1066.4            |

- **mAP<sup>val</sup>** values are for single-model single-scale on [COCO Keypoints val2017](https://cocodataset.org) dataset. <br>Reproduce by `yolo val pose data=coco-pose.yaml device=0`
- **Speed** averaged over COCO val images using an [Amazon EC2 P4d](https://aws.amazon.com/ec2/instance-types/p4/) instance. <br>Reproduce by `yolo val pose data=coco8-pose.yaml batch=1 device=0|cpu`

## Train

Train a YOLOv8-pose model on the COCO128-pose dataset.

!!! Example

    === "Python"

        ```python
        from ultralytics import YOLO

        # Load a model
        model = YOLO("yolov8n-pose.yaml")  # build a new model from YAML
        model = YOLO("yolov8n-pose.pt")  # load a pretrained model (recommended for training)
        model = YOLO("yolov8n-pose.yaml").load("yolov8n-pose.pt")  # build from YAML and transfer weights

        # Train the model
        results = model.train(data="coco8-pose.yaml", epochs=100, imgsz=640)
        ```

    === "CLI"

        ```bash
        # Build a new model from YAML and start training from scratch
        yolo pose train data=coco8-pose.yaml model=yolov8n-pose.yaml epochs=100 imgsz=640

        # Start training from a pretrained *.pt model
        yolo pose train data=coco8-pose.yaml model=yolov8n-pose.pt epochs=100 imgsz=640

        # Build a new model from YAML, transfer pretrained weights to it and start training
        yolo pose train data=coco8-pose.yaml model=yolov8n-pose.yaml pretrained=yolov8n-pose.pt epochs=100 imgsz=640
        ```

### Dataset format

YOLO pose dataset format can be found in detail in the [Dataset Guide](../datasets/pose/index.md). To convert your existing dataset from other formats (like COCO etc.) to YOLO format, please use [JSON2YOLO](https://github.com/ultralytics/JSON2YOLO) tool by Ultralytics.

## Val

Validate trained YOLOv8n-pose model accuracy on the COCO128-pose dataset. No argument need to passed as the `model`
retains its training `data` and arguments as model attributes.

!!! Example

    === "Python"

        ```python
        from ultralytics import YOLO

        # Load a model
        model = YOLO("yolov8n-pose.pt")  # load an official model
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
        yolo pose val model=yolov8n-pose.pt  # val official model
        yolo pose val model=path/to/best.pt  # val custom model
        ```

## Predict

Use a trained YOLOv8n-pose model to run predictions on images.

!!! Example

    === "Python"

        ```python
        from ultralytics import YOLO

        # Load a model
        model = YOLO("yolov8n-pose.pt")  # load an official model
        model = YOLO("path/to/best.pt")  # load a custom model

        # Predict with the model
        results = model("https://ultralytics.com/images/bus.jpg")  # predict on an image
        ```
    === "CLI"

        ```bash
        yolo pose predict model=yolov8n-pose.pt source='https://ultralytics.com/images/bus.jpg'  # predict with official model
        yolo pose predict model=path/to/best.pt source='https://ultralytics.com/images/bus.jpg'  # predict with custom model
        ```

See full `predict` mode details in the [Predict](../modes/predict.md) page.

## Export

Export a YOLOv8n Pose model to a different format like ONNX, CoreML, etc.

!!! Example

    === "Python"

        ```python
        from ultralytics import YOLO

        # Load a model
        model = YOLO("yolov8n-pose.pt")  # load an official model
        model = YOLO("path/to/best.pt")  # load a custom trained model

        # Export the model
        model.export(format="onnx")
        ```
    === "CLI"

        ```bash
        yolo export model=yolov8n-pose.pt format=onnx  # export official model
        yolo export model=path/to/best.pt format=onnx  # export custom trained model
        ```

Available YOLOv8-pose export formats are in the table below. You can export to any format using the `format` argument, i.e. `format='onnx'` or `format='engine'`. You can predict or validate directly on exported models, i.e. `yolo predict model=yolov8n-pose.onnx`. Usage examples are shown for your model after export completes.

| Format                                            | `format` Argument | Model                          | Metadata | Arguments                                                            |
| ------------------------------------------------- | ----------------- | ------------------------------ | -------- | -------------------------------------------------------------------- |
| [PyTorch](https://pytorch.org/)                   | -                 | `yolov8n-pose.pt`              | ✅       | -                                                                    |
| [TorchScript](../integrations/torchscript.md)     | `torchscript`     | `yolov8n-pose.torchscript`     | ✅       | `imgsz`, `optimize`, `batch`                                         |
| [ONNX](../integrations/onnx.md)                   | `onnx`            | `yolov8n-pose.onnx`            | ✅       | `imgsz`, `half`, `dynamic`, `simplify`, `opset`, `batch`             |
| [OpenVINO](../integrations/openvino.md)           | `openvino`        | `yolov8n-pose_openvino_model/` | ✅       | `imgsz`, `half`, `int8`, `batch`                                     |
| [TensorRT](../integrations/tensorrt.md)           | `engine`          | `yolov8n-pose.engine`          | ✅       | `imgsz`, `half`, `dynamic`, `simplify`, `workspace`, `int8`, `batch` |
| [CoreML](../integrations/coreml.md)               | `coreml`          | `yolov8n-pose.mlpackage`       | ✅       | `imgsz`, `half`, `int8`, `nms`, `batch`                              |
| [TF SavedModel](../integrations/tf-savedmodel.md) | `saved_model`     | `yolov8n-pose_saved_model/`    | ✅       | `imgsz`, `keras`, `int8`, `batch`                                    |
| [TF GraphDef](../integrations/tf-graphdef.md)     | `pb`              | `yolov8n-pose.pb`              | ❌       | `imgsz`, `batch`                                                     |
| [TF Lite](../integrations/tflite.md)              | `tflite`          | `yolov8n-pose.tflite`          | ✅       | `imgsz`, `half`, `int8`, `batch`                                     |
| [TF Edge TPU](../integrations/edge-tpu.md)        | `edgetpu`         | `yolov8n-pose_edgetpu.tflite`  | ✅       | `imgsz`                                                              |
| [TF.js](../integrations/tfjs.md)                  | `tfjs`            | `yolov8n-pose_web_model/`      | ✅       | `imgsz`, `half`, `int8`, `batch`                                     |
| [PaddlePaddle](../integrations/paddlepaddle.md)   | `paddle`          | `yolov8n-pose_paddle_model/`   | ✅       | `imgsz`, `batch`                                                     |
| [NCNN](../integrations/ncnn.md)                   | `ncnn`            | `yolov8n-pose_ncnn_model/`     | ✅       | `imgsz`, `half`, `batch`                                             |

See full `export` details in the [Export](../modes/export.md) page.

## FAQ

### What is pose estimation with Ultralytics YOLOv8?

Pose estimation with Ultralytics YOLOv8 involves identifying the locations of specific keypoints in an image. These keypoints can represent parts of the human body, such as joints or landmarks. The outputs are usually 2D `[x, y]` coordinates or 3D `[x, y, visible]` coordinates. YOLOv8-pose models, like `yolov8n-pose.pt`, are trained on the [COCO keypoints](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/datasets/coco-pose.yaml) dataset and provide robust performance for various pose estimation tasks.

### How can I train a YOLOv8-pose model on my custom dataset?

To train a YOLOv8-pose model on your custom dataset, you need to follow these steps:

1. Prepare your dataset in the YOLO format. If your dataset is in a different format (like COCO), convert it using the [JSON2YOLO](https://github.com/ultralytics/JSON2YOLO) tool.
2. Load a model and train it using the `yolo pose train` command or the Python API:

**Python:**

```python
from ultralytics import YOLO

model = YOLO("yolov8n-pose.pt")
results = model.train(data="path/to/your-dataset.yaml", epochs=100, imgsz=640)
```

**CLI:**

```bash
yolo pose train data=path/to/your-dataset.yaml model=yolov8n-pose.pt epochs=100 imgsz=640
```

For more details on training, check the [Dataset Guide](../datasets/pose/index.md).

### How do I validate a trained YOLOv8-pose model?

Validating a trained YOLOv8-pose model can be performed using the `yolo pose val` command or the Python API:

**Python:**

```python
from ultralytics import YOLO

model = YOLO("path/to/best.pt")
metrics = model.val()
print(metrics.box.map)  # map50-95
```

**CLI:**

```bash
yolo pose val model=path/to/best.pt
```

Validation ensures your model's accuracy and helps identify areas for improvement. For detailed validation steps, visit the [Val](#val) section.

### How can I use a YOLOv8-pose model for making predictions?

You can use a YOLOv8-pose model to make predictions with the following commands:

**Python:**

```python
from ultralytics import YOLO

model = YOLO("yolov8n-pose.pt")
results = model("https://ultralytics.com/images/bus.jpg")
```

**CLI:**

```bash
yolo pose predict model=yolov8n-pose.pt source="https://ultralytics.com/images/bus.jpg"
```

Prediction enables you to apply your trained model to new images and visualize the keypoint detections. For more details, see the [Predict](../modes/predict.md) page.

### How can I export a YOLOv8-pose model to different formats?

To export a YOLOv8-pose model to formats such as ONNX or TensorRT, use the `yolo export` command or the Python API:

**Python:**

```python
from ultralytics import YOLO

model = YOLO("yolov8n-pose.pt")
model.export(format="onnx")
```

**CLI:**

```bash
yolo export model=yolov8n-pose.pt format=onnx
```

Exporting models facilitates deployment to different platforms. For more information on supported export formats, see the [Export](../modes/export.md) page.
