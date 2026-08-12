---
title: Pose Estimation with Ultralytics YOLO
comments: true
description: Discover how to use YOLO26 for pose estimation tasks. Learn about model training, validation, prediction, and exporting in various formats.
keywords: pose estimation, YOLO26, Ultralytics, keypoints, model training, image recognition, deep learning, human pose detection, computer vision, real-time tracking
model_name: yolo26n-pose
---

# Pose Estimation with Ultralytics YOLO {#pose-estimation}

<img width="1024" src="https://cdn.jsdelivr.net/gh/ultralytics/assets@main/docs/pose-estimation-examples.avif" alt="Ultralytics YOLO pose estimation with human body keypoint detection">

Pose estimation is a task that involves identifying the location of specific points in an image, usually referred to as keypoints. The keypoints can represent various parts of the object such as joints, landmarks, or other distinctive features. The locations of the keypoints are usually represented as a set of 2D `[x, y]` coordinates, optionally with a visibility flag `[x, y, visible]`.

The output of a pose estimation model is a set of points that represent the keypoints on an object in the image, usually along with the confidence scores for each point. Pose estimation is a good choice when you need to identify specific parts of an object in a scene, and their location in relation to each other.

<p align="center">
  <br>
  <iframe loading="lazy" width="720" height="405" src="https://www.youtube.com/embed/4VTuqfrOIws"
    title="YouTube video player" frameborder="0"
    allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share"
    allowfullscreen>
  </iframe>
  <br>
  <strong>Watch:</strong> How to Run Real-Time Pose Estimation with Ultralytics YOLO26 | Tracking & Keypoints Extraction 🕺
</p>

!!! tip

    YOLO26 _pose_ models use the `-pose` suffix, i.e., `yolo26n-pose.pt`. These models are trained on the [COCO keypoints](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/datasets/coco-pose.yaml) dataset and are suitable for a variety of pose estimation tasks.

    In the default YOLO26 pose model, there are 17 keypoints, each representing a different part of the human body. Here is the mapping of each index to its respective body joint:

    0. Nose
    1. Left Eye
    2. Right Eye
    3. Left Ear
    4. Right Ear
    5. Left Shoulder
    6. Right Shoulder
    7. Left Elbow
    8. Right Elbow
    9. Left Wrist
    10. Right Wrist
    11. Left Hip
    12. Right Hip
    13. Left Knee
    14. Right Knee
    15. Left Ankle
    16. Right Ankle

## [Models](https://github.com/ultralytics/ultralytics/tree/main/ultralytics/cfg/models/26)

Ultralytics YOLO26 pretrained Pose models are shown here. Detect, Segment and Pose models are pretrained on the [COCO](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/datasets/coco.yaml) dataset, [Semantic](semantic.md) models are pretrained on [Cityscapes](../datasets/semantic/cityscapes.md), and Classify models are pretrained on the [ImageNet](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/datasets/ImageNet.yaml) dataset.

[Models](https://github.com/ultralytics/ultralytics/tree/main/ultralytics/cfg/models) download automatically from the latest Ultralytics [release](https://github.com/ultralytics/assets/releases) on first use.

| Model                                                                            | size<br><sup>(pixels)</sup> | mAP<sup>pose<br>50-95(e2e)</sup> | mAP<sup>pose<br>50(e2e)</sup> | Speed<br><sup>CPU ONNX<br>(ms)</sup> | Speed<br><sup>T4 TensorRT10<br>(ms)</sup> | params<br><sup>(M)</sup> | FLOPs<br><sup>(B)</sup> |
| -------------------------------------------------------------------------------- | --------------------------- | -------------------------------- | ----------------------------- | ------------------------------------ | ----------------------------------------- | ------------------------ | ----------------------- |
| [YOLO26n-pose](https://platform.ultralytics.com/ultralytics/yolo26/yolo26n-pose) | 640                         | 57.2                             | 83.3                          | 40.3 ± 0.5                           | 1.8 ± 0.0                                 | 2.9                      | 7.5                     |
| [YOLO26s-pose](https://platform.ultralytics.com/ultralytics/yolo26/yolo26s-pose) | 640                         | 63.0                             | 86.6                          | 85.3 ± 0.9                           | 2.7 ± 0.0                                 | 10.4                     | 23.9                    |
| [YOLO26m-pose](https://platform.ultralytics.com/ultralytics/yolo26/yolo26m-pose) | 640                         | 68.8                             | 89.6                          | 218.0 ± 1.5                          | 5.0 ± 0.1                                 | 21.5                     | 73.1                    |
| [YOLO26l-pose](https://platform.ultralytics.com/ultralytics/yolo26/yolo26l-pose) | 640                         | 70.4                             | 90.5                          | 275.4 ± 2.4                          | 6.5 ± 0.1                                 | 25.9                     | 91.3                    |
| [YOLO26x-pose](https://platform.ultralytics.com/ultralytics/yolo26/yolo26x-pose) | 640                         | 71.6                             | 91.6                          | 565.4 ± 3.0                          | 12.2 ± 0.2                                | 57.6                     | 201.7                   |

- **mAP<sup>val</sup>** values are for single-model single-scale on [COCO Keypoints val2017](https://cocodataset.org/) dataset. <br>Reproduce by `yolo val pose data=coco-pose.yaml device=0`
- **Speed** averaged over COCO val images using an [Amazon EC2 P4d](https://aws.amazon.com/ec2/instance-types/p4/) instance. <br>Reproduce by `yolo val pose data=coco-pose.yaml batch=1 device=0|cpu`
- **Params** and **FLOPs** values are for the fused model after `model.fuse()`, which merges Conv and BatchNorm layers and, for end2end models, removes the auxiliary one-to-many detection head. Pretrained checkpoints retain the full training architecture and may show higher counts.

## Train

Train a YOLO26-pose model on the COCO8-pose dataset. The [COCO8-pose dataset](../datasets/pose/coco8-pose.md) is a small sample dataset that's perfect for testing and debugging your pose estimation models.

!!! example

    === "Python"

        ```python
        from ultralytics import YOLO

        # Load a model
        model = YOLO("yolo26n-pose.yaml")  # build a new model from YAML
        model = YOLO("yolo26n-pose.pt")  # load a pretrained model (recommended for training)
        model = YOLO("yolo26n-pose.yaml").load("yolo26n-pose.pt")  # build from YAML and transfer weights

        # Train the model
        results = model.train(data="coco8-pose.yaml", epochs=100, imgsz=640)
        ```

    === "CLI"

        ```bash
        # Build a new model from YAML and start training from scratch
        yolo pose train data=coco8-pose.yaml model=yolo26n-pose.yaml epochs=100 imgsz=640

        # Start training from a pretrained *.pt model
        yolo pose train data=coco8-pose.yaml model=yolo26n-pose.pt epochs=100 imgsz=640

        # Build a new model from YAML, transfer pretrained weights to it and start training
        yolo pose train data=coco8-pose.yaml model=yolo26n-pose.yaml pretrained=yolo26n-pose.pt epochs=100 imgsz=640
        ```

See full `train` mode details in the [Train](../modes/train.md) page. Pose models can also be trained with [Ultralytics Platform cloud training](../platform/train/cloud-training.md).

### Dataset format

YOLO pose dataset format can be found in detail in the [Dataset Guide](../datasets/pose/index.md). To convert your existing dataset from other formats (like [COCO](../datasets/pose/coco.md) etc.) to YOLO format, please use the [JSON2YOLO](https://github.com/ultralytics/JSON2YOLO) tool by Ultralytics. [Ultralytics Platform annotation](../platform/data/annotation.md) also supports pose labels with built-in skeleton templates for person, hand, face, and custom keypoint layouts.

For custom pose estimation tasks, you can also explore specialized datasets like [Tiger-Pose](../datasets/pose/tiger-pose.md) for animal pose estimation, [Hand Keypoints](../datasets/pose/hand-keypoints.md) for hand tracking, or [Dog-Pose](../datasets/pose/dog-pose.md) for canine pose analysis.

## Val

Validate trained YOLO26n-pose model [accuracy](https://www.ultralytics.com/glossary/accuracy) on the COCO8-pose dataset. No arguments are needed as the `model` retains its training `data` and arguments as model attributes.

!!! example

    === "Python"

        ```python
        from ultralytics import YOLO

        # Load a model
        model = YOLO("yolo26n-pose.pt")  # load an official model
        model = YOLO("path/to/best.pt")  # load a custom model

        # Validate the model
        metrics = model.val()  # no arguments needed, dataset and settings remembered
        metrics.box.map  # map50-95
        metrics.box.map50  # map50
        metrics.box.map75  # map75
        metrics.box.maps  # a list containing mAP50-95 for each category
        metrics.box.image_metrics  # per-image metrics dictionary for box with precision, recall, F1, TP, FP, and FN
        metrics.pose.map  # map50-95(P)
        metrics.pose.map50  # map50(P)
        metrics.pose.map75  # map75(P)
        metrics.pose.maps  # a list containing mAP50-95(P) for each category
        metrics.pose.image_metrics  # per-image metrics dictionary for pose with precision, recall, F1, TP, FP, and FN
        ```

    === "CLI"

        ```bash
        yolo pose val model=yolo26n-pose.pt # val official model
        yolo pose val model=path/to/best.pt # val custom model
        ```

## Predict

Use a trained YOLO26n-pose model to run predictions on images. The [predict mode](../modes/predict.md) allows you to perform inference on images, videos, or real-time streams.

!!! example

    === "Python"

        ```python
        from ultralytics import YOLO

        # Load a model
        model = YOLO("yolo26n-pose.pt")  # load an official model
        model = YOLO("path/to/best.pt")  # load a custom model

        # Predict with the model
        results = model("https://ultralytics.com/images/bus.jpg")  # predict on an image

        # Access the results
        for result in results:
            xy = result.keypoints.xy  # x and y coordinates
            xyn = result.keypoints.xyn  # normalized
            kpts = result.keypoints.data  # x, y, visibility (if available)
        ```

    === "CLI"

        ```bash
        yolo pose predict model=yolo26n-pose.pt source='https://ultralytics.com/images/bus.jpg' # predict with official model
        yolo pose predict model=path/to/best.pt source='https://ultralytics.com/images/bus.jpg' # predict with custom model
        ```

See full `predict` mode details in the [Predict](../modes/predict.md) page.

### Results Output

Pose estimation returns one `Results` object per image. The primary prediction fields are `result.keypoints` for pose
coordinates and `result.boxes` for the detected instances that those keypoints belong to.

| Attribute               | Type            | Shape       | Description                                |
| ----------------------- | --------------- | ----------- | ------------------------------------------ |
| `result.keypoints`      | `Keypoints`     | `(N)`       | Keypoints.                                 |
| `result.keypoints.data` | `torch.float32` | `(N,K,2/3)` | `x,y` plus optional visibility/confidence. |
| `result.keypoints.xy`   | `torch.float32` | `(N,K,2)`   | Pixel keypoints.                           |
| `result.keypoints.xyn`  | `torch.float32` | `(N,K,2)`   | Normalized keypoints.                      |
| `result.boxes`          | `Boxes`         | `(N)`       | Instance boxes.                            |

For task-specific `Results` fields across every task, see the [Predict Results by Task](../modes/predict.md#results-by-task) section.

## Export

Export a YOLO26n Pose model to a different format like ONNX, CoreML, etc. This allows you to deploy your model on various platforms and devices for [real-time inference](https://www.ultralytics.com/glossary/real-time-inference).

!!! example

    === "Python"

        ```python
        from ultralytics import YOLO

        # Load a model
        model = YOLO("yolo26n-pose.pt")  # load an official model
        model = YOLO("path/to/best.pt")  # load a custom-trained model

        # Export the model
        model.export(format="onnx")
        ```

    === "CLI"

        ```bash
        yolo export model=yolo26n-pose.pt format=onnx # export official model
        yolo export model=path/to/best.pt format=onnx # export custom-trained model
        ```

Available YOLO26-pose export formats are in the table below. You can export to any format using the `format` argument, i.e., `format='onnx'` or `format='engine'`. You can predict or validate directly on exported models, i.e., `yolo predict model=yolo26n-pose.onnx`. Usage examples are shown for your model after export completes.

| Format | `format` Argument | Model | Metadata | Arguments |
| ---------------------------------------------------------- | ----------------- | ------------------------------------------------- | -------- | ------------------------------------------------------------------------------------------------------------------------ |
| [PyTorch](https://pytorch.org/) | - | `yolo26n-pose.pt` | ✅ | - |
| [TorchScript](../integrations/torchscript.md) | `torchscript` | `yolo26n-pose.torchscript` | ✅ | `imgsz`, `quantize`, `dynamic`, `nms`:material-information-outline:{ title="conf, iou, agnostic_nms are also available when nms=True" }, `batch`, `device` |
| [ONNX](../integrations/onnx.md) | `onnx` | `yolo26n-pose.onnx` | ✅ | `imgsz`, `quantize`, `dynamic`, `simplify`, `opset`, `nms`:material-information-outline:{ title="conf, iou, agnostic_nms are also available when nms=True" }, `batch`, `data`, `fraction`, `device` |
| [OpenVINO](../integrations/openvino.md) | `openvino` | `yolo26n-pose_openvino_model/` | ✅ | `imgsz`, `quantize`, `dynamic`, `nms`:material-information-outline:{ title="conf, iou, agnostic_nms are also available when nms=True" }, `batch`, `data`, `fraction`, `device` |
| [TensorRT](../integrations/tensorrt.md) | `engine` | `yolo26n-pose.engine` | ✅ | `imgsz`, `quantize`, `dynamic`, `simplify`, `opset`, `workspace`, `nms`:material-information-outline:{ title="conf, iou, agnostic_nms are also available when nms=True" }, `batch`, `data`, `fraction`, `device` |
| [CoreML](../integrations/coreml.md) | `coreml` | `yolo26n-pose.mlpackage` | ✅ | `imgsz`, `dynamic`, `quantize`, `nms`:material-information-outline:{ title="conf, iou, agnostic_nms are also available when nms=True" }, `batch`, `device` |
| [TF SavedModel](../integrations/tf-savedmodel.md) | `saved_model` | `yolo26n-pose_saved_model/` | ✅ | `imgsz`, `keras`, `quantize`, `opset`, `nms`:material-information-outline:{ title="conf, iou, agnostic_nms are also available when nms=True" }, `batch`, `data`, `fraction`, `device` |
| [TF GraphDef](../integrations/tf-graphdef.md) | `pb` | `yolo26n-pose.pb` | ❌ | `imgsz`, `opset`, `batch`, `device` |
| [TF Edge TPU](../integrations/edge-tpu.md) | `edgetpu` | `yolo26n-pose_edgetpu.tflite` | ✅ | `imgsz`, `quantize`, `opset`, `data`, `fraction`, `device` |
| [PaddlePaddle](../integrations/paddlepaddle.md) | `paddle` | `yolo26n-pose_paddle_model/` | ✅ | `imgsz`, `batch`, `device` |
| [MNN](../integrations/mnn.md) | `mnn` | `yolo26n-pose.mnn` | ✅ | `imgsz`, `batch`, `dynamic`, `quantize`, `simplify`, `opset`, `nms`:material-information-outline:{ title="conf, iou, agnostic_nms are also available when nms=True" }, `device` |
| [NCNN](../integrations/ncnn.md) | `ncnn` | `yolo26n-pose_ncnn_model/` | ✅ | `imgsz`, `quantize`, `batch`, `device` |
| [IMX500](../integrations/sony-imx500.md):material-information-outline:{ title="IMX format is currently only supported for YOLOv8n, YOLO11n models" } | `imx` | `yolo26n-pose_imx_model/` | ✅ | `imgsz`, `quantize`, `data`, `fraction`, `nms`:material-information-outline:{ title="conf, iou, agnostic_nms are also available when nms=True" }, `device` |
| [RKNN](../integrations/rockchip-rknn.md) | `rknn` | `yolo26n-pose_rknn_model/` | ✅ | `imgsz`, `batch`, `name`, `quantize`, `simplify`, `opset`, `data`, `fraction`, `device` |
| [ExecuTorch](../integrations/executorch.md) | `executorch` | `yolo26n-pose_executorch_model/` | ✅ | `imgsz`, `batch`, `device` |
| [Axelera](../integrations/axelera.md) | `axelera` | `yolo26n-pose_axelera_model/` | ✅ | `imgsz`, `batch`, `quantize`, `data`, `fraction`, `device` |
| [DEEPX](../integrations/deepx.md) | `deepx` | `yolo26n-pose_deepx_model/` | ✅ | `imgsz`, `quantize`, `simplify`, `opset`, `data`, `optimize`, `device` |
| [Qualcomm QNN](../integrations/qnn.md) | `qnn` | `yolo26n-pose_qnn.onnx` | ✅ | `imgsz`, `batch`, `name`, `quantize`, `simplify`, `opset`, `data`, `fraction`, `device` |
| [LiteRT](../integrations/litert.md) | `litert` | `yolo26n-pose.tflite` | ✅ | `imgsz`, `quantize`, `batch`, `data`, `fraction`, `device` |
| [Hailo](../integrations/hailo.md) | `hailo` | `yolo26n-pose_hailo_model/` | ✅ | `imgsz`, `name`, `quantize`, `data`, `fraction`, `simplify`, `conf`, `iou` |
| [Huawei Ascend](../integrations/ascend.md) | `ascend` | `yolo26n-pose_ascend_model/` | ✅ | `imgsz`, `batch`, `name`, `quantize`, `opset`, `simplify`, `nms`:material-information-outline:{ title="conf, iou, agnostic_nms are also available when nms=True" } |

See full `export` details in the [Export](../modes/export.md) page.

## FAQ

### What is Pose Estimation with Ultralytics YOLO26 and how does it work?

Pose estimation with Ultralytics YOLO26 involves identifying specific points, known as keypoints, in an image. These keypoints typically represent joints or other important features of the object. The output includes the `[x, y]` coordinates and confidence scores for each point. YOLO26-pose models are specifically designed for this task and use the `-pose` suffix, such as `yolo26n-pose.pt`. These models are pretrained on datasets like [COCO keypoints](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/datasets/coco-pose.yaml) and can be used for various pose estimation tasks. For more information, visit the [Pose Estimation Page](#pose-estimation).

### How can I train a YOLO26-pose model on a custom dataset?

Training a YOLO26-pose model on a custom dataset involves loading a model, either a new model defined by a YAML file or a pretrained model. You can then start the training process using your specified dataset and parameters.

```python
from ultralytics import YOLO

# Load a model
model = YOLO("yolo26n-pose.yaml")  # build a new model from YAML
model = YOLO("yolo26n-pose.pt")  # load a pretrained model (recommended for training)

# Train the model
results = model.train(data="your-dataset.yaml", epochs=100, imgsz=640)
```

For comprehensive details on training, refer to the [Train Section](#train). You can also use [Ultralytics Platform cloud training](../platform/train/cloud-training.md) for a no-code approach to training custom pose estimation models.

### How do I validate a trained YOLO26-pose model?

Validation of a YOLO26-pose model involves assessing its accuracy using the same dataset parameters retained during training. Here's an example:

```python
from ultralytics import YOLO

# Load a model
model = YOLO("yolo26n-pose.pt")  # load an official model
model = YOLO("path/to/best.pt")  # load a custom model

# Validate the model
metrics = model.val()  # no arguments needed, dataset and settings remembered
```

For more information, visit the [Val Section](#val).

### Can I export a YOLO26-pose model to other formats, and how?

Yes, you can export a YOLO26-pose model to various formats like ONNX, CoreML, TensorRT, and more. This can be done using either Python or the Command Line Interface (CLI).

```python
from ultralytics import YOLO

# Load a model
model = YOLO("yolo26n-pose.pt")  # load an official model
model = YOLO("path/to/best.pt")  # load a custom-trained model

# Export the model
model.export(format="onnx")
```

Refer to the [Export Section](#export) for more details. Exported models can be deployed on edge devices for [real-time applications](https://www.ultralytics.com/blog/real-time-inferences-in-vision-ai-solutions-are-making-an-impact) like fitness tracking, sports analysis, or [robotics](https://www.ultralytics.com/blog/from-algorithms-to-automation-ais-role-in-robotics).

### What are the available Ultralytics YOLO26-pose models and their performance metrics?

Ultralytics YOLO26 offers various pretrained pose models such as YOLO26n-pose, YOLO26s-pose, YOLO26m-pose, among others. These models differ in size, accuracy (mAP), and speed. For instance, the YOLO26n-pose model achieves a mAP<sup>pose</sup>50-95 of 57.2 and an mAP<sup>pose</sup>50 of 83.3. For a complete list and performance details, visit the [Models Section](#models).
