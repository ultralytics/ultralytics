---
comments: true
description: 学习如何使用Ultralytics YOLO进行实例分割模型。包括训练、验证、图像预测和模型导出的说明。
keywords: yolov8, 实例分割, Ultralytics, COCO数据集, 图像分割, 物体检测, 模型训练, 模型验证, 图像预测, 模型导出
---

# 实例分割

<img width="1024" src="https://user-images.githubusercontent.com/26833433/243418644-7df320b8-098d-47f1-85c5-26604d761286.png" alt="实例分割示例">

实例分割比物体检测有所深入，它涉及到识别图像中的个别物体并将它们从图像的其余部分中分割出来。

实例分割模型的输出是一组蒙版或轮廓，用于勾画图像中每个物体，以及每个物体的类别标签和置信度分数。实例分割在您需要不仅知道图像中的物体位置，还需要知道它们确切形状时非常有用。

<p align="center">
  <br>
  <iframe width="720" height="405" src="https://www.youtube.com/embed/o4Zd-IeMlSY?si=37nusCzDTd74Obsp"
    title="YouTube视频播放器" frameborder="0"
    allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share"
    allowfullscreen>
  </iframe>
  <br>
  <strong>观看:</strong> 在Python中使用预训练的Ultralytics YOLOv8模型运行分割。
</p>

!!! Tip "提示"

    YOLOv8分割模型使用`-seg`后缀，即`yolov8n-seg.pt`，并在[COCO](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/datasets/coco.yaml)上进行预训练。

## [模型](https://github.com/ultralytics/ultralytics/tree/main/ultralytics/cfg/models/v8)

这里展示了预训练的YOLOv8分割模型。Detect、Segment和Pose模型都是在[COCO](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/datasets/coco.yaml)数据集上进行预训练的，而Classify模型则是在[ImageNet](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/datasets/ImageNet.yaml)数据集上进行预训练的。

[模型](https://github.com/ultralytics/ultralytics/tree/main/ultralytics/cfg/models)会在首次使用时自动从Ultralytics的最新[版本](https://github.com/ultralytics/assets/releases)下载。

| 模型                                                                                         | 尺寸<br><sup>(像素) | mAP<sup>box<br>50-95 | mAP<sup>mask<br>50-95 | 速度<br><sup>CPU ONNX<br>(ms) | 速度<br><sup>A100 TensorRT<br>(ms) | 参数<br><sup>(M) | FLOPs<br><sup>(B) |
| -------------------------------------------------------------------------------------------- | ------------------- | -------------------- | --------------------- | ----------------------------- | ---------------------------------- | ---------------- | ----------------- |
| [YOLOv8n-seg](https://github.com/ultralytics/assets/releases/download/v8.1.0/yolov8n-seg.pt) | 640                 | 36.7                 | 30.5                  | 96.1                          | 1.21                               | 3.4              | 12.6              |
| [YOLOv8s-seg](https://github.com/ultralytics/assets/releases/download/v8.1.0/yolov8s-seg.pt) | 640                 | 44.6                 | 36.8                  | 155.7                         | 1.47                               | 11.8             | 42.6              |
| [YOLOv8m-seg](https://github.com/ultralytics/assets/releases/download/v8.1.0/yolov8m-seg.pt) | 640                 | 49.9                 | 40.8                  | 317.0                         | 2.18                               | 27.3             | 110.2             |
| [YOLOv8l-seg](https://github.com/ultralytics/assets/releases/download/v8.1.0/yolov8l-seg.pt) | 640                 | 52.3                 | 42.6                  | 572.4                         | 2.79                               | 46.0             | 220.5             |
| [YOLOv8x-seg](https://github.com/ultralytics/assets/releases/download/v8.1.0/yolov8x-seg.pt) | 640                 | 53.4                 | 43.4                  | 712.1                         | 4.02                               | 71.8             | 344.1             |

- **mAP<sup>val</sup>** 值针对[COCO val2017](http://cocodataset.org)数据集的单模型单尺度。
    <br>通过`yolo val segment data=coco.yaml device=0`复现。
- **速度** 基于在[Amazon EC2 P4d](https://aws.amazon.com/ec2/instance-types/p4/)实例上运行的COCO val图像的平均值。
    <br>通过`yolo val segment data=coco128-seg.yaml batch=1 device=0|cpu`复现。

## 训练

在COCO128-seg数据集上以640的图像尺寸训练YOLOv8n-seg模型共100个周期。想了解更多可用的参数，请查阅[配置](/../usage/cfg.md)页面。

!!! Example "示例"

    === "Python"

        ```python
        from ultralytics import YOLO

        # 载入一个模型
        model = YOLO("yolov8n-seg.yaml")  # 从YAML构建一个新模型
        model = YOLO("yolov8n-seg.pt")  # 载入预训练模型（推荐用于训练）
        model = YOLO("yolov8n-seg.yaml").load("yolov8n.pt")  # 从YAML构建并传递权重

        # 训练模型
        results = model.train(data="coco128-seg.yaml", epochs=100, imgsz=640)
        ```
    === "CLI"

        ```bash
        # 从YAML构建新模型并从头开始训练
        yolo segment train data=coco128-seg.yaml model=yolov8n-seg.yaml epochs=100 imgsz=640

        # 从预训练*.pt模型开始训练
        yolo segment train data=coco128-seg.yaml model=yolov8n-seg.pt epochs=100 imgsz=640

        # 从YAML构建新模型，传递预训练权重，开始训练
        yolo segment train data=coco128-seg.yaml model=yolov8n-seg.yaml pretrained=yolov8n-seg.pt epochs=100 imgsz=640
        ```

### 数据集格式

可以在[数据集指南](/../datasets/segment/index.md)中详细了解YOLO分割数据集格式。要将现有数据集从其他格式（如COCO等）转换为YOLO格式，请使用Ultralytics的[JSON2YOLO](https://github.com/ultralytics/JSON2YOLO)工具。

## 验证

在COCO128-seg数据集上验证已训练的YOLOv8n-seg模型的准确性。不需要传递任何参数，因为`model`保留了其训练的`data`和作为模型属性的设置。

!!! Example "示例"

    === "Python"

        ```python
        from ultralytics import YOLO

        # 载入一个模型
        model = YOLO("yolov8n-seg.pt")  # 载入官方模型
        model = YOLO("path/to/best.pt")  # 载入自定义模型

        # 验证模型
        metrics = model.val()  # 不需要参数，数据集和设置被记住了
        metrics.box.map  # map50-95(B)
        metrics.box.map50  # map50(B)
        metrics.box.map75  # map75(B)
        metrics.box.maps  # 各类别map50-95(B)列表
        metrics.seg.map  # map50-95(M)
        metrics.seg.map50  # map50(M)
        metrics.seg.map75  # map75(M)
        metrics.seg.maps  # 各类别map50-95(M)列表
        ```
    === "CLI"

        ```bash
        yolo segment val model=yolov8n-seg.pt  # 验证官方模型
        yolo segment val model=path/to/best.pt  # 验证自定义模型
        ```

## 预测

使用已训练的YOLOv8n-seg模型在图像上进行预测。

!!! Example "示例"

    === "Python"

        ```python
        from ultralytics import YOLO

        # 载入一个模型
        model = YOLO("yolov8n-seg.pt")  # 载入官方模型
        model = YOLO("path/to/best.pt")  # 载入自定义模型

        # 使用模型进行预测
        results = model("https://ultralytics.com/images/bus.jpg")  # 对一张图像进行预测
        ```
    === "CLI"

        ```bash
        yolo segment predict model=yolov8n-seg.pt source='https://ultralytics.com/images/bus.jpg'  # 使用官方模型进行预测
        yolo segment predict model=path/to/best.pt source='https://ultralytics.com/images/bus.jpg'  # 使用自定义模型进行预测
        ```

预测模式的完整详情请参见[Predict](https://docs.ultralytics.com/modes/predict/)页面。

## 导出

将YOLOv8n-seg模型导出为ONNX、CoreML等不同格式。

!!! Example "示例"

    === "Python"

        ```python
        from ultralytics import YOLO

        # 载入一个模型
        model = YOLO("yolov8n-seg.pt")  # 载入官方模型
        model = YOLO("path/to/best.pt")  # 载入自定义训练模型

        # 导出模型
        model.export(format="onnx")
        ```
    === "CLI"

        ```bash
        yolo export model=yolov8n-seg.pt format=onnx  # 导出官方模型
        yolo export model=path/to/best.pt format=onnx  # 导出自定义训练模型
        ```

YOLOv8-seg导出格式的可用表格如下所示。您可以直接在导出的模型上进行预测或验证，例如`yolo predict model=yolov8n-seg.onnx`。导出完成后，示例用法将显示您的模型。

| 格式                                                               | `format` 参数 | 模型                          | 元数据 | 参数                                                |
| ------------------------------------------------------------------ | ------------- | ----------------------------- | ------ | --------------------------------------------------- |
| [PyTorch](https://pytorch.org/)                                    | -             | `yolov8n-seg.pt`              | ✅     | -                                                   |
| [TorchScript](https://pytorch.org/docs/stable/jit.html)            | `torchscript` | `yolov8n-seg.torchscript`     | ✅     | `imgsz`, `optimize`                                 |
| [ONNX](https://onnx.ai/)                                           | `onnx`        | `yolov8n-seg.onnx`            | ✅     | `imgsz`, `half`, `dynamic`, `simplify`, `opset`     |
| [OpenVINO](https://docs.openvino.ai/latest/index.html)             | `openvino`    | `yolov8n-seg_openvino_model/` | ✅     | `imgsz`, `half`                                     |
| [TensorRT](https://developer.nvidia.com/tensorrt)                  | `engine`      | `yolov8n-seg.engine`          | ✅     | `imgsz`, `half`, `dynamic`, `simplify`, `workspace` |
| [CoreML](https://github.com/apple/coremltools)                     | `coreml`      | `yolov8n-seg.mlpackage`       | ✅     | `imgsz`, `half`, `int8`, `nms`                      |
| [TF SavedModel](https://www.tensorflow.org/guide/saved_model)      | `saved_model` | `yolov8n-seg_saved_model/`    | ✅     | `imgsz`, `keras`                                    |
| [TF GraphDef](https://www.tensorflow.org/api_docs/python/tf/Graph) | `pb`          | `yolov8n-seg.pb`              | ❌     | `imgsz`                                             |
| [TF Lite](https://www.tensorflow.org/lite)                         | `tflite`      | `yolov8n-seg.tflite`          | ✅     | `imgsz`, `half`, `int8`                             |
| [TF Edge TPU](https://coral.ai/docs/edgetpu/models-intro/)         | `edgetpu`     | `yolov8n-seg_edgetpu.tflite`  | ✅     | `imgsz`                                             |
| [TF.js](https://www.tensorflow.org/js)                             | `tfjs`        | `yolov8n-seg_web_model/`      | ✅     | `imgsz`                                             |
| [PaddlePaddle](https://github.com/PaddlePaddle)                    | `paddle`      | `yolov8n-seg_paddle_model/`   | ✅     | `imgsz`                                             |
| [ncnn](https://github.com/Tencent/ncnn)                            | `ncnn`        | `yolov8n-seg_ncnn_model/`     | ✅     | `imgsz`, `half`                                     |

导出模式的完整详情请参见[Export](https://docs.ultralytics.com/modes/export/)页面。
