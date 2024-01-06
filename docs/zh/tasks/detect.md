---
comments: true
description: Ultralytics 官方YOLOv8文档。学习如何训练、验证、预测并以各种格式导出模型。包括详尽的性能统计。
keywords: YOLOv8, Ultralytics, 目标检测, 预训练模型, 训练, 验证, 预测, 导出模型, COCO, ImageNet, PyTorch, ONNX, CoreML
---

# 目标检测

<img width="1024" src="https://user-images.githubusercontent.com/26833433/243418624-5785cb93-74c9-4541-9179-d5c6782d491a.png" alt="目标检测示例">

目标检测是一项任务，涉及辨识图像或视频流中物体的位置和类别。

目标检测器的输出是一组围绕图像中物体的边界框，以及每个框的类别标签和置信度得分。当您需要识别场景中的感兴趣对象，但不需要准确了解物体的位置或其确切形状时，目标检测是一个很好的选择。

<p align="center">
  <br>
  <iframe width="720" height="405" src="https://www.youtube.com/embed/5ku7npMrW40?si=6HQO1dDXunV8gekh"
    title="YouTube视频播放器" frameborder="0"
    allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share"
    allowfullscreen>
  </iframe>
  <br>
  <strong>观看：</strong>使用预训练的Ultralytics YOLOv8模型进行目标检测。
</p>

!!! Tip "提示"

    YOLOv8 Detect 模型是默认的 YOLOv8 模型，即 `yolov8n.pt` ，并在 [COCO](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/datasets/coco.yaml) 数据集上进行了预训练。

## [模型](https://github.com/ultralytics/ultralytics/tree/main/ultralytics/cfg/models/v8)

此处展示了预训练的YOLOv8 Detect模型。Detect、Segment和Pose模型在 [COCO](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/datasets/coco.yaml) 数据集上预训练，而Classify模型在 [ImageNet](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/datasets/ImageNet.yaml) 数据集上预训练。

[模型](https://github.com/ultralytics/ultralytics/tree/main/ultralytics/cfg/models) 会在首次使用时自动从Ultralytics的最新 [发布](https://github.com/ultralytics/assets/releases) 中下载。

| 模型                                                                                   | 尺寸<br><sup>(像素) | mAP<sup>val<br>50-95 | 速度<br><sup>CPU ONNX<br>(毫秒) | 速度<br><sup>A100 TensorRT<br>(毫秒) | 参数<br><sup>(M) | FLOPs<br><sup>(B) |
|--------------------------------------------------------------------------------------|-----------------|----------------------|-----------------------------|----------------------------------|----------------|-------------------|
| [YOLOv8n](https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n.pt) | 640             | 37.3                 | 80.4                        | 0.99                             | 3.2            | 8.7               |
| [YOLOv8s](https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8s.pt) | 640             | 44.9                 | 128.4                       | 1.20                             | 11.2           | 28.6              |
| [YOLOv8m](https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8m.pt) | 640             | 50.2                 | 234.7                       | 1.83                             | 25.9           | 78.9              |
| [YOLOv8l](https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8l.pt) | 640             | 52.9                 | 375.2                       | 2.39                             | 43.7           | 165.2             |
| [YOLOv8x](https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8x.pt) | 640             | 53.9                 | 479.1                       | 3.53                             | 68.2           | 257.8             |

- **mAP<sup>val</sup>** 值适用于 [COCO val2017](http://cocodataset.org) 数据集上的单模型单尺度。
  <br>通过 `yolo val detect data=coco.yaml device=0` 复现。
- **速度** 是在使用 [Amazon EC2 P4d](https://aws.amazon.com/ec2/instance-types/p4/) 云实例对COCO val图像的平均值。
  <br>通过 `yolo val detect data=coco128.yaml batch=1 device=0|cpu` 复现。

## 训练

在COCO128数据集上使用图像尺寸640将YOLOv8n训练100个epochs。要查看可用参数的完整列表，请参阅 [配置](/../usage/cfg.md) 页面。

!!! Example "示例"

    === "Python"

        ```python
        from ultralytics import YOLO

        # 加载模型
        model = YOLO('yolov8n.yaml')  # 从YAML构建新模型
        model = YOLO('yolov8n.pt')    # 加载预训练模型（推荐用于训练）
        model = YOLO('yolov8n.yaml').load('yolov8n.pt')  # 从YAML构建并转移权重

        # 训练模型
        results = model.train(data='coco128.yaml', epochs=100, imgsz=640)
        ```
    === "CLI"

        ```bash
        # 从YAML构建新模型并从头开始训练
        yolo detect train data=coco128.yaml model=yolov8n.yaml epochs=100 imgsz=640

        # 从预训练的*.pt模型开始训练
        yolo detect train data=coco128.yaml model=yolov8n.pt epochs=100 imgsz=640

        # 从YAML构建新模型，传递预训练权重并开始训练
        yolo detect train data=coco128.yaml model=yolov8n.yaml pretrained=yolov8n.pt epochs=100 imgsz=640
        ```

### 数据集格式

YOLO检测数据集格式可以在 [数据集指南](/../datasets/detect/index.md) 中详细找到。要将您现有的数据集从其他格式（如COCO等）转换为YOLO格式，请使用Ultralytics的 [JSON2YOLO](https://github.com/ultralytics/JSON2YOLO) 工具。

## 验证

在COCO128数据集上验证训练好的YOLOv8n模型准确性。无需传递参数，`model` 作为模型属性保留其训练的 `data` 和参数。

!!! Example "示例"

    === "Python"

        ```python
        from ultralytics import YOLO

        # 加载模型
        model = YOLO('yolov8n.pt')  # 加载官方模型
        model = YOLO('path/to/best.pt')  # 加载自定义模型

        # 验证模型
        metrics = model.val()  # 无需参数，数据集和设置通过模型属性记住
        metrics.box.map    # map50-95
        metrics.box.map50  # map50
        metrics.box.map75  # map75
        metrics.box.maps   # 包含每个类别map50-95的列表
        ```
    === "CLI"

        ```bash
        yolo detect val model=yolov8n.pt  # 验证官方模型
        yolo detect val model=path/to/best.pt  # 验证自定义模型
        ```

## 预测

使用训练好的YOLOv8n模型在图像上进行预测。

!!! Example "示例"

    === "Python"

        ```python
        from ultralytics import YOLO

        # 加载模型
        model = YOLO('yolov8n.pt')  # 加载官方模型
        model = YOLO('path/to/best.pt')  # 加载自定义模型

        # 使用模型进行预测
        results = model('https://ultralytics.com/images/bus.jpg')  # 对图像进行预测
        ```
    === "CLI"

        ```bash
        yolo detect predict model=yolov8n.pt source='https://ultralytics.com/images/bus.jpg'  # 使用官方模型进行预测
        yolo detect predict model=path/to/best.pt source='https://ultralytics.com/images/bus.jpg'  # 使用自定义模型进行预测
        ```

完整的 `predict` 模式细节请见 [预测](https://docs.ultralytics.com/modes/predict/) 页面。

## 导出

将YOLOv8n模型导出为ONNX、CoreML等不同格式。

!!! Example "示例"

    === "Python"

        ```python
        from ultralytics import YOLO

        # 加载模型
        model = YOLO('yolov8n.pt')  # 加载官方模型
        model = YOLO('path/to/best.pt')  # 加载自定义训练模型

        # 导出模型
        model.export(format='onnx')
        ```
    === "CLI"

        ```bash
        yolo export model=yolov8n.pt format=onnx  # 导出官方模型
        yolo export model=path/to/best.pt format=onnx  # 导出自定义训练模型
        ```

下表中提供了可用的YOLOv8导出格式。您可以直接在导出的模型上进行预测或验证，即 `yolo predict model=yolov8n.onnx`。导出完成后，会为您的模型显示使用示例。

| 格式                                                                 | `format` 参数   | 模型                        | 元数据 | 参数                                              |
|--------------------------------------------------------------------|---------------|---------------------------|-----|-------------------------------------------------|
| [PyTorch](https://pytorch.org/)                                    | -             | `yolov8n.pt`              | ✅   | -                                               |
| [TorchScript](https://pytorch.org/docs/stable/jit.html)            | `torchscript` | `yolov8n.torchscript`     | ✅   | `imgsz`，`optimize`                              |
| [ONNX](https://onnx.ai/)                                           | `onnx`        | `yolov8n.onnx`            | ✅   | `imgsz`，`half`，`dynamic`，`simplify`，`opset`     |
| [OpenVINO](https://docs.openvino.ai/latest/index.html)             | `openvino`    | `yolov8n_openvino_model/` | ✅   | `imgsz`，`half`                                  |
| [TensorRT](https://developer.nvidia.com/tensorrt)                  | `engine`      | `yolov8n.engine`          | ✅   | `imgsz`，`half`，`dynamic`，`simplify`，`workspace` |
| [CoreML](https://github.com/apple/coremltools)                     | `coreml`      | `yolov8n.mlpackage`       | ✅   | `imgsz`，`half`，`int8`，`nms`                     |
| [TF SavedModel](https://www.tensorflow.org/guide/saved_model)      | `saved_model` | `yolov8n_saved_model/`    | ✅   | `imgsz`，`keras`                                 |
| [TF GraphDef](https://www.tensorflow.org/api_docs/python/tf/Graph) | `pb`          | `yolov8n.pb`              | ❌   | `imgsz`                                         |
| [TF Lite](https://www.tensorflow.org/lite)                         | `tflite`      | `yolov8n.tflite`          | ✅   | `imgsz`，`half`，`int8`                           |
| [TF Edge TPU](https://coral.ai/docs/edgetpu/models-intro/)         | `edgetpu`     | `yolov8n_edgetpu.tflite`  | ✅   | `imgsz`                                         |
| [TF.js](https://www.tensorflow.org/js)                             | `tfjs`        | `yolov8n_web_model/`      | ✅   | `imgsz`                                         |
| [PaddlePaddle](https://github.com/PaddlePaddle)                    | `paddle`      | `yolov8n_paddle_model/`   | ✅   | `imgsz`                                         |
| [ncnn](https://github.com/Tencent/ncnn)                            | `ncnn`        | `yolov8n_ncnn_model/`     | ✅   | `imgsz`，`half`                                  |

完整的 `export` 详情请见 [导出](https://docs.ultralytics.com/modes/export/) 页面。
