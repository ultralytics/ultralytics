---
comments: true
description: 学习YOLOv8分类模型进行图像分类。获取关于预训练模型列表及如何训练、验证、预测、导出模型的详细信息。
keywords: Ultralytics, YOLOv8, 图像分类, 预训练模型, YOLOv8n-cls, 训练, 验证, 预测, 模型导出
---

# 图像分类

<img width="1024" src="https://user-images.githubusercontent.com/26833433/243418606-adf35c62-2e11-405d-84c6-b84e7d013804.png" alt="图像分类示例">

图像分类是三项任务中最简单的，它涉及将整个图像分类为一组预定义类别中的一个。

图像分类器的输出是单个类别标签和一个置信度分数。当您只需要知道一幅图像属于哪个类别、而不需要知道该类别对象的位置或它们的确切形状时，图像分类非常有用。

!!! Tip "提示"

    YOLOv8分类模型使用`-cls`后缀，即`yolov8n-cls.pt`，并预先训练在[ImageNet](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/datasets/ImageNet.yaml)上。

## [模型](https://github.com/ultralytics/ultralytics/tree/main/ultralytics/cfg/models/v8)

这里展示了预训练的YOLOv8分类模型。Detect、Segment和Pose模型是在[COCO](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/datasets/coco.yaml)数据集上预训练的，而分类模型则是在[ImageNet](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/datasets/ImageNet.yaml)数据集上预训练的。

[模型](https://github.com/ultralytics/ultralytics/tree/main/ultralytics/cfg/models)会在首次使用时自动从Ultralytics的最新[发布版本](https://github.com/ultralytics/assets/releases)中下载。

| 模型                                                                                           | 尺寸<br><sup>(像素) | 准确率<br><sup>top1 | 准确率<br><sup>top5 | 速度<br><sup>CPU ONNX<br>(ms) | 速度<br><sup>A100 TensorRT<br>(ms) | 参数<br><sup>(M) | FLOPs<br><sup>(B) at 640 |
|----------------------------------------------------------------------------------------------|-----------------|------------------|------------------|-----------------------------|----------------------------------|----------------|--------------------------|
| [YOLOv8n-cls](https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n-cls.pt) | 224             | 66.6             | 87.0             | 12.9                        | 0.31                             | 2.7            | 4.3                      |
| [YOLOv8s-cls](https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8s-cls.pt) | 224             | 72.3             | 91.1             | 23.4                        | 0.35                             | 6.4            | 13.5                     |
| [YOLOv8m-cls](https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8m-cls.pt) | 224             | 76.4             | 93.2             | 85.4                        | 0.62                             | 17.0           | 42.7                     |
| [YOLOv8l-cls](https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8l-cls.pt) | 224             | 78.0             | 94.1             | 163.0                       | 0.87                             | 37.5           | 99.7                     |
| [YOLOv8x-cls](https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8x-cls.pt) | 224             | 78.4             | 94.3             | 232.0                       | 1.01                             | 57.4           | 154.8                    |

- **准确率** 是模型在[ImageNet](https://www.image-net.org/)数据集验证集上的准确度。
  <br>通过`yolo val classify data=path/to/ImageNet device=0`复现结果。
- **速度** 是在使用[Amazon EC2 P4d](https://aws.amazon.com/ec2/instance-types/p4/)实例时，ImageNet验证图像的平均处理速度。
  <br>通过`yolo val classify data=path/to/ImageNet batch=1 device=0|cpu`复现结果。

## 训练

在MNIST160数据集上训练YOLOv8n-cls模型100个时期，图像尺寸为64。有关可用参数的完整列表，请参见[配置](/../usage/cfg.md)页面。

!!! Example "示例"

    === "Python"

        ```python
        from ultralytics import YOLO

        # 加载模型
        model = YOLO('yolov8n-cls.yaml')  # 从YAML构建新模型
        model = YOLO('yolov8n-cls.pt')  # 加载预训练模型（推荐用于训练）
        model = YOLO('yolov8n-cls.yaml').load('yolov8n-cls.pt')  # 从YAML构建并转移权重

        # 训练模型
        results = model.train(data='mnist160', epochs=100, imgsz=64)
        ```

    === "CLI"

        ```bash
        # 从YAML构建新模型并从头开始训练
        yolo classify train data=mnist160 model=yolov8n-cls.yaml epochs=100 imgsz=64

        # 从预训练的*.pt模型开始训练
        yolo classify train data=mnist160 model=yolov8n-cls.pt epochs=100 imgsz=64

        # 从YAML构建新模型，转移预训练权重并开始训练
        yolo classify train data=mnist160 model=yolov8n-cls.yaml pretrained=yolov8n-cls.pt epochs=100 imgsz=64
        ```

### 数据集格式

YOLO分类数据集的格式详情请参见[数据集指南](/../datasets/classify/index.md)。

## 验证

在MNIST160数据集上验证训练好的YOLOv8n-cls模型准确性。不需要传递任何参数，因为`model`保留了它的训练`data`和参数作为模型属性。

!!! Example "示例"

    === "Python"

        ```python
        from ultralytics import YOLO

        # 加载模型
        model = YOLO('yolov8n-cls.pt')  # 加载官方模型
        model = YOLO('path/to/best.pt')  # 加载自定义模型

        # 验证模型
        metrics = model.val()  # 无需参数，数据集和设置已记忆
        metrics.top1   # top1准确率
        metrics.top5   # top5准确率
        ```
    === "CLI"

        ```bash
        yolo classify val model=yolov8n-cls.pt  # 验证官方模型
        yolo classify val model=path/to/best.pt  # 验证自定义模型
        ```

## 预测

使用训练过的YOLOv8n-cls模型对图像进行预测。

!!! Example "示例"

    === "Python"

        ```python
        from ultralytics import YOLO

        # 加载模型
        model = YOLO('yolov8n-cls.pt')  # 加载官方模型
        model = YOLO('path/to/best.pt')  # 加载自定义模型

        # 使用模型进行预测
        results = model('https://ultralytics.com/images/bus.jpg')  # 对图像进行预测
        ```
    === "CLI"

        ```bash
        yolo classify predict model=yolov8n-cls.pt source='https://ultralytics.com/images/bus.jpg'  # 使用官方模型进行预测
        yolo classify predict model=path/to/best.pt source='https://ultralytics.com/images/bus.jpg'  # 使用自定义模型进行预测
        ```

有关`predict`模式的完整详细信息，请参见[预测](https://docs.ultralytics.com/modes/predict/)页面。

## 导出

将YOLOv8n-cls模型导出为其他格式，如ONNX、CoreML等。

!!! Example "示例"

    === "Python"

        ```python
        from ultralytics import YOLO

        # 加载模型
        model = YOLO('yolov8n-cls.pt')  # 加载官方模型
        model = YOLO('path/to/best.pt')  # 加载自定义训练模型

        # 导出模型
        model.export(format='onnx')
        ```
    === "CLI"

        ```bash
        yolo export model=yolov8n-cls.pt format=onnx  # 导出官方模型
        yolo export model=path/to/best.pt format=onnx  # 导出自定义训练模型
        ```

下表中提供了YOLOv8-cls模型可导出的格式。您可以直接在导出的模型上进行预测或验证，即`yolo predict model=yolov8n-cls.onnx`。导出完成后，示例用法会显示您的模型。

| 格式                                                                 | `format` 参数   | 模型                            | 元数据 | 参数                                                  |
|--------------------------------------------------------------------|---------------|-------------------------------|-----|-----------------------------------------------------|
| [PyTorch](https://pytorch.org/)                                    | -             | `yolov8n-cls.pt`              | ✅   | -                                                   |
| [TorchScript](https://pytorch.org/docs/stable/jit.html)            | `torchscript` | `yolov8n-cls.torchscript`     | ✅   | `imgsz`, `optimize`                                 |
| [ONNX](https://onnx.ai/)                                           | `onnx`        | `yolov8n-cls.onnx`            | ✅   | `imgsz`, `half`, `dynamic`, `simplify`, `opset`     |
| [OpenVINO](https://docs.openvino.ai/latest/index.html)             | `openvino`    | `yolov8n-cls_openvino_model/` | ✅   | `imgsz`, `half`                                     |
| [TensorRT](https://developer.nvidia.com/tensorrt)                  | `engine`      | `yolov8n-cls.engine`          | ✅   | `imgsz`, `half`, `dynamic`, `simplify`, `workspace` |
| [CoreML](https://github.com/apple/coremltools)                     | `coreml`      | `yolov8n-cls.mlpackage`       | ✅   | `imgsz`, `half`, `int8`, `nms`                      |
| [TF SavedModel](https://www.tensorflow.org/guide/saved_model)      | `saved_model` | `yolov8n-cls_saved_model/`    | ✅   | `imgsz`, `keras`                                    |
| [TF GraphDef](https://www.tensorflow.org/api_docs/python/tf/Graph) | `pb`          | `yolov8n-cls.pb`              | ❌   | `imgsz`                                             |
| [TF Lite](https://www.tensorflow.org/lite)                         | `tflite`      | `yolov8n-cls.tflite`          | ✅   | `imgsz`, `half`, `int8`                             |
| [TF Edge TPU](https://coral.ai/docs/edgetpu/models-intro/)         | `edgetpu`     | `yolov8n-cls_edgetpu.tflite`  | ✅   | `imgsz`                                             |
| [TF.js](https://www.tensorflow.org/js)                             | `tfjs`        | `yolov8n-cls_web_model/`      | ✅   | `imgsz`                                             |
| [PaddlePaddle](https://github.com/PaddlePaddle)                    | `paddle`      | `yolov8n-cls_paddle_model/`   | ✅   | `imgsz`                                             |
| [ncnn](https://github.com/Tencent/ncnn)                            | `ncnn`        | `yolov8n-cls_ncnn_model/`     | ✅   | `imgsz`, `half`                                     |

有关`export`的完整详细信息，请参见[导出](https://docs.ultralytics.com/modes/export/)页面。
