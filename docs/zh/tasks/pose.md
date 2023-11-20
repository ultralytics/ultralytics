---
comments: true
description: 学习如何使用Ultralytics YOLOv8进行姿态估计任务。找到预训练模型，学习如何训练、验证、预测以及导出你自己的模型。
keywords: Ultralytics, YOLO, YOLOv8, 姿态估计, 关键点检测, 物体检测, 预训练模型, 机器学习, 人工智能
---

# 姿态估计

<img width="1024" src="https://user-images.githubusercontent.com/26833433/243418616-9811ac0b-a4a7-452a-8aba-484ba32bb4a8.png" alt="姿态估计示例">

姿态估计是一项任务，其涉及识别图像中特定点的位置，通常被称为关键点。这些关键点可以代表物体的各种部位，如关节、地标或其他显著特征。关键点的位置通常表示为一组2D `[x, y]` 或3D `[x, y, visible]` 坐标。

姿态估计模型的输出是一组点集，这些点代表图像中物体上的关键点，通常还包括每个点的置信度得分。当你需要在场景中识别物体的特定部位及其相互之间的位置时，姿态估计是一个不错的选择。

<p align="center">
  <br>
  <iframe width="720" height="405" src="https://www.youtube.com/embed/Y28xXQmju64?si=pCY4ZwejZFu6Z4kZ"
    title="YouTube视频播放器" frameborder="0"
    allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share"
    允许全屏>
  </iframe>
  <br>
  <strong>观看：</strong>使用Ultralytics YOLOv8进行姿态估计。
</p>

!!! Tip "提示"

    YOLOv8 _姿态_ 模型使用 `-pose` 后缀，例如 `yolov8n-pose.pt`。这些模型在 [COCO关键点](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/datasets/coco-pose.yaml) 数据集上进行了训练，并且适用于各种姿态估计任务。

## [模型](https://github.com/ultralytics/ultralytics/tree/main/ultralytics/cfg/models/v8)

这里展示了YOLOv8预训练的姿态模型。检测、分割和姿态模型在 [COCO](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/datasets/coco.yaml) 数据集上进行预训练，而分类模型则在 [ImageNet](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/datasets/ImageNet.yaml) 数据集上进行预训练。

[模型](https://github.com/ultralytics/ultralytics/tree/main/ultralytics/cfg/models) 在首次使用时将自动从最新的Ultralytics [发布版本](https://github.com/ultralytics/assets/releases)中下载。

| 模型                                                                                                 | 尺寸<br><sup>(像素) | mAP<sup>姿态<br>50-95 | mAP<sup>姿态<br>50 | 速度<br><sup>CPU ONNX<br>(毫秒) | 速度<br><sup>A100 TensorRT<br>(毫秒) | 参数<br><sup>(M) | 浮点数运算<br><sup>(B) |
|----------------------------------------------------------------------------------------------------|-----------------|---------------------|------------------|-----------------------------|----------------------------------|----------------|-------------------|
| [YOLOv8n-姿态](https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n-pose.pt)       | 640             | 50.4                | 80.1             | 131.8                       | 1.18                             | 3.3            | 9.2               |
| [YOLOv8s-姿态](https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8s-pose.pt)       | 640             | 60.0                | 86.2             | 233.2                       | 1.42                             | 11.6           | 30.2              |
| [YOLOv8m-姿态](https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8m-pose.pt)       | 640             | 65.0                | 88.8             | 456.3                       | 2.00                             | 26.4           | 81.0              |
| [YOLOv8l-姿态](https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8l-pose.pt)       | 640             | 67.6                | 90.0             | 784.5                       | 2.59                             | 44.4           | 168.6             |
| [YOLOv8x-姿态](https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8x-pose.pt)       | 640             | 69.2                | 90.2             | 1607.1                      | 3.73                             | 69.4           | 263.2             |
| [YOLOv8x-姿态-p6](https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8x-pose-p6.pt) | 1280            | 71.6                | 91.2             | 4088.7                      | 10.04                            | 99.1           | 1066.4            |

- **mAP<sup>val</sup>** 值适用于[COCO 关键点 val2017](http://cocodataset.org)数据集上的单模型单尺度。
  <br>通过执行 `yolo val pose data=coco-pose.yaml device=0` 来复现。
- **速度** 是在 [亚马逊EC2 P4d](https://aws.amazon.com/ec2/instance-types/p4/)实例上使用COCO val图像的平均值。
  <br>通过执行 `yolo val pose data=coco8-pose.yaml batch=1 device=0|cpu` 来复现。

## 训练

在COCO128姿态数据集上训练一个YOLOv8姿态模型。

!!! Example "示例"

    === "Python"

        ```python
        from ultralytics import YOLO

        # 加载模型
        model = YOLO('yolov8n-pose.yaml')  # 从YAML构建一个新模型
        model = YOLO('yolov8n-pose.pt')  # 加载一个预训练模型（推荐用于训练）
        model = YOLO('yolov8n-pose.yaml').load('yolov8n-pose.pt')  # 从YAML构建并传输权重

        # 训练模型
        results = model.train(data='coco8-pose.yaml', epochs=100, imgsz=640)
        ```
    === "CLI"

        ```bash
        # 从YAML构建一个新模型并从头开始训练
        yolo pose train data=coco8-pose.yaml model=yolov8n-pose.yaml epochs=100 imgsz=640

        # 从一个预训练的*.pt模型开始训练
        yolo pose train data=coco8-pose.yaml model=yolov8n-pose.pt epochs=100 imgsz=640

        # 从YAML构建一个新模型，传输预训练权重并开始训练
        yolo pose train data=coco8-pose.yaml model=yolov8n-pose.yaml pretrained=yolov8n-pose.pt epochs=100 imgsz=640
        ```

### 数据集格式

YOLO姿态数据集格式可详细找到在[数据集指南](/../datasets/pose/index.md)中。若要将您现有的数据集从其他格式（如COCO等）转换为YOLO格式，请使用Ultralytics的 [JSON2YOLO](https://github.com/ultralytics/JSON2YOLO) 工具。

## 验证

在COCO128姿态数据集上验证训练好的YOLOv8n姿态模型的准确性。没有参数需要传递，因为`模型`保存了其训练`数据`和参数作为模型属性。

!!! Example "示例"

    === "Python"

        ```python
        from ultralytics import YOLO

        # 加载模型
        model = YOLO('yolov8n-pose.pt')  # 加载官方模型
        model = YOLO('path/to/best.pt')  # 加载自定义模型

        # 验证模型
        metrics = model.val()  # 无需参数，数据集和设置都记住了
        metrics.box.map    # map50-95
        metrics.box.map50  # map50
        metrics.box.map75  # map75
        metrics.box.maps   # 包含每个类别map50-95的列表
        ```
    === "CLI"

        ```bash
        yolo pose val model=yolov8n-pose.pt  # 验证官方模型
        yolo pose val model=path/to/best.pt  # 验证自定义模型
        ```

## 预测

使用训练好的YOLOv8n姿态模型在图片上运行预测。

!!! Example "示例"

    === "Python"

        ```python
        from ultralytics import YOLO

        # 加载模型
        model = YOLO('yolov8n-pose.pt')  # 加载官方模型
        model = YOLO('path/to/best.pt')  # 加载自定义模型

        # 用模型进行预测
        results = model('https://ultralytics.com/images/bus.jpg')  # 在一张图片上预测
        ```
    === "CLI"

        ```bash
        yolo pose predict model=yolov8n-pose.pt source='https://ultralytics.com/images/bus.jpg'  # 用官方模型预测
        yolo pose predict model=path/to/best.pt source='https://ultralytics.com/images/bus.jpg'  # 用自定义模型预测
        ```

在[预测](https://docs.ultralytics.com/modes/predict/)页面中查看完整的`预测`模式细节。

## 导出

将YOLOv8n姿态模型导出为ONNX、CoreML等不同格式。

!!! Example "示例"

    === "Python"

        ```python
        from ultralytics import YOLO

        # 加载模型
        model = YOLO('yolov8n-pose.pt')  # 加载官方模型
        model = YOLO('path/to/best.pt')  # 加载自定义训练好的模型

        # 导出模型
        model.export(format='onnx')
        ```
    === "CLI"

        ```bash
        yolo export model=yolov8n-pose.pt format=onnx  # 导出官方模型
        yolo export model=path/to/best.pt format=onnx  # 导出自定义训练好的模型
        ```

以下表格中有可用的YOLOv8姿态导出格式。您可以直接在导出的模型上进行预测或验证，例如 `yolo predict model=yolov8n-pose.onnx`。导出完成后，为您的模型显示用法示例。

| 格式                                                                 | `format` 参数   | 模型                             | 元数据 | 参数                                                  |
|--------------------------------------------------------------------|---------------|--------------------------------|-----|-----------------------------------------------------|
| [PyTorch](https://pytorch.org/)                                    | -             | `yolov8n-pose.pt`              | ✅   | -                                                   |
| [TorchScript](https://pytorch.org/docs/stable/jit.html)            | `torchscript` | `yolov8n-pose.torchscript`     | ✅   | `imgsz`, `optimize`                                 |
| [ONNX](https://onnx.ai/)                                           | `onnx`        | `yolov8n-pose.onnx`            | ✅   | `imgsz`, `half`, `dynamic`, `simplify`, `opset`     |
| [OpenVINO](https://docs.openvino.ai/latest/index.html)             | `openvino`    | `yolov8n-pose_openvino_model/` | ✅   | `imgsz`, `half`                                     |
| [TensorRT](https://developer.nvidia.com/tensorrt)                  | `engine`      | `yolov8n-pose.engine`          | ✅   | `imgsz`, `half`, `dynamic`, `simplify`, `workspace` |
| [CoreML](https://github.com/apple/coremltools)                     | `coreml`      | `yolov8n-pose.mlpackage`       | ✅   | `imgsz`, `half`, `int8`, `nms`                      |
| [TF SavedModel](https://www.tensorflow.org/guide/saved_model)      | `saved_model` | `yolov8n-pose_saved_model/`    | ✅   | `imgsz`, `keras`                                    |
| [TF GraphDef](https://www.tensorflow.org/api_docs/python/tf/Graph) | `pb`          | `yolov8n-pose.pb`              | ❌   | `imgsz`                                             |
| [TF Lite](https://www.tensorflow.org/lite)                         | `tflite`      | `yolov8n-pose.tflite`          | ✅   | `imgsz`, `half`, `int8`                             |
| [TF Edge TPU](https://coral.ai/docs/edgetpu/models-intro/)         | `edgetpu`     | `yolov8n-pose_edgetpu.tflite`  | ✅   | `imgsz`                                             |
| [TF.js](https://www.tensorflow.org/js)                             | `tfjs`        | `yolov8n-pose_web_model/`      | ✅   | `imgsz`                                             |
| [PaddlePaddle](https://github.com/PaddlePaddle)                    | `paddle`      | `yolov8n-pose_paddle_model/`   | ✅   | `imgsz`                                             |
| [ncnn](https://github.com/Tencent/ncnn)                            | `ncnn`        | `yolov8n-pose_ncnn_model/`     | ✅   | `imgsz`, `half`                                     |

在[导出](https://docs.ultralytics.com/modes/export/) 页面中查看完整的`导出`细节。
