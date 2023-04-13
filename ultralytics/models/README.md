## Models

Welcome to the Ultralytics Models directory! Here you will find a wide variety of pre-configured model configuration
files (`*.yaml`s) that can be used to create custom YOLO models. The models in this directory have been expertly crafted
and fine-tuned by the Ultralytics team to provide the best performance for a wide range of object detection and image
segmentation tasks.

These model configurations cover a wide range of scenarios, from simple object detection to more complex tasks like
instance segmentation and object tracking. They are also designed to run efficiently on a variety of hardware platforms,
from CPUs to GPUs. Whether you are a seasoned machine learning practitioner or just getting started with YOLO, this
directory provides a great starting point for your custom model development needs.

To get started, simply browse through the models in this directory and find one that best suits your needs. Once you've
selected a model, you can use the provided `*.yaml` file to train and deploy your custom YOLO model with ease. See full
details at the Ultralytics [Docs](https://docs.ultralytics.com), and if you need help or have any questions, feel free
to reach out to the Ultralytics team for support. So, don't wait, start creating your custom YOLO model now!

### Usage

Model `*.yaml` files may be used directly in the Command Line Interface (CLI) with a `yolo` command:

```bash
yolo task=detect mode=train model=yolov8n.yaml data=coco128.yaml epochs=100
```

They may also be used directly in a Python environment, and accepts the same
[arguments](https://docs.ultralytics.com/usage/cfg/) as in the CLI example above:

```python
from ultralytics import YOLO

model = YOLO("model.yaml")  # build a YOLOv8n model from scratch
# YOLO("model.pt")  use pre-trained model if available
model.info()  # display model information
model.train(data="coco128.yaml", epochs=100)  # train the model
```

## Pre-trained Model Architectures

Ultralytics supports many model architectures. Visit [models](#) page to view detailed information and usage.
Any of these models can be used by loading their configs or pretrained checkpoints if available.

<b>What to add your model architecture?</b> [Here's](#) how you can contribute

### 1. YOLOv8

**About** - Cutting edge Detection, Segmentation, Classification and Pose models developed by Ultralytics. </br>

Available Models:

- Detection - `yolov8n`, `yolov8s`, `yolov8m`, `yolov8l`, `yolov8x`
- Instance Segmentation - `yolov8n-seg`, `yolov8s-seg`, `yolov8m-seg`, `yolov8l-seg`, `yolov8x-seg`
- Classification - `yolov8n-cls`, `yolov8s-cls`, `yolov8m-cls`, `yolov8l-cls`, `yolov8x-cls`
- Pose - `yolov8n-pose`, `yolov8s-pose`, `yolov8m-pose`, `yolov8l-pose`, `yolov8x-pose`, `yolov8x-pose-p6`

<details><summary>Performance</summary>

### Detection

| Model                                                                                | size<br><sup>(pixels) | mAP<sup>val<br>50-95 | Speed<br><sup>CPU ONNX<br>(ms) | Speed<br><sup>A100 TensorRT<br>(ms) | params<br><sup>(M) | FLOPs<br><sup>(B) |
| ------------------------------------------------------------------------------------ | --------------------- | -------------------- | ------------------------------ | ----------------------------------- | ------------------ | ----------------- |
| [YOLOv8n](https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n.pt) | 640                   | 37.3                 | 80.4                           | 0.99                                | 3.2                | 8.7               |
| [YOLOv8s](https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8s.pt) | 640                   | 44.9                 | 128.4                          | 1.20                                | 11.2               | 28.6              |
| [YOLOv8m](https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8m.pt) | 640                   | 50.2                 | 234.7                          | 1.83                                | 25.9               | 78.9              |
| [YOLOv8l](https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8l.pt) | 640                   | 52.9                 | 375.2                          | 2.39                                | 43.7               | 165.2             |
| [YOLOv8x](https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8x.pt) | 640                   | 53.9                 | 479.1                          | 3.53                                | 68.2               | 257.8             |

### Segmentation

| Model                                                                                        | size<br><sup>(pixels) | mAP<sup>box<br>50-95 | mAP<sup>mask<br>50-95 | Speed<br><sup>CPU ONNX<br>(ms) | Speed<br><sup>A100 TensorRT<br>(ms) | params<br><sup>(M) | FLOPs<br><sup>(B) |
| -------------------------------------------------------------------------------------------- | --------------------- | -------------------- | --------------------- | ------------------------------ | ----------------------------------- | ------------------ | ----------------- |
| [YOLOv8n-seg](https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n-seg.pt) | 640                   | 36.7                 | 30.5                  | 96.1                           | 1.21                                | 3.4                | 12.6              |
| [YOLOv8s-seg](https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8s-seg.pt) | 640                   | 44.6                 | 36.8                  | 155.7                          | 1.47                                | 11.8               | 42.6              |
| [YOLOv8m-seg](https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8m-seg.pt) | 640                   | 49.9                 | 40.8                  | 317.0                          | 2.18                                | 27.3               | 110.2             |
| [YOLOv8l-seg](https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8l-seg.pt) | 640                   | 52.3                 | 42.6                  | 572.4                          | 2.79                                | 46.0               | 220.5             |
| [YOLOv8x-seg](https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8x-seg.pt) | 640                   | 53.4                 | 43.4                  | 712.1                          | 4.02                                | 71.8               | 344.1             |

### Classification

| Model                                                                                        | size<br><sup>(pixels) | acc<br><sup>top1 | acc<br><sup>top5 | Speed<br><sup>CPU ONNX<br>(ms) | Speed<br><sup>A100 TensorRT<br>(ms) | params<br><sup>(M) | FLOPs<br><sup>(B) at 640 |
| -------------------------------------------------------------------------------------------- | --------------------- | ---------------- | ---------------- | ------------------------------ | ----------------------------------- | ------------------ | ------------------------ |
| [YOLOv8n-cls](https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n-cls.pt) | 224                   | 66.6             | 87.0             | 12.9                           | 0.31                                | 2.7                | 4.3                      |
| [YOLOv8s-cls](https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8s-cls.pt) | 224                   | 72.3             | 91.1             | 23.4                           | 0.35                                | 6.4                | 13.5                     |
| [YOLOv8m-cls](https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8m-cls.pt) | 224                   | 76.4             | 93.2             | 85.4                           | 0.62                                | 17.0               | 42.7                     |
| [YOLOv8l-cls](https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8l-cls.pt) | 224                   | 78.0             | 94.1             | 163.0                          | 0.87                                | 37.5               | 99.7                     |
| [YOLOv8x-cls](https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8x-cls.pt) | 224                   | 78.4             | 94.3             | 232.0                          | 1.01                                | 57.4               | 154.8                    |

### Pose

| Model                                                                                                | size<br><sup>(pixels) | mAP<sup>pose<br>50-95 | mAP<sup>pose<br>50 | Speed<br><sup>CPU ONNX<br>(ms) | Speed<br><sup>A100 TensorRT<br>(ms) | params<br><sup>(M) | FLOPs<br><sup>(B) |
| ---------------------------------------------------------------------------------------------------- | --------------------- | --------------------- | ------------------ | ------------------------------ | ----------------------------------- | ------------------ | ----------------- |
| [YOLOv8n-pose](https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n-pose.pt)       | 640                   | 49.7                  | 79.7               | 131.8                          | 1.18                                | 3.3                | 9.2               |
| [YOLOv8s-pose](https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8s-pose.pt)       | 640                   | 59.2                  | 85.8               | 233.2                          | 1.42                                | 11.6               | 30.2              |
| [YOLOv8m-pose](https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8m-pose.pt)       | 640                   | 63.6                  | 88.8               | 456.3                          | 2.00                                | 26.4               | 81.0              |
| [YOLOv8l-pose](https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8l-pose.pt)       | 640                   | 67.0                  | 89.9               | 784.5                          | 2.59                                | 44.4               | 168.6             |
| [YOLOv8x-pose](https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8x-pose.pt)       | 640                   | 68.9                  | 90.4               | 1607.1                         | 3.73                                | 69.4               | 263.2             |
| [YOLOv8x-pose-p6](https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8x-pose-p6.pt) | 1280                  | 71.5                  | 91.3               | 4088.7                         | 10.04                               | 99.1               | 1066.4            |

</details>

### 2. YOLOv5u

**About** - Anchor-free YOLOv5 models with new detection head and better speed-accuracy tradeoff </br>

Available Models:

- Detection P5/32 - `yolov5nu`, `yolov5su`, `yolov5mu`, `yolov5lu`, `yolov5xu`
- Detection P6/64 - `yolov5n6u`, `yolov5s6u`, `yolov5m6u`, `yolov5l6u`, `yolov5x6u`

<details><summary>Performance</summary>

### Detection

| Model                                                                                    | size<br><sup>(pixels) | mAP<sup>val<br>50-95 | Speed<br><sup>CPU ONNX<br>(ms) | Speed<br><sup>A100 TensorRT<br>(ms) | params<br><sup>(M) | FLOPs<br><sup>(B) |
| ---------------------------------------------------------------------------------------- | --------------------- | -------------------- | ------------------------------ | ----------------------------------- | ------------------ | ----------------- |
| [YOLOv5nu](https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov5nu.pt)   | 640                   | 34.3                 | 73.6                           | 1.06                                | 2.6                | 7.7               |
| [YOLOv5su](https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov5su.pt)   | 640                   | 43.0                 | 120.7                          | 1.27                                | 9.1                | 24.0              |
| [YOLOv5mu](https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov5mu.pt)   | 640                   | 49.0                 | 233.9                          | 1.86                                | 25.1               | 64.2              |
| [YOLOv5lu](https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov5lu.pt)   | 640                   | 52.2                 | 408.4                          | 2.50                                | 53.2               | 135.0             |
| [YOLOv5xu](https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov5xu.pt)   | 640                   | 53.2                 | 763.2                          | 3.81                                | 97.2               | 246.4             |
|                                                                                          |                       |                      |                                |                                     |                    |                   |
| [YOLOv5n6u](https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov5n6u.pt) | 1280                  | 42.1                 | -                              | -                                   | 4.3                | 7.8               |
| [YOLOv5s6u](https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov5s6u.pt) | 1280                  | 48.6                 | -                              | -                                   | 15.3               | 24.6              |
| [YOLOv5m6u](https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov5m6u.pt) | 1280                  | 53.6                 | -                              | -                                   | 41.2               | 65.7              |
| [YOLOv5l6u](https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov5l6u.pt) | 1280                  | 55.7                 | -                              | -                                   | 86.1               | 137.4             |
| [YOLOv5x6u](https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov5x6u.pt) | 1280                  | 56.8                 | -                              | -                                   | 155.4              | 250.7             |

</details>
