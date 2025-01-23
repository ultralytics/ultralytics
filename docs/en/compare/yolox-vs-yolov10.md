---
comments: true
description: Explore the detailed comparison between YOLOX and YOLOv10, two powerful models in real-time object detection. Discover how YOLOv10, with innovations like NMS-free training and enhanced efficiency, stacks up against YOLOX in terms of speed, accuracy, and performance for edge AI and computer vision applications.
keywords: YOLOX, YOLOv10, Ultralytics, object detection, real-time AI, edge AI, computer vision, model comparison
---

# YOLOX VS YOLOv10

The comparison between YOLOX and YOLOv10 highlights the evolution of object detection models, focusing on advancements in speed, accuracy, and efficiency. Both models represent significant milestones in the YOLO series, offering unique capabilities tailored for diverse applications in computer vision.

YOLOX emphasizes robust performance and flexibility, making it suitable for real-world challenges requiring high adaptability. On the other hand, YOLOv10 introduces innovations like NMS-free architecture and optimized design, delivering superior accuracy and reduced computational overhead as detailed in the [YOLOv10 documentation](https://docs.ultralytics.com/models/yolov10/).

## mAP Comparison

This section highlights the mAP values of YOLOX and YOLOv10 across their variants, showcasing their accuracy in object detection tasks. mAP, or mean Average Precision, is a comprehensive metric that evaluates a model’s precision and recall, making it essential for comparing models like YOLOX and [YOLOv10](https://docs.ultralytics.com/models/yolov10/) on benchmarks like the [COCO dataset](https://docs.ultralytics.com/datasets/detect/coco/).

!!! tip "Accuracy"

    === "Detection (COCO)"

    	| Variant | mAP<sup>val<br>50<br>YOLOX | mAP<sup>val<br>50<br>YOLOv10 |
    	|---------------------|-------------------------------------------------------|-------------------------------------------------------|
    	| n | N/A | 39.5 |
    	| s | 40.5 | 46.7 |
    	| m | 46.9 | 51.3 |
    	| b | N/A | 52.7 |
    	| l | 49.7 | 53.3 |
    	| x | 51.1 | 54.4 |


## Speed Comparison

This section highlights the speed metrics of YOLOX and YOLOv10 across various model sizes, measured in milliseconds. By comparing inference times, it demonstrates how YOLOv10 achieves superior efficiency and lower latency, making it ideal for real-time applications. Learn more about [Ultralytics YOLOv10](https://docs.ultralytics.com/models/yolov10/) or explore the architecture of [YOLOX](https://github.com/Megvii-BaseDetection/YOLOX).

!!! tip "Speed"

    === "Detection (COCO)"

    	| Variant | Speed<br><sup>T4 TensorRT10<br>(ms)</sup><br>YOLOX | Speed<br><sup>T4 TensorRT10<br>(ms)</sup><br>YOLOv10 |
    	|---------------------|-------------------------------------------------------|-------------------------------------------------------|
    	| n | N/A | 1.56 |
    	| s | 2.56 | 2.66 |
    	| m | 5.43 | 5.48 |
    	| b | N/A | 6.54 |
    	| l | 9.04 | 8.33 |
    	| x | 16.1 | 12.2 |

## Train with Ultralytics YOLO11

Ultralytics YOLO11 excels in training models on diverse datasets, providing unparalleled flexibility and efficiency. With its advanced architecture and optimization capabilities, YOLO11 ensures high accuracy and speed during the training process. Whether you're working with general-purpose datasets like COCO8 or more specialized datasets, YOLO11 simplifies the training pipeline while delivering exceptional results.

To get started with training, utilize the [Ultralytics Python package](https://pypi.org/project/ultralytics/), which offers an intuitive interface for loading datasets, configuring hyperparameters, and monitoring performance in real-time. For more insights and guidance on training, explore [this guide on custom training](https://docs.ultralytics.com/modes/train/).

### Python Code Example

```python
from ultralytics import YOLO

# Load a pre-trained YOLO11 model
model = YOLO("yolo11.pt")

# Train the model on a custom dataset
model.train(data="custom_dataset.yaml", epochs=100, imgsz=640, batch=16)

# Validate the trained model
model.val()
```

This streamlined approach allows you to train YOLO11 models effectively while leveraging Ultralytics' cutting-edge tools.
