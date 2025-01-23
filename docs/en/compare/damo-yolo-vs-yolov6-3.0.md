---
comments: true
description: Explore the detailed comparison between DAMO-YOLO and YOLOv6-3.0, two advanced object detection models. Discover how they stack up in terms of speed, accuracy, and performance for real-time AI, edge AI, and computer vision applications. Learn which model fits your use case best with insights into their unique features and capabilities.
keywords: DAMO-YOLO, YOLOv6-3.0, Ultralytics, object detection, real-time AI, edge AI, computer vision, model comparison
---

# DAMO-YOLO VS YOLOv6-3.0

The comparison between DAMO-YOLO and YOLOv6-3.0 highlights the advancements in object detection technology and their applications across industries. Both models push the boundaries of accuracy and speed, making them valuable tools for real-time computer vision tasks.

DAMO-YOLO focuses on delivering high efficiency and adaptability, optimized for edge deployments and diverse scenarios. Meanwhile, YOLOv6-3.0 builds on the strengths of its predecessors, offering precision and scalability for large-scale projects. Explore how these models differ and excel in various use cases.

## mAP Comparison

The mAP (Mean Average Precision) metric evaluates the accuracy of object detection models like DAMO-YOLO and YOLOv6-3.0 across various thresholds, providing a comprehensive measure of performance. By comparing these models, you can assess their effectiveness in identifying and localizing objects across different datasets. Learn more about [mAP here](https://www.ultralytics.com/glossary/mean-average-precision-map).

!!! tip "Accuracy"

    === "Detection (COCO)"

    	| Variant | mAP<sup>val<br>50<br>DAMO-YOLO | mAP<sup>val<br>50<br>YOLOv6-3.0 |
    	|---------------------|-------------------------------------------------------|-------------------------------------------------------|
    	| n | 42.0 | 37.5 |
    	| s | 46.0 | 45.0 |
    	| m | 49.2 | 50.0 |
    	| l | 50.8 | 52.8 |


## Speed Comparison

This section highlights the speed metrics of DAMO-YOLO and YOLOv6-3.0 models, measured in milliseconds, to reflect their performance across various sizes. Analyzing these metrics provides insights into the efficiency and suitability of each model for real-time applications. For more details on YOLOv6-3.0, visit the [official benchmarks page](https://docs.ultralytics.com/reference/utils/benchmarks/).

!!! tip "Speed"

    === "Detection (COCO)"

    	| Variant | Speed<br><sup>T4 TensorRT10<br>(ms)</sup><br>DAMO-YOLO | Speed<br><sup>T4 TensorRT10<br>(ms)</sup><br>YOLOv6-3.0 |
    	|---------------------|-------------------------------------------------------|-------------------------------------------------------|
    	| n | 2.32 | 1.17 |
    	| s | 3.45 | 2.66 |
    	| m | 5.09 | 5.28 |
    	| l | 7.18 | 8.95 |

## Train with Ultralytics YOLO11

Training models with Ultralytics YOLO11 is a streamlined process that enables users to fine-tune the model on custom datasets for a variety of applications. Whether you're working on datasets like COCO8, African wildlife, or a niche dataset like signature detection, YOLO11 provides exceptional flexibility and performance.

With the Ultralytics Python package, you can start training in just a few lines of code. The model supports advanced features like automated hyperparameter tuning and K-Fold cross-validation for improved generalization. For comprehensive guidance, refer to the [Ultralytics Training Guide](https://docs.ultralytics.com/modes/train/), which offers detailed instructions on optimizing training processes and achieving the best results.

```python
from ultralytics import YOLO

# Load a pretrained YOLO11 model
model = YOLO("yolo11.pt")

# Train the model on a custom dataset
model.train(data="custom_dataset.yaml", epochs=50, batch=16, imgsz=640)
```

Explore more about training and enhancing YOLO11 models in the [Common Issues Guide](https://docs.ultralytics.com/guides/yolo-common-issues/) to troubleshoot effectively.
