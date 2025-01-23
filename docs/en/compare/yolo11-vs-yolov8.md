---
comments: true
description: Discover the key differences between Ultralytics YOLO11 and YOLOv8 in this comprehensive comparison. Learn how these cutting-edge models stack up in terms of accuracy, speed, and efficiency for object detection, real-time AI, and various computer vision tasks. Explore their performance across diverse applications, from edge AI to large-scale deployments.
keywords: Ultralytics, YOLO11, YOLOv8, object detection, real-time AI, edge AI, computer vision, model comparison, artificial intelligence, AI models
---

# Ultralytics YOLO11 VS YOLOv8

# Ultralytics YOLO11 VS Ultralytics YOLOv8

Ultralytics YOLO11 and YOLOv8 represent significant milestones in the evolution of real-time object detection, each bringing unique advances to the field. This comparison highlights their strengths, enabling users to make informed decisions based on their specific requirements and application needs.

While YOLOv8 introduced groundbreaking features like an anchor-free architecture and optimized accuracy-speed tradeoffs, YOLO11 builds upon this foundation with enhanced feature extraction and greater efficiency. Both models excel in diverse tasks, from object detection to pose estimation, ensuring versatility for a wide range of use cases. [Learn more about YOLOv8](https://docs.ultralytics.com/models/yolov8/) or [explore YOLO11's capabilities](https://docs.ultralytics.com/models/yolo11/).

## mAP Comparison

This section compares the mean Average Precision (mAP) of Ultralytics YOLO11 and YOLOv8 across various model variants, highlighting their relative accuracy. mAP serves as a comprehensive metric for evaluating object detection performance, combining precision and recall to assess how effectively each model identifies objects. Learn more about [mAP](https://www.ultralytics.com/glossary/mean-average-precision-map) and its significance in model evaluation.

!!! tip "Accuracy"

    === "Detection (COCO)"

    	| Variant | mAP<sup>val<br>50<br>YOLO11 | mAP<sup>val<br>50<br>YOLOv8 |
    	|---------------------|-------------------------------------------------------|-------------------------------------------------------|
    	| n | 39.5 | 37.3 |
    	| s | 47.0 | 44.9 |
    	| m | 51.4 | 50.2 |
    	| l | 53.2 | 52.9 |
    	| x | 54.7 | 53.9 |

## Speed Comparison

This section highlights the performance differences between Ultralytics YOLO11 and YOLOv8, focusing on speed metrics in milliseconds. Faster inference times in YOLO11 demonstrate its optimization for real-time applications, ensuring efficiency across various model sizes. Learn more about [YOLOv8](https://docs.ultralytics.com/models/yolov8/) and [YOLO11](https://www.ultralytics.com/blog/ultralytics-yolo11-has-arrived-redefine-whats-possible-in-ai).

!!! tip "Speed"

    === "Detection (COCO)"

    	| Variant | Speed<br><sup>T4 TensorRT10<br>(ms)</sup><br>YOLO11 | Speed<br><sup>T4 TensorRT10<br>(ms)</sup><br>YOLOv8 |
    	|---------------------|-------------------------------------------------------|-------------------------------------------------------|
    	| n | 1.55 | 1.47 |
    	| s | 2.63 | 2.66 |
    	| m | 5.27 | 5.86 |
    	| l | 6.84 | 9.06 |
    	| x | 12.49 | 14.37 |

## Train with Ultralytics YOLO11

Training models with Ultralytics YOLO11 is a seamless experience, allowing users to fine-tune pre-trained models such as those available for COCO8 or other custom datasets. With its advanced training pipeline, YOLO11 ensures optimal performance for specific tasks, including object detection, segmentation, and more. The model supports customization through hyperparameter tuning, enabling users to adapt it to various use cases like wildlife monitoring or signature detection.

For a comprehensive guide on training, visit [Ultralytics YOLO11 Training Documentation](https://docs.ultralytics.com/modes/train/).

### Python Code Example

```python
from ultralytics import YOLO

# Load a pretrained YOLO11 model
model = YOLO("yolo11.pt")

# Train on a custom dataset
model.train(data="coco8.yaml", epochs=50, batch=16, imgsz=640)
```

This code snippet demonstrates how to initiate training for your dataset using the COCO8 configuration file. Ensure your dataset is correctly formatted for YOLO to achieve the best results.
