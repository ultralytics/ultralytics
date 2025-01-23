---
comments: true
description: Compare YOLOv7 and Ultralytics YOLOv8, two cutting-edge models in object detection and real-time AI. Discover their performance in computer vision, edge AI applications, and how they excel in accuracy, speed, and versatility.
keywords: YOLOv7, Ultralytics YOLOv8, object detection, real-time AI, edge AI, computer vision, machine learning, model comparison
---

# YOLOv7 VS Ultralytics YOLOv8

The comparison between YOLOv7 and Ultralytics YOLOv8 highlights the strides in computer vision technology, particularly in object detection and segmentation. These models represent significant advancements, each offering unique features and performance metrics tailored for diverse real-world applications.

YOLOv7 is renowned for its balance of speed and accuracy, making it a strong contender for real-time tasks. Meanwhile, Ultralytics YOLOv8 introduces state-of-the-art innovations, including an anchor-free architecture and seamless integration with all YOLO versions, positioning it as a versatile choice for cutting-edge AI solutions. Learn more about [Ultralytics YOLOv8](https://www.ultralytics.com/blog/introducing-ultralytics-yolov8) and its capabilities.

## mAP Comparison

This section highlights the Mean Average Precision (mAP) of YOLOv7 and Ultralytics YOLOv8 across their variants, showcasing their accuracy in detecting objects consistently across classes and thresholds. Learn more about [mAP](https://www.ultralytics.com/glossary/mean-average-precision-map) and its role in evaluating object detection models.

!!! tip "Accuracy"

    === "Detection (COCO)"

    	| Variant | mAP<sup>val<br>50<br>YOLOv7 | mAP<sup>val<br>50<br>YOLOv8 |
    	|---------------------|-------------------------------------------------------|-------------------------------------------------------|
    	| n | N/A | 37.3 |
    	| s | N/A | 44.9 |
    	| m | N/A | 50.2 |
    	| l | 51.4 | 52.9 |
    	| x | 53.1 | 53.9 |

## Speed Comparison

This section highlights the speed performance of YOLOv7 and Ultralytics YOLOv8 models across various sizes, measured in milliseconds per inference. These metrics provide a clear understanding of how each model handles real-time tasks. Learn more about [YOLOv7](https://docs.ultralytics.com/models/yolov7/) and [Ultralytics YOLOv8](https://www.ultralytics.com/blog/object-detection-with-a-pre-trained-ultralytics-yolov8-model) for detailed insights.

!!! tip "Speed"

    === "Detection (COCO)"

    	| Variant | Speed<br><sup>T4 TensorRT10<br>(ms)</sup><br>YOLOv7 | Speed<br><sup>T4 TensorRT10<br>(ms)</sup><br>YOLOv8 |
    	|---------------------|-------------------------------------------------------|-------------------------------------------------------|
    	| n | N/A | 1.47 |
    	| s | N/A | 2.66 |
    	| m | N/A | 5.86 |
    	| l | 6.84 | 9.06 |
    	| x | 11.57 | 14.37 |

## Benchmark Functionalities in Ultralytics YOLO11

Ultralytics YOLO11 offers an advanced benchmarking feature to evaluate model performance across various tasks. This functionality enables users to measure metrics like speed, accuracy, and resource efficiency, providing insights into model effectiveness for real-world applications. Benchmarking is crucial for identifying the best configurations and optimizing deployments in diverse environments.

With YOLO11, you can benchmark models on different datasets, such as COCO8 or custom datasets, ensuring they meet your specific requirements. This feature is particularly valuable when comparing performance across multiple hardware platforms or integration formats like ONNX or TensorFlow Lite.

For more details, explore the [Ultralytics YOLO11 documentation](https://docs.ultralytics.com/guides/) to understand benchmarking best practices and tools.

```python
from ultralytics import YOLO

# Load the YOLO11 model
model = YOLO("yolo11.pt")

# Benchmark the model
results = model.benchmark(data="coco8.yaml")
print(results)
```
