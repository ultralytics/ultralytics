---
comments: true
description: Compare YOLOv9 and YOLOX, two cutting-edge object detection models, to explore their performance in real-time AI applications. Discover how these models excel in computer vision tasks, balancing accuracy, speed, and efficiency for edge AI solutions.
keywords: YOLOv9, YOLOX, Ultralytics, object detection, real-time AI, edge AI, computer vision
---

# YOLOv9 VS YOLOX

The comparison between YOLOv9 and YOLOX highlights the evolution of object detection models and their unique approaches to tackling real-world challenges. Both models represent significant milestones in AI development, with YOLOv9 focusing on efficiency and adaptability, and YOLOX introducing innovative design choices for enhanced performance.

YOLOv9, part of the [Ultralytics YOLO](https://docs.ultralytics.com/models/yolov10/) family, is designed for optimized deployment across diverse platforms, making it a robust choice for edge AI. On the other hand, YOLOX emphasizes dynamic training strategies and modular architecture, offering flexibility for custom applications. Explore how these models excel in different scenarios and redefine object detection benchmarks.

## mAP Comparison

This section evaluates the performance of YOLOv9 and YOLOX models based on their mAP (Mean Average Precision) values, which reflect their accuracy across various object classes and IoU thresholds. For a deeper understanding of mAP and its role in object detection, visit the [Ultralytics Glossary on mAP](https://www.ultralytics.com/glossary/mean-average-precision-map).

!!! tip "Accuracy"

    === "Detection (COCO)"

    	| Variant | mAP<sup>val<br>50<br>YOLOv9 | mAP<sup>val<br>50<br>YOLOX |
    	|---------------------|-------------------------------------------------------|-------------------------------------------------------|
    	| n | 37.8 | N/A |
    	| s | 46.5 | 40.5 |
    	| m | 51.5 | 46.9 |
    	| l | 52.8 | 49.7 |
    	| x | 55.1 | 51.1 |

## Speed Comparison

This section highlights the speed differences between YOLOv9 and YOLOX across various model sizes. Measured in milliseconds, these metrics demonstrate the efficiency of each model in delivering real-time performance, with insights into their suitability for different deployment scenarios. Explore more about [Ultralytics YOLO models](https://docs.ultralytics.com/models/yolov10/) for speed optimization.

!!! tip "Speed"

    === "Detection (COCO)"

    	| Variant | Speed<br><sup>T4 TensorRT10<br>(ms)</sup><br>YOLOv9 | Speed<br><sup>T4 TensorRT10<br>(ms)</sup><br>YOLOX |
    	|---------------------|-------------------------------------------------------|-------------------------------------------------------|
    	| n | 2.3 | N/A |
    	| s | 3.54 | 2.56 |
    	| m | 6.43 | 5.43 |
    	| l | 7.16 | 9.04 |
    	| x | 16.77 | 16.1 |

## Using YOLO11 for Object Detection With OpenVINO Integration

Ultralytics YOLO11 seamlessly integrates with OpenVINO, enabling optimized inference for edge devices while maintaining high accuracy and efficiency. OpenVINO is particularly beneficial for deploying YOLO11 models in resource-constrained environments, such as IoT devices or embedded systems. By leveraging this integration, users can achieve reduced latency and improved throughput for real-time applications like surveillance and smart city infrastructure.

For more details on deploying YOLO11 models with OpenVINO, refer to the [OpenVINO Latency vs Throughput Modes Guide](https://docs.ultralytics.com/guides/) to optimize performance effectively.

### Python Code Snippet for OpenVINO Export

```python
from ultralytics import YOLO

# Load a pretrained YOLO11 model
model = YOLO("yolov8n.pt")

# Export the model to OpenVINO format
model.export(format="openvino")
```

This code highlights how easily YOLO11 models can be exported to OpenVINO format, enabling you to deploy optimized models for edge computing. Explore more about [model deployment options](https://docs.ultralytics.com/guides/).
