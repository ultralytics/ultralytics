---
comments: true
description: Explore the comprehensive comparison between Ultralytics YOLOv8 and RTDETRv2, two cutting-edge models in object detection. Discover their performance, speed, and accuracy for real-time AI and edge AI applications in computer vision.
keywords: Ultralytics, YOLOv8, RTDETRv2, object detection, real-time AI, edge AI, computer vision, model comparison
---

# Ultralytics YOLOv8 VS RTDETRv2

The comparison between Ultralytics YOLOv8 and RTDETRv2 highlights two cutting-edge models in the field of computer vision, showcasing their strengths in real-time object detection tasks. Both models cater to diverse applications, offering unique features that push the boundaries of speed, accuracy, and efficiency.

Ultralytics YOLOv8 excels with its state-of-the-art architecture, including an anchor-free detection head and optimized workflows for seamless integration and extensibility. On the other hand, RTDETRv2 focuses on delivering robust performance with minimal latency, making it a strong contender for real-time applications in dynamic environments. For more details about YOLOv8, visit the [Ultralytics YOLOv8 documentation](https://docs.ultralytics.com/models/yolov8/).

## mAP Comparison

This section highlights the mAP (mean Average Precision) performance of Ultralytics YOLOv8 and RT-DETRv2 across various model variants. mAP, a critical metric in object detection, evaluates model accuracy by balancing precision and recall. Learn more about [mAP and its significance](https://www.ultralytics.com/glossary/mean-average-precision-map).

!!! tip "Accuracy"

    === "Detection (COCO)"

    	| Variant | mAP<sup>val<br>50<br>YOLOv8 | mAP<sup>val<br>50<br>RTDETRv2 |
    	|---------------------|-------------------------------------------------------|-------------------------------------------------------|
    	| n | 37.3 | N/A |
    	| s | 44.9 | 48.1 |
    	| m | 50.2 | 51.9 |
    	| l | 52.9 | 53.4 |
    	| x | 53.9 | 54.3 |

## Speed Comparison

This section highlights the performance differences between Ultralytics YOLOv8 and RTDETRv2 in terms of speed, measured in milliseconds. Faster inference times across various model sizes demonstrate the efficiency of YOLOv8, which is designed for real-time applications. For more details on YOLOv8's performance, visit the [Ultralytics YOLOv8 documentation](https://docs.ultralytics.com/models/yolov8/).

!!! tip "Speed"

    === "Detection (COCO)"

    	| Variant | Speed<br><sup>T4 TensorRT10<br>(ms)</sup><br>YOLOv8 | Speed<br><sup>T4 TensorRT10<br>(ms)</sup><br>RTDETRv2 |
    	|---------------------|-------------------------------------------------------|-------------------------------------------------------|
    	| n | 1.47 | N/A |
    	| s | 2.66 | 5.03 |
    	| m | 5.86 | 7.51 |
    	| l | 9.06 | 9.76 |
    	| x | 14.37 | 15.03 |

## Enhancing Object Detection With YOLO11 Integration: OpenVINO

Ultralytics YOLO11 seamlessly integrates with [OpenVINO](https://www.ultralytics.com/glossary/model-deployment) to optimize performance for real-time object detection tasks. OpenVINO, developed by Intel, allows YOLO11 models to leverage hardware acceleration, significantly improving inference speed and efficiency on Intel processors and other compatible devices. This integration is ideal for deploying YOLO11 in edge computing scenarios, such as smart city infrastructure or retail environments.

OpenVINO’s latency and throughput optimization modes ensure that YOLO11 delivers precise and fast predictions, even in resource-constrained environments. By combining YOLO11’s advanced capabilities with OpenVINO’s deployment features, developers can achieve high-performance object detection with minimal latency.

For more details on integrating YOLO11 with OpenVINO, explore the [OpenVINO Latency vs Throughput Modes Guide](https://docs.ultralytics.com/guides/).

### Python Code Example: Export YOLO11 to OpenVINO

```python
from ultralytics import YOLO

# Load a pretrained YOLO11 model
model = YOLO("yolo11.pt")

# Export the model to OpenVINO format
model.export(format="openvino")
```
