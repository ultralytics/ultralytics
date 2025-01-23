---
comments: true
description: Explore an in-depth comparison between YOLOv7 and RTDETRv2, highlighting their strengths in real-time object detection, speed, accuracy, and applicability in edge AI and computer vision tasks. Discover how these models perform under various scenarios and their suitability for real-time AI applications.
keywords: YOLOv7, RTDETRv2, Ultralytics, object detection, real-time AI, edge AI, computer vision, deep learning, AI models comparison
---

# YOLOv7 VS RTDETRv2

The comparison between YOLOv7 and RTDETRv2 highlights a critical evaluation of two advanced object detection models, each excelling in unique aspects of speed, accuracy, and versatility. YOLOv7 is renowned for its groundbreaking training optimizations and efficient architecture, making it a leader in real-time object detection tasks across diverse applications.

RTDETRv2, on the other hand, leverages Vision Transformer-based design to deliver high accuracy with adaptable inference speed. By combining innovative encoding techniques and IoU-aware query selection, RTDETRv2 sets a new standard for transformer-based detectors. Explore more about [YOLOv7](https://docs.ultralytics.com/models/yolov7/) and [RTDETR](https://docs.ultralytics.com/reference/models/rtdetr/model/) for deeper insights into their capabilities.

## mAP Comparison

This section evaluates the accuracy of YOLOv7 and RTDETRv2 models through their mAP (Mean Average Precision) scores, a critical metric for object detection performance. By comparing mAP values across different variants, we can better understand how these models balance precision and recall. Learn more about [mAP](https://www.ultralytics.com/glossary/mean-average-precision-map) in object detection.

!!! tip "Accuracy"

    === "Detection (COCO)"

    	| Variant | mAP<sup>val<br>50<br>YOLOv7 | mAP<sup>val<br>50<br>RTDETRv2 |
    	|---------------------|-------------------------------------------------------|-------------------------------------------------------|
    	| s | N/A | 48.1 |
    	| m | N/A | 51.9 |
    	| l | 51.4 | 53.4 |
    	| x | 53.1 | 54.3 |


## Speed Comparison

This section highlights the performance differences between YOLOv7 and RT-DETRv2 models, focusing on speed metrics measured in milliseconds. These results demonstrate how each model scales across various input sizes and their efficiency in real-time applications. For more details, refer to the [YOLOv7 documentation](https://docs.ultralytics.com/models/yolov7/) or explore further on [RT-DETR benchmarks](https://arxiv.org/pdf/2207.02696).

!!! tip "Speed"

    === "Detection (COCO)"

    	| Variant | Speed<br><sup>T4 TensorRT10<br>(ms)</sup><br>YOLOv7 | Speed<br><sup>T4 TensorRT10<br>(ms)</sup><br>RTDETRv2 |
    	|---------------------|-------------------------------------------------------|-------------------------------------------------------|
    	| s | N/A | 5.03 |
    	| m | N/A | 7.51 |
    	| l | 6.84 | 9.76 |
    	| x | 11.57 | 15.03 |

## Benchmarking With Ultralytics YOLO11

Benchmarking is a critical functionality in Ultralytics YOLO11, enabling users to assess model performance effectively. This feature allows you to measure inference time, accuracy, and other key metrics across various devices and frameworks. Whether you're deploying models on edge devices or cloud-based systems, benchmarking helps ensure that YOLO11 delivers optimal performance tailored to your requirements.

With compatibility for multiple hardware and software environments, Ultralytics YOLO11 streamlines the comparison process, offering insights for informed decision-making. For detailed guidance, explore the [Ultralytics Documentation](https://docs.ultralytics.com/guides/) for benchmarking best practices.

### Example Python Code for Benchmarking

```python
from ultralytics import YOLO

# Load the YOLO11 model
model = YOLO("yolo11.pt")

# Perform benchmarking
results = model.benchmark(device="cuda")  # Specify 'cuda' for GPU or 'cpu' for CPU
print(results)
```

This snippet demonstrates how to perform benchmarking using YOLO11, showcasing its efficiency across different hardware configurations.
