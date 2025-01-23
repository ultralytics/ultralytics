---
comments: true
description: Explore the ultimate comparison between Ultralytics YOLOv5 and YOLOX, two cutting-edge object detection models. Discover their performance in real-time AI, edge AI applications, and computer vision tasks, highlighting speed, accuracy, and versatility for diverse use cases.
keywords: Ultralytics, YOLOv5, YOLOX, object detection, real-time AI, edge AI, computer vision, model comparison, deep learning
---

# Ultralytics YOLOv5 VS YOLOX

The comparison of Ultralytics YOLOv5 and YOLOX highlights two leading object detection frameworks, each representing significant advancements in computer vision. Both models are widely recognized for their accuracy, speed, and versatility, making this evaluation crucial for developers and researchers seeking optimal solutions for diverse use cases.

Ultralytics YOLOv5 is celebrated for its simplicity, performance, and user-focused design, supported by extensive documentation and community engagement. On the other hand, YOLOX introduces innovative anchor-free mechanisms and modular designs that enhance detection efficiency. Dive into this comparison to explore the unique strengths and applications of each framework. Learn more about [Ultralytics YOLOv5](https://github.com/ultralytics/yolov5) and [YOLOX](https://github.com/Megvii-BaseDetection/YOLOX).

## mAP Comparison

This section evaluates the mAP values of Ultralytics YOLOv5 and YOLOX models, highlighting their accuracy across various configurations. Mean Average Precision (mAP) is a key metric that reflects the models' ability to detect and classify objects effectively across different thresholds. Learn more about [mAP and its significance](https://www.ultralytics.com/glossary/mean-average-precision-map) in object detection.

!!! tip "Accuracy"

    === "Detection (COCO)"

    	| Variant | mAP<sup>val<br>50<br>YOLOv5 | mAP<sup>val<br>50<br>YOLOX |
    	|---------------------|-------------------------------------------------------|-------------------------------------------------------|
    	| s | 37.4 | 40.5 |
    	| m | 45.4 | 46.9 |
    	| l | 49.0 | 49.7 |
    	| x | 50.7 | 51.1 |


## Speed Comparison

This section highlights the speed performance differences between Ultralytics YOLOv5 and YOLOX models across various sizes. Speed metrics in milliseconds provide insights into inference efficiency, helping you evaluate how these models perform in real-world scenarios. For more details, explore the [YOLOv5 documentation](https://docs.ultralytics.com/models/yolov5/) and [YOLOX resources](https://github.com/Megvii-BaseDetection/YOLOX).

!!! tip "Speed"

    === "Detection (COCO)"

    	| Variant | Speed<br><sup>T4 TensorRT10<br>(ms)</sup><br>YOLOv5 | Speed<br><sup>T4 TensorRT10<br>(ms)</sup><br>YOLOX |
    	|---------------------|-------------------------------------------------------|-------------------------------------------------------|
    	| s | 1.92 | 2.56 |
    	| m | 4.03 | 5.43 |
    	| l | 6.61 | 9.04 |
    	| x | 11.89 | 16.1 |

## Export Functionality in Ultralytics YOLO11

Ultralytics YOLO11 provides robust export capabilities, enabling seamless integration of models into various deployment environments. The export functionality supports multiple formats, including ONNX, OpenVINO, TensorFlow Lite, and more. This flexibility ensures that YOLO11 models can be deployed efficiently across diverse hardware and platforms, from edge devices to cloud-based systems.

One of the key benefits of this feature is the ability to optimize models for inference, reducing latency and ensuring real-time performance. For example, exporting to TensorFlow Lite can enable lightweight deployment on mobile and IoT devices, while OpenVINO enhances performance for Intel-based hardware.

For more details on export options, refer to the [Ultralytics documentation](https://docs.ultralytics.com/guides/).

### Python Code Example

```python
from ultralytics import YOLO

# Load the YOLO model
model = YOLO("yolov11.pt")

# Export the model to ONNX format
model.export(format="onnx", dynamic=True)
```

This code snippet demonstrates how to export a YOLO11 model to ONNX format, making it ready for deployment in supported environments.
