---
comments: true
description: Compare YOLOv9 and YOLOv6-3.0, two cutting-edge object detection models by Ultralytics. Explore their performance, efficiency, and advancements in real-time AI, edge AI, and computer vision, highlighting their capabilities for diverse applications.
keywords: YOLOv9, YOLOv6-3.0, Ultralytics, object detection, real-time AI, edge AI, computer vision, model comparison
---

# YOLOv9 VS YOLOv6-3.0

The comparison between YOLOv9 and YOLOv6-3.0 highlights two significant advancements in the evolution of YOLO models. Both models have made remarkable contributions to object detection, showcasing unique strengths in efficiency, speed, and accuracy for various computer vision tasks.

YOLOv9 focuses on delivering cutting-edge performance through refined architectures and enhanced feature extraction techniques. Meanwhile, YOLOv6-3.0 emphasizes holistic efficiency and accuracy-driven design, incorporating lightweight classification heads and spatial-channel decoupled downsampling to optimize performance. Dive into this comparison to explore their capabilities and identify the ideal solution for your needs. Learn more about [YOLOv9](https://www.ultralytics.com/blog/ultralytics-yolo11-has-arrived-redefine-whats-possible-in-ai) and [YOLOv6-3.0](https://docs.ultralytics.com/models/yolov10/).

## mAP Comparison

This section highlights the mAP values of YOLOv9 and YOLOv6-3.0, showcasing their accuracy across various model variants. Mean Average Precision (mAP) is a critical metric in object detection, balancing precision and recall to evaluate model performance comprehensively. Learn more about [mAP calculation](https://www.ultralytics.com/glossary/mean-average-precision-map) and its significance in computer vision.

!!! tip "Accuracy"

    === "Detection (COCO)"

    	| Variant | mAP<sup>val<br>50<br>YOLOv9 | mAP<sup>val<br>50<br>YOLOv6-3.0 |
    	|---------------------|-------------------------------------------------------|-------------------------------------------------------|
    	| n | 37.8 | 37.5 |
    	| s | 46.5 | 45.0 |
    	| m | 51.5 | 50.0 |
    	| l | 52.8 | 52.8 |
    	| x | 55.1 | N/A |

## Speed Comparison

This section highlights the speed differences between YOLOv9 and YOLOv6-3.0 models across various sizes, measured in milliseconds. These metrics, tested on advanced frameworks like TensorRT, underscore the efficiency of each model in real-time object detection scenarios. For more on YOLO versions, visit the [Ultralytics YOLOv9 documentation](https://docs.ultralytics.com/models/yolov9/).

!!! tip "Speed"

    === "Detection (COCO)"

    	| Variant | Speed<br><sup>T4 TensorRT10<br>(ms)</sup><br>YOLOv9 | Speed<br><sup>T4 TensorRT10<br>(ms)</sup><br>YOLOv6-3.0 |
    	|---------------------|-------------------------------------------------------|-------------------------------------------------------|
    	| n | 2.3 | 1.17 |
    	| s | 3.54 | 2.66 |
    	| m | 6.43 | 5.28 |
    	| l | 7.16 | 8.95 |
    	| x | 16.77 | N/A |

## YOLO11 Functionalities: Predict

The Predict functionality in Ultralytics YOLO11 allows users to perform real-time inference on images, videos, or streams, making it ideal for a variety of applications like security systems, traffic monitoring, and wildlife observation. With its cutting-edge architecture, YOLO11 ensures high-speed predictions without compromising accuracy.

To get started with predictions, simply load a trained model and pass your input data. For instance, in Python:

```python
from ultralytics import YOLO

# Load a pre-trained YOLO11 model
model = YOLO("yolo11.pt")

# Run predictions on an image
results = model.predict(source="image.jpg", save=True)

# Display results
results.show()
```

The framework supports multiple export options, allowing users to deploy their models across various platforms, including ONNX and TensorFlow Lite. For more insights on how the Predict functionality works, visit the [Ultralytics Documentation](https://docs.ultralytics.com/modes/predict/). This feature is designed to simplify tasks like object detection, segmentation, and classification, ensuring efficiency for diverse real-world scenarios.
