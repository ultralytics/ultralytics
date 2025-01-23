---
comments: true
description: Compare YOLOX and YOLOv9 to uncover their strengths in real-time object detection, efficiency, and performance. Explore how these models excel in computer vision and edge AI applications, powered by Ultralytics' advancements in state-of-the-art technology.
keywords: YOLOX, YOLOv9, Ultralytics, object detection, real-time AI, edge AI, computer vision, model comparison, efficiency, performance benchmarks
---

# YOLOX VS YOLOv9

YOLOX and YOLOv9 represent significant advancements in the field of object detection, each bringing unique innovations to computer vision tasks. This comparison aims to explore their strengths, performance benchmarks, and practical applications, offering insights into their suitability for diverse use cases.

While YOLOX is celebrated for its efficiency and robust adaptability across platforms, YOLOv9 introduces enhanced architectural designs and improved accuracy-speed tradeoffs. Understanding these key differences will help developers make informed decisions for real-time AI deployment. Learn more about YOLO models in the [Ultralytics documentation](https://docs.ultralytics.com/models/yolov8/).

## mAP Comparison

This section highlights the mAP performance of YOLOX and YOLOv9 across their respective variants, showcasing their accuracy in detecting and classifying objects. mAP values, such as mAP@.5 and mAP@.5:.95, provide a comprehensive measure of how effectively these models perform on datasets like [COCO](https://docs.ultralytics.com/datasets/detect/coco/). For more on mAP metrics, explore the [Ultralytics Glossary](https://www.ultralytics.com/glossary/mean-average-precision-map).

!!! tip "Accuracy"

    === "Detection (COCO)"

    	| Variant | mAP<sup>val<br>50<br>YOLOX | mAP<sup>val<br>50<br>YOLOv9 |
    	|---------------------|-------------------------------------------------------|-------------------------------------------------------|
    	| n | N/A | 37.8 |
    	| s | 40.5 | 46.5 |
    	| m | 46.9 | 51.5 |
    	| l | 49.7 | 52.8 |
    	| x | 51.1 | 55.1 |


## Speed Comparison

This section highlights the speed metrics of YOLOX and YOLOv9 models, measured in milliseconds across different sizes. These metrics provide valuable insights into the real-time performance and efficiency of both models, helping to identify the best fit for various applications. For more on YOLOv9, visit the [Ultralytics YOLOv9 documentation](https://docs.ultralytics.com/models/yolov9/).

!!! tip "Speed"

    === "Detection (COCO)"

    	| Variant | Speed<br><sup>T4 TensorRT10<br>(ms)</sup><br>YOLOX | Speed<br><sup>T4 TensorRT10<br>(ms)</sup><br>YOLOv9 |
    	|---------------------|-------------------------------------------------------|-------------------------------------------------------|
    	| n | N/A | 2.3 |
    	| s | 2.56 | 3.54 |
    	| m | 5.43 | 6.43 |
    	| l | 9.04 | 7.16 |
    	| x | 16.1 | 16.77 |

## Exploring the Predict Functionality in Ultralytics YOLO11

The **Predict** functionality in Ultralytics YOLO11 enables users to perform real-time inference on images or videos, identifying objects, poses, or other visual elements with remarkable accuracy. YOLO11 builds on its predecessors by offering enhanced speed and precision, making it suitable for both edge devices and high-performance applications.

To get started with predicting using YOLO11, you can use the Ultralytics Python package. Below is a simple example to run predictions on an image:

```python
from ultralytics import YOLO

# Load a pre-trained YOLO11 model
model = YOLO("yolo11.pt")

# Run predictions on an image
results = model.predict(source="image.jpg", save=True)

# Display results
results.show()
```

This functionality supports various use cases such as object detection, segmentation, and pose estimation. For more details, check the [Ultralytics YOLO11 Prediction Guide](https://docs.ultralytics.com/modes/predict/). Additionally, explore the [Ultralytics HUB](https://www.ultralytics.com/hub) for a no-code prediction experience.
