---
comments: true
description: Explore the detailed comparison between DAMO-YOLO and Ultralytics YOLOv5, two leading models in object detection and real-time AI. Discover their performance, efficiency, and applications in computer vision and edge AI scenarios.
keywords: DAMO-YOLO, Ultralytics YOLOv5, object detection, real-time AI, edge AI, computer vision, Ultralytics, AI models comparison
---

# DAMO-YOLO VS Ultralytics YOLOv5

Comparing DAMO-YOLO and Ultralytics YOLOv5 offers valuable insights into the evolving capabilities of object detection models. This analysis highlights their unique architectures, performance metrics, and suitability for diverse computer vision applications.

DAMO-YOLO is recognized for its efficient design and speed enhancements, while Ultralytics YOLOv5 is celebrated for its balance between accuracy and versatility. Understanding these differences is crucial for selecting the right solution for your specific use case, whether it's real-time detection or large-scale applications. Explore more about [YOLOv5's features](https://docs.ultralytics.com/models/yolov5/) and advancements.

## mAP Comparison

This section compares the mAP values of DAMO-YOLO and Ultralytics YOLOv5 models to highlight their object detection accuracy across different variants. mAP, or Mean Average Precision, evaluates the balance between precision and recall, offering a comprehensive measure of model performance. Learn more about [mAP](https://www.ultralytics.com/glossary/mean-average-precision-map) and its significance in object detection.

!!! tip "Accuracy"

    === "Detection (COCO)"

    	| Variant | mAP<sup>val<br>50<br>DAMO-YOLO | mAP<sup>val<br>50<br>YOLOv5 |
    	|---------------------|-------------------------------------------------------|-------------------------------------------------------|
    	| n | 42.0 | N/A |
    	| s | 46.0 | 37.4 |
    	| m | 49.2 | 45.4 |
    	| l | 50.8 | 49.0 |
    	| x | N/A | 50.7 |


## Speed Comparison

This section highlights the speed metrics of DAMO-YOLO and Ultralytics YOLOv5 across various model sizes, showcasing their performance in milliseconds. Detailed insights into inference times help evaluate their efficiency on diverse hardware configurations. For more on YOLOv5, visit the [Ultralytics YOLOv5 documentation](https://docs.ultralytics.com/models/yolov5/).

!!! tip "Speed"

    === "Detection (COCO)"

    	| Variant | Speed<br><sup>T4 TensorRT10<br>(ms)</sup><br>DAMO-YOLO | Speed<br><sup>T4 TensorRT10<br>(ms)</sup><br>YOLOv5 |
    	|---------------------|-------------------------------------------------------|-------------------------------------------------------|
    	| n | 2.32 | N/A |
    	| s | 3.45 | 1.92 |
    	| m | 5.09 | 4.03 |
    	| l | 7.18 | 6.61 |
    	| x | N/A | 11.89 |

## Leveraging YOLO11 for Object Counting

Ultralytics YOLO11 offers robust solutions for object counting, enabling precise identification and tallying of objects in various applications. This functionality is crucial for industries such as retail, agriculture, and traffic management, where accurate counting impacts operations and decision-making. By leveraging YOLO11's high-speed and accuracy capabilities, users can perform object counting in real-time scenarios, even in challenging environments.

For more details on how YOLO models can be used for object counting, visit the [Ultralytics Object Counting Guide](https://docs.ultralytics.com/guides/object-counting/).

### Python Code Example

```python
from ultralytics import YOLO

# Load a pretrained YOLO11 model
model = YOLO("yolo11.pt")

# Perform object detection and count objects
results = model.predict(source="video.mp4", save=True)
object_counts = results[0].boxes.cls.count()

print(f"Total Objects Detected: {object_counts}")
```

This example demonstrates how to use YOLO11 to detect and count objects in a video source effortlessly.
