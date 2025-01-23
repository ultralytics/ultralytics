---
comments: true
description: Compare Ultralytics YOLO11 and YOLOv9 to discover advancements in object detection, real-time AI, and edge AI. Explore how these cutting-edge computer vision models redefine accuracy, speed, and efficiency in diverse applications.
keywords: Ultralytics, YOLO11, YOLOv9, object detection, real-time AI, edge AI, computer vision, model comparison
---

# Ultralytics YOLO11 VS YOLOv9

Choosing the right model for your computer vision projects can be challenging, especially when comparing two groundbreaking innovations like Ultralytics YOLO11 and YOLOv9. This page explores the advancements and trade-offs between these models, helping you determine which one aligns best with your specific needs.

Ultralytics YOLO11 represents the pinnacle of accuracy, speed, and efficiency, building on the robust foundation of YOLOv9. While YOLOv9 introduced significant improvements in feature extraction and architectural design, YOLO11 takes it further with enhanced processing speeds and reduced parameter requirements. Discover how these models redefine what's possible in AI by exploring [Ultralytics YOLO11's capabilities](https://docs.ultralytics.com/models/yolo11/) and [YOLOv9's impact](https://www.youtube.com/watch?v=ZF7EAodHn1U&t=1s).

## mAP Comparison

This section highlights the mAP values, a key metric for evaluating the accuracy of object detection models, to compare Ultralytics YOLO11 and YOLOv9 across their variants. The comparison showcases YOLO11's advancements in precision and efficiency, building on YOLOv9's strong performance on datasets like [COCO](https://docs.ultralytics.com/datasets/detect/coco/).

!!! tip "Accuracy"

    === "Detection (COCO)"

    	| Variant | mAP<sup>val<br>50<br>YOLO11 | mAP<sup>val<br>50<br>YOLOv9 |
    	|---------------------|-------------------------------------------------------|-------------------------------------------------------|
    	| n | 39.5 | 37.8 |
    	| s | 47.0 | 46.5 |
    	| m | 51.4 | 51.5 |
    	| l | 53.2 | 52.8 |
    	| x | 54.7 | 55.1 |


## Speed Comparison

This section highlights the speed metrics of Ultralytics YOLO11 and YOLOv9 across various model sizes, measured in milliseconds. Faster inference times in YOLO11 make it an ideal choice for applications requiring real-time performance, as detailed in [Ultralytics YOLO11 documentation](https://docs.ultralytics.com/models/yolo11/).

!!! tip "Speed"

    === "Detection (COCO)"

    	| Variant | Speed<br><sup>T4 TensorRT10<br>(ms)</sup><br>YOLO11 | Speed<br><sup>T4 TensorRT10<br>(ms)</sup><br>YOLOv9 |
    	|---------------------|-------------------------------------------------------|-------------------------------------------------------|
    	| n | 1.55 | 2.3 |
    	| s | 2.63 | 3.54 |
    	| m | 5.27 | 6.43 |
    	| l | 6.84 | 7.16 |
    	| x | 12.49 | 16.77 |

## Export Functionality in Ultralytics YOLO11

Ultralytics YOLO11 offers seamless **export functionality**, enabling models to be converted into various formats like ONNX, OpenVINO, TensorFlow Lite, and more. This feature ensures compatibility across diverse deployment environments, from edge devices to cloud platforms. For instance, exporting models to ONNX or TensorFlow Lite makes them suitable for mobile and embedded applications, improving scalability and performance.

Exporting is straightforward and can be done with a single command using the Ultralytics Python package. For more details, explore the [Ultralytics YOLO Export Guide](https://docs.ultralytics.com/modes/export/).

### Python Code Snippet for Exporting a Model

```python
from ultralytics import YOLO

# Load a YOLO11 model
model = YOLO("yolov11.pt")

# Export the model to ONNX format
model.export(format="onnx")
```

This code snippet demonstrates how to quickly export a YOLO11 model to ONNX format, enabling integration into a broad range of applications.
