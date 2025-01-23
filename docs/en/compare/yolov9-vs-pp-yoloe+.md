---
comments: true
description: Explore an in-depth comparison between YOLOv9 and PP-YOLOE+, two leading models in real-time object detection. Discover how these cutting-edge solutions perform on metrics like mAP, speed, and computational efficiency, making them ideal for diverse computer vision applications, including edge AI and real-time AI tasks. Learn how YOLOv9's advancements in accuracy and efficiency stack up against PP-YOLOE+'s performance optimizations for various use cases.
keywords: YOLOv9, PP-YOLOE+, Ultralytics, object detection, real-time AI, edge AI, computer vision, mAP, AI models comparison, performance analysis
---

# YOLOv9 VS PP-YOLOE+

Choosing the right object detection model is critical for achieving optimal performance in real-world applications. This comparison between YOLOv9 and PP-YOLOE+ provides a detailed analysis of their strengths, helping developers make informed decisions based on accuracy, speed, and efficiency.

YOLOv9 stands out with its advanced architectural optimizations like Programmable Gradient Information, enabling greater efficiency and accuracy on datasets like [COCO](https://docs.ultralytics.com/datasets/detect/coco/). Meanwhile, PP-YOLOE+ offers robust performance with a focus on lightweight deployment, making it a strong contender for edge AI applications. For a deeper dive into YOLOv9’s innovations, check out its [documentation](https://docs.ultralytics.com/models/yolov9/).

## mAP Comparison

This section evaluates the performance of YOLOv9 and PP-YOLOE+ models by comparing their mAP values. Mean Average Precision (mAP) serves as a crucial metric, reflecting the accuracy of these models in detecting and localizing objects across various variants. Explore more about [mAP metrics](https://www.ultralytics.com/glossary/mean-average-precision-map) to understand their significance in model evaluation.

!!! tip "Accuracy"

    === "Detection (COCO)"

    	| Variant | mAP<sup>val<br>50<br>YOLOv9 | mAP<sup>val<br>50<br>PP-YOLOE+ |
    	|---------------------|-------------------------------------------------------|-------------------------------------------------------|
    	| n | 37.8 | 39.9 |
    	| s | 46.5 | 43.7 |
    	| m | 51.5 | 49.8 |
    	| l | 52.8 | 52.9 |
    	| x | 55.1 | 54.7 |

## Speed Comparison

This section highlights the speed performance of YOLOv9 and PP-YOLOE+ across various model sizes, measured in milliseconds per inference. These metrics provide a clear understanding of how each model balances efficiency and computational requirements for real-time applications. For more details, explore the [Ultralytics YOLOv9 documentation](https://docs.ultralytics.com/models/yolov9/) or [PP-YOLOE+ performance benchmarks](https://github.com/PaddlePaddle/PaddleDetection).

!!! tip "Speed"

    === "Detection (COCO)"

    	| Variant | Speed<br><sup>T4 TensorRT10<br>(ms)</sup><br>YOLOv9 | Speed<br><sup>T4 TensorRT10<br>(ms)</sup><br>PP-YOLOE+ |
    	|---------------------|-------------------------------------------------------|-------------------------------------------------------|
    	| n | 2.3 | 2.84 |
    	| s | 3.54 | 2.62 |
    	| m | 6.43 | 5.56 |
    	| l | 7.16 | 8.36 |
    	| x | 16.77 | 14.3 |

## Exploring the Export Functionality in YOLO11

Ultralytics YOLO11 offers advanced **Export** functionality, allowing users to convert models into various formats compatible with different deployment environments. Supported formats include ONNX, OpenVINO, TensorFlow Lite, and more. This feature ensures seamless integration into diverse platforms, from cloud-based solutions to edge devices. Exporting models is straightforward, making it ideal for developers aiming to deploy YOLO11 in production efficiently.

For instance, exporting to ONNX enables compatibility with a wide range of AI frameworks and tools, enhancing flexibility in real-world applications. Learn more about YOLO11's export capabilities by visiting the [Model Deployment Options](https://docs.ultralytics.com/guides/) guide.

### Python Code Example for Exporting YOLO11 Models

```python
from ultralytics import YOLO

# Load a YOLO11 model
model = YOLO("yolo11.pt")

# Export the model to ONNX format
model.export(format="onnx")
```

This simple code snippet demonstrates how to export a YOLO11 model for use in ONNX-supported platforms, ensuring optimal performance and compatibility.
