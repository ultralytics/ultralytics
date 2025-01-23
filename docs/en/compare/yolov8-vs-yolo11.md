---
comments: true
description: Compare Ultralytics YOLOv8 and YOLO11 to discover the advancements in real-time AI, object detection, and edge AI. Explore how YOLO11 redefines computer vision with superior accuracy, speed, and efficiency, building on the foundation of YOLOv8's capabilities.
keywords: Ultralytics YOLOv8, Ultralytics YOLO11, object detection, real-time AI, edge AI, computer vision, YOLO comparison, AI models
---

# Ultralytics YOLOv8 VS YOLO11

# Ultralytics YOLOv8 VS Ultralytics YOLO11

Ultralytics YOLOv8 and YOLO11 represent significant advancements in real-time object detection, segmentation, and classification. This comparison explores their unique strengths, providing insights into their efficiency, accuracy, and adaptability across a variety of applications.

YOLOv8, released in early 2023, set a benchmark with its optimized accuracy-speed tradeoff and anchor-free detection capabilities. Meanwhile, the newer Ultralytics YOLO11 takes innovation further with enhanced feature extraction, fewer parameters, and improved performance on the [COCO dataset](https://docs.ultralytics.com/datasets/detect/coco/), making it ideal for cutting-edge and resource-efficient applications.

## mAP Comparison

This section compares the mAP values of Ultralytics YOLOv8 and Ultralytics YOLO11 to highlight their accuracy across various tasks. Mean Average Precision (mAP) serves as a critical metric for evaluating object detection performance, with YOLO11 achieving higher precision while using fewer parameters. Learn more about [mAP](https://www.ultralytics.com/glossary/mean-average-precision-map) and its importance in benchmarking.

!!! tip "Accuracy"

    === "Detection (COCO)"

    	| Variant | mAP<sup>val<br>50<br>YOLOv8 | mAP<sup>val<br>50<br>YOLO11 |
    	|---------------------|-------------------------------------------------------|-------------------------------------------------------|
    	| n | 37.3 | 39.5 |
    	| s | 44.9 | 47.0 |
    	| m | 50.2 | 51.4 |
    	| l | 52.9 | 53.2 |
    	| x | 53.9 | 54.7 |

## Speed Comparison

This section highlights the performance of Ultralytics YOLOv8 and YOLO11 models by analyzing speed metrics in milliseconds across various model sizes. Faster inference times, as showcased by YOLO11, make it ideal for real-time applications like [object detection](https://docs.ultralytics.com/tasks/detect/) and [edge deployment](https://docs.ultralytics.com/guides/model-deployment-options/), reflecting its efficiency and precision advancements.

!!! tip "Speed"

    === "Detection (COCO)"

    	| Variant | Speed<br><sup>T4 TensorRT10<br>(ms)</sup><br>YOLOv8 | Speed<br><sup>T4 TensorRT10<br>(ms)</sup><br>YOLO11 |
    	|---------------------|-------------------------------------------------------|-------------------------------------------------------|
    	| n | 1.47 | 1.55 |
    	| s | 2.66 | 2.63 |
    	| m | 5.86 | 5.27 |
    	| l | 9.06 | 6.84 |
    	| x | 14.37 | 12.49 |

## Benchmarking With Ultralytics YOLO11

Benchmarking is a critical functionality offered by Ultralytics YOLO11, enabling users to evaluate the model's performance across various datasets and environments. By analyzing metrics like speed, accuracy, and memory usage, you can optimize your workflows and ensure the model meets specific project requirements. This feature is particularly useful when comparing YOLO11's performance with other models or when deploying it on edge devices.

The benchmarking process in YOLO11 is streamlined, providing insights into how the model handles tasks like object detection, segmentation, and classification. For a detailed guide on YOLO performance metrics like mAP and F1 score, visit [YOLO Performance Metrics](https://docs.ultralytics.com/guides/).

Here's an example of how you can benchmark a YOLO11 model using Python:

```python
from ultralytics import YOLO

# Load a YOLO11 model
model = YOLO("yolo11.pt")

# Benchmark the model
results = model.benchmark(data="coco.yaml", imgsz=640, batch=16)

# Display results
print(results)
```

This ensures you get actionable insights to fine-tune your model effectively.
