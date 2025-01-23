---
comments: true
description: Discover the key differences between DAMO-YOLO and YOLOv7 in this detailed comparison. Explore their performance in object detection, real-time AI, and edge AI applications to understand how these cutting-edge models excel in computer vision tasks for modern use cases.
keywords: DAMO-YOLO, YOLOv7, Ultralytics, object detection, real-time AI, edge AI, computer vision, model comparison
---

# DAMO-YOLO VS YOLOv7

When it comes to cutting-edge object detection, comparing DAMO-YOLO and YOLOv7 reveals unique strengths tailored to different use cases. Both models excel in real-time applications but bring distinct architectural innovations and optimizations to the table.

YOLOv7, a prominent member of the YOLO family, is renowned for its performance and efficiency in various computer vision tasks. On the other hand, DAMO-YOLO emphasizes accuracy and scalability, offering competitive solutions for demanding AI projects. Explore how each model redefines object detection through advanced features and capabilities.

## mAP Comparison

This section compares the mAP values of DAMO-YOLO and YOLOv7, showcasing their accuracy across different variants. Mean Average Precision (mAP) is a critical metric that evaluates object detection performance, balancing precision and recall to provide a comprehensive measure of model effectiveness. Learn more about [mAP](https://www.ultralytics.com/glossary/mean-average-precision-map) in object detection.

!!! tip "Accuracy"

    === "Detection (COCO)"

    	| Variant | mAP<sup>val<br>50<br>DAMO-YOLO | mAP<sup>val<br>50<br>YOLOv7 |
    	|---------------------|-------------------------------------------------------|-------------------------------------------------------|
    	| n | 42.0 | N/A |
    	| s | 46.0 | N/A |
    	| m | 49.2 | N/A |
    	| l | 50.8 | 51.4 |
    	| x | N/A | 53.1 |


## Speed Comparison

This section highlights the speed metrics of DAMO-YOLO and YOLOv7 models across various sizes, measuring performance in milliseconds. These comparisons demonstrate how each model balances efficiency and real-time inference capabilities. For more insights, explore the [YOLOv7 documentation](https://docs.ultralytics.com/models/yolov7/) and [benchmarking tools](https://docs.ultralytics.com/reference/utils/benchmarks/).

!!! tip "Speed"

    === "Detection (COCO)"

    	| Variant | Speed<br><sup>T4 TensorRT10<br>(ms)</sup><br>DAMO-YOLO | Speed<br><sup>T4 TensorRT10<br>(ms)</sup><br>YOLOv7 |
    	|---------------------|-------------------------------------------------------|-------------------------------------------------------|
    	| n | 2.32 | N/A |
    	| s | 3.45 | N/A |
    	| m | 5.09 | N/A |
    	| l | 7.18 | 6.84 |
    	| x | N/A | 11.57 |

## Leveraging YOLO11 for Custom Training on African Wildlife Dataset

Ultralytics YOLO11 empowers users to fine-tune models on custom datasets, including the African Wildlife dataset. This capability is ideal for wildlife monitoring and conservation efforts, enabling precise detection and classification of diverse species in their natural habitats. By training on this dataset, users can improve the model's performance for specific tasks like identifying endangered species or monitoring animal behavior.

Custom training with YOLO11 is streamlined, offering tools for dataset preparation, augmentation, and evaluation. Learn more about custom training and datasets in the [Ultralytics documentation](https://docs.ultralytics.com/modes/train/).

### Python Code Example for Custom Training

```python
from ultralytics import YOLO

# Load the YOLO11 model
model = YOLO("yolo11.pt")

# Train the model on the African Wildlife dataset
model.train(
    data="african_wildlife.yaml",  # Path to dataset configuration
    epochs=50,
    batch=16,
    imgsz=640,
)

# Evaluate the model
metrics = model.val()
print(metrics)
```

This code snippet demonstrates how to leverage YOLO11 for fine-tuning on the African Wildlife dataset. For more details, visit the [Ultralytics YOLO11 guide](https://docs.ultralytics.com/guides/).
