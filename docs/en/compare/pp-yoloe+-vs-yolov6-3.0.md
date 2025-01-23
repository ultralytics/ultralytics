---
comments: true
description: Explore an in-depth comparison between PP-YOLOE+ and YOLOv6-3.0, two leading object detection models excelling in real-time AI, edge AI, and computer vision applications. Discover their performance, accuracy, and efficiency to determine the best fit for your needs in advanced AI solutions.
keywords: PP-YOLOE+, YOLOv6-3.0, Ultralytics, object detection, real-time AI, edge AI, computer vision, model comparison, AI performance, COCO dataset.
---

# PP-YOLOE+ VS YOLOv6-3.0

In this comprehensive comparison, we delve into the capabilities of PP-YOLOE+ and YOLOv6-3.0, two advanced object detection models. By evaluating their performance, speed, and precision, this analysis aims to highlight how they cater to diverse computer vision applications.

PP-YOLOE+ stands out with its efficient architecture and adaptability across various tasks, while YOLOv6-3.0 continues to push boundaries with its optimized accuracy and resource efficiency. Explore how these models perform in real-world scenarios and which one best suits your needs.

## mAP Comparison

This section compares the mAP values of PP-YOLOE+ and YOLOv6-3.0, showcasing their accuracy across different variants. Higher mAP scores indicate better precision and recall, making it a critical metric for evaluating object detection performance. Learn more about [Mean Average Precision (mAP)](https://www.ultralytics.com/glossary/mean-average-precision-map).

!!! tip "Accuracy"

    === "Detection (COCO)"

    	| Variant | mAP<sup>val<br>50<br>PP-YOLOE+ | mAP<sup>val<br>50<br>YOLOv6-3.0 |
    	|---------------------|-------------------------------------------------------|-------------------------------------------------------|
    	| n | 39.9 | 37.5 |
    	| s | 43.7 | 45.0 |
    	| m | 49.8 | 50.0 |
    	| l | 52.9 | 52.8 |
    	| x | 54.7 | N/A |


## Speed Comparison

This section highlights the speed metrics in milliseconds for models such as PP-YOLOE+ and YOLOv6-3.0 across various sizes. By comparing their latency, we observe critical differences in performance, offering insights into their efficiency for real-time applications. Explore more about [PP-YOLOE+](https://github.com/PaddlePaddle/PaddleDetection) and [YOLOv6](https://github.com/meituan/YOLOv6) for detailed specifications.

!!! tip "Speed"

    === "Detection (COCO)"

    	| Variant | Speed<br><sup>T4 TensorRT10<br>(ms)</sup><br>PP-YOLOE+ | Speed<br><sup>T4 TensorRT10<br>(ms)</sup><br>YOLOv6-3.0 |
    	|---------------------|-------------------------------------------------------|-------------------------------------------------------|
    	| n | 2.84 | 1.17 |
    	| s | 2.62 | 2.66 |
    	| m | 5.56 | 5.28 |
    	| l | 8.36 | 8.95 |
    	| x | 14.3 | N/A |

## Fine-Tuning With the COCO8 Dataset

Ultralytics YOLO11 supports fine-tuning on diverse datasets, including the widely used [COCO8 dataset](https://docs.ultralytics.com/datasets/detect/coco/). The COCO8 dataset, derived from the original COCO dataset, contains a smaller subset of 80 classes, making it a powerful option for training models efficiently on a variety of objects. This dataset is particularly useful for tasks like object detection, segmentation, and classification.

Fine-tuning YOLO11 on COCO8 enables the model to adapt pretrained weights to your specific use case, enhancing accuracy and performance. Whether you're building solutions for retail analytics or autonomous systems, COCO8 provides a solid foundation for customization.

### Python Code Example: Training YOLO11 on COCO8

```python
from ultralytics import YOLO

# Load the YOLO11 model
model = YOLO("yolo11.pt")

# Train the model on the COCO8 dataset
results = model.train(data="coco8.yaml", epochs=50, batch=16, imgsz=640)

# Evaluate the trained model
metrics = model.val()
print("Validation Metrics:", metrics)
```

Learn more about [training YOLO11](https://docs.ultralytics.com/modes/train/) with custom datasets to unlock its full potential for your projects.
