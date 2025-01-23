---
comments: true
description: Dive into a detailed comparison of Ultralytics YOLOv8 and YOLOv10, highlighting their performance, accuracy, and efficiency for real-time object detection. Explore how these models excel in computer vision tasks, edge AI applications, and more.
keywords: Ultralytics, YOLOv8, YOLOv10, object detection, real-time AI, edge AI, computer vision
---

# Ultralytics YOLOv8 VS YOLOv10

The evolution of YOLO models continues to redefine computer vision, and comparing Ultralytics YOLOv8 and YOLOv10 highlights the strides made in speed, accuracy, and efficiency. This page delves into the critical advancements and trade-offs between these two state-of-the-art models, showcasing their unique capabilities for diverse applications.

Ultralytics YOLOv8 is celebrated for its exceptional real-time performance and ease of use, making it ideal for developers at all levels. On the other hand, YOLOv10 incorporates innovative methodologies like dual label assignments and efficiency-driven designs, achieving superior metrics and versatility across edge and cloud deployments. For more details on YOLOv8, visit the [Ultralytics YOLOv8 documentation](https://docs.ultralytics.com/models/yolov8/).

## mAP Comparison

Mean Average Precision (mAP) is a critical metric for evaluating model accuracy in object detection tasks. This section compares the mAP values of Ultralytics YOLOv8 and YOLOv10 across their variants, showcasing advancements in precision and recall. Learn more about mAP in the [Ultralytics Glossary](https://www.ultralytics.com/glossary/mean-average-precision-map).

!!! tip "Accuracy"

    === "Detection (COCO)"

    	| Variant | mAP<sup>val<br>50<br>YOLOv8 | mAP<sup>val<br>50<br>YOLOv10 |
    	|---------------------|-------------------------------------------------------|-------------------------------------------------------|
    	| n | 37.3 | 39.5 |
    	| s | 44.9 | 46.7 |
    	| m | 50.2 | 51.3 |
    	| b | N/A | 52.7 |
    	| l | 52.9 | 53.3 |
    	| x | 53.9 | 54.4 |

## Speed Comparison

This section highlights the performance differences between Ultralytics YOLOv8 and YOLOv10 by comparing their latency metrics across various model sizes. Measured in milliseconds, these figures demonstrate the efficiency improvements achieved with YOLOv10, offering faster inference speeds for tasks like [object detection](https://docs.ultralytics.com/tasks/detect/) and real-time applications. Explore more about YOLOv10's advancements [here](https://docs.ultralytics.com/models/yolov10/).

!!! tip "Speed"

    === "Detection (COCO)"

    	| Variant | Speed<br><sup>T4 TensorRT10<br>(ms)</sup><br>YOLOv8 | Speed<br><sup>T4 TensorRT10<br>(ms)</sup><br>YOLOv10 |
    	|---------------------|-------------------------------------------------------|-------------------------------------------------------|
    	| n | 1.47 | 1.56 |
    	| s | 2.66 | 2.66 |
    	| m | 5.86 | 5.48 |
    	| b | N/A | 6.54 |
    	| l | 9.06 | 8.33 |
    	| x | 14.37 | 12.2 |

## Train with Ultralytics YOLO11

Ultralytics YOLO11 offers an intuitive and efficient training pipeline for various computer vision tasks. With support for custom datasets like COCO8, African Wildlife, and more, YOLO11 allows users to fine-tune models for their specific use cases. Training can be seamlessly executed using the Ultralytics Python package or the Ultralytics HUB, ensuring accessibility for both beginners and experts. The training process includes built-in validation and monitoring tools to track performance metrics like loss and accuracy, enabling better model optimization.

For a detailed guide on training your models, explore the [Ultralytics YOLO Training Documentation](https://docs.ultralytics.com/modes/train/).

### Example: Python Code for Training

```python
from ultralytics import YOLO

# Load a pre-trained YOLO model
model = YOLO("yolov11.pt")

# Train the model on a custom dataset
model.train(data="path/to/dataset.yaml", epochs=50, imgsz=640, batch=32)
```
