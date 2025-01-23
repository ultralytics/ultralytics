---
comments: true
description: Explore the detailed comparison between YOLOv9 and YOLOv6-3.0, two cutting-edge models in real-time object detection. Discover how these models differ in performance, efficiency, and suitability for edge AI and computer vision tasks.
keywords: YOLOv9, YOLOv6-3.0, Ultralytics, object detection, real-time AI, edge AI, computer vision, model comparison
---

# YOLOv9 VS YOLOv6-3.0

The comparison between YOLOv9 and YOLOv6-3.0 highlights the evolution of object detection models, focusing on advancements in accuracy, speed, and efficiency. Both models represent significant milestones in the YOLO family, showcasing unique innovations tailored for diverse applications.

YOLOv9 emphasizes enhanced architectural designs and superior performance on large-scale datasets, while YOLOv6-3.0 excels in achieving high efficiency with optimized resource utilization. Understanding their differences offers valuable insights for selecting the right model for your specific needs. Explore more about [YOLOv9](https://www.youtube.com/watch?v=ZF7EAodHn1U&t=1s) and [YOLOv6-3.0](https://www.ultralytics.com/blog/introducing-ultralytics-yolov8).

## mAP Comparison

The mAP values highlight the accuracy of YOLOv9 and YOLOv6-3.0 across various model variants, offering a detailed view of their performance on object detection tasks. These metrics, like mAP@.5 and mAP@.5:.95, reflect the models' ability to balance precision and recall effectively. Learn more about [mean average precision](https://www.ultralytics.com/glossary/mean-average-precision-map) and its significance in model evaluation.

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

This section highlights the speed performance of YOLOv9 and YOLOv6-3.0 across various model sizes, with latency measured in milliseconds. These metrics demonstrate the efficiency of each model, offering insights into their suitability for real-time applications. Explore more about [YOLOv9](https://docs.ultralytics.com/models/yolov9/) and [YOLOv6-3.0](https://docs.ultralytics.com/reference/utils/benchmarks/).

!!! tip "Speed"

    === "Detection (COCO)"

    	| Variant | Speed<br><sup>T4 TensorRT10<br>(ms)</sup><br>YOLOv9 | Speed<br><sup>T4 TensorRT10<br>(ms)</sup><br>YOLOv6-3.0 |
    	|---------------------|-------------------------------------------------------|-------------------------------------------------------|
    	| n | 2.3 | 1.17 |
    	| s | 3.54 | 2.66 |
    	| m | 6.43 | 5.28 |
    	| l | 7.16 | 8.95 |
    	| x | 16.77 | N/A |

## Train With Ultralytics YOLO11

Ultralytics YOLO11 excels in training models for various computer vision tasks, including object detection, segmentation, and classification. This functionality allows users to fine-tune pre-trained models or build custom models tailored to their specific datasets, such as COCO8 or African wildlife. With its streamlined API and advanced capabilities, YOLO11 ensures high accuracy and efficiency during the training process.

To get started with training, Ultralytics YOLO11 provides an intuitive interface through the Python package. You can customize hyperparameters, monitor metrics, and visualize training progress effortlessly. For a detailed guide on custom training, refer to the [Ultralytics documentation](https://docs.ultralytics.com/modes/train/).

### Python Code Example

```python
from ultralytics import YOLO

# Load a YOLO11 model
model = YOLO("yolo11.pt")

# Train the model on a custom dataset
model.train(data="path/to/dataset.yaml", epochs=50, imgsz=640, batch=16)
```

This snippet demonstrates how to initiate training with YOLO11, specifying the dataset, number of epochs, image size, and batch size to optimize performance.
