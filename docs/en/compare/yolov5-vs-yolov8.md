---
comments: true
description: Compare Ultralytics YOLOv5 and YOLOv8 to explore advancements in real-time object detection, speed, and accuracy. Discover how YOLOv8 builds upon YOLOv5 with state-of-the-art features for edge AI and computer vision applications.
keywords: Ultralytics, YOLOv5, YOLOv8, object detection, real-time AI, edge AI, computer vision, Ultralytics YOLOv8, Ultralytics YOLOv5 comparison
---

# Ultralytics YOLOv5 VS YOLOv8

# Ultralytics YOLOv5 VS Ultralytics YOLOv8

The comparison between Ultralytics YOLOv5 and YOLOv8 highlights the evolution of cutting-edge object detection models. Both models have redefined real-time detection, offering unique strengths tailored to diverse applications.

Ultralytics YOLOv5 is renowned for its simplicity and widespread adoption, while YOLOv8 takes performance and usability to new heights. By building on YOLOv5's foundation, YOLOv8 introduces innovations like anchor-free detection and enhanced accuracy-speed tradeoffs. Learn more about [YOLOv8's advancements](https://docs.ultralytics.com/models/yolov8/) and [YOLOv5's capabilities](https://github.com/ultralytics/yolov5).

## mAP Comparison

The mAP (Mean Average Precision) metric evaluates the accuracy of object detection models across different thresholds and classes. Comparing the mAP values of Ultralytics YOLOv5 and YOLOv8 highlights advancements in detection precision, with YOLOv8 achieving superior performance across its variants. To learn more about mAP and its significance, visit the [Ultralytics Glossary on mAP](https://www.ultralytics.com/glossary/mean-average-precision-map).

!!! tip "Accuracy"

    === "Detection (COCO)"

    	| Variant | mAP<sup>val<br>50<br>YOLOv5 | mAP<sup>val<br>50<br>YOLOv8 |
    	|---------------------|-------------------------------------------------------|-------------------------------------------------------|
    	| n | N/A | 37.3 |
    	| s | 37.4 | 44.9 |
    	| m | 45.4 | 50.2 |
    	| l | 49.0 | 52.9 |
    	| x | 50.7 | 53.9 |


## Speed Comparison

This section highlights the speed metrics of Ultralytics YOLOv5 versus Ultralytics YOLOv8 across various model sizes, measured in milliseconds per inference. YOLOv8 showcases significant advancements in processing speed, making it ideal for real-time applications. Explore more about [YOLOv8's performance](https://docs.ultralytics.com/models/yolov8/) and its optimized accuracy-speed tradeoff.

!!! tip "Speed"

    === "Detection (COCO)"

    	| Variant | Speed<br><sup>T4 TensorRT10<br>(ms)</sup><br>YOLOv5 | Speed<br><sup>T4 TensorRT10<br>(ms)</sup><br>YOLOv8 |
    	|---------------------|-------------------------------------------------------|-------------------------------------------------------|
    	| n | N/A | 1.47 |
    	| s | 1.92 | 2.66 |
    	| m | 4.03 | 5.86 |
    	| l | 6.61 | 9.06 |
    	| x | 11.89 | 14.37 |

## Train with Ultralytics YOLO11

Ultralytics YOLO11 offers an intuitive and efficient training process, allowing users to fine-tune models for specific tasks and datasets. Whether you're working with pre-trained weights or starting from scratch, YOLO11 ensures high performance and adaptability. The training process supports diverse datasets, such as African Wildlife, COCO8, and more, empowering users to address unique challenges in computer vision.

For detailed guidance on training YOLO models, explore the [Ultralytics Training Documentation](https://docs.ultralytics.com/modes/train/). This resource provides step-by-step instructions, covering everything from dataset preparation to hyperparameter tuning.

### Python Code Example for Training

```python
from ultralytics import YOLO

# Load a YOLO11 model
model = YOLO("yolo11.yaml")

# Train the model on a custom dataset
model.train(data="path/to/dataset.yaml", epochs=50, imgsz=640)
```

This code snippet demonstrates how to load the YOLO11 model and train it on a custom dataset. Adjust the parameters to suit your project requirements for optimal results.
