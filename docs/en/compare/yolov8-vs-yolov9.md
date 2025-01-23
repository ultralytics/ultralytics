---
comments: true
description: Compare Ultralytics YOLOv8 and YOLOv9 to discover advancements in object detection, real-time AI, and edge AI. Explore their performance, features, and capabilities in the world of computer vision.
keywords: Ultralytics, YOLOv8, YOLOv9, object detection, real-time AI, edge AI, computer vision, model comparison, AI advancements
---

# Ultralytics YOLOv8 VS YOLOv9

Ultralytics YOLOv8 and YOLOv9 represent significant milestones in the evolution of object detection models. Both versions showcase advancements in speed, accuracy, and flexibility, making them essential tools for real-time AI applications across various industries.

This comparison highlights YOLOv8's optimal balance of simplicity and performance against YOLOv9's enhanced architectural innovations and refined efficiency. By exploring their capabilities, we aim to provide insights into selecting the right model for your specific use case. Learn more about [YOLOv8's features](https://docs.ultralytics.com/models/yolov8/) and [YOLOv9's advancements](https://www.youtube.com/watch?v=ZF7EAodHn1U&t=1s).

## mAP Comparison

This section highlights the mAP values of Ultralytics YOLOv8 and YOLOv9 across their respective variants, showcasing their accuracy on tasks like object detection. mAP, or Mean Average Precision, evaluates performance by balancing precision and recall, as detailed [here](https://www.ultralytics.com/glossary/mean-average-precision-map), making it a critical metric for comparing model effectiveness.

!!! tip "Accuracy"

    === "Detection (COCO)"

    	| Variant | mAP<sup>val<br>50<br>YOLOv8 | mAP<sup>val<br>50<br>YOLOv9 |
    	|---------------------|-------------------------------------------------------|-------------------------------------------------------|
    	| n | 37.3 | 37.8 |
    	| s | 44.9 | 46.5 |
    	| m | 50.2 | 51.5 |
    	| l | 52.9 | 52.8 |
    	| x | 53.9 | 55.1 |


## Speed Comparison

This section highlights the speed performance of Ultralytics YOLOv8 and YOLOv9 across various model sizes, measured in milliseconds. These metrics provide a clear understanding of how each model balances computational efficiency and real-time application needs. For more details, visit the [YOLOv8 documentation](https://docs.ultralytics.com/models/yolov8/) and [YOLOv9 documentation](https://docs.ultralytics.com/models/yolov9/).

!!! tip "Speed"

    === "Detection (COCO)"

    	| Variant | Speed<br><sup>T4 TensorRT10<br>(ms)</sup><br>YOLOv8 | Speed<br><sup>T4 TensorRT10<br>(ms)</sup><br>YOLOv9 |
    	|---------------------|-------------------------------------------------------|-------------------------------------------------------|
    	| n | 1.47 | 2.3 |
    	| s | 2.66 | 3.54 |
    	| m | 5.86 | 6.43 |
    	| l | 9.06 | 7.16 |
    	| x | 14.37 | 16.77 |

## Train Ultralytics YOLO11 Models

Ultralytics YOLO11 supports advanced training capabilities, enabling you to fine-tune the model on custom datasets for specific applications. Whether you're working with datasets like COCO8 or African wildlife, YOLO11 ensures flexibility and adaptability to meet diverse project requirements. Training involves optimizing hyperparameters, monitoring loss, and validating performance for high accuracy.

For detailed guidance on training, explore the [Ultralytics YOLO Training Guide](https://docs.ultralytics.com/modes/train/).

### Example Python Code for Training

```python
from ultralytics import YOLO

# Load a pre-trained YOLO model
model = YOLO("yolo11.pt")

# Train the model on a custom dataset
model.train(data="custom_dataset.yaml", epochs=50, batch=16, imgsz=640)
```

This process ensures that your YOLO11 model is tailored for the unique requirements of your application. For more insights, visit our [Comprehensive Tutorials](https://docs.ultralytics.com/guides/).
