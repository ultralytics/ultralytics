---
comments: true
description: Explore the key differences between Ultralytics YOLOv5 and YOLOX in this comprehensive comparison. Discover how these models perform in object detection, real-time AI, edge AI, and computer vision tasks, and understand their suitability for various applications.
keywords: Ultralytics, YOLOv5, YOLOX, object detection, real-time AI, edge AI, computer vision, machine learning, deep learning
---

# Ultralytics YOLOv5 VS YOLOX

Choosing the right object detection model is critical for achieving optimal performance in computer vision tasks. This comparison between Ultralytics YOLOv5 and YOLOX highlights their unique strengths, helping you determine which model best suits your specific needs.

Ultralytics YOLOv5 is renowned for its streamlined workflows, excellent documentation, and adaptability across diverse applications. On the other hand, YOLOX offers a fully anchor-free design, emphasizing innovative detection strategies and delivering robust real-time performance in various use cases.

## mAP Comparison

Mean Average Precision (mAP) evaluates the accuracy of object detection models by balancing precision and recall. This section compares the mAP values of Ultralytics YOLOv5 and YOLOX across various variants to highlight their performance differences. Learn more about [mAP and its significance](https://www.ultralytics.com/glossary/mean-average-precision-map).

!!! tip "Accuracy"

    === "Detection (COCO)"

    	| Variant | mAP<sup>val<br>50<br>YOLOv5 | mAP<sup>val<br>50<br>YOLOX |
    	|---------------------|-------------------------------------------------------|-------------------------------------------------------|
    	| s | 37.4 | 40.5 |
    	| m | 45.4 | 46.9 |
    	| l | 49.0 | 49.7 |
    	| x | 50.7 | 51.1 |


## Speed Comparison

This section highlights the speed metrics of Ultralytics YOLOv5 and YOLOX models, measured in milliseconds, to showcase their performance across various sizes. These metrics provide valuable insights for selecting the optimal model for real-time applications. Learn more about [YOLOv5](https://docs.ultralytics.com/models/yolov5/) and [YOLOX](https://github.com/Megvii-BaseDetection/YOLOX).

!!! tip "Speed"

    === "Detection (COCO)"

    	| Variant | Speed<br><sup>T4 TensorRT10<br>(ms)</sup><br>YOLOv5 | Speed<br><sup>T4 TensorRT10<br>(ms)</sup><br>YOLOX |
    	|---------------------|-------------------------------------------------------|-------------------------------------------------------|
    	| s | 1.92 | 2.56 |
    	| m | 4.03 | 5.43 |
    	| l | 6.61 | 9.04 |
    	| x | 11.89 | 16.1 |

## Train with Ultralytics YOLO11

Training a model with Ultralytics YOLO11 is a straightforward process that enables you to fine-tune the model for various datasets, including COCO8, African wildlife, and more. YOLO11 offers exceptional flexibility and performance, making it ideal for diverse computer vision applications. Using the Ultralytics Python package, you can initiate training with a simple command and customize parameters like epochs, batch size, and learning rate.

To explore more about training strategies and best practices, visit the [YOLO Common Issues Guide](https://docs.ultralytics.com/guides/yolo-common-issues/) for tips on troubleshooting and improving your model's performance.

### Example: Training a YOLO11 Model

```python
from ultralytics import YOLO

# Load a YOLO11 model
model = YOLO("yolov11.pt")

# Train the model on a custom dataset
model.train(data="coco128.yaml", epochs=50, batch=16, imgsz=640)
```

For a detailed walkthrough on training Ultralytics YOLO11, check out the [training documentation](https://docs.ultralytics.com/modes/train/).
