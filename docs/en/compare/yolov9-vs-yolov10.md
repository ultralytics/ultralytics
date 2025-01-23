---
comments: true
description: Compare YOLOv9 and YOLOv10, two advanced models from Ultralytics, to explore their improvements in object detection, real-time AI, and edge AI. Discover how these models redefine efficiency and accuracy in computer vision applications.
keywords: YOLOv9, YOLOv10, Ultralytics, object detection, real-time AI, edge AI, computer vision
---

# YOLOv9 VS YOLOv10

Comparing YOLOv9 and YOLOv10 highlights the evolution of real-time object detection technology, showcasing how each model builds upon its predecessor's strengths. This page delves into their unique capabilities, enabling users to make informed decisions for their specific applications.

YOLOv9 introduced advanced features like enhanced efficiency and improved architecture, setting a strong foundation for real-time tasks. YOLOv10 takes this further with innovations such as NMS-free training and a holistic design, resulting in superior accuracy-speed trade-offs, as detailed in the [YOLOv10 documentation](https://docs.ultralytics.com/models/yolov10/).

## mAP Comparison

This section highlights the mAP (mean Average Precision) performance of YOLOv9 and YOLOv10 across their respective model variants. As a key metric for evaluating accuracy, mAP reflects how well these models balance precision and recall in detecting objects. Learn more about [mAP](https://www.ultralytics.com/glossary/mean-average-precision-map) and its importance in object detection.

!!! tip "Accuracy"

    === "Detection (COCO)"

    	| Variant | mAP<sup>val<br>50<br>YOLOv9 | mAP<sup>val<br>50<br>YOLOv10 |
    	|---------------------|-------------------------------------------------------|-------------------------------------------------------|
    	| n | 37.8 | 39.5 |
    	| s | 46.5 | 46.7 |
    	| m | 51.5 | 51.3 |
    	| b | N/A | 52.7 |
    	| l | 52.8 | 53.3 |
    	| x | 55.1 | 54.4 |


## Speed Comparison

This section highlights the speed performance of YOLOv9 and YOLOv10 across various model sizes, measured in milliseconds. YOLOv10 demonstrates significant latency improvements, leveraging its optimized architecture for faster inference times without compromising accuracy. Learn more about YOLOv10's advancements in its [documentation](https://docs.ultralytics.com/models/yolov10/).

!!! tip "Speed"

    === "Detection (COCO)"

    	| Variant | Speed<br><sup>T4 TensorRT10<br>(ms)</sup><br>YOLOv9 | Speed<br><sup>T4 TensorRT10<br>(ms)</sup><br>YOLOv10 |
    	|---------------------|-------------------------------------------------------|-------------------------------------------------------|
    	| n | 2.3 | 1.56 |
    	| s | 3.54 | 2.66 |
    	| m | 6.43 | 5.48 |
    	| b | N/A | 6.54 |
    	| l | 7.16 | 8.33 |
    	| x | 16.77 | 12.2 |

## Train with Ultralytics YOLO11

The "Train" functionality in Ultralytics YOLO11 empowers users to fine-tune models on custom datasets for various tasks like object detection, segmentation, and classification. Whether you're working with datasets such as COCO8 or more specialized ones like car parts segmentation, YOLO11 provides the flexibility to optimize your model for specific use cases. Its robust integration with frameworks like PyTorch ensures seamless and efficient training processes.

With easy-to-use configuration files and command-line options, training your model becomes straightforward. For example, you can adjust hyperparameters, incorporate pre-trained weights, and monitor metrics such as loss and accuracy during the training process. Explore more details on [custom training](https://docs.ultralytics.com/modes/train/) for YOLO11.

### Python Code Example: Training a YOLO11 Model

```python
from ultralytics import YOLO

# Load YOLO11 model
model = YOLO("yolo11n.pt")

# Train the model on a custom dataset
model.train(data="custom_dataset.yaml", epochs=50, imgsz=640, batch=16)
```

This snippet demonstrates how to load a pre-trained YOLO11 model and train it on a custom dataset with specified parameters.
