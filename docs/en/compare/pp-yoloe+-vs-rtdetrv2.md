---
comments: true
description: Compare PP-YOLOE+ and RTDETRv2, two cutting-edge object detection models, to explore their performance, efficiency, and real-time AI capabilities in computer vision tasks. Discover how these models excel in edge AI applications with Ultralytics' advanced tools and frameworks.
keywords: PP-YOLOE+, RTDETRv2, Ultralytics, object detection, real-time AI, edge AI, computer vision, model comparison, AI performance, real-time object detection
---

# PP-YOLOE+ VS RTDETRv2

The comparison of PP-YOLOE+ and RTDETRv2 brings to light the diverse advancements in object detection technology. By evaluating these models, we aim to highlight their performance across parameters like speed, accuracy, and scalability to guide their application in real-world scenarios.

PP-YOLOE+ is renowned for its efficiency and optimized architecture tailored for rapid inference, making it a strong contender in edge-based applications. On the other hand, RTDETRv2 leverages Vision Transformer techniques, offering high accuracy and flexibility in deployment, as detailed in the [RTDETR model reference](https://docs.ultralytics.com/reference/models/rtdetr/model/).

## mAP Comparison

This section compares the mAP values of PP-YOLOE+ and RTDETRv2 models, showcasing their accuracy in detecting objects across variants. Mean Average Precision (mAP) is a key metric for evaluating model performance, reflecting their balance between precision and recall. Learn more about [mAP](https://www.ultralytics.com/glossary/mean-average-precision-map) and its importance in object detection.

!!! tip "Accuracy"

    === "Detection (COCO)"

    	| Variant | mAP<sup>val<br>50<br>PP-YOLOE+ | mAP<sup>val<br>50<br>RTDETRv2 |
    	|---------------------|-------------------------------------------------------|-------------------------------------------------------|
    	| n | 39.9 | N/A |
    	| s | 43.7 | 48.1 |
    	| m | 49.8 | 51.9 |
    	| l | 52.9 | 53.4 |
    	| x | 54.7 | 54.3 |

## Speed Comparison

This section highlights the speed performance of PP-YOLOE+ and RTDETRv2 across various model sizes. Measured in milliseconds, these metrics provide critical insights into the models' latency and efficiency on diverse hardware setups. For more about model benchmarking, see [Ultralytics Benchmarking](https://docs.ultralytics.com/modes/benchmark/).

!!! tip "Speed"

    === "Detection (COCO)"

    	| Variant | Speed<br><sup>T4 TensorRT10<br>(ms)</sup><br>PP-YOLOE+ | Speed<br><sup>T4 TensorRT10<br>(ms)</sup><br>RTDETRv2 |
    	|---------------------|-------------------------------------------------------|-------------------------------------------------------|
    	| n | 2.84 | N/A |
    	| s | 2.62 | 5.03 |
    	| m | 5.56 | 7.51 |
    	| l | 8.36 | 9.76 |
    	| x | 14.3 | 15.03 |

## Train with Ultralytics YOLO11

Ultralytics YOLO11 offers seamless training functionality, allowing users to fine-tune models on custom datasets such as COCO8, car parts segmentation, and many more. With its intuitive interface and advanced features, training a YOLO11 model is both efficient and user-friendly. This functionality is ideal for tasks like object detection, image classification, and segmentation across a wide variety of industries.

To get started, explore the [custom training guide](https://docs.ultralytics.com/modes/train/) to learn how to prepare datasets, configure hyperparameters, and monitor performance metrics during training.

### Example Code for Training YOLO11

```python
from ultralytics import YOLO

# Load a YOLO11 model
model = YOLO("yolo11.pt")

# Train the model on a custom dataset
model.train(data="path/to/dataset.yaml", epochs=50, imgsz=640)
```

This code snippet demonstrates how easy it is to initiate training with YOLO11 using the Ultralytics Python package. For more details, visit the [training documentation](https://docs.ultralytics.com/modes/train/).
