---
comments: true
description: Explore a detailed comparison between YOLOv6-3.0 and Ultralytics YOLOv5, two cutting-edge models in object detection and real-time AI. Understand their performance in computer vision tasks, edge AI applications, and advancements in speed and accuracy.
keywords: YOLOv6-3.0, Ultralytics YOLOv5, object detection, real-time AI, edge AI, computer vision, machine learning, AI models, Ultralytics
---

# YOLOv6-3.0 VS Ultralytics YOLOv5

Comparing YOLOv6-3.0 and Ultralytics YOLOv5 highlights the rapid advancements in object detection technologies. Both models have shaped the AI landscape, offering unique strengths in speed, accuracy, and efficiency for various applications.

Ultralytics YOLOv5 is celebrated for its simplicity, extensive community support, and robust performance across diverse tasks. On the other hand, YOLOv6-3.0 introduces innovative features aimed at enhanced precision and scalability, showcasing its potential in cutting-edge real-time scenarios. For more on YOLOv5, visit its [documentation page](https://docs.ultralytics.com/models/yolov5/).

## mAP Comparison

This section highlights the mAP values of YOLOv6-3.0 and Ultralytics YOLOv5, showcasing their performance across different variants. Mean Average Precision (mAP) serves as a key metric for evaluating the accuracy of object detection models, balancing precision and recall. Learn more about [mAP](https://www.ultralytics.com/glossary/mean-average-precision-map) and its role in model performance.

!!! tip "Accuracy"

    === "Detection (COCO)"

    	| Variant | mAP<sup>val<br>50<br>YOLOv6-3.0 | mAP<sup>val<br>50<br>YOLOv5 |
    	|---------------------|-------------------------------------------------------|-------------------------------------------------------|
    	| n | 37.5 | N/A |
    	| s | 45.0 | 37.4 |
    	| m | 50.0 | 45.4 |
    	| l | 52.8 | 49.0 |
    	| x | N/A | 50.7 |

## Speed Comparison

This section highlights the speed performance of YOLOv6-3.0 and Ultralytics YOLOv5 across various model sizes, measured in milliseconds. These comparisons demonstrate how latency impacts real-time applications, showcasing the efficiency of each model variant. For more details on YOLOv5's architecture, refer to the [YOLOv5 Architecture Summary](https://docs.ultralytics.com/yolov5/tutorials/architecture_description/).

!!! tip "Speed"

    === "Detection (COCO)"

    	| Variant | Speed<br><sup>T4 TensorRT10<br>(ms)</sup><br>YOLOv6-3.0 | Speed<br><sup>T4 TensorRT10<br>(ms)</sup><br>YOLOv5 |
    	|---------------------|-------------------------------------------------------|-------------------------------------------------------|
    	| n | 1.17 | N/A |
    	| s | 2.66 | 1.92 |
    	| m | 5.28 | 4.03 |
    	| l | 8.95 | 6.61 |
    	| x | N/A | 11.89 |

## Fine-Tuning With Car Parts Segmentation Dataset

Ultralytics YOLO11 offers seamless fine-tuning capabilities, enabling users to adapt the model for specific use cases like car parts segmentation. This functionality is particularly useful in industries such as automotive manufacturing, repair, and e-commerce, where precise identification and categorization of vehicle components are essential. By customizing the pre-trained YOLO11 model with labeled examples from the car parts segmentation dataset, you can achieve enhanced accuracy and relevance for your unique tasks.

To learn more about car parts segmentation and how to leverage YOLO11 for this purpose, explore the [official documentation on segmentation datasets](https://docs.ultralytics.com/datasets/segment/carparts-seg/).

### Python Code Example

```python
from ultralytics import YOLO

# Load a pre-trained YOLO11 model
model = YOLO("yolo11.pt")

# Fine-tune the model with a car parts segmentation dataset
model.train(data="carparts.yaml", epochs=50, imgsz=640)

# Validate the model
metrics = model.val()

# Perform predictions
results = model.predict(source="test_images/", save=True)
```

This example demonstrates how to fine-tune YOLO11 using a car parts segmentation dataset and evaluate its performance for specific applications.
