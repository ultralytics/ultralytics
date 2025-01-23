---
comments: true
description: Explore the comprehensive comparison between YOLOv7 and YOLOv6-3.0, two state-of-the-art models in object detection. Learn about their performance metrics, architectural innovations, and suitability for real-time AI applications in edge AI and computer vision tasks.
keywords: YOLOv7, YOLOv6-3.0, Ultralytics, object detection, real-time AI, edge AI, computer vision, model comparison
---

# YOLOv7 VS YOLOv6-3.0

The comparison between YOLOv7 and YOLOv6-3.0 offers valuable insights into the evolution of real-time object detection technologies. Both models epitomize cutting-edge advancements, addressing diverse computational needs with remarkable efficiency in speed and accuracy.

YOLOv7 excels with innovations like dynamic label assignment and extended scaling, pushing the boundaries of real-time detection. On the other hand, YOLOv6-3.0 introduces unique features such as the Bi-directional Concatenation (BiC) module and anchor-aided training, delivering state-of-the-art performance across applications ranging from edge devices to cloud systems. Explore more about [YOLOv7](https://arxiv.org/pdf/2207.02696) and [YOLOv6-3.0](https://arxiv.org/abs/2301.05586) for deeper understanding.

## mAP Comparison

This section compares the Mean Average Precision (mAP) values of YOLOv7 and YOLOv6-3.0 across various model variants, highlighting their accuracy in detecting and classifying objects. mAP, a critical metric in object detection, evaluates model performance by balancing precision and recall at different thresholds. Learn more about mAP [here](https://www.ultralytics.com/glossary/mean-average-precision-map).

!!! tip "Accuracy"

    === "Detection (COCO)"

    	| Variant | mAP<sup>val<br>50<br>YOLOv7 | mAP<sup>val<br>50<br>YOLOv6-3.0 |
    	|---------------------|-------------------------------------------------------|-------------------------------------------------------|
    	| n | N/A | 37.5 |
    	| s | N/A | 45.0 |
    	| m | N/A | 50.0 |
    	| l | 51.4 | 52.8 |
    	| x | 53.1 | N/A |


## Speed Comparison

This section highlights the speed performance of YOLOv7 and YOLOv6-3.0 models across various sizes, measured in milliseconds. These metrics provide a clear understanding of their efficiency on tasks, showcasing real-time capabilities for different use cases. For more details on YOLOv7's innovations, visit the [YOLOv7 documentation](https://docs.ultralytics.com/models/yolov7/).

!!! tip "Speed"

    === "Detection (COCO)"

    	| Variant | Speed<br><sup>T4 TensorRT10<br>(ms)</sup><br>YOLOv7 | Speed<br><sup>T4 TensorRT10<br>(ms)</sup><br>YOLOv6-3.0 |
    	|---------------------|-------------------------------------------------------|-------------------------------------------------------|
    	| n | N/A | 1.17 |
    	| s | N/A | 2.66 |
    	| m | N/A | 5.28 |
    	| l | 6.84 | 8.95 |
    	| x | 11.57 | N/A |

## Train with Ultralytics YOLO11

Ultralytics YOLO11 offers a seamless training process, allowing users to fine-tune models on custom datasets for optimal performance. Whether you're working on object detection, segmentation, or classification tasks, YOLO11's training functionality ensures high-quality results with minimal effort. By leveraging pre-trained weights, users can dramatically reduce training time while maintaining accuracy. For a step-by-step guide on setting up and running training sessions, explore the [YOLO11 training documentation](https://docs.ultralytics.com/modes/train/).

### Python Code Snippet

```python
from ultralytics import YOLO

# Load a YOLO11 model
model = YOLO("yolov11.pt")

# Train the model on a custom dataset
model.train(data="path/to/dataset.yaml", epochs=50, imgsz=640, batch=16)
```

This code snippet demonstrates how to load a YOLO11 model and train it on a custom dataset. Modify parameters like `epochs` and `batch` size as needed for your project requirements.
