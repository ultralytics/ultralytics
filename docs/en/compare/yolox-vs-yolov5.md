---
comments: true
description: Explore the comprehensive comparison between YOLOX and Ultralytics YOLOv5, two leading models in object detection and computer vision. Discover their performance in real-time AI, edge AI, and practical applications to help you choose the best model for your needs.
keywords: YOLOX, Ultralytics YOLOv5, object detection, real-time AI, edge AI, computer vision, model comparison
---

# YOLOX VS Ultralytics YOLOv5

The comparison between YOLOX and Ultralytics YOLOv5 highlights the evolution of object detection technology. Both models represent significant milestones, offering unique capabilities tailored for diverse computer vision applications.

YOLOX is known for its anchor-free design and simplified training process, delivering remarkable efficiency. In contrast, Ultralytics YOLOv5 has set benchmarks in ease of use and versatile deployment, making it a favorite among developers globally. Explore the differences to better understand their strengths and applications.

## mAP Comparison

This section highlights the mAP values of YOLOX and Ultralytics YOLOv5 models, representing their accuracy in detecting objects across different variants. Mean Average Precision (mAP) serves as a key metric for evaluating a model's precision and recall performance. Learn more about [mAP](https://www.ultralytics.com/glossary/mean-average-precision-map) and its importance in object detection.

!!! tip "Accuracy"

    === "Detection (COCO)"

    	| Variant | mAP<sup>val<br>50<br>YOLOX | mAP<sup>val<br>50<br>YOLOv5 |
    	|---------------------|-------------------------------------------------------|-------------------------------------------------------|
    	| s | 40.5 | 37.4 |
    	| m | 46.9 | 45.4 |
    	| l | 49.7 | 49.0 |
    	| x | 51.1 | 50.7 |


## Speed Comparison

This section highlights the performance of YOLOX and Ultralytics YOLOv5 models in terms of inference speed, measured in milliseconds, across various model sizes. These metrics provide critical insights into the efficiency of each model for real-world applications. For more details on YOLOv5, visit the [Ultralytics YOLOv5 Documentation](https://docs.ultralytics.com/models/yolov5/).

!!! tip "Speed"

    === "Detection (COCO)"

    	| Variant | Speed<br><sup>T4 TensorRT10<br>(ms)</sup><br>YOLOX | Speed<br><sup>T4 TensorRT10<br>(ms)</sup><br>YOLOv5 |
    	|---------------------|-------------------------------------------------------|-------------------------------------------------------|
    	| s | 2.56 | 1.92 |
    	| m | 5.43 | 4.03 |
    	| l | 9.04 | 6.61 |
    	| x | 16.1 | 11.89 |

## Predict Functionality in Ultralytics YOLO11

The Predict functionality in Ultralytics YOLO11 enables users to perform real-time inference across a wide range of computer vision tasks. Whether you’re working on object detection, segmentation, or classification, this feature ensures efficient and precise predictions. YOLO11 offers seamless integration with pre-trained models and custom datasets, making it versatile for diverse applications like retail analytics, wildlife monitoring, and industrial inspection.

To get started with predictions, you can load the model, input your image or video, and retrieve results with bounding boxes, masks, or class labels. For more details, explore the [Ultralytics YOLO11 documentation](https://docs.ultralytics.com/models/yolo11/).

### Example Python Code for Predict Functionality

```python
from ultralytics import YOLO

# Load a pre-trained YOLO11 model
model = YOLO("yolo11.pt")

# Perform prediction on an image
results = model.predict(source="image.jpg", save=True)

# Display results
results.show()
```

This code snippet demonstrates how to make predictions on an image using YOLO11, with options to save and visualize the output.
