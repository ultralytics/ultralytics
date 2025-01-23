---
comments: true
description: Explore a detailed comparison between YOLOv9 and YOLOX, two cutting-edge models in object detection. Discover their performance in terms of accuracy, efficiency, and suitability for real-time AI and edge AI applications, powered by advancements in computer vision.
keywords: YOLOv9, YOLOX, Ultralytics, object detection, real-time AI, edge AI, computer vision, model comparison
---

# YOLOv9 VS YOLOX

YOLOv9 and YOLOX represent significant advancements in the field of object detection, each offering unique strengths tailored to different applications. This comparison explores their performance, architecture, and efficiency to guide users in selecting the best model for their needs.

While YOLOv9 emphasizes a balance of speed and precision, making it suitable for real-time applications, YOLOX showcases versatility with its adaptable design and robust performance. Both models push the boundaries of [computer vision](https://docs.ultralytics.com/tasks/) and are widely adopted across industries such as autonomous driving and smart surveillance.


## mAP Comparison

This section compares the mAP values of YOLOv9 and YOLOX, showcasing their accuracy across various model sizes. Mean Average Precision (mAP) is a critical metric for evaluating object detection performance, balancing precision and recall to highlight model efficiency. Learn more about [mAP and its significance](https://www.ultralytics.com/glossary/mean-average-precision-map).


!!! tip "Accuracy"

	=== "Detection (COCO)"

		| Variant | mAP<sup>val<br>50<br>YOLOv9 | mAP<sup>val<br>50<br>YOLOX |
		|---------------------|-------------------------------------------------------|-------------------------------------------------------|
		| n | 37.8 | N/A |
		| s | 46.5 | 40.5 |
		| m | 51.5 | 46.9 |
		| l | 52.8 | 49.7 |
		| x | 55.1 | 51.1 |
		

## Speed Comparison

This section highlights the speed metrics of YOLOv9 versus YOLOX across various model sizes, measured in milliseconds. These comparisons demonstrate the efficiency of each model in real-time applications, enabling users to make informed decisions about performance trade-offs. Learn more about YOLOX [here](https://github.com/Megvii-BaseDetection/YOLOX) and YOLOv9 advancements [here](https://docs.ultralytics.com/models/yolov10/).


!!! tip "Speed"

	=== "Detection (COCO)"

		| Variant | Speed<br><sup>T4 TensorRT10<br>(ms)</sup><br>YOLOv9 | Speed<br><sup>T4 TensorRT10<br>(ms)</sup><br>YOLOX |
		|---------------------|-------------------------------------------------------|-------------------------------------------------------|
		| n | 2.3 | N/A |
		| s | 3.54 | 2.56 |
		| m | 6.43 | 5.43 |
		| l | 7.16 | 9.04 |
		| x | 16.77 | 16.1 |

## YOLO11 Functionalities: Predict

Ultralytics YOLO11's **predict** functionality is a cornerstone of real-time object detection, classification, and segmentation. This feature allows users to leverage pre-trained models or their custom-trained models to analyze new data effectively. Whether you're working with images, videos, or streaming data, YOLO11 ensures accurate predictions with unparalleled speed and efficiency.

To perform predictions using YOLO11, simply load your model and execute the prediction command. The following Python snippet demonstrates how to predict on an image:

```python
from ultralytics import YOLO

# Load a pre-trained YOLO11 model
model = YOLO('yolo11.pt')

# Predict on an image
results = model.predict(source='image.jpg', show=True, save=True)
```

For more details on prediction workflows and advanced use cases, visit the [Ultralytics Modes Documentation](https://docs.ultralytics.com/modes/). This guide covers everything from input formats to optimizing prediction performance, ensuring users can maximize the potential of YOLO11 in their projects.
