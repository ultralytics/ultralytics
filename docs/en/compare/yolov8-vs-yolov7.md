---
comments: true
description: Compare the performance, speed, and features of ULTRALYTICS YOLOv8 and YOLOv7. Explore how these state-of-the-art models advance object detection, real-time AI, and edge AI applications in computer vision.
keywords: YOLOv8, YOLOv7, Ultralytics, object detection, real-time AI, edge AI, computer vision
---

# Ultralytics YOLOv8 VS YOLOv7

The evolution of YOLO models has brought significant advancements in computer vision, making comparisons between versions essential for understanding their unique capabilities. This page delves into the differences between Ultralytics YOLOv8 and YOLOv7, showcasing how each model excels in specific aspects of real-time object detection.

Ultralytics YOLOv8 introduces cutting-edge features like an anchor-free detection head and optimized speed-accuracy balance, setting a new benchmark in AI performance. Conversely, YOLOv7 remains a robust and efficient option, widely recognized for its accuracy and versatility in various object detection tasks. For more insights, explore the [Ultralytics YOLOv8 documentation](https://docs.ultralytics.com/models/yolov8/) or learn about [YOLOv7 advancements](https://github.com/WongKinYiu/yolov7).


## mAP Comparison

This section highlights the mAP values to compare the accuracy of Ultralytics YOLOv8 and YOLOv7 across different model variants. Mean Average Precision (mAP) is a critical metric in evaluating object detection performance, balancing precision and recall for comprehensive accuracy. Learn more about [mAP metrics](https://www.ultralytics.com/glossary/mean-average-precision-map).


!!! tip "Accuracy"

	=== "Detection (COCO)"

		| Variant | mAP<sup>val<br>50<br>YOLOv8 | mAP<sup>val<br>50<br>YOLOv7 |
		|---------------------|-------------------------------------------------------|-------------------------------------------------------|
		| n | 37.3 | N/A |
		| s | 44.9 | N/A |
		| m | 50.2 | N/A |
		| l | 52.9 | 51.4 |
		| x | 53.9 | 53.1 |
		

## Speed Comparison

This section highlights the speed metrics of Ultralytics YOLOv8 and YOLOv7 across various model sizes, measured in milliseconds. The comparison underscores YOLOv8's advancements in inference efficiency, making it ideal for real-time applications. For further details, explore the [YOLOv8 documentation](https://docs.ultralytics.com/models/yolov8/) and [YOLOv7 performance analysis](https://docs.ultralytics.com/models/yolov7/).


!!! tip "Speed"

	=== "Detection (COCO)"

		| Variant | Speed<br><sup>T4 TensorRT10<br>(ms)</sup><br>YOLOv8 | Speed<br><sup>T4 TensorRT10<br>(ms)</sup><br>YOLOv7 |
		|---------------------|-------------------------------------------------------|-------------------------------------------------------|
		| n | 1.47 | N/A |
		| s | 2.66 | N/A |
		| m | 5.86 | N/A |
		| l | 9.06 | 6.84 |
		| x | 14.37 | 11.57 |

## YOLO11 Functionalities: Predict

The Predict functionality of Ultralytics YOLO11 empowers users to perform real-time inference on images, videos, and streams. This feature allows you to harness YOLO11's capabilities for applications such as object detection, classification, and segmentation. Whether you're analyzing wildlife in African savannahs or detecting objects in industrial settings, YOLO11 delivers reliable predictions with exceptional speed and accuracy.

To get started with this functionality, you can use the Ultralytics Python package. Below is an example of how to run predictions:

```python
from ultralytics import YOLO

# Load a pre-trained YOLO11 model
model = YOLO('yolo11.pt')

# Predict on an image and display results
results = model.predict(source='image.jpg', save=True)
```

For more advanced usage, visit the [Ultralytics Predict Documentation](https://docs.ultralytics.com/modes/predict/) to learn about options like confidence thresholds, batch processing, and saving outputs. Dive into the power of YOLO11 to explore real-world solutions for your projects!
