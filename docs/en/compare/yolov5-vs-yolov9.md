---
comments: true
description: Compare Ultralytics YOLOv5 and YOLOv9 in this in-depth analysis of performance, accuracy, and efficiency. Discover how these state-of-the-art models excel in object detection, real-time AI applications, and edge AI deployments, driving advancements in computer vision technology.
keywords: YOLOv5, YOLOv9, Ultralytics, object detection, real-time AI, edge AI, computer vision, AI model comparison
---

# Ultralytics YOLOv5 VS YOLOv9

The comparison between YOLOv5 and YOLOv9 highlights the evolution of computer vision models over time. As two significant milestones in the YOLO series, these models showcase advancements in speed, accuracy, and deployment versatility.

YOLOv5 is celebrated for its balance of simplicity and performance, making it a favorite for many real-time applications. In contrast, YOLOv9 introduces cutting-edge enhancements in feature extraction and training pipelines, pushing the boundaries of efficiency and precision for modern AI tasks. For more insights, explore [Ultralytics YOLOv9](https://www.youtube.com/watch?v=ZF7EAodHn1U&t=1s) and [Ultralytics YOLOv5 architecture](https://docs.ultralytics.com/yolov5/tutorials/architecture_description/).


## mAP Comparison

This section highlights the mAP values of Ultralytics YOLOv5 and YOLOv9, showcasing their accuracy across different model sizes and configurations. Mean Average Precision (mAP) is a critical metric that evaluates object detection performance by balancing precision and recall. Learn more about [mAP](https://www.ultralytics.com/glossary/mean-average-precision-map) and its significance in model evaluation.


!!! tip "Accuracy"

	=== "Detection (COCO)"

		| Variant | mAP<sup>val<br>50<br>YOLOv5 | mAP<sup>val<br>50<br>YOLOv9 |
		|---------------------|-------------------------------------------------------|-------------------------------------------------------|
		| n | N/A | 37.8 |
		| s | 37.4 | 46.5 |
		| m | 45.4 | 51.5 |
		| l | 49.0 | 52.8 |
		| x | 50.7 | 55.1 |
		

## Speed Comparison

This section highlights the speed performance of Ultralytics YOLOv5 and YOLOv9 across various model sizes. Measured in milliseconds, these metrics reflect real-time processing capabilities, showcasing YOLOv9's efficiency improvements over YOLOv5 for diverse applications. Explore detailed comparisons in the [YOLOv5 docs](https://docs.ultralytics.com/models/yolov5/) and [YOLOv9 docs](https://docs.ultralytics.com/models/yolov9/).


!!! tip "Speed"

	=== "Detection (COCO)"

		| Variant | Speed<br><sup>T4 TensorRT10<br>(ms)</sup><br>YOLOv5 | Speed<br><sup>T4 TensorRT10<br>(ms)</sup><br>YOLOv9 |
		|---------------------|-------------------------------------------------------|-------------------------------------------------------|
		| n | N/A | 2.3 |
		| s | 1.92 | 3.54 |
		| m | 4.03 | 6.43 |
		| l | 6.61 | 7.16 |
		| x | 11.89 | 16.77 |

## Understanding YOLO11's Predict Functionality

<<<<<<< HEAD
The predict functionality in Ultralytics YOLO11 allows users to efficiently perform inference on images, videos, or streams. Leveraging its state-of-the-art architecture, YOLO11 ensures exceptional accuracy and speed for real-time object detection tasks. This feature is highly versatile, supporting tasks like object detection, segmentation, pose estimation, and object tracking.
=======
Ultralytics YOLO11 simplifies the training process, allowing users to fine-tune models for various tasks using custom datasets. With its robust framework, YOLO11 supports diverse datasets like COCO8, African wildlife, and signature detection, enabling high adaptability for unique use cases. The model's efficient architecture ensures faster convergence and improved accuracy during training.
>>>>>>> 95d73b193a43ffc9b65d81b5b93f6d2acf3cb195

To get started with the predict functionality, you can utilize the Ultralytics Python package. Here's an example:

```python
from ultralytics import YOLO

# Load a pretrained YOLO11 model
model = YOLO('yolo11.pt')

# Perform prediction on an image
results = model.predict(source='image.jpg', save=True, conf=0.5)

# Display results
results.show()
```

This functionality is ideal for various applications, including monitoring wildlife, retail analytics, and autonomous systems. For more information on using YOLO11 predict effectively, check out the [Ultralytics documentation](https://docs.ultralytics.com/modes/predict/). To explore additional advanced features like threading and deployment strategies, refer to the [YOLO Thread-Safe Inference Guide](https://docs.ultralytics.com/guides/yolo-thread-safe-inference/).
