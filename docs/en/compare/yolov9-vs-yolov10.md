---
comments: true
description: Compare YOLOv9 and YOLOv10, two cutting-edge models from Ultralytics, to explore advancements in real-time object detection, efficiency, and accuracy. Discover how YOLOv10's innovative architecture and NMS-free training outperform YOLOv9 in speed and performance, making it a top choice for computer vision and edge AI applications.
keywords: YOLOv9, YOLOv10, Ultralytics, object detection, real-time AI, edge AI, computer vision, NMS-free training, model comparison
---

# YOLOv9 VS YOLOv10

The comparison between YOLOv9 and YOLOv10 showcases the rapid evolution of object detection models developed by Ultralytics. Both models represent significant milestones in balancing speed, accuracy, and efficiency, making them key tools for real-time AI applications.

YOLOv9 introduced innovative enhancements in feature extraction and efficiency, setting a strong foundation for real-time tasks. Meanwhile, YOLOv10 builds on these advancements by integrating NMS-free training and a holistic design strategy, achieving superior accuracy-latency trade-offs for diverse use cases. Learn more about [YOLOv9](https://www.youtube.com/watch?v=ZF7EAodHn1U&t=1s) and [YOLOv10](https://docs.ultralytics.com/models/yolov10/) to explore their unique capabilities.

## mAP Comparison

The mAP values highlight the accuracy of YOLOv9 and YOLOv10 models across their various iterations, showcasing advancements in precision and efficiency. YOLOv10 variants consistently outperform YOLOv9, offering better accuracy while reducing computational overhead. Learn more about [mean Average Precision (mAP)](https://www.ultralytics.com/glossary/mean-average-precision-map) and its importance in evaluating object detection models.

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

This section highlights the speed performance differences between YOLOv9 and YOLOv10 across various model sizes. By analyzing latency metrics in milliseconds, it demonstrates how YOLOv10 achieves faster inference times while maintaining efficiency and accuracy. For more on YOLOv10's innovations, visit the [Ultralytics YOLOv10 documentation](https://docs.ultralytics.com/models/yolov10/).

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

## YOLO11 Functionalities: Predict

The **Predict** functionality in Ultralytics YOLO11 enables users to perform object detection, segmentation, classification, and more with unparalleled speed and accuracy. This functionality is designed to handle a wide range of tasks, from simple image classification to advanced edge applications like real-time object detection in streaming video.

Using the Ultralytics Python package, the process to predict is straightforward. By loading a pre-trained YOLO11 model or fine-tuning a model on a custom dataset, users can easily run inference on any image or video. The prediction outputs include bounding boxes, labels, and confidence scores, making it ideal for applications such as monitoring wildlife, automating quality control, or enhancing security systems.

For more details, visit the [YOLO11 Prediction Guide](https://docs.ultralytics.com/modes/predict/).

### Example: Running Predictions With YOLO11

```python
from ultralytics import YOLO

# Load a pre-trained YOLO11 model
model = YOLO("yolo11.pt")

# Run predictions on an image
results = model.predict(source="image.jpg", save=True)

# Display results
results.show()
```

This simple example demonstrates how YOLO11 empowers users to quickly achieve actionable insights from visual data. Learn more about customization and advanced prediction techniques in the [Ultralytics Documentation](https://docs.ultralytics.com/).
