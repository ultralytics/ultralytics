---
comments: true
description: Explore the detailed comparison between Ultralytics YOLOv8 and YOLOv7, highlighting advancements in object detection, real-time AI performance, and edge AI capabilities. Discover how these models push the boundaries of computer vision applications with enhanced speed, accuracy, and versatility.
keywords: Ultralytics, YOLOv8, YOLOv7, object detection, real-time AI, edge AI, computer vision, AI models comparison, machine learning, advanced AI tools
---

# Ultralytics YOLOv8 VS YOLOv7

Ultralytics YOLOv8 and YOLOv7 represent two pivotal advancements in the YOLO family, redefining real-time object detection and segmentation. This comparison highlights the evolution of features, speed, and accuracy, offering insights into their respective capabilities.

YOLOv7 brought remarkable improvements in efficiency and accuracy, setting a new standard upon its release. Building on this foundation, [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics) integrates state-of-the-art innovations, such as anchor-free detection and optimized architectures, ensuring superior performance for diverse applications.

## mAP Comparison

The mAP (Mean Average Precision) metric provides a comprehensive evaluation of model accuracy across different object detection tasks. This section highlights the performance of Ultralytics YOLOv8 compared to YOLOv7, showcasing advancements in precision and recall. Learn more about [mAP](https://www.ultralytics.com/glossary/mean-average-precision-map) for detailed insights.

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

This section highlights the speed performance of Ultralytics YOLOv8 and YOLOv7 across various model sizes, measured in milliseconds per image. These metrics showcase how efficiently each model performs in real-time applications, providing insights into their suitability for tasks requiring high-speed object detection. Learn more about [YOLOv8's advancements](https://docs.ultralytics.com/models/yolov8/) and [YOLOv7's performance](https://docs.ultralytics.com/models/yolov7/).

!!! tip "Speed"

    === "Detection (COCO)"

    	| Variant | Speed<br><sup>T4 TensorRT10<br>(ms)</sup><br>YOLOv8 | Speed<br><sup>T4 TensorRT10<br>(ms)</sup><br>YOLOv7 |
    	|---------------------|-------------------------------------------------------|-------------------------------------------------------|
    	| n | 1.47 | N/A |
    	| s | 2.66 | N/A |
    	| m | 5.86 | N/A |
    	| l | 9.06 | 6.84 |
    	| x | 14.37 | 11.57 |

## Segment With Ultralytics YOLO11

Ultralytics YOLO11 excels in **image segmentation**, allowing users to separate objects from their backgrounds with precision. This functionality is particularly valuable in tasks like automotive manufacturing, where identifying specific car parts through segmentation can streamline processes such as repair, quality assessment, and e-commerce cataloging.

With YOLO11, you can perform both pre-trained segmentation using datasets like COCO or customize the model for unique datasets, such as **Car Parts Segmentation**. This flexibility ensures robust performance across diverse applications.

For a detailed step-by-step guide on image segmentation, including practical examples, check out the [image segmentation tutorial with YOLO11 on Google Colab](https://www.ultralytics.com/blog/image-segmentation-with-ultralytics-yolo11-on-google-colab).

### Sample Python Code for Segmentation with YOLO11

```python
from ultralytics import YOLO

# Load the YOLO11 model
model = YOLO("yolo11-seg.pt")

# Perform segmentation on an image
results = model.predict(source="car_parts.jpg", task="segment")

# Visualize the segmentation results
results.show()
```
