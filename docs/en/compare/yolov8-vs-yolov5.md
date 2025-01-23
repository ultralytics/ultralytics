---
comments: true
description: Compare Ultralytics YOLOv8 and YOLOv5 to explore advancements in object detection, real-time AI, and edge AI. Discover how these state-of-the-art computer vision models differ in speed, accuracy, and versatility for various applications. 
keywords: Ultralytics, YOLOv8, YOLOv5, object detection, real-time AI, edge AI, computer vision, model comparison
---
# Ultralytics YOLOv8 VS YOLOv5
# Ultralytics YOLOv8 VS Ultralytics YOLOv5

Ultralytics YOLOv8 and YOLOv5 are pivotal advancements in the YOLO family, designed to address evolving needs in real-time object detection, segmentation, and classification. This comparison highlights the strengths of each model, helping users select the best option for diverse applications.

While YOLOv5 gained widespread adoption for its speed and simplicity, YOLOv8 takes innovation further with its state-of-the-art architecture and enhanced usability. Explore how these models excel in performance, accuracy, and flexibility to meet modern AI demands. Learn more about [YOLOv8's advancements](https://docs.ultralytics.com/models/yolov8/) and [YOLOv5's legacy](https://github.com/ultralytics/yolov5).


## mAP Comparison

The mAP (mean average precision) metric highlights the accuracy of object detection across different Ultralytics YOLOv8 and YOLOv5 variants. It evaluates precision and recall, providing a comprehensive performance measure for identifying and localizing objects. Learn more about [mAP evaluation](https://www.ultralytics.com/glossary/mean-average-precision-map) and its significance.


!!! tip "Accuracy"

	=== "Detection (COCO)"

		| Variant | mAP<sup>val<br>50<br>YOLOv8 | mAP<sup>val<br>50<br>YOLOv5 |
		|---------------------|-------------------------------------------------------|-------------------------------------------------------|
		| n | 37.3 | N/A |
		| s | 44.9 | 37.4 |
		| m | 50.2 | 45.4 |
		| l | 52.9 | 49.0 |
		| x | 53.9 | 50.7 |
		

## Speed Comparison

This section highlights the speed performance of Ultralytics YOLOv8 versus YOLOv5 across different model sizes, measured in milliseconds. These metrics demonstrate how efficiently each model processes data, with YOLOv8 offering significant improvements in real-time applications. For more details on YOLOv8's advancements, visit the [Ultralytics YOLOv8 overview](https://docs.ultralytics.com/models/yolov8/).


!!! tip "Speed"

	=== "Detection (COCO)"

		| Variant | Speed<br><sup>T4 TensorRT10<br>(ms)</sup><br>YOLOv8 | Speed<br><sup>T4 TensorRT10<br>(ms)</sup><br>YOLOv5 |
		|---------------------|-------------------------------------------------------|-------------------------------------------------------|
		| n | 1.47 | N/A |
		| s | 2.66 | 1.92 |
		| m | 5.86 | 4.03 |
		| l | 9.06 | 6.61 |
		| x | 14.37 | 11.89 |

## Segment With Ultralytics YOLO11

Ultralytics YOLO11 excels in **segmentation**, a critical functionality in computer vision that identifies and isolates objects within an image. This feature is particularly useful in applications like automotive manufacturing, where segmenting car parts or other components can streamline processes. With YOLO11, you can segment objects with high accuracy and even customize the model to suit specific datasets, such as the [Car Parts Segmentation dataset](https://docs.ultralytics.com/datasets/segment/carparts-seg/).

### Python Code Example for Segmentation

```python
from ultralytics import YOLO

# Load the pre-trained YOLO11 model
model = YOLO("yolo11-seg.pt")

# Perform segmentation on an image
results = model("car_parts.jpg", task="segment")

# Display the segmented results
results.show()
```

Leverage YOLO11's segmentation capabilities to achieve precise object differentiation in diverse industries. Explore more about segmentation tasks in [Ultralytics documentation](https://docs.ultralytics.com/tasks/segment/).
