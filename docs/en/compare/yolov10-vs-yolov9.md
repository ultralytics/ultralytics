---
comments: true  
description: Discover the key differences and advancements between YOLOv10 and YOLOv9 in this detailed comparison. Learn how these Ultralytics models push the boundaries of real-time AI, object detection, and edge AI with enhanced accuracy, efficiency, and speed for cutting-edge computer vision applications.  
keywords: YOLOv10, YOLOv9, Ultralytics, object detection, real-time AI, edge AI, computer vision, model comparison
---

# YOLOv10 VS YOLOv9

YOLOv10 and YOLOv9 represent significant milestones in the evolution of real-time object detection, showcasing innovative approaches to balancing speed and accuracy. This comparison highlights their architectural advancements and performance metrics to help users understand the unique strengths of each model.

YOLOv9 introduced noteworthy optimizations in feature extraction and detection efficiency, setting new benchmarks for object detection tasks. On the other hand, [YOLOv10](https://docs.ultralytics.com/models/yolov10/) builds upon this foundation with NMS-free training and a holistic design strategy, further improving [accuracy and latency](https://www.ultralytics.com/blog/ultralytics-yolo11-has-arrived-redefine-whats-possible-in-ai) across diverse applications.


## mAP Comparison

This section highlights the mAP values of YOLOv10 and YOLOv9 across different model variants, showcasing their respective accuracies on tasks such as object detection. Mean Average Precision (mAP) provides a holistic measure of a model's precision and recall, offering insights into performance improvements between these versions. Learn more about [mAP](https://www.ultralytics.com/glossary/mean-average-precision-map) and its role in evaluating object detection models.


!!! tip "Accuracy"

	=== "Detection (COCO)"

		| Variant | mAP<sup>val<br>50<br>YOLOv10 | mAP<sup>val<br>50<br>YOLOv9 |
		|---------------------|-------------------------------------------------------|-------------------------------------------------------|
		| n | 39.5 | 37.8 |
		| s | 46.7 | 46.5 |
		| m | 51.3 | 51.5 |
		| b | 52.7 | N/A |
		| l | 53.3 | 52.8 |
		| x | 54.4 | 55.1 |
		

## Speed Comparison

This section highlights the speed performance of YOLOv10 versus YOLOv9 across multiple model sizes, measured in milliseconds. By analyzing inference times, such as those optimized for TensorRT, it showcases the efficiency gains achieved with [YOLOv10](https://docs.ultralytics.com/models/yolov10/). These improvements are critical for real-time applications requiring low-latency processing.


!!! tip "Speed"

	=== "Detection (COCO)"

		| Variant | Speed<br><sup>T4 TensorRT10<br>(ms)</sup><br>YOLOv10 | Speed<br><sup>T4 TensorRT10<br>(ms)</sup><br>YOLOv9 |
		|---------------------|-------------------------------------------------------|-------------------------------------------------------|
		| n | 1.56 | 2.3 |
		| s | 2.66 | 3.54 |
		| m | 5.48 | 6.43 |
		| b | 6.54 | N/A |
		| l | 8.33 | 7.16 |
		| x | 12.2 | 16.77 |

## Exploring YOLO11 Functionalities: Segment

Ultralytics YOLO11 offers robust segmentation capabilities, making it an excellent choice for tasks that require precise object identification and separation. Segmentation divides an image into distinct regions, helping machines understand and analyze visual data more effectively. This is particularly useful in applications like autonomous driving, medical imaging, and manufacturing.

For example, YOLO11 can be custom-trained to segment car parts, enabling automated inspection and cataloging in the automotive industry. It also supports pre-trained models such as those trained on the [COCO dataset](https://docs.ultralytics.com/datasets/segment/coco/) for general-purpose segmentation tasks. With built-in tools for validation and evaluation, YOLO11 simplifies the entire segmentation workflow.

To explore how to set up segmentation tasks with YOLO11, check out the [image segmentation guide](https://www.ultralytics.com/blog/image-segmentation-with-ultralytics-yolo11-on-google-colab).

### Python Code Example for Segmentation

```python
from ultralytics import YOLO

# Load a pre-trained YOLO11 segmentation model
model = YOLO('yolov11-seg.pt')

# Perform segmentation on an image
results = model('image.jpg', task='segment')

# Visualize results
results.show()
```

This example demonstrates how easy it is to perform image segmentation using YOLO11 and visualize the results.
