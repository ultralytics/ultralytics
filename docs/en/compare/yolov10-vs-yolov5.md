---
comments: true
description: Discover an in-depth comparison between YOLOv10 and Ultralytics YOLOv5, highlighting their performance in object detection, real-time AI applications, and advancements in computer vision. Learn how these models excel in edge AI scenarios and redefine efficiency and accuracy in AI-based solutions.
keywords: YOLOv10, Ultralytics YOLOv5, object detection, real-time AI, edge AI, computer vision, Ultralytics, AI models comparison, YOLO models
---

# YOLOv10 VS Ultralytics YOLOv5

The comparison between YOLOv10 and Ultralytics YOLOv5 showcases the evolution of object detection models, highlighting advancements in speed, accuracy, and efficiency. Each model represents a significant milestone in AI, offering distinct strengths suited for diverse applications in computer vision.

YOLOv10 introduces innovative architectural enhancements, improving performance with fewer parameters and lower latency. In contrast, Ultralytics YOLOv5 has set a benchmark for reliability and versatility, excelling in real-time object detection across various domains. Learn more about [YOLOv10](https://docs.ultralytics.com/models/yolov10/) and [Ultralytics YOLOv5](https://docs.ultralytics.com/models/yolov5/).


## mAP Comparison

This section evaluates the mAP (mean Average Precision) performance of YOLOv10 and Ultralytics YOLOv5 across their respective variants. mAP serves as a key metric for assessing the accuracy of object detection models, balancing precision and recall to provide a comprehensive performance overview. Learn more about [mAP](https://www.ultralytics.com/glossary/mean-average-precision-map) and its significance in model evaluation.


!!! tip "Accuracy"

	=== "Detection (COCO)"

		| Variant | mAP<sup>val<br>50<br>YOLOv10 | mAP<sup>val<br>50<br>YOLOv5 |
		|---------------------|-------------------------------------------------------|-------------------------------------------------------|
		| n | 39.5 | N/A |
		| s | 46.7 | 37.4 |
		| m | 51.3 | 45.4 |
		| b | 52.7 | N/A |
		| l | 53.3 | 49.0 |
		| x | 54.4 | 50.7 |
		

## Speed Comparison

Explore the inference times in milliseconds to compare the performance of YOLOv10 and Ultralytics YOLOv5 across various model sizes. This analysis highlights the advancements in speed and efficiency achieved by YOLOv10, designed for real-time applications. Learn more about [YOLOv10's architecture](https://docs.ultralytics.com/models/yolov10/) and how it achieves lower latency.


!!! tip "Speed"

	=== "Detection (COCO)"

		| Variant | Speed<br><sup>T4 TensorRT10<br>(ms)</sup><br>YOLOv10 | Speed<br><sup>T4 TensorRT10<br>(ms)</sup><br>YOLOv5 |
		|---------------------|-------------------------------------------------------|-------------------------------------------------------|
		| n | 1.56 | N/A |
		| s | 2.66 | 1.92 |
		| m | 5.48 | 4.03 |
		| b | 6.54 | N/A |
		| l | 8.33 | 6.61 |
		| x | 12.2 | 11.89 |

## Segment Functionality in YOLO11  

The **segment functionality** in Ultralytics YOLO11 empowers developers to perform precise image segmentation tasks by identifying and isolating specific objects within an image. This feature is particularly crucial for applications requiring detailed object boundaries, such as medical imaging, automotive parts identification, and e-commerce cataloging. YOLO11 supports both general-purpose segmentation using pre-trained models and custom training for specialized datasets.  

For instance, custom segmentation can be performed on datasets like [car parts segmentation](https://docs.ultralytics.com/datasets/segment/carparts-seg/) to streamline manufacturing and repair processes. With YOLO11, segmentation tasks are simplified by leveraging its robust architecture and user-friendly tools.  

### Python Code Example  

```python
from ultralytics import YOLO  

# Load a pre-trained YOLO11 segmentation model  
model = YOLO('yolo11-seg.pt')  

# Perform segmentation on an image  
results = model.segment(source='image.jpg', save=True, show=True)  
```  

Learn more about [image segmentation with YOLO11](https://www.ultralytics.com/blog/image-segmentation-with-ultralytics-yolo11-on-google-colab).
