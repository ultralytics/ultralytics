---
comments: true
description: Dive into a detailed comparison of RTDETRv2 and Ultralytics YOLOv5, two cutting-edge models in object detection and real-time AI. Explore their performance, accuracy, and suitability for edge AI applications in computer vision tasks.
keywords: RTDETRv2, Ultralytics YOLOv5, object detection, real-time AI, edge AI, computer vision, Ultralytics, model comparison
---

# RTDETRv2 VS Ultralytics YOLOv5

In the rapidly evolving field of computer vision, comparing cutting-edge models like RTDETRv2 and Ultralytics YOLOv5 is crucial to understanding their capabilities and applications. Both models represent significant advancements, offering unique strengths in terms of speed, accuracy, and efficiency for object detection tasks.

RTDETRv2 leverages its real-time transformer-based architecture for high precision in diverse scenarios, while Ultralytics YOLOv5 is renowned for its balance of performance and scalability across real-world applications. This page explores key differences and similarities, helping you choose the best solution for your specific needs. For more details on YOLOv5, visit the [Ultralytics YOLO Docs](https://docs.ultralytics.com/guides/).

## mAP Comparison

This section compares the mAP values of RTDETRv2 and Ultralytics YOLOv5, providing insights into their detection accuracy across different model variants. mAP, or Mean Average Precision, serves as a key metric to evaluate the precision and recall balance of these object detection models. Learn more about [Mean Average Precision (mAP)](https://www.ultralytics.com/glossary/mean-average-precision-map).

!!! tip "Accuracy"

    === "Detection (COCO)"

    	| Variant | mAP<sup>val<br>50<br>RTDETRv2 | mAP<sup>val<br>50<br>YOLOv5 |
    	|---------------------|-------------------------------------------------------|-------------------------------------------------------|
    	| s | 48.1 | 37.4 |
    	| m | 51.9 | 45.4 |
    	| l | 53.4 | 49.0 |
    	| x | 54.3 | 50.7 |


## Speed Comparison

This section highlights the speed performance of RTDETRv2 and Ultralytics YOLOv5 models across various sizes, measured in milliseconds. Faster inference times, as seen in [YOLOv5's performance](https://docs.ultralytics.com/models/yolov5/), reflect enhanced computational efficiency, critical for real-time applications.

!!! tip "Speed"

    === "Detection (COCO)"

    	| Variant | Speed<br><sup>T4 TensorRT10<br>(ms)</sup><br>RTDETRv2 | Speed<br><sup>T4 TensorRT10<br>(ms)</sup><br>YOLOv5 |
    	|---------------------|-------------------------------------------------------|-------------------------------------------------------|
    	| s | 5.03 | 1.92 |
    	| m | 7.51 | 4.03 |
    	| l | 9.76 | 6.61 |
    	| x | 15.03 | 11.89 |

## Segment With Ultralytics YOLO11

Ultralytics YOLO11 excels in segmentation tasks, allowing users to identify and classify individual objects within an image by segmenting them into meaningful regions. This functionality is highly beneficial for industries such as automotive, healthcare, and retail, where precise object identification is crucial. For example, YOLO11 can segment car parts for streamlined manufacturing or e-commerce cataloging.

YOLO11’s segmentation capabilities are further enhanced through custom training. By fine-tuning on datasets such as the [Car Parts Segmentation dataset](https://docs.ultralytics.com/datasets/segment/carparts-seg/), users can achieve tailored results with increased accuracy. The model supports pre-trained weights for general tasks and fine-tuning for specific use cases, making it adaptable and versatile.

To learn more about segmentation with YOLO11, explore the [Image Segmentation guide on Google Colab](https://www.ultralytics.com/blog/image-segmentation-with-ultralytics-yolo11-on-google-colab).

### Python Code Example

```python
from ultralytics import YOLO

# Load the YOLO11 model
model = YOLO("yolo11-seg.pt")

# Perform segmentation on an image
results = model("image.jpg", task="segment")

# Visualize results
results.show()
```
