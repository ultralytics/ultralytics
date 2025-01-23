---
comments: true
description: Discover how Ultralytics YOLOv8 and YOLOX compare in terms of accuracy, speed, and real-time object detection performance. Explore their strengths in computer vision, edge AI, and real-world applications to determine the best choice for your needs.
keywords: Ultralytics, YOLOv8, YOLOX, object detection, real-time AI, edge AI, computer vision, model comparison
---

# Ultralytics YOLOv8 VS YOLOX

Ultralytics YOLOv8 and YOLOX represent two of the most advanced object detection models in the industry, each excelling in unique ways. This comparison explores their performance, architecture, and real-world applications, offering insights for researchers and developers to make informed choices.

While YOLOX emphasizes flexibility with its decoupled head and anchor-free design, Ultralytics YOLOv8 stands out with its state-of-the-art accuracy and speed. Dive deeper into how these models address diverse [object detection](https://www.ultralytics.com/glossary/object-detection) challenges and push the boundaries of real-time AI solutions.

## mAP Comparison

This section highlights the mAP values of Ultralytics YOLOv8 and YOLOX across their variants, showcasing their accuracy in detecting and classifying objects. Mean Average Precision (mAP) is a vital metric for evaluating model performance, as it combines precision and recall to assess detection accuracy comprehensively. Learn more about [mAP](https://www.ultralytics.com/glossary/mean-average-precision-map) and its significance in object detection.

!!! tip "Accuracy"

    === "Detection (COCO)"

    	| Variant | mAP<sup>val<br>50<br>YOLOv8 | mAP<sup>val<br>50<br>YOLOX |
    	|---------------------|-------------------------------------------------------|-------------------------------------------------------|
    	| n | 37.3 | N/A |
    	| s | 44.9 | 40.5 |
    	| m | 50.2 | 46.9 |
    	| l | 52.9 | 49.7 |
    	| x | 53.9 | 51.1 |

## Speed Comparison

This section highlights the speed metrics of Ultralytics YOLOv8 and YOLOX models, measured in milliseconds across different model sizes. Faster inference times achieved by YOLOv8 emphasize its efficiency for real-time applications compared to YOLOX. Learn more about [Ultralytics YOLOv8](https://docs.ultralytics.com/models/yolov8/) and its benchmarks.

!!! tip "Speed"

    === "Detection (COCO)"

    	| Variant | Speed<br><sup>T4 TensorRT10<br>(ms)</sup><br>YOLOv8 | Speed<br><sup>T4 TensorRT10<br>(ms)</sup><br>YOLOX |
    	|---------------------|-------------------------------------------------------|-------------------------------------------------------|
    	| n | 1.47 | N/A |
    	| s | 2.66 | 2.56 |
    	| m | 5.86 | 5.43 |
    	| l | 9.06 | 9.04 |
    	| x | 14.37 | 16.1 |

## Segment: Leveraging YOLO11 for Image Segmentation

Ultralytics YOLO11 excels in **image segmentation**, enabling precise identification and delineation of objects within images. This functionality is ideal for applications such as car parts segmentation, package analysis, and more. For example, [car parts segmentation](https://docs.ultralytics.com/datasets/segment/carparts-seg/) simplifies automotive manufacturing processes by identifying individual components efficiently.

YOLO11's segmentation capabilities are built on a robust framework, allowing for seamless integration of custom datasets like COCO and others. With support for both pre-trained and fine-tuned models, it offers unparalleled flexibility for tasks requiring detailed object outlines.

To start with segmentation using YOLO11, you can leverage its custom training mode on datasets tailored to your specific needs. Visit the [Image Segmentation Guide](https://www.ultralytics.com/blog/image-segmentation-with-ultralytics-yolo11-on-google-colab) for a detailed walkthrough.

### Python Code Example: YOLO11 Segmentation

```python
from ultralytics import YOLO

# Load a pre-trained YOLO11 model
model = YOLO("yolov11-seg.pt")

# Perform segmentation on an image
results = model.segment(source="path/to/image.jpg", save=True)

# Save results to an output folder
results.show()
```

Explore more about YOLO11's segmentation capabilities to redefine your project outcomes!
