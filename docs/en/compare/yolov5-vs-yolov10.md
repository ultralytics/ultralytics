---
comments: true
description: Discover the key differences between Ultralytics YOLOv5 and YOLOv10 in this comprehensive comparison. Explore advancements in object detection, real-time AI performance, and edge AI capabilities, as we highlight their unique strengths and applications in modern computer vision tasks.
keywords: YOLOv5, YOLOv10, Ultralytics, object detection, real-time AI, edge AI, computer vision, model comparison
---

# Ultralytics YOLOv5 VS YOLOv10

The evolution of YOLO models has consistently pushed the boundaries of real-time object detection, and this comparison of Ultralytics YOLOv5 and YOLOv10 highlights key advancements. Each model introduces unique strengths, offering researchers and developers critical insights into performance, efficiency, and real-world applicability.

Ultralytics YOLOv5 is celebrated for its simplicity and widespread adoption, while YOLOv10 delivers cutting-edge architecture optimized for accuracy and latency. With this analysis, we aim to showcase how these models cater to diverse use cases, from resource-constrained environments to large-scale applications. For further insights, explore [YOLOv10 documentation](https://docs.ultralytics.com/models/yolov10/) and the [Ultralytics YOLOv5 overview](https://docs.ultralytics.com/models/yolov8/).

## mAP Comparison

This section highlights the mAP values of Ultralytics YOLOv5 and YOLOv10, showcasing their accuracy across different model variants. Mean Average Precision (mAP) evaluates detection performance, balancing precision and recall, and provides a clear metric to compare the advancements of YOLOv10 over its predecessor. Learn more about [mAP](https://www.ultralytics.com/glossary/mean-average-precision-map) and its significance in model evaluation.

!!! tip "Accuracy"

    === "Detection (COCO)"

    	| Variant | mAP<sup>val<br>50<br>YOLOv5 | mAP<sup>val<br>50<br>YOLOv10 |
    	|---------------------|-------------------------------------------------------|-------------------------------------------------------|
    	| n | N/A | 39.5 |
    	| s | 37.4 | 46.7 |
    	| m | 45.4 | 51.3 |
    	| b | N/A | 52.7 |
    	| l | 49.0 | 53.3 |
    	| x | 50.7 | 54.4 |

## Speed Comparison

This section highlights the speed metrics in milliseconds for Ultralytics YOLOv5 and YOLOv10 across various model sizes. By comparing inference times, it showcases the advancements in performance and efficiency offered by YOLOv10. For more details, explore the official [YOLOv5 documentation](https://docs.ultralytics.com/models/yolov5/) and [YOLOv10 overview](https://docs.ultralytics.com/models/yolov10/).

!!! tip "Speed"

    === "Detection (COCO)"

    	| Variant | Speed<br><sup>T4 TensorRT10<br>(ms)</sup><br>YOLOv5 | Speed<br><sup>T4 TensorRT10<br>(ms)</sup><br>YOLOv10 |
    	|---------------------|-------------------------------------------------------|-------------------------------------------------------|
    	| n | N/A | 1.56 |
    	| s | 1.92 | 2.66 |
    	| m | 4.03 | 5.48 |
    	| b | N/A | 6.54 |
    	| l | 6.61 | 8.33 |
    	| x | 11.89 | 12.2 |

## Object Cropping with Ultralytics YOLO11

Object cropping is a powerful feature of Ultralytics YOLO11 that allows users to isolate and extract objects from images or video frames. This functionality is especially useful in applications such as retail analytics, automated quality control, and surveillance, where cropped images of detected objects can be further analyzed or stored for record-keeping.

Using YOLO11’s object cropping feature, users can easily automate the process of detecting and saving object-specific regions, reducing manual effort and increasing efficiency. To leverage this feature, you can integrate it with the [Ultralytics Python package](https://pypi.org/project/ultralytics/) or use the intuitive [Ultralytics HUB](https://www.ultralytics.com/hub) platform for a no-code approach.

Learn more about object cropping and other solutions by exploring our [comprehensive guides](https://docs.ultralytics.com/guides/). Whether for real-time or batch processing, YOLO11 simplifies tasks with its cutting-edge capabilities.
