---
comments: true
description: Explore an in-depth comparison of RT-DETRv2 and YOLOv6-3.0, two advanced models in real-time AI and object detection. Discover their performance, efficiency, and adaptability for edge AI applications and computer vision tasks.
keywords: RT-DETRv2, YOLOv6-3.0, Ultralytics, object detection, real-time AI, edge AI, computer vision, model comparison
---

# RTDETRv2 VS YOLOv6-3.0

The comparison between RTDETRv2 and YOLOv6-3.0 explores two leading-edge object detection models designed to balance accuracy and efficiency. As advancements in AI-driven detection models continue, understanding their unique strengths and trade-offs is crucial for developers and researchers.

RTDETRv2 builds on Vision Transformer principles to deliver real-time performance with high accuracy, while YOLOv6-3.0 focuses on optimizing speed and parameter efficiency. This analysis highlights their performance across benchmarks like COCO, enabling informed decisions for diverse applications. For more details on RTDETR, visit the [RT-DETR model documentation](https://docs.ultralytics.com/reference/models/rtdetr/model/), and to learn about YOLOv6-3.0, explore the [YOLOv6 overview](https://docs.ultralytics.com/models/yolov6/).

## mAP Comparison

This section evaluates the accuracy of RTDETRv2 and YOLOv6-3.0 models by comparing their mAP values, a critical metric for assessing object detection performance. Learn more about [mAP](https://www.ultralytics.com/glossary/mean-average-precision-map) to understand how it reflects the models' precision and recall across various classes and thresholds.

!!! tip "Accuracy"

    === "Detection (COCO)"

    	| Variant | mAP<sup>val<br>50<br>RTDETRv2 | mAP<sup>val<br>50<br>YOLOv6-3.0 |
    	|---------------------|-------------------------------------------------------|-------------------------------------------------------|
    	| n | N/A | 37.5 |
    	| s | 48.1 | 45.0 |
    	| m | 51.9 | 50.0 |
    	| l | 53.4 | 52.8 |
    	| x | 54.3 | N/A |

## Speed Comparison

This section highlights the speed performance of RTDETRv2 and YOLOv6-3.0 across various model sizes. Measured in milliseconds, these metrics provide critical insights into the inference efficiency of both models, particularly on hardware-optimized platforms like TensorRT. Learn more about benchmarking techniques [here](https://docs.ultralytics.com/reference/utils/benchmarks/).

!!! tip "Speed"

    === "Detection (COCO)"

    	| Variant | Speed<br><sup>T4 TensorRT10<br>(ms)</sup><br>RTDETRv2 | Speed<br><sup>T4 TensorRT10<br>(ms)</sup><br>YOLOv6-3.0 |
    	|---------------------|-------------------------------------------------------|-------------------------------------------------------|
    	| n | N/A | 1.17 |
    	| s | 5.03 | 2.66 |
    	| m | 7.51 | 5.28 |
    	| l | 9.76 | 8.95 |
    	| x | 15.03 | N/A |

## SAHI Tiled Inference

SAHI (Sliced Aided Hyper Inference) tiled inference is a powerful technique supported by Ultralytics YOLO11, designed to enhance object detection in high-resolution images. It works by slicing large images into smaller tiles, enabling the detection of small objects that might otherwise be missed in the original resolution. This makes it an invaluable tool for applications such as satellite imagery analysis, medical imaging, and surveillance.

Using SAHI with YOLO11 ensures improved detection accuracy without sacrificing speed. For step-by-step guidance on implementing SAHI tiled inference, explore the [SAHI Tiled Inference Guide](https://docs.ultralytics.com/guides/sahi-tiled-inference/). This comprehensive resource covers setup, usage, and best practices for integrating tiled inference into your workflows.

To learn more about enhancing detection on challenging datasets or scenarios, check out the [Ultralytics YOLO documentation](https://docs.ultralytics.com/).
