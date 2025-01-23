---
comments: true
description: Compare YOLOv7 and YOLOX, two leading object detection models, to explore their performance, efficiency, and applications in real-time AI and computer vision. Discover how these models excel in edge AI and other advanced use cases.
keywords: YOLOv7, YOLOX, Ultralytics, object detection, real-time AI, edge AI, computer vision
---

# YOLOv7 VS YOLOX

Comparing YOLOv7 and YOLOX is essential for understanding the advancements in real-time object detection. Both models represent significant milestones in computer vision, offering unique features that cater to diverse application needs and performance benchmarks.

YOLOv7 stands out for its speed and accuracy, leveraging innovations like model re-parameterization and dynamic label assignment. On the other hand, YOLOX introduces anchor-free mechanisms that simplify training and improve detection efficiency, making it a versatile choice for modern AI workflows. For more, explore [YOLOv7 documentation](https://docs.ultralytics.com/models/yolov7/) and [object detection insights](https://www.ultralytics.com/glossary/object-detection).

## mAP Comparison

This section evaluates the mAP scores of YOLOv7 and YOLOX, highlighting their accuracy in object detection tasks across various model variants. mAP, a key metric in model performance, reflects the balance of precision and recall, offering insights into detection reliability. Learn more about [mAP in object detection](https://www.ultralytics.com/glossary/mean-average-precision-map).

!!! tip "Accuracy"

    === "Detection (COCO)"

    	| Variant | mAP<sup>val<br>50<br>YOLOv7 | mAP<sup>val<br>50<br>YOLOX |
    	|---------------------|-------------------------------------------------------|-------------------------------------------------------|
    	| s | N/A | 40.5 |
    	| m | N/A | 46.9 |
    	| l | 51.4 | 49.7 |
    	| x | 53.1 | 51.1 |


## Speed Comparison

This section highlights the speed metrics of YOLOv7 and YOLOX models across various sizes, showcasing their performance in milliseconds. YOLOv7 demonstrates superior inference speed compared to YOLOX, as evidenced by detailed benchmarks available on the [Ultralytics YOLOv7 documentation](https://docs.ultralytics.com/models/yolov7/) and other [relevant sources](https://github.com/Megvii-BaseDetection/YOLOX).

!!! tip "Speed"

    === "Detection (COCO)"

    	| Variant | Speed<br><sup>T4 TensorRT10<br>(ms)</sup><br>YOLOv7 | Speed<br><sup>T4 TensorRT10<br>(ms)</sup><br>YOLOX |
    	|---------------------|-------------------------------------------------------|-------------------------------------------------------|
    	| s | N/A | 2.56 |
    	| m | N/A | 5.43 |
    	| l | 6.84 | 9.04 |
    	| x | 11.57 | 16.1 |

## YOLO11 Functionalities: Segment

Ultralytics YOLO11's segmentation functionality enables precise identification and separation of objects within images, making it an exceptional tool for tasks requiring pixel-level accuracy. This feature is particularly beneficial in industries like healthcare, agriculture, and retail, where detailed object boundaries are critical.

With seamless integration into the [Ultralytics Python package](https://pypi.org/project/ultralytics/), users can easily load pre-trained models or fine-tune YOLO11 for custom segmentation tasks. Supported datasets, such as COCO8-Seg or Package Segmentation, allow users to experiment with diverse applications. YOLO11 also supports exporting models to formats like ONNX and TensorFlow Lite, ensuring compatibility with deployment pipelines.

For a step-by-step guide on using segmentation models, refer to the [Ultralytics YOLO Guides](https://docs.ultralytics.com/guides/). Dive deeper into this functionality and unlock new possibilities for your computer vision projects.
