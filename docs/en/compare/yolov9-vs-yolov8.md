---
comments: true
description: Compare YOLOv9 and Ultralytics YOLOv8 to uncover their advancements in object detection, real-time AI, and edge applications. Explore how these models excel in speed, accuracy, and versatility, empowering computer vision tasks across diverse industries.
keywords: YOLOv9, Ultralytics YOLOv8, object detection, real-time AI, edge AI, computer vision, Ultralytics, model comparison, AI advancements
---

# YOLOv9 VS Ultralytics YOLOv8

Comparing YOLOv9 and Ultralytics YOLOv8 reveals key advancements in real-time object detection and computer vision capabilities. Both models showcase substantial improvements in speed, accuracy, and efficiency, making them pivotal tools for modern AI applications.

YOLOv9 introduces enhanced architectural designs and improved feature extraction, while Ultralytics YOLOv8 stands out for its simplicity and flexibility, supporting diverse tasks like detection, segmentation, and classification. Explore how these models meet the demands of cutting-edge [object detection](https://www.ultralytics.com/glossary/object-detection) across varied industries.

## mAP Comparison

Mean Average Precision (mAP) evaluates the performance of object detection models, highlighting their accuracy across various thresholds. This section compares YOLOv9 and Ultralytics YOLOv8, showcasing their mAP values to illustrate improvements in precision and efficiency. Learn more about [mAP](https://www.ultralytics.com/glossary/mean-average-precision-map) and its importance in model evaluation.

!!! tip "Accuracy"

    === "Detection (COCO)"

    	| Variant | mAP<sup>val<br>50<br>YOLOv9 | mAP<sup>val<br>50<br>YOLOv8 |
    	|---------------------|-------------------------------------------------------|-------------------------------------------------------|
    	| n | 37.8 | 37.3 |
    	| s | 46.5 | 44.9 |
    	| m | 51.5 | 50.2 |
    	| l | 52.8 | 52.9 |
    	| x | 55.1 | 53.9 |

## Speed Comparison

Compare the inference speeds of YOLOv9 and Ultralytics YOLOv8 across various model sizes, measured in milliseconds. These speed metrics highlight the efficiency and suitability of each model for real-time [object detection tasks](https://docs.ultralytics.com/tasks/detect/). For further insights into YOLOv8's performance, visit the [YOLOv8 documentation](https://docs.ultralytics.com/models/yolov8/).

!!! tip "Speed"

    === "Detection (COCO)"

    	| Variant | Speed<br><sup>T4 TensorRT10<br>(ms)</sup><br>YOLOv9 | Speed<br><sup>T4 TensorRT10<br>(ms)</sup><br>YOLOv8 |
    	|---------------------|-------------------------------------------------------|-------------------------------------------------------|
    	| n | 2.3 | 1.47 |
    	| s | 3.54 | 2.66 |
    	| m | 6.43 | 5.86 |
    	| l | 7.16 | 9.06 |
    	| x | 16.77 | 14.37 |

## Export Functionality in Ultralytics YOLO11

<<<<<<< HEAD
Ultralytics YOLO11 offers a robust export functionality, enabling seamless integration with various inference engines and formats, such as ONNX, OpenVINO, TensorFlow Lite, and more. This feature ensures that trained models can be efficiently deployed across diverse hardware platforms, including edge devices and cloud environments.
=======
Ultralytics YOLO11 empowers users to solve advanced computer vision challenges, including **object counting**. This functionality is particularly valuable in scenarios like retail analytics, traffic monitoring, and inventory management, where accurate counting is critical. By leveraging YOLO11's real-time detection capabilities, users can achieve precise object counts even in dynamic environments.

> > > > > > > 95d73b193a43ffc9b65d81b5b93f6d2acf3cb195

Exporting a YOLO11 model is straightforward. Using the Ultralytics Python package, you can export models with a single command. The export functionality supports optimization techniques that enhance performance and reduce latency, making it ideal for real-time applications.

For a step-by-step guide on exporting models and supported formats, explore the [Export Models Documentation](https://docs.ultralytics.com/modes/export/). Additionally, you can check out the [ONNX Integration Guide](https://docs.ultralytics.com/integrations/onnx/) for details on leveraging ONNX for deployment.

By simplifying the export process, YOLO11 empowers developers to focus on building innovative solutions with maximum efficiency.
