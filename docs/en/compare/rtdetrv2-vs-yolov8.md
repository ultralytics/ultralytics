---
comments: true
description: Explore the detailed comparison between RT-DETRv2 and Ultralytics YOLOv8, two cutting-edge models in object detection. Discover how these models excel in real-time AI, edge AI, and computer vision applications with their unique features, performance metrics, and deployment capabilities.
keywords: RTDETRv2, Ultralytics YOLOv8, object detection, real-time AI, edge AI, computer vision, model comparison, AI performance, Ultralytics
---

# RTDETRv2 VS Ultralytics YOLOv8

The comparison between RTDETRv2 and Ultralytics YOLOv8 underscores the advancements in real-time object detection technologies. Both models represent cutting-edge solutions, designed to deliver optimal performance in speed, accuracy, and efficiency for diverse applications.

RTDETRv2 stands out with its innovative methodologies like consistent dual assignments for NMS-free training, while Ultralytics YOLOv8 excels in combining state-of-the-art accuracy with user-friendly workflows. Explore their unique capabilities to determine the best fit for your computer vision needs. Learn more about [Ultralytics YOLOv8](https://docs.ultralytics.com/models/yolov8/) and [RTDETR](https://github.com/ultralytics/ultralytics).

## mAP Comparison

The mAP values highlight the accuracy of RTDETRv2 and Ultralytics YOLOv8 across different variants, showcasing their ability to detect and classify objects effectively. For a deeper understanding of mAP and its role in evaluating object detection models, explore [mean average precision](https://www.ultralytics.com/glossary/mean-average-precision-map) and its significance in model performance.

!!! tip "Accuracy"

    === "Detection (COCO)"

    	| Variant | mAP<sup>val<br>50<br>RTDETRv2 | mAP<sup>val<br>50<br>YOLOv8 |
    	|---------------------|-------------------------------------------------------|-------------------------------------------------------|
    	| n | N/A | 37.3 |
    	| s | 48.1 | 44.9 |
    	| m | 51.9 | 50.2 |
    	| l | 53.4 | 52.9 |
    	| x | 54.3 | 53.9 |

## Speed Comparison

This section highlights the speed performance of RT-DETRv2 and Ultralytics YOLOv8 models, measured in milliseconds across various sizes. Faster inference speeds, especially those achieved by YOLOv8, demonstrate its capability for real-time applications, as noted in [Ultralytics' documentation](https://docs.ultralytics.com/models/yolov10/). Explore how these metrics impact practical use cases like [object detection](https://docs.ultralytics.com/tasks/detect/) and more.

!!! tip "Speed"

    === "Detection (COCO)"

    	| Variant | Speed<br><sup>T4 TensorRT10<br>(ms)</sup><br>RTDETRv2 | Speed<br><sup>T4 TensorRT10<br>(ms)</sup><br>YOLOv8 |
    	|---------------------|-------------------------------------------------------|-------------------------------------------------------|
    	| n | N/A | 1.47 |
    	| s | 5.03 | 2.66 |
    	| m | 7.51 | 5.86 |
    	| l | 9.76 | 9.06 |
    	| x | 15.03 | 14.37 |

## Leveraging Ultralytics YOLO11 for Object Counting

Ultralytics YOLO11 offers advanced capabilities for object counting, making it a powerful tool for industries like retail, traffic management, and event analytics. With its real-time detection and counting features, YOLO11 can accurately manage scenarios such as monitoring crowd density, counting vehicles in intersections, or analyzing stock levels in warehouses. This functionality is highly valuable for optimizing operations and improving decision-making processes.

Explore the [object counting guide](https://docs.ultralytics.com/guides/object-counting/) to understand how YOLO11 can be implemented in various real-world use cases. The guide provides detailed steps on configuring the model for object counting tasks and demonstrates its integration into workflows. With YOLO11’s flexibility and precision, businesses can extract actionable insights from their visual data effortlessly.

For more on YOLO11’s applications, visit the [Ultralytics documentation](https://docs.ultralytics.com/).
