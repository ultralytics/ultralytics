---
comments: true
description: Discover the key differences between YOLOv6-3.0 and RT-DETRv2, two cutting-edge models in real-time object detection. This comparison highlights their performance, efficiency, and adaptability for applications in edge AI and computer vision.
keywords: YOLOv6-3.0, RT-DETRv2, Ultralytics, object detection, real-time AI, edge AI, computer vision, model comparison
---

# YOLOv6-3.0 VS RTDETRv2

The comparison between YOLOv6-3.0 and RTDETRv2 highlights the advancements in real-time object detection technology. Both models represent cutting-edge solutions, each excelling in specific areas of accuracy, efficiency, and deployment flexibility.

YOLOv6-3.0 offers remarkable performance with lightweight architecture optimized for speed and accuracy, while RTDETRv2 leverages Vision Transformer-based design for adaptable inference and multiscale feature processing. Explore their unique strengths and use cases to determine the ideal model for your needs. Learn more about [RT-DETR](https://docs.ultralytics.com/models/rtdetr/) or [YOLOv6](https://docs.ultralytics.com/models/yolov8/).

## mAP Comparison

This section highlights the mAP (Mean Average Precision) values of YOLOv6-3.0 and RTDETRv2, showcasing their accuracy in object detection across various thresholds. mAP, a key metric in evaluating models like [Ultralytics YOLO](https://docs.ultralytics.com/models/yolov10/), balances precision and recall to provide a comprehensive assessment of detection performance.

!!! tip "Accuracy"

    === "Detection (COCO)"

    	| Variant | mAP<sup>val<br>50<br>YOLOv6-3.0 | mAP<sup>val<br>50<br>RTDETRv2 |
    	|---------------------|-------------------------------------------------------|-------------------------------------------------------|
    	| n | 37.5 | N/A |
    	| s | 45.0 | 48.1 |
    	| m | 50.0 | 51.9 |
    	| l | 52.8 | 53.4 |
    	| x | N/A | 54.3 |

## Speed Comparison

This section highlights the performance differences between YOLOv6-3.0 and RTDETRv2, focusing on speed metrics measured in milliseconds. These metrics, evaluated across various input sizes, provide insights into the real-time capabilities of each model. For more, explore the [Ultralytics Benchmarks](https://docs.ultralytics.com/reference/utils/benchmarks/) and [RT-DETR details](https://docs.ultralytics.com/models/yolov10/).

!!! tip "Speed"

    === "Detection (COCO)"

    	| Variant | Speed<br><sup>T4 TensorRT10<br>(ms)</sup><br>YOLOv6-3.0 | Speed<br><sup>T4 TensorRT10<br>(ms)</sup><br>RTDETRv2 |
    	|---------------------|-------------------------------------------------------|-------------------------------------------------------|
    	| n | 1.17 | N/A |
    	| s | 2.66 | 5.03 |
    	| m | 5.28 | 7.51 |
    	| l | 8.95 | 9.76 |
    	| x | N/A | 15.03 |

## Object Counting with Ultralytics YOLO11

Object counting is a powerful solution offered by Ultralytics YOLO11, enabling precise identification and enumeration of objects in various applications. From monitoring pedestrian traffic to managing inventory in warehouses, YOLO11's object counting capabilities provide accurate real-time insights. This feature is particularly beneficial for industries such as retail, manufacturing, and urban planning, where understanding object density and distribution is critical.

Leveraging YOLO11’s advanced detection algorithms, users can integrate object counting into their workflows seamlessly. With support for edge devices and real-time processing, YOLO11 ensures that object counting tasks are both efficient and scalable.

For more details on how to implement object counting and other functionalities, explore the [Ultralytics YOLO11 documentation](https://docs.ultralytics.com/guides/) or check out the [Ultralytics HUB](https://www.ultralytics.com/hub) for an intuitive no-code solution.
