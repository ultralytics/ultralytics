---
comments: true
description: Compare the advanced capabilities of Ultralytics YOLO11 and YOLOv8 to discover which model delivers superior performance in object detection, real-time AI, and edge AI applications. Explore their key differences in accuracy, speed, and adaptability across various computer vision tasks.
keywords: Ultralytics YOLO11, YOLOv8, object detection, real-time AI, edge AI, computer vision, Ultralytics models, AI comparison
---

# Ultralytics YOLO11 VS Ultralytics YOLOv8

The comparison between Ultralytics YOLO11 and YOLOv8 highlights the evolution of object detection models, showcasing advancements in accuracy, speed, and efficiency. Both models represent significant milestones in computer vision, offering cutting-edge solutions for diverse real-world applications.

Ultralytics YOLOv8 introduced a robust architecture with an anchor-free design, optimized for real-time tasks. In contrast, YOLO11 builds on these strengths, offering enhanced feature extraction and reduced computational costs, making it an ideal choice for demanding scenarios. Learn more about [Ultralytics YOLOv8](https://docs.ultralytics.com/models/yolov8/) and [YOLO11](https://docs.ultralytics.com/models/yolo11/) to explore their unique capabilities.

# <<<<<<< HEAD

While YOLOv8 introduced groundbreaking features like an anchor-free architecture and optimized accuracy-speed tradeoffs, YOLO11 builds upon this foundation with enhanced feature extraction and greater efficiency. Both models excel in diverse tasks, from object detection to pose estimation, ensuring versatility for a wide range of use cases. [Learn more about YOLOv8](https://docs.ultralytics.com/models/yolov8/) or [explore YOLO11's capabilities](https://docs.ultralytics.com/models/yolo11/).

> > > > > > > 95d73b193a43ffc9b65d81b5b93f6d2acf3cb195

## mAP Comparison

This section highlights the mAP values achieved by Ultralytics YOLO11 and YOLOv8 across different model variants, showcasing their accuracy in object detection tasks. Mean Average Precision (mAP) is a widely used metric that evaluates model performance by balancing precision and recall; learn more about [mAP](https://www.ultralytics.com/glossary/mean-average-precision-map). YOLO11 demonstrates a notable improvement, achieving higher mAP scores with fewer parameters, making it a more efficient and precise model. Explore the advancements of [Ultralytics YOLO11 here](https://docs.ultralytics.com/models/yolo11/).

!!! tip "Accuracy"

    === "Detection (COCO)"

    	| Variant | mAP<sup>val<br>50<br>YOLO11 | mAP<sup>val<br>50<br>YOLOv8 |
    	|---------------------|-------------------------------------------------------|-------------------------------------------------------|
    	| n | 39.5 | 37.3 |
    	| s | 47.0 | 44.9 |
    	| m | 51.4 | 50.2 |
    	| l | 53.2 | 52.9 |
    	| x | 54.7 | 53.9 |


## Speed Comparison

This section highlights the speed metrics of Ultralytics YOLO11 and YOLOv8 across various model sizes, showcasing performance improvements in milliseconds per inference. With YOLO11 offering faster processing times, it is ideal for real-time applications, as shown in benchmarks like [TensorRT FP16](https://docs.ultralytics.com/integrations/tensorrt/) on NVIDIA GPUs. Explore the detailed performance of [Ultralytics YOLO models](https://docs.ultralytics.com/models/).

!!! tip "Speed"

    === "Detection (COCO)"

    	| Variant | Speed<br><sup>T4 TensorRT10<br>(ms)</sup><br>YOLO11 | Speed<br><sup>T4 TensorRT10<br>(ms)</sup><br>YOLOv8 |
    	|---------------------|-------------------------------------------------------|-------------------------------------------------------|
    	| n | 1.55 | 1.47 |
    	| s | 2.63 | 2.66 |
    	| m | 5.27 | 5.86 |
    	| l | 6.84 | 9.06 |
    	| x | 12.49 | 14.37 |

## YOLO Thread-Safe Inference

Thread-safe inference is a critical aspect when deploying Ultralytics YOLO11 models in multi-threaded environments. By ensuring thread safety, you can prevent race conditions, maintain consistent outputs, and improve the reliability of your object detection or classification tasks. This capability is particularly useful in applications like real-time monitoring, robotics, and IoT devices.

Ultralytics YOLO11 provides comprehensive support for thread-safe inference. Developers can leverage this functionality to optimize model performance while ensuring stability. To explore best practices for implementing thread-safe inference with YOLO11, consult the [YOLO Thread-Safe Inference Guide](https://docs.ultralytics.com/guides/yolo-thread-safe-inference/). This guide offers detailed instructions and examples to help you integrate this feature into your workflow effectively.

For more information on optimizing model performance and deployment, check out the [Model Deployment Options](https://docs.ultralytics.com/guides/model-deployment-options/) guide for insights into supported formats like ONNX and TensorRT.
