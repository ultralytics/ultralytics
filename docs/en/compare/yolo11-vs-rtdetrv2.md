---
comments: true
description: Compare Ultralytics YOLO11 and RTDETRv2 to explore their advancements in object detection, real-time AI, and computer vision. Discover which model excels in speed, accuracy, and efficiency for applications like edge AI and real-time deployments.
keywords: Ultralytics, YOLO11, RTDETRv2, object detection, real-time AI, edge AI, computer vision, efficiency, model comparison
---

# Ultralytics YOLO11 VS RTDETRv2

The comparison between Ultralytics YOLO11 and RTDETRv2 represents a critical evaluation of two cutting-edge models in the field of object detection. Both models aim to redefine performance benchmarks in terms of accuracy, speed, and efficiency for real-world applications.

Ultralytics YOLO11 excels with its advanced feature extraction and optimized architecture, ensuring unparalleled precision and efficiency. On the other hand, RTDETRv2 offers robust real-time capabilities, making it a strong contender for applications requiring rapid inference. Explore how these models stack up in key metrics and deployment scenarios.

## mAP Comparison

This section compares the mAP values of Ultralytics YOLO11 and RTDETRv2, showcasing their accuracy across various model variants. Mean Average Precision (mAP) is a critical metric that evaluates a model's ability to detect objects precisely, balancing between recall and precision. Learn more about [mAP and its significance](https://www.ultralytics.com/glossary/mean-average-precision-map).

!!! tip "Accuracy"

    === "Detection (COCO)"

    	| Variant | mAP<sup>val<br>50<br>YOLO11 | mAP<sup>val<br>50<br>RTDETRv2 |
    	|---------------------|-------------------------------------------------------|-------------------------------------------------------|
    	| n | 39.5 | N/A |
    	| s | 47.0 | 48.1 |
    	| m | 51.4 | 51.9 |
    	| l | 53.2 | 53.4 |
    	| x | 54.7 | 54.3 |


## Speed Comparison

This section highlights the speed performance of Ultralytics YOLO11 versus RTDETRv2 across various model sizes. Speed metrics, measured in milliseconds, provide insights into their efficiency for tasks like object detection, emphasizing the trade-offs between rapid inference and computational demands. For more details, explore [Ultralytics Benchmarking Tools](https://docs.ultralytics.com/modes/benchmark/).

!!! tip "Speed"

    === "Detection (COCO)"

    	| Variant | Speed<br><sup>T4 TensorRT10<br>(ms)</sup><br>YOLO11 | Speed<br><sup>T4 TensorRT10<br>(ms)</sup><br>RTDETRv2 |
    	|---------------------|-------------------------------------------------------|-------------------------------------------------------|
    	| n | 1.55 | N/A |
    	| s | 2.63 | 5.03 |
    	| m | 5.27 | 7.51 |
    	| l | 6.84 | 9.76 |
    	| x | 12.49 | 15.03 |

## Leveraging YOLO Performance Metrics for Model Optimization

Performance metrics are crucial when evaluating and improving the accuracy and efficiency of your Ultralytics YOLO11 models. By understanding metrics like mAP (mean Average Precision), IoU (Intersection over Union), and the F1 score, you can fine-tune your model for better detection and segmentation results.

To explore these metrics in detail, refer to the [YOLO Performance Metrics guide](https://docs.ultralytics.com/guides/yolo-performance-metrics/). This guide provides practical examples, insightful tips, and strategies to optimize your model's performance. Whether you're working on object detection, segmentation, or classification, these metrics help you assess the effectiveness of your model in real-world scenarios.

For additional learning, check out the [Comprehensive Tutorials to Ultralytics YOLO](https://docs.ultralytics.com/guides/) to gain deeper insights into model evaluation and improvement. These resources ensure you’re well-equipped to achieve the accuracy and speed required for your computer vision tasks.
