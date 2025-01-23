---
comments: true
description: Discover the detailed comparison between YOLOv9 and RTDETRv2, two state-of-the-art models for real-time object detection. Explore their performance, efficiency, and adaptability for computer vision and edge AI applications, powered by Ultralytics' cutting-edge advancements.
keywords: YOLOv9, RTDETRv2, Ultralytics, object detection, real-time AI, edge AI, computer vision, model comparison
---

# YOLOv9 VS RTDETRv2

The comparison between YOLOv9 and RTDETRv2 highlights the evolution of object detection technologies, showcasing advancements in speed, accuracy, and efficiency. These models cater to different technical needs, making it essential to evaluate their capabilities for specific applications like real-time detection and resource-constrained environments.

YOLOv9, a continuation of the YOLO series, excels in achieving a balance between precision and lightweight design, making it ideal for diverse tasks. In contrast, RTDETRv2 leverages Vision Transformer-based architecture, offering high accuracy and adaptable inference speeds, as detailed in its [documentation](https://docs.ultralytics.com/reference/models/rtdetr/model/).

## mAP Comparison

This section highlights the performance differences between YOLOv9 and RTDETRv2 by comparing their mAP values, a critical metric for evaluating object detection accuracy. Mean Average Precision (mAP) reflects how effectively each model identifies and localizes objects across various datasets and thresholds, ensuring a fair assessment of their precision and recall capabilities. Learn more about [mAP metrics](https://www.ultralytics.com/glossary/mean-average-precision-map) and their role in evaluating model performance.

!!! tip "Accuracy"

    === "Detection (COCO)"

    	| Variant | mAP<sup>val<br>50<br>YOLOv9 | mAP<sup>val<br>50<br>RTDETRv2 |
    	|---------------------|-------------------------------------------------------|-------------------------------------------------------|
    	| n | 37.8 | N/A |
    	| s | 46.5 | 48.1 |
    	| m | 51.5 | 51.9 |
    	| l | 52.8 | 53.4 |
    	| x | 55.1 | 54.3 |

## Speed Comparison

This section highlights the speed performance of YOLOv9 and RTDETRv2 across various sizes, assessed in milliseconds. These metrics, tested with formats like TensorRT, provide valuable insights into their real-world efficiency on tasks such as object detection. Learn more about [benchmarking models](https://docs.ultralytics.com/modes/benchmark/) for detailed comparisons.

!!! tip "Speed"

    === "Detection (COCO)"

    	| Variant | Speed<br><sup>T4 TensorRT10<br>(ms)</sup><br>YOLOv9 | Speed<br><sup>T4 TensorRT10<br>(ms)</sup><br>RTDETRv2 |
    	|---------------------|-------------------------------------------------------|-------------------------------------------------------|
    	| n | 2.3 | N/A |
    	| s | 3.54 | 5.03 |
    	| m | 6.43 | 7.51 |
    	| l | 7.16 | 9.76 |
    	| x | 16.77 | 15.03 |

## Fine-Tuning With African Wildlife Dataset

The African Wildlife dataset is a popular choice for training and fine-tuning object detection models, particularly for conservation and wildlife monitoring applications. By leveraging Ultralytics YOLO11, you can efficiently detect and classify various animal species in their natural habitats. This dataset provides a diverse set of labeled images, enabling researchers and developers to create robust models for tasks like species identification, population monitoring, and poaching prevention.

To get started with fine-tuning Ultralytics YOLO11 on the African Wildlife dataset, you can utilize the [Ultralytics Python package](https://pypi.org/project/ultralytics/) for seamless integration. This process involves dataset preparation, model configuration, and training. For a detailed walkthrough, explore the [custom training guide](https://www.ultralytics.com/blog/custom-training-ultralytics-yolo11-with-computer-vision-datasets).

By fine-tuning on this dataset, YOLO11 can achieve higher mAP scores, making it an ideal solution for real-time wildlife monitoring and conservation efforts. Learn more about dataset preparation and annotations in the [Ultralytics documentation](https://docs.ultralytics.com/datasets/).
