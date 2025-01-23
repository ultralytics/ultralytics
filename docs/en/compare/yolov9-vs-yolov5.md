---
comments: true  
description: Explore a comprehensive comparison between YOLOv9 and Ultralytics YOLOv5, highlighting advancements in object detection, real-time AI performance, and edge AI capabilities. Learn about their efficiency, accuracy, and applications in computer vision.  
keywords: YOLOv9, Ultralytics YOLOv5, object detection, real-time AI, edge AI, computer vision, Ultralytics, model comparison
---

# YOLOv9 VS Ultralytics YOLOv5

YOLOv9 and Ultralytics YOLOv5 are pivotal models in the evolution of AI-driven object detection. This comparison highlights the technological advancements and distinctive features that make each model stand out in the computer vision landscape.

While YOLOv9 showcases improved accuracy and efficiency with modern design techniques, Ultralytics YOLOv5 remains a benchmark for simplicity and widespread adoption. Both models cater to diverse use cases, ensuring robust performance across various applications like [real-time detection](https://www.ultralytics.com/glossary/object-detection) and [AI deployment](https://docs.ultralytics.com/guides/model-deployment-options/).


## mAP Comparison

This section evaluates the performance of YOLOv9 and Ultralytics YOLOv5 models using mean Average Precision (mAP) as a metric. mAP provides a comprehensive measure of accuracy, balancing precision and recall across various object classes and thresholds. Learn more about mAP and its significance in [object detection metrics](https://docs.ultralytics.com/guides/yolo-performance-metrics/).


!!! tip "Accuracy"

	=== "Detection (COCO)"

		| Variant | mAP<sup>val<br>50<br>YOLOv9 | mAP<sup>val<br>50<br>YOLOv5 |
		|---------------------|-------------------------------------------------------|-------------------------------------------------------|
		| n | 37.8 | N/A |
		| s | 46.5 | 37.4 |
		| m | 51.5 | 45.4 |
		| l | 52.8 | 49.0 |
		| x | 55.1 | 50.7 |
		

## Speed Comparison

This section highlights the speed performance of YOLOv9 and Ultralytics YOLOv5 across various model sizes, measured in milliseconds. By comparing inference times, it provides insights into their efficiency for real-time applications, leveraging advancements in [TensorRT optimization](https://docs.ultralytics.com/reference/utils/benchmarks/) and lightweight design.


!!! tip "Speed"

	=== "Detection (COCO)"

		| Variant | Speed<br><sup>T4 TensorRT10<br>(ms)</sup><br>YOLOv9 | Speed<br><sup>T4 TensorRT10<br>(ms)</sup><br>YOLOv5 |
		|---------------------|-------------------------------------------------------|-------------------------------------------------------|
		| n | 2.3 | N/A |
		| s | 3.54 | 1.92 |
		| m | 6.43 | 4.03 |
		| l | 7.16 | 6.61 |
		| x | 16.77 | 11.89 |

## Mastering YOLO Performance Metrics

<<<<<<< HEAD
Understanding and optimizing performance metrics is crucial when working with Ultralytics YOLO11 models. Metrics such as mAP (mean Average Precision), IoU (Intersection over Union), and F1 Score evaluate the effectiveness of your model in tasks like object detection, classification, and segmentation. These metrics provide insights into how well your model is performing and help identify areas for improvement.
=======
Ultralytics YOLO11 excels at segmentation tasks, enabling users to identify and isolate specific objects within an image. This functionality is particularly useful in applications like car parts segmentation, where precise object boundaries are necessary for tasks such as automotive repairs, manufacturing, or e-commerce cataloging. YOLO11's segmentation capabilities are powered by its robust architecture, ensuring high accuracy and efficiency.
>>>>>>> 95d73b193a43ffc9b65d81b5b93f6d2acf3cb195

For instance, mAP is commonly used to assess detection accuracy, while IoU evaluates the overlap between predicted and actual bounding boxes. F1 Score combines precision and recall, offering a balanced performance measure. By analyzing these metrics, you can fine-tune your model's hyperparameters and enhance its accuracy and speed.

Learn more about performance metrics and how they influence your YOLO11 model's success in our [YOLO Performance Metrics Guide](https://docs.ultralytics.com/guides/yolo-performance-metrics/), which includes practical examples and optimization tips.
