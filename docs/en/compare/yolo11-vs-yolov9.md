---
comments: true
description: Compare Ultralytics YOLO11 and YOLOv9 to discover how these advanced models revolutionize object detection, real-time AI, and computer vision. Explore their performance, accuracy, and efficiency, and find out which model suits your edge AI applications best.  
keywords: Ultralytics, YOLO11, YOLOv9, object detection, real-time AI, edge AI, computer vision, model comparison
---

# Ultralytics YOLO11 VS YOLOv9

The comparison between Ultralytics YOLO11 and YOLOv9 represents a significant milestone in understanding advancements in computer vision. Both models have reshaped the field with their unique capabilities, but YOLO11's arrival pushes the boundaries of speed, accuracy, and efficiency further than ever before.

Ultralytics YOLO11 stands out with its enhanced feature extraction, optimized architecture, and superior adaptability across platforms like edge devices and cloud systems. Meanwhile, YOLOv9 laid the groundwork for these innovations, excelling in real-time applications. Explore [YOLO11's key features](https://docs.ultralytics.com/models/yolo11/) to see how it builds upon the solid foundation set by YOLOv9's pioneering design.


## mAP Comparison

This section highlights the mAP performance of Ultralytics YOLO11 and YOLOv9 across various model variants, showcasing advancements in detection accuracy. Mean Average Precision (mAP), a key evaluation metric in object detection, reflects how effectively each model identifies and localizes objects across diverse datasets like [COCO](https://docs.ultralytics.com/datasets/detect/coco/). Learn more about [mAP](https://www.ultralytics.com/glossary/mean-average-precision-map) and its significance in evaluating AI models.


!!! tip "Accuracy"

	=== "Detection (COCO)"

		| Variant | mAP<sup>val<br>50<br>YOLO11 | mAP<sup>val<br>50<br>YOLOv9 |
		|---------------------|-------------------------------------------------------|-------------------------------------------------------|
		| n | 39.5 | 37.8 |
		| s | 47.0 | 46.5 |
		| m | 51.4 | 51.5 |
		| l | 53.2 | 52.8 |
		| x | 54.7 | 55.1 |
		

## Speed Comparison

This section highlights the speed performance of Ultralytics YOLO11 versus YOLOv9 across various model sizes, showcasing inference time in milliseconds. These metrics underscore the efficiency improvements of YOLO11, making it ideal for real-time applications. Learn more about [Ultralytics YOLO11](https://docs.ultralytics.com/models/yolo11/) and its advancements.


!!! tip "Speed"

	=== "Detection (COCO)"

		| Variant | Speed<br><sup>T4 TensorRT10<br>(ms)</sup><br>YOLO11 | Speed<br><sup>T4 TensorRT10<br>(ms)</sup><br>YOLOv9 |
		|---------------------|-------------------------------------------------------|-------------------------------------------------------|
		| n | 1.55 | 2.3 |
		| s | 2.63 | 3.54 |
		| m | 5.27 | 6.43 |
		| l | 6.84 | 7.16 |
		| x | 12.49 | 16.77 |

## YOLO Performance Metrics

Understanding performance metrics is crucial for evaluating and improving the accuracy of your YOLO11 models. Metrics like mean Average Precision (mAP), Intersection over Union (IoU), and F1 Score provide a quantitative way to assess how well your model is performing in object detection tasks.

For instance, mAP is widely used to evaluate the precision-recall trade-off, while IoU measures the overlap between predicted and actual bounding boxes. These metrics help identify areas for refinement in both training and inference phases. Leveraging these insights enables you to fine-tune your model for better results.

For a detailed explanation and practical examples of these metrics, check out [YOLO Performance Metrics](https://docs.ultralytics.com/guides/yolo-performance-metrics/). This guide also includes tips to enhance detection accuracy and speed. Whether you're a beginner or an advanced user, mastering these metrics will elevate your computer vision projects.
