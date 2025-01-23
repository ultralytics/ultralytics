---
comments: true
description: Explore an in-depth comparison between Ultralytics YOLOv5 and YOLOv8, highlighting advancements in object detection, real-time AI performance, and cutting-edge computer vision features. Learn how YOLOv8 surpasses its predecessor with enhanced speed, accuracy, and flexibility for diverse applications, including edge AI.
keywords: Ultralytics, YOLOv5, YOLOv8, object detection, real-time AI, edge AI, computer vision
---

# Ultralytics YOLOv5 VS Ultralytics YOLOv8

Ultralytics YOLOv5 and YOLOv8 represent two groundbreaking milestones in the evolution of real-time object detection and computer vision. This comparison dives into their unique features, showcasing their advancements and suitability for diverse applications in AI.

While YOLOv5 established itself as a widely adopted, user-friendly model, YOLOv8 takes innovation further with enhanced speed, accuracy, and flexibility. Explore how these models differ and determine which fits your needs by visiting the [YOLOv8 documentation](https://docs.ultralytics.com/models/yolov8/).

## mAP Comparison

Mean Average Precision (mAP) is a critical metric used to evaluate the accuracy of object detection models like Ultralytics YOLOv5 and YOLOv8. By comparing mAP values across variants, we can assess their ability to detect and classify objects effectively. Learn more about [mAP and its significance](https://www.ultralytics.com/glossary/mean-average-precision-map).

!!! tip "Accuracy"

    === "Detection (COCO)"

    	| Variant | mAP<sup>val<br>50<br>YOLOv5 | mAP<sup>val<br>50<br>YOLOv8 |
    	|---------------------|-------------------------------------------------------|-------------------------------------------------------|
    	| n | N/A | 37.3 |
    	| s | 37.4 | 44.9 |
    	| m | 45.4 | 50.2 |
    	| l | 49.0 | 52.9 |
    	| x | 50.7 | 53.9 |

## Speed Comparison

This section highlights the speed performance of Ultralytics YOLOv5 and YOLOv8 models across various sizes, measured in milliseconds. YOLOv8 demonstrates significant advancements in efficiency while maintaining its state-of-the-art detection capabilities, as detailed in the [YOLOv8 documentation](https://docs.ultralytics.com/models/yolov8/).

!!! tip "Speed"

    === "Detection (COCO)"

    	| Variant | Speed<br><sup>T4 TensorRT10<br>(ms)</sup><br>YOLOv5 | Speed<br><sup>T4 TensorRT10<br>(ms)</sup><br>YOLOv8 |
    	|---------------------|-------------------------------------------------------|-------------------------------------------------------|
    	| n | N/A | 1.47 |
    	| s | 1.92 | 2.66 |
    	| m | 4.03 | 5.86 |
    	| l | 6.61 | 9.06 |
    	| x | 11.89 | 14.37 |

## Hyperparameter Tuning

Fine-tuning hyperparameters is a crucial step in optimizing the performance of Ultralytics YOLO11 models. By carefully adjusting parameters like learning rate, batch size, momentum, and others, you can significantly enhance the model's accuracy and efficiency for specific tasks and datasets. YOLO11 provides robust tools, including the Tuner class and genetic evolution algorithms, to simplify and automate this process.

The Tuner class leverages advanced optimization techniques to explore the hyperparameter space effectively, saving time and computational resources. This feature is particularly valuable for large datasets like COCO8 or complex tasks like segmentation and pose estimation.

For a deeper dive into hyperparameter optimization and practical examples, explore the [Hyperparameter Tuning Guide](https://docs.ultralytics.com/guides/hyperparameter-tuning/). Whether you're working on object detection, tracking, or classification, these tools can help you extract the best performance from your YOLO11 model.
