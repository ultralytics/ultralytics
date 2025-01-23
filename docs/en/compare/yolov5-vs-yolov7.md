---
comments: true
description: Explore a comprehensive comparison between Ultralytics YOLOv5 and YOLOv7, highlighting advancements in object detection, real-time AI performance, and edge AI capabilities. Discover how these models excel in computer vision applications with metrics like accuracy, speed, and efficiency.
keywords: Ultralytics, YOLOv5, YOLOv7, object detection, real-time AI, edge AI, computer vision, AI models, performance comparison
---

# Ultralytics YOLOv5 VS YOLOv7

Ultralytics YOLOv5 and YOLOv7 are two landmark models in real-time object detection, each representing significant advancements in AI technology. This comparison page delves into their unique capabilities, helping you understand which model suits your specific requirements.

YOLOv5, developed by Ultralytics, is renowned for its ease of use and robust performance across a variety of tasks. Meanwhile, YOLOv7 introduces cutting-edge innovations like model re-parameterization and dynamic label assignment, pushing the boundaries of speed and accuracy [as highlighted in the YOLOv7 paper](https://arxiv.org/pdf/2207.02696). Explore their strengths to make an informed decision for your next project.

## mAP Comparison

This section compares the mAP values of Ultralytics YOLOv5 and YOLOv7, showcasing their accuracy across different variants. Mean Average Precision (mAP) serves as a key metric in evaluating the models' performance in detecting and classifying objects, balancing precision and recall effectively. Learn more about [mAP and its significance](https://www.ultralytics.com/glossary/mean-average-precision-map).

!!! tip "Accuracy"

    === "Detection (COCO)"

    	| Variant | mAP<sup>val<br>50<br>YOLOv5 | mAP<sup>val<br>50<br>YOLOv7 |
    	|---------------------|-------------------------------------------------------|-------------------------------------------------------|
    	| s | 37.4 | N/A |
    	| m | 45.4 | N/A |
    	| l | 49.0 | 51.4 |
    	| x | 50.7 | 53.1 |

## Speed Comparison

The speed comparison highlights the performance of Ultralytics YOLOv5 and YOLOv7 in terms of inference time, measured in milliseconds across various model sizes. This metric is crucial for applications requiring real-time object detection, showcasing the models' efficiency on different hardware configurations. For details on YOLOv7 innovations, refer to the [YOLOv7 GitHub repository](https://github.com/WongKinYiu/yolov7) or explore the [Ultralytics YOLOv5 documentation](https://docs.ultralytics.com/models/yolov5/).

!!! tip "Speed"

    === "Detection (COCO)"

    	| Variant | Speed<br><sup>T4 TensorRT10<br>(ms)</sup><br>YOLOv5 | Speed<br><sup>T4 TensorRT10<br>(ms)</sup><br>YOLOv7 |
    	|---------------------|-------------------------------------------------------|-------------------------------------------------------|
    	| s | 1.92 | N/A |
    	| m | 4.03 | N/A |
    	| l | 6.61 | 6.84 |
    	| x | 11.89 | 11.57 |

## Insights on Model Evaluation and Fine-Tuning

Model evaluation and fine-tuning are critical steps in enhancing the performance of your Ultralytics YOLO11 models. These processes involve assessing the model's accuracy, identifying areas for improvement, and iteratively refining the parameters to achieve optimal results. By leveraging evaluation metrics like mAP, IoU, and F1 score, you can gain insights into your model's performance and make informed decisions about adjustments.

To dive deeper into strategies for improving your YOLO11 models, check out the [YOLO Performance Metrics Guide](https://docs.ultralytics.com/guides/yolo-performance-metrics/). This guide provides practical tips and examples for achieving higher accuracy and efficiency. Additionally, explore [Hyperparameter Tuning](https://docs.ultralytics.com/guides/hyperparameter-tuning/) to learn how to adjust parameters like batch size and learning rate for maximum effectiveness.

<<<<<<< HEAD
Fine-tuning your YOLO11 model ensures it is well-suited for specific tasks, enabling better real-world applications.
=======
To learn more about YOLO11's solutions, visit the [Ultralytics YOLO Guides](https://docs.ultralytics.com/guides/) for comprehensive tutorials or explore its practical applications in privacy management.

### Example Python Code

```python
from ultralytics import YOLO

# Load YOLO11 model
model = YOLO("yolo11-model.pt")

# Perform object detection and apply blur
results = model.predict(source="input_video.mp4", save=True, conf=0.5)

# Save output with blurred objects
results.save_blurred("output_video.mp4")
```

Explore how YOLO11 can streamline privacy tasks with its easy-to-use interface and powerful features.

> > > > > > > 95d73b193a43ffc9b65d81b5b93f6d2acf3cb195
