---
comments: true
description: Discover the key differences between RTDETRv2 and YOLOv9 in this comprehensive comparison. Explore how these state-of-the-art models from Ultralytics excel in object detection, balancing real-time AI performance, edge AI efficiency, and computer vision accuracy to meet diverse application needs.
keywords: RTDETRv2, YOLOv9, Ultralytics, object detection, real-time AI, edge AI, computer vision, model comparison, AI performance, COCO dataset
---

# RTDETRv2 VS YOLOv9

The comparison between RTDETRv2 and YOLOv9 highlights two significant advancements in object detection technology, each excelling in unique ways. Both models set new benchmarks in speed, accuracy, and efficiency, making them ideal for a variety of real-world applications, including real-time and resource-constrained environments.

RTDETRv2 introduces refined architectural enhancements for improved inference speeds and robust performance, while YOLOv9 builds on [Ultralytics](https://www.ultralytics.com/)’ legacy with cutting-edge accuracy and efficient design. This comparison sheds light on their respective strengths, equipping researchers and developers with the knowledge to choose the best model for their specific needs. For more details on YOLOv9's evolution, refer to its [documentation](https://docs.ultralytics.com/models/).

## mAP Comparison

This section compares the mAP values of RTDETRv2 and YOLOv9, showcasing their accuracy across various model variants. Mean Average Precision (mAP) evaluates the models' ability to detect and classify objects effectively, balancing precision and recall. Learn more about [mAP](https://www.ultralytics.com/glossary/mean-average-precision-map) and its significance in model evaluation.

!!! tip "Accuracy"

    === "Detection (COCO)"

    	| Variant | mAP<sup>val<br>50<br>RTDETRv2 | mAP<sup>val<br>50<br>YOLOv9 |
    	|---------------------|-------------------------------------------------------|-------------------------------------------------------|
    	| n | N/A | 37.8 |
    	| s | 48.1 | 46.5 |
    	| m | 51.9 | 51.5 |
    	| l | 53.4 | 52.8 |
    	| x | 54.3 | 55.1 |


## Speed Comparison

This section highlights the speed performance of RTDETRv2 and YOLOv9 models across different sizes, measured in milliseconds. Comparing latency metrics, such as inference time, demonstrates their efficiency on real-world tasks, offering insights into trade-offs between speed and accuracy. For more details, explore [YOLOv9](https://docs.ultralytics.com/models/yolov9/) or [benchmarking tools](https://docs.ultralytics.com/reference/utils/benchmarks/).

!!! tip "Speed"

    === "Detection (COCO)"

    	| Variant | Speed<br><sup>T4 TensorRT10<br>(ms)</sup><br>RTDETRv2 | Speed<br><sup>T4 TensorRT10<br>(ms)</sup><br>YOLOv9 |
    	|---------------------|-------------------------------------------------------|-------------------------------------------------------|
    	| n | N/A | 2.3 |
    	| s | 5.03 | 3.54 |
    	| m | 7.51 | 6.43 |
    	| l | 9.76 | 7.16 |
    	| x | 15.03 | 16.77 |

## Fine-Tuning With COCO8 Dataset

Ultralytics YOLO11 offers exceptional flexibility for fine-tuning on various datasets, including the COCO8 dataset. COCO8 is a smaller subset of the COCO dataset, ideal for quick experimentation and prototyping. By leveraging COCO8, you can rapidly test your model's performance and optimize it for specific object detection tasks.

To fine-tune YOLO11 on COCO8, you can use the `ultralytics` Python package. The process involves loading the dataset, configuring training parameters, and initiating training. This allows the model to adapt to new data efficiently, ensuring high accuracy for your specific requirements. Learn more about [custom training with COCO datasets](https://docs.ultralytics.com/datasets/detect/coco/).

### Python Code: Fine-Tuning on COCO8

```python
from ultralytics import YOLO

# Load the YOLO11 model
model = YOLO("yolo11.yaml")

# Fine-tune on COCO8 dataset
model.train(data="coco8.yaml", epochs=50, imgsz=640)

# Validate the model
metrics = model.val()

# Save the fine-tuned model
model.export(format="onnx")
```

This simple process empowers users to achieve tailored performance for their computer vision tasks.
