---
comments: true
description: Compare RTDETRv2 and DAMO-YOLO, two cutting-edge models in real-time object detection and computer vision. Explore their performance, features, and capabilities for real-time AI and edge AI applications. Learn how these models excel in advanced scenarios and redefine the benchmarks for object detection tasks.
keywords: RTDETRv2, DAMO-YOLO, Ultralytics, object detection, real-time AI, edge AI, computer vision, real-time detection, AI models comparison
---

# RTDETRv2 VS DAMO-YOLO

The comparison between RTDETRv2 and DAMO-YOLO highlights the advancements in real-time object detection, focusing on speed and accuracy. These models represent cutting-edge solutions, catering to the demands of AI applications in diverse industries such as autonomous driving and smart retail.

RTDETRv2 stands out with its Vision Transformer-based architecture, delivering high accuracy with adaptable inference speeds, as detailed in its [reference documentation](https://docs.ultralytics.com/reference/models/rtdetr/model/). Meanwhile, DAMO-YOLO leverages innovative model designs to balance computational efficiency and precision, making it ideal for high-performance tasks.

## mAP Comparison

This section highlights the mAP values of RTDETRv2 and DAMO-YOLO, showcasing their accuracy across various model variants. Mean Average Precision (mAP) is a critical metric for evaluating object detection performance, balancing precision and recall. Learn more about [mAP](https://www.ultralytics.com/glossary/mean-average-precision-map) and its importance in model benchmarking.

!!! tip "Accuracy"

    === "Detection (COCO)"

    	| Variant | mAP<sup>val<br>50<br>RTDETRv2 | mAP<sup>val<br>50<br>DAMO-YOLO |
    	|---------------------|-------------------------------------------------------|-------------------------------------------------------|
    	| n | N/A | 42.0 |
    	| s | 48.1 | 46.0 |
    	| m | 51.9 | 49.2 |
    	| l | 53.4 | 50.8 |
    	| x | 54.3 | N/A |


## Speed Comparison

This section highlights the speed performance of RTDETRv2 and DAMO-YOLO models, measured in milliseconds across different sizes. By comparing their latency metrics, you can evaluate their suitability for time-critical applications and hardware configurations. For more on benchmarking, explore [Ultralytics benchmarking tools](https://docs.ultralytics.com/reference/utils/benchmarks/).

!!! tip "Speed"

    === "Detection (COCO)"

    	| Variant | Speed<br><sup>T4 TensorRT10<br>(ms)</sup><br>RTDETRv2 | Speed<br><sup>T4 TensorRT10<br>(ms)</sup><br>DAMO-YOLO |
    	|---------------------|-------------------------------------------------------|-------------------------------------------------------|
    	| n | N/A | 2.32 |
    	| s | 5.03 | 3.45 |
    	| m | 7.51 | 5.09 |
    	| l | 9.76 | 7.18 |
    	| x | 15.03 | N/A |

## Training With Ultralytics YOLO11

Ultralytics YOLO11 offers a seamless and powerful training functionality, enabling users to fine-tune models on custom datasets for specific use cases. Whether you are working with datasets like COCO8, African wildlife, or car parts segmentation, YOLO11 makes the process efficient and intuitive. The model leverages advanced optimization techniques, such as mixed precision training, to deliver high accuracy and performance.

To get started, explore the [Training Mode Documentation](https://docs.ultralytics.com/modes/train/) for step-by-step guidance on setting up your training process. With support for various data formats and augmentation strategies, YOLO11 ensures flexibility and scalability for diverse applications.

### Example: Python Code for Training

```python
from ultralytics import YOLO

# Load a pre-trained YOLO11 model
model = YOLO("yolov11.pt")

# Train the model on a custom dataset
model.train(data="path/to/dataset.yaml", epochs=50, imgsz=640, batch=16)

# Evaluate model performance
metrics = model.val()
print(metrics)
```

By following these steps, you can customize Ultralytics YOLO11 to meet your project requirements with ease.
