---
comments: true
description: Explore the detailed comparison between YOLOv7 and DAMO-YOLO, two cutting-edge models in real-time AI and object detection. Learn about their performance, speed, and suitability for various computer vision applications, including edge AI deployments.
keywords: YOLOv7, DAMO-YOLO, Ultralytics, object detection, real-time AI, computer vision, edge AI, AI models comparison
---

# YOLOv7 VS DAMO-YOLO

Comparing YOLOv7 and DAMO-YOLO showcases the advancements in real-time object detection, segmentation, and classification. Both models represent cutting-edge innovation, offering unique strengths suited to diverse applications in AI.

YOLOv7, a member of the YOLO family, is known for its balance of speed and accuracy, excelling in real-time tasks. DAMO-YOLO, on the other hand, brings competitive performance metrics with a focus on efficiency and scalability for complex deployments. For more on YOLO models, explore [Ultralytics YOLO models](https://docs.ultralytics.com/models/yolov8/).

## mAP Comparison

This section examines the mAP values of YOLOv7 and DAMO-YOLO, offering a direct comparison of their accuracy in object detection across various model variants. Mean Average Precision (mAP) serves as a critical metric, balancing precision and recall to evaluate the effectiveness of these models comprehensively. Learn more about [Mean Average Precision](https://www.ultralytics.com/glossary/mean-average-precision-map) and its relevance in model evaluations.

!!! tip "Accuracy"

    === "Detection (COCO)"

    	| Variant | mAP<sup>val<br>50<br>YOLOv7 | mAP<sup>val<br>50<br>DAMO-YOLO |
    	|---------------------|-------------------------------------------------------|-------------------------------------------------------|
    	| n | N/A | 42.0 |
    	| s | N/A | 46.0 |
    	| m | N/A | 49.2 |
    	| l | 51.4 | 50.8 |
    	| x | 53.1 | N/A |

## Speed Comparison

This section highlights the speed performance of YOLOv7 and DAMO-YOLO models across different sizes, measured in milliseconds per image. These metrics provide a clear understanding of the inference efficiency of each model, helping users assess their suitability for real-time applications. For further insights, explore the [YOLOv7 documentation](https://docs.ultralytics.com/models/yolov7/) or learn about [benchmarking tools](https://docs.ultralytics.com/reference/utils/benchmarks/).

!!! tip "Speed"

    === "Detection (COCO)"

    	| Variant | Speed<br><sup>T4 TensorRT10<br>(ms)</sup><br>YOLOv7 | Speed<br><sup>T4 TensorRT10<br>(ms)</sup><br>DAMO-YOLO |
    	|---------------------|-------------------------------------------------------|-------------------------------------------------------|
    	| n | N/A | 2.32 |
    	| s | N/A | 3.45 |
    	| m | N/A | 5.09 |
    	| l | 6.84 | 7.18 |
    	| x | 11.57 | N/A |

## Train With Ultralytics YOLO11

Training models with Ultralytics YOLO11 is efficient and highly customizable, enabling users to fine-tune the model on various datasets for optimal performance. With support for diverse datasets like COCO8 and custom datasets, YOLO11 adapts seamlessly to different applications, ensuring accurate results. The training process is streamlined using the Ultralytics Python package, which includes tools for monitoring metrics like loss and accuracy.

For a deeper dive into training YOLO models, check out the [YOLO Training Guide](https://docs.ultralytics.com/modes/train/).

### Python Code Example

```python
from ultralytics import YOLO

# Load a YOLO model
model = YOLO("yolov11.pt")

# Train the model on a custom dataset
model.train(data="custom_dataset.yaml", epochs=50, batch=16, imgsz=640)
```

This snippet demonstrates setting up YOLO11 for training, making it easy to get started with your custom computer vision project.
