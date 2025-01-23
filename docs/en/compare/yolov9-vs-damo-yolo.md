---
comments: true
description: Discover the key differences between YOLOv9 and DAMO-YOLO in this comprehensive comparison. Explore their performance in object detection, real-time AI capabilities, and suitability for edge AI applications, all designed to enhance advancements in computer vision.
keywords: YOLOv9, DAMO-YOLO, Ultralytics, object detection, real-time AI, edge AI, computer vision
---

# YOLOv9 VS DAMO-YOLO

The comparison between YOLOv9 and DAMO-YOLO highlights two powerful advancements in object detection technology. Both models have been pivotal in driving innovation, offering unique capabilities suited for various real-world applications in computer vision.

YOLOv9, developed by Ultralytics, emphasizes efficiency and precision, delivering cutting-edge performance in tasks like object detection and image segmentation. On the other hand, DAMO-YOLO, designed by Alibaba DAMO Academy, integrates advanced optimizations tailored for competitive speed and accuracy in diverse environments. Explore how these models stack up against each other.

## mAP Comparison

This section compares the mAP values of YOLOv9 and DAMO-YOLO across different model variants, illustrating their accuracy in object detection tasks. Mean Average Precision (mAP) serves as a key metric for evaluating the precision and recall of these models, providing insights into their real-world performance. Learn more about [mAP in object detection](https://www.ultralytics.com/glossary/mean-average-precision-map).

!!! tip "Accuracy"

    === "Detection (COCO)"

    	| Variant | mAP<sup>val<br>50<br>YOLOv9 | mAP<sup>val<br>50<br>DAMO-YOLO |
    	|---------------------|-------------------------------------------------------|-------------------------------------------------------|
    	| n | 37.8 | 42.0 |
    	| s | 46.5 | 46.0 |
    	| m | 51.5 | 49.2 |
    	| l | 52.8 | 50.8 |
    	| x | 55.1 | N/A |

## Speed Comparison

This section evaluates the speed performance of YOLOv9 and DAMO-YOLO across various model sizes, measured in milliseconds. These speed metrics highlight the efficiency of both models, offering valuable insights for real-time applications. For more details on YOLOv9's advancements, visit the [YOLOv9 documentation](https://docs.ultralytics.com/models/yolov9/).

!!! tip "Speed"

    === "Detection (COCO)"

    	| Variant | Speed<br><sup>T4 TensorRT10<br>(ms)</sup><br>YOLOv9 | Speed<br><sup>T4 TensorRT10<br>(ms)</sup><br>DAMO-YOLO |
    	|---------------------|-------------------------------------------------------|-------------------------------------------------------|
    	| n | 2.3 | 2.32 |
    	| s | 3.54 | 3.45 |
    	| m | 6.43 | 5.09 |
    	| l | 7.16 | 7.18 |
    	| x | 16.77 | N/A |

## Benchmark Functionality in YOLO11

Ultralytics YOLO11 provides an advanced **benchmark** functionality, allowing users to evaluate model performance across various tasks efficiently. This feature helps analyze metrics like speed, accuracy, and memory usage, ensuring models meet the requirements for real-world applications. By benchmarking, developers can fine-tune their models for optimal performance on specific hardware or datasets.

To explore more about YOLO11’s benchmarking capabilities, visit the [Ultralytics Documentation](https://docs.ultralytics.com/guides/).

### Python Code Example for Benchmarking

```python
from ultralytics import YOLO

# Load a YOLO11 model
model = YOLO("yolo11.pt")

# Run benchmarking on the COCO dataset
results = model.benchmark(data="coco.yaml", imgsz=640, device=0)

# Print results
print(results)
```

This functionality provides actionable insights to improve deployment strategies, making YOLO11 a versatile choice for diverse computer vision projects.
