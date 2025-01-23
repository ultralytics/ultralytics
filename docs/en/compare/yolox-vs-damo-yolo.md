---
comments: true
description: Explore a detailed comparison between YOLOX and DAMO-YOLO, highlighting their performance, accuracy, and efficiency in object detection. Learn how these models stack up for real-time AI, edge AI applications, and computer vision tasks.
keywords: YOLOX, DAMO-YOLO, Ultralytics, object detection, real-time AI, edge AI, computer vision, model comparison
---

# YOLOX VS DAMO-YOLO

The evolution of object detection has given rise to powerful models like YOLOX and DAMO-YOLO, both designed to excel in real-time applications. This comparison explores their unique capabilities, helping you choose the best fit for your specific computer vision needs.

YOLOX is celebrated for its anchor-free design and balanced accuracy-speed tradeoff, while DAMO-YOLO focuses on advanced feature extraction and resource efficiency. By analyzing their performance across tasks and datasets, we highlight the strengths and trade-offs of these cutting-edge models. Explore more about [object detection](https://www.ultralytics.com/glossary/object-detection) and how these models redefine AI innovation.

## mAP Comparison

This section highlights the mAP values of YOLOX and DAMO-YOLO, showcasing their accuracy across different variants. Mean Average Precision (mAP) is a critical metric for evaluating object detection models, measuring how well they balance precision and recall across multiple classes. Learn more about [mAP and its calculation process](https://www.ultralytics.com/glossary/mean-average-precision-map).

!!! tip "Accuracy"

    === "Detection (COCO)"

    	| Variant | mAP<sup>val<br>50<br>YOLOX | mAP<sup>val<br>50<br>DAMO-YOLO |
    	|---------------------|-------------------------------------------------------|-------------------------------------------------------|
    	| n | N/A | 42.0 |
    	| s | 40.5 | 46.0 |
    	| m | 46.9 | 49.2 |
    	| l | 49.7 | 50.8 |
    	| x | 51.1 | N/A |


## Speed Comparison

This section highlights the speed performance of YOLOX and DAMO-YOLO models across various sizes, measured in milliseconds. The comparison provides insights into their efficiency, especially for applications requiring real-time object detection. For further details, explore [Ultralytics YOLO documentation](https://docs.ultralytics.com/models/yolov7/) and [DAMO-YOLO resources](https://github.com/tinyvision/damo-yolo).

!!! tip "Speed"

    === "Detection (COCO)"

    	| Variant | Speed<br><sup>T4 TensorRT10<br>(ms)</sup><br>YOLOX | Speed<br><sup>T4 TensorRT10<br>(ms)</sup><br>DAMO-YOLO |
    	|---------------------|-------------------------------------------------------|-------------------------------------------------------|
    	| n | N/A | 2.32 |
    	| s | 2.56 | 3.45 |
    	| m | 5.43 | 5.09 |
    	| l | 9.04 | 7.18 |
    	| x | 16.1 | N/A |

## Train Functionality

The **Train** functionality in Ultralytics YOLO11 allows users to fine-tune models on custom datasets to achieve optimal performance for specific use cases. Whether working with datasets like COCO8 or African wildlife, YOLO11 ensures efficient training and improved accuracy through advanced optimization techniques. This feature is ideal for users looking to adapt YOLO11 to unique tasks such as object detection, classification, or segmentation.

For a thorough guide on training YOLO models, refer to the [YOLO Training Guide](https://docs.ultralytics.com/modes/train/).

### Python Code Example

```python
from ultralytics import YOLO

# Load a YOLO11 model
model = YOLO('yolo11n.pt')

# Train the model on a custom dataset
model.train(data='path/to/dataset.yaml', epochs=50, imgsz=640, batch=16)
```

This code initializes a YOLO11 model, sets up a dataset, and begins the training process with configurable parameters like epochs and image size.
