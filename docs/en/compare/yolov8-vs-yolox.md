---
comments: true
description: Dive into a detailed comparison between Ultralytics YOLOv8 and YOLOX, two advanced models in the realm of object detection and computer vision. Explore their performance in terms of accuracy, speed, and real-time AI capabilities, and understand their applications from edge AI to diverse industries.
keywords: Ultralytics, YOLOv8, YOLOX, object detection, real-time AI, edge AI, computer vision, model comparison, machine learning, deep learning
---

# Ultralytics YOLOv8 VS YOLOX

Choosing the right object detection model is crucial for ensuring optimal performance in computer vision tasks. This comparison between Ultralytics YOLOv8 and YOLOX highlights the advancements in speed, accuracy, and versatility offered by these state-of-the-art frameworks.

Ultralytics YOLOv8 brings cutting-edge innovations like anchor-free detection heads and seamless compatibility with previous YOLO versions. On the other hand, YOLOX is known for its robust decoupled head design and efficiency in real-time applications. Dive deeper to explore their unique strengths and discover which model best fits your needs.

## mAP Comparison

This section highlights the mAP values of Ultralytics YOLOv8 and YOLOX across different model variants, showcasing their accuracy in detecting and classifying objects. mAP, a key metric in object detection, combines precision and recall to evaluate performance comprehensively. Learn more about [Mean Average Precision (mAP)](https://www.ultralytics.com/glossary/mean-average-precision-map) and its role in model evaluation.

!!! tip "Accuracy"

    === "Detection (COCO)"

    	| Variant | mAP<sup>val<br>50<br>YOLOv8 | mAP<sup>val<br>50<br>YOLOX |
    	|---------------------|-------------------------------------------------------|-------------------------------------------------------|
    	| n | 37.3 | N/A |
    	| s | 44.9 | 40.5 |
    	| m | 50.2 | 46.9 |
    	| l | 52.9 | 49.7 |
    	| x | 53.9 | 51.1 |

## Speed Comparison

This section highlights the speed performance of Ultralytics YOLOv8 and YOLOX models across various sizes, measured in milliseconds. These metrics provide insights into the efficiency of each model for real-time applications, helping users determine the best fit for their needs. For more details, explore the [Ultralytics YOLOv8 documentation](https://docs.ultralytics.com/models/yolov8/) or learn about [benchmarking methods](https://docs.ultralytics.com/reference/utils/benchmarks/).

!!! tip "Speed"

    === "Detection (COCO)"

    	| Variant | Speed<br><sup>T4 TensorRT10<br>(ms)</sup><br>YOLOv8 | Speed<br><sup>T4 TensorRT10<br>(ms)</sup><br>YOLOX |
    	|---------------------|-------------------------------------------------------|-------------------------------------------------------|
    	| n | 1.47 | N/A |
    	| s | 2.66 | 2.56 |
    	| m | 5.86 | 5.43 |
    	| l | 9.06 | 9.04 |
    	| x | 14.37 | 16.1 |

## Fine-Tuning Ultralytics YOLO11 on African Wildlife Dataset

Ultralytics YOLO11 allows seamless fine-tuning on various datasets, including the African Wildlife dataset, which is ideal for conservation and research projects. This dataset includes diverse images of wildlife in their natural habitats, enabling the development of robust models for tasks like species identification, population monitoring, and anti-poaching efforts.

<<<<<<< HEAD
To fine-tune Ultralytics YOLO11 on the African Wildlife dataset, users can leverage the pre-trained weights and customize the training process to their specific needs. The Ultralytics Python package simplifies this process, ensuring efficient data preparation, annotation, and training.
=======
YOLO11's segmentation capabilities are built on a robust framework, allowing for seamless integration of custom datasets like COCO and others. With support for both pre-trained and fine-tuned models, it offers unparalleled flexibility for tasks requiring detailed object outlines.

> > > > > > > 95d73b193a43ffc9b65d81b5b93f6d2acf3cb195

For a step-by-step guide on fine-tuning YOLO models, explore the [Custom Training Guide](https://www.ultralytics.com/blog/custom-training-ultralytics-yolo11-with-computer-vision-datasets). This guide covers everything from dataset preparation to hyperparameter optimization, helping you achieve high accuracy for your wildlife detection tasks.

<<<<<<< HEAD
Learn more about dataset options and integrations on the [Ultralytics Datasets Page](https://docs.ultralytics.com/datasets/). Start building impactful solutions for wildlife conservation today!
=======

### Python Code Example: YOLO11 Segmentation

```python
from ultralytics import YOLO

# Load a pre-trained YOLO11 model
model = YOLO("yolov11-seg.pt")

# Perform segmentation on an image
results = model.segment(source="path/to/image.jpg", save=True)

# Save results to an output folder
results.show()
```

Explore more about YOLO11's segmentation capabilities to redefine your project outcomes!

> > > > > > > 95d73b193a43ffc9b65d81b5b93f6d2acf3cb195
