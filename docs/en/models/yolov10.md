---
comments: true
description: Explore the YOLOv10, a real-time object detector. Understand its superior speed, impressive accuracy, and unique approach to end-to-end object detection optimization.
keywords: YOLOv10, real-time object detector, state-of-the-art, Tsinghua University, COCO dataset, NMS-free training, holistic model design, efficient architecture
---

# YOLOv10: Real-Time End-to-End Object Detection

YOLOv10, built on the [Ultralytics](https://ultralytics.com) [Python package](https://pypi.org/project/ultralytics/) by researchers at [Tsinghua University](https://www.tsinghua.edu.cn/en/), introduces a new approach to real-time object detection, addressing both the post-processing and model architecture deficiencies found in previous YOLO versions. By eliminating non-maximum suppression (NMS) and optimizing various model components, YOLOv10 achieves state-of-the-art performance with significantly reduced computational overhead. Extensive experiments demonstrate its superior accuracy-latency trade-offs across multiple model scales.

![YOLOv10 consistent dual assignment for NMS-free training](https://github.com/ultralytics/ultralytics/assets/26833433/f9b1bec0-928e-41ce-a205-e12db3c4929a)

## Overview

Real-time object detection aims to accurately predict object categories and positions in images with low latency. The YOLO series has been at the forefront of this research due to its balance between performance and efficiency. However, reliance on NMS and architectural inefficiencies have hindered optimal performance. YOLOv10 addresses these issues by introducing consistent dual assignments for NMS-free training and a holistic efficiency-accuracy driven model design strategy.

### Architecture

The architecture of YOLOv10 builds upon the strengths of previous YOLO models while introducing several key innovations. The model architecture consists of the following components:

1. **Backbone**: Responsible for feature extraction, the backbone in YOLOv10 uses an enhanced version of CSPNet (Cross Stage Partial Network) to improve gradient flow and reduce computational redundancy.
2. **Neck**: The neck is designed to aggregate features from different scales and passes them to the head. It includes PAN (Path Aggregation Network) layers for effective multiscale feature fusion.
3. **One-to-Many Head**: Generates multiple predictions per object during training to provide rich supervisory signals and improve learning accuracy.
4. **One-to-One Head**: Generates a single best prediction per object during inference to eliminate the need for NMS, thereby reducing latency and improving efficiency.

## Key Features

1. **NMS-Free Training**: Utilizes consistent dual assignments to eliminate the need for NMS, reducing inference latency.
2. **Holistic Model Design**: Comprehensive optimization of various components from both efficiency and accuracy perspectives, including lightweight classification heads, spatial-channel decoupled down sampling, and rank-guided block design.
3. **Enhanced Model Capabilities**: Incorporates large-kernel convolutions and partial self-attention modules to improve performance without significant computational cost.

## Model Variants

YOLOv10 comes in various model scales to cater to different application needs:

- **YOLOv10-N**: Nano version for extremely resource-constrained environments.
- **YOLOv10-S**: Small version balancing speed and accuracy.
- **YOLOv10-M**: Medium version for general-purpose use.
- **YOLOv10-B**: Balanced version with increased width for higher accuracy.
- **YOLOv10-L**: Large version for higher accuracy at the cost of increased computational resources.
- **YOLOv10-X**: Extra-large version for maximum accuracy and performance.

## Performance

YOLOv10 outperforms previous YOLO versions and other state-of-the-art models in terms of accuracy and efficiency. For example, YOLOv10-S is 1.8x faster than RT-DETR-R18 with similar AP on the COCO dataset, and YOLOv10-B has 46% less latency and 25% fewer parameters than YOLOv9-C with the same performance.

| Model     | Input Size | AP<sup>val</sup> | FLOPs (G) | Latency (ms) |
|-----------|------------|------------------|-----------|--------------|
| YOLOv10-N | 640        | 38.5             | **6.7**   | **1.84**     | 
| YOLOv10-S | 640        | 46.3             | 21.6      | 2.49         |
| YOLOv10-M | 640        | 51.1             | 59.1      | 4.74         |
| YOLOv10-B | 640        | 52.5             | 92.0      | 5.74         |  
| YOLOv10-L | 640        | 53.2             | 120.3     | 7.28         |
| YOLOv10-X | 640        | **54.4**         | 160.4     | 10.70        |

Latency measured with TensorRT FP16 on T4 GPU.

## Methodology

### Consistent Dual Assignments for NMS-Free Training

YOLOv10 employs dual label assignments, combining one-to-many and one-to-one strategies during training to ensure rich supervision and efficient end-to-end deployment. The consistent matching metric aligns the supervision between both strategies, enhancing the quality of predictions during inference.

### Holistic Efficiency-Accuracy Driven Model Design

#### Efficiency Enhancements

1. **Lightweight Classification Head**: Reduces the computational overhead of the classification head by using depth-wise separable convolutions.
2. **Spatial-Channel Decoupled Down sampling**: Decouples spatial reduction and channel modulation to minimize information loss and computational cost.
3. **Rank-Guided Block Design**: Adapts block design based on intrinsic stage redundancy, ensuring optimal parameter utilization.

#### Accuracy Enhancements

1. **Large-Kernel Convolution**: Enlarges the receptive field to enhance feature extraction capability.
2. **Partial Self-Attention (PSA)**: Incorporates self-attention modules to improve global representation learning with minimal overhead.

## Experiments and Results

YOLOv10 has been extensively tested on standard benchmarks like COCO, demonstrating superior performance and efficiency. The model achieves state-of-the-art results across different variants, showcasing significant improvements in latency and accuracy compared to previous versions and other contemporary detectors.

## Comparisons

![YOLOv10 comparison with SOTA object detectors](https://github.com/ultralytics/ultralytics/assets/26833433/e0360eb4-3589-4cd1-b362-a8970bceada6)

Compared to other state-of-the-art detectors:

- YOLOv10-S / X are 1.8× / 1.3× faster than RT-DETR-R18 / R101 with similar accuracy
- YOLOv10-B has 25% fewer parameters and 46% lower latency than YOLOv9-C at same accuracy
- YOLOv10-L / X outperform YOLOv8-L / X by 0.3 AP / 0.5 AP with 1.8× / 2.3× fewer parameters

Here is a detailed comparison of YOLOv10 variants with other state-of-the-art models:

| Model         | Params (M) | FLOPs (G) | APval (%) | Latency (ms) | Latency (Forward) (ms) |
|---------------|------------|-----------|-----------|--------------|------------------------|
| YOLOv6-3.0-N  | 4.7        | 11.4      | 37.0      | 2.69         | **1.76**               |
| Gold-YOLO-N   | 5.6        | 12.1      | **39.6**  | 2.92         | 1.82                   |
| YOLOv8-N      | 3.2        | 8.7       | 37.3      | 6.16         | 1.77                   |
| **YOLOv10-N** | **2.3**    | **6.7**   | 39.5      | **1.84**     | 1.79                   |
|               |            |           |           |              |                        |
| YOLOv6-3.0-S  | 18.5       | 45.3      | 44.3      | 3.42         | 2.35                   |
| Gold-YOLO-S   | 21.5       | 46.0      | 45.4      | 3.82         | 2.73                   |
| YOLOv8-S      | 11.2       | 28.6      | 44.9      | 7.07         | **2.33**               |
| **YOLOv10-S** | **7.2**    | **21.6**  | **46.8**  | **2.49**     | 2.39                   |
|               |            |           |           |              |                        |
| RT-DETR-R18   | 20.0       | 60.0      | 46.5      | **4.58**     | **4.49**               |
| YOLOv6-3.0-M  | 34.9       | 85.8      | 49.1      | 5.63         | 4.56                   |
| Gold-YOLO-M   | 41.3       | 87.5      | 49.8      | 6.38         | 5.45                   |
| YOLOv8-M      | 25.9       | 78.9      | 50.6      | 9.50         | 5.09                   |
| **YOLOv10-M** | **15.4**   | **59.1**  | **51.3**  | 4.74         | 4.63                   |
|               |            |           |           |              |                        |
| YOLOv6-3.0-L  | 59.6       | 150.7     | 51.8      | 9.02         | 7.90                   |
| Gold-YOLO-L   | 75.1       | 151.7     | 51.8      | 10.65        | 9.78                   |
| YOLOv8-L      | 43.7       | 165.2     | 52.9      | 12.39        | 8.06                   |
| RT-DETR-R50   | 42.0       | 136.0     | 53.1      | 9.20         | 9.07                   |
| **YOLOv10-L** | **24.4**   | **120.3** | **53.4**  | **7.28**     | **7.21**               |
|               |            |           |           |              |
| YOLOv8-X      | 68.2       | 257.8     | 53.9      | 16.86        | 12.83                  |
| RT-DETR-R101  | 76.0       | 259.0     | 54.3      | 13.71        | 13.58                  |
| **YOLOv10-X** | **29.5**   | **160.4** | **54.4**  | **10.70**    | **10.60**              |

## Usage Examples

!!! tip "Coming Soon"

    The Ultralytics team is actively working on officially integrating the YOLOv10 models into the `ultralytics` package. Once the integration is complete, the usage examples shown below will be fully functional. Please stay tuned by following our social media and [GitHub repository](https://github.com/ultralytics/ultralytics) for the latest updates on YOLOv10 integration. We appreciate your patience and excitement! 🚀

For predicting new images with YOLOv10:

```python
from ultralytics import YOLO

# Load a pretrained YOLOv10n model
model = YOLO("yolov10n.pt")

# Perform object detection on an image
results = model("image.jpg")

# Display the results
results[0].show()
```

For training YOLOv10 on a custom dataset:

```python
from ultralytics import YOLO

# Load YOLOv10n model from scratch
model = YOLO("yolov10n.yaml")

# Train the model
model.train(data="coco8.yaml", epochs=100, imgsz=640)
```

## Conclusion

YOLOv10 sets a new standard in real-time object detection by addressing the shortcomings of previous YOLO versions and incorporating innovative design strategies. Its ability to deliver high accuracy with low computational cost makes it an ideal choice for a wide range of real-world applications.

## Citations and Acknowledgements

We would like to acknowledge the YOLOv10 authors from [Tsinghua University](https://www.tsinghua.edu.cn/en/) for their extensive research and significant contributions to the [Ultralytics](https://ultralytics.com) framework:

!!! Quote ""

    === "BibTeX"

        ```bibtex
        @article{THU-MIGyolov10,
          title={YOLOv10: Real-Time End-to-End Object Detection},
          author={Ao Wang, Hui Chen, Lihao Liu, et al.},
          journal={arXiv preprint arXiv:2405.14458},
          year={2024},
          institution={Tsinghua University},
          license = {AGPL-3.0}
        }
        ```

For detailed implementation, architectural innovations, and experimental results, please refer to the YOLOv10 [research paper](https://arxiv.org/pdf/2405.14458) and [GitHub repository](https://github.com/THU-MIG/yolov10) by the Tsinghua University team.
