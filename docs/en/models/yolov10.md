---
comments: true
description: Explore the YOLOv10, a real-time object detector. Understand its superior speed, impressive accuracy, and unique approach to end-to-end object detection optimization.
keywords: YOLOv10, real-time object detector, state-of-the-art, Tsinghua University, COCO dataset, NMS-free training, holistic model design, efficient architecture
---

# YOLOv10: Real-Time End-to-End Object Detection

!!! tip "Coming Soon"

    YOLOv10 will be integrated into the Ultralytics library as quickly as possible. Make sure to follow us on all social media channels and follow our [GitHub repository](https://github.com/ultralytics/ultralytics) for updates 🚀

YOLOv10 introduces a new approach to real-time object detection, addressing both the post-processing and model architecture deficiencies found in previous YOLO versions. By eliminating non-maximum suppression (NMS) and optimizing various model components, YOLOv10 achieves state-of-the-art performance with significantly reduced computational overhead. Extensive experiments demonstrate its superior accuracy-latency trade-offs across multiple model scales.

![YOLOv10 comparison with SOTA object detectors](https://github.com/RizwanMunawar/RizwanMunawar/assets/62513924/f68032d5-8311-4ef6-ac11-fefdd7db72c2)

## Overview

Real-time object detection aims to accurately predict object categories and positions in images with low latency. The YOLO series has been at the forefront of this research due to its balance between performance and efficiency. However, reliance on NMS and architectural inefficiencies have hindered optimal performance. YOLOv10 addresses these issues by introducing consistent dual assignments for NMS-free training and a holistic efficiency-accuracy driven model design strategy.

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

## Conclusion

YOLOv10 sets a new standard in real-time object detection by addressing the shortcomings of previous YOLO versions and incorporating innovative design strategies. Its ability to deliver high accuracy with low computational cost makes it an ideal choice for a wide range of real-world applications.

## Citations and Acknowledgements

We would like to acknowledge the YOLOv10 authors for their extensive research to build upon the Ultralytics framework:

!!! Quote ""

    === "BibTeX"

        ```bibtex
        @article{THU-MIGyolov10,
          title={YOLOv10: Real-Time End-to-End Object Detection},
          author={Ao Wang, Hui Chen, Lihao Liu, et al.},
          journal={arXiv preprint arXiv:2405.14458},
          year={2024},
          license = {AGPL-3.0}
        }
        ```

For detailed implementation and experimental results, please refer to the YOLOv10 [research paper](https://arxiv.org/pdf/2405.14458).
