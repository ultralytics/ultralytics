---
comments: true
description: Discover YOLOv9, the latest addition to the real-time object detection arsenal, leveraging Programmable Gradient Information and GELAN architecture for unparalleled performance.
keywords: YOLOv9, real-time object detection, Programmable Gradient Information, GELAN architecture, Ultralytics, MS COCO dataset, open-source, lightweight model, computer vision, AI
---

# YOLOv9: A Leap Forward in Object Detection Technology

YOLOv9 marks a significant advancement in real-time object detection, introducing groundbreaking techniques such as Programmable Gradient Information (PGI) and the Generalized Efficient Layer Aggregation Network (GELAN). This model demonstrates remarkable improvements in efficiency, accuracy, and adaptability, setting new benchmarks on the MS COCO dataset. The YOLOv9 project, while developed by a separate open-source team, builds upon the robust codebase provided by [Ultralytics](https://ultralytics.com) [YOLOv5](yolov5.md), showcasing the collaborative spirit of the AI research community.

![YOLOv9 performance comparison](https://github.com/ultralytics/ultralytics/assets/26833433/9f41ef7b-6008-43eb-8ba1-0a9b89600100)

## Introduction to YOLOv9

In the quest for optimal real-time object detection, YOLOv9 stands out with its innovative approach to overcoming information loss challenges inherent in deep neural networks. By integrating PGI and the versatile GELAN architecture, YOLOv9 not only enhances the model's learning capacity but also ensures the retention of crucial information throughout the detection process, thereby achieving exceptional accuracy and performance.

## Core Innovations of YOLOv9

YOLOv9's advancements are deeply rooted in addressing the challenges posed by information loss in deep neural networks. The Information Bottleneck Principle and the innovative use of Reversible Functions are central to its design, ensuring YOLOv9 maintains high efficiency and accuracy.

### Information Bottleneck Principle

The Information Bottleneck Principle reveals a fundamental challenge in deep learning: as data passes through successive layers of a network, the potential for information loss increases. This phenomenon is mathematically represented as:

```python
I(X, X) >= I(X, f_theta(X)) >= I(X, g_phi(f_theta(X)))
```

where `I` denotes mutual information, and `f` and `g` represent transformation functions with parameters `theta` and `phi`, respectively. YOLOv9 counters this challenge by implementing Programmable Gradient Information (PGI), which aids in preserving essential data across the network's depth, ensuring more reliable gradient generation and, consequently, better model convergence and performance.

### Reversible Functions

The concept of Reversible Functions is another cornerstone of YOLOv9's design. A function is deemed reversible if it can be inverted without any loss of information, as expressed by:

```python
X = v_zeta(r_psi(X))
```

with `psi` and `zeta` as parameters for the reversible and its inverse function, respectively. This property is crucial for deep learning architectures, as it allows the network to retain a complete information flow, thereby enabling more accurate updates to the model's parameters. YOLOv9 incorporates reversible functions within its architecture to mitigate the risk of information degradation, especially in deeper layers, ensuring the preservation of critical data for object detection tasks.

### Impact on Lightweight Models

Addressing information loss is particularly vital for lightweight models, which are often under-parameterized and prone to losing significant information during the feedforward process. YOLOv9's architecture, through the use of PGI and reversible functions, ensures that even with a streamlined model, the essential information required for accurate object detection is retained and effectively utilized.

### Programmable Gradient Information (PGI)

PGI is a novel concept introduced in YOLOv9 to combat the information bottleneck problem, ensuring the preservation of essential data across deep network layers. This allows for the generation of reliable gradients, facilitating accurate model updates and improving the overall detection performance.

### Generalized Efficient Layer Aggregation Network (GELAN)

GELAN represents a strategic architectural advancement, enabling YOLOv9 to achieve superior parameter utilization and computational efficiency. Its design allows for flexible integration of various computational blocks, making YOLOv9 adaptable to a wide range of applications without sacrificing speed or accuracy.

![YOLOv9 architecture comparison](https://github.com/ultralytics/ultralytics/assets/26833433/286a3971-677b-45e6-a90b-4b6bd565a7af)

## Performance on MS COCO Dataset

The performance of YOLOv9 on the [COCO dataset](../datasets/detect/coco.md) exemplifies its significant advancements in real-time object detection, setting new benchmarks across various model sizes. Table 1 presents a comprehensive comparison of state-of-the-art real-time object detectors, illustrating YOLOv9's superior efficiency and accuracy.

**Table 1. Comparison of State-of-the-Art Real-Time Object Detectors**

| Model    | Parameters (M) | FLOPs (G) | APval 50:95 (%) | APval 50 (%) | APval 75 (%) | APval S (%) | APval M (%) | APval L (%) |
|----------|----------------|-----------|-----------------|--------------|--------------|-------------|-------------|-------------|
| YOLOv9-S | 7.2            | 26.7      | 46.8            | 63.4         | 50.7         | 26.6        | 56.0        | 64.5        |
| YOLOv9-M | 20.1           | 76.8      | 51.4            | 68.1         | 56.1         | 33.6        | 57.0        | 68.0        |
| YOLOv9-C | 25.5           | 102.8     | 53.0            | 70.2         | 57.8         | 36.2        | 58.5        | 69.3        |
| YOLOv9-E | 58.1           | 192.5     | 55.6            | 72.8         | 60.6         | 40.2        | 61.0        | 71.4        |

YOLOv9's iterations, ranging from the smaller S variant to the extensive E model, demonstrate improvements not only in accuracy (AP metrics) but also in efficiency with a reduced number of parameters and computational needs (FLOPs). This table underscores YOLOv9's ability to deliver high precision while maintaining or reducing the computational overhead compared to prior versions and competing models.

Comparatively, YOLOv9 exhibits remarkable gains:

- **Lightweight Models**: YOLOv9-S surpasses the YOLO MS-S in parameter efficiency and computational load while achieving an improvement of 0.4∼0.6% in AP.
- **Medium to Large Models**: YOLOv9-M and YOLOv9-E show notable advancements in balancing the trade-off between model complexity and detection performance, offering significant reductions in parameters and computations against the backdrop of improved accuracy.

The YOLOv9-C model, in particular, highlights the effectiveness of the architecture's optimizations. It operates with 42% fewer parameters and 21% less computational demand than YOLOv7 AF, yet it achieves comparable accuracy, demonstrating YOLOv9's significant efficiency improvements. Furthermore, the YOLOv9-E model sets a new standard for large models, with 15% fewer parameters and 25% less computational need than YOLOv8-X, alongside a substantial 1.7% improvement in AP.

These results showcase YOLOv9's strategic advancements in model design, emphasizing its enhanced efficiency without compromising on the precision essential for real-time object detection tasks. The model not only pushes the boundaries of performance metrics but also emphasizes the importance of computational efficiency, making it a pivotal development in the field of computer vision.

## Integration and Future Directions

YOLOv9 embodies the spirit of open-source collaboration that is central to the advancement of AI technology. With plans for future integration into the Ultralytics package, YOLOv9 is poised to become an accessible tool for researchers and practitioners alike, further enhancing its impact on the field of computer vision.

## Conclusion

YOLOv9 represents a pivotal development in real-time object detection, offering significant improvements in terms of efficiency, accuracy, and adaptability. By addressing critical challenges through innovative solutions like PGI and GELAN, YOLOv9 sets a new precedent for future research and application in the field. As the AI community continues to evolve, YOLOv9 stands as a testament to the power of collaboration and innovation in driving technological progress.

Stay tuned for updates on Ultralytics package integration and explore the possibilities that YOLOv9 brings to the realm of computer vision.

## Citations and Acknowledgements

We would like to acknowledge the YOLOv9 authors for their significant contributions in the field of real-time object detection:

!!! Quote ""

    === "BibTeX"

        ```bibtex
        @article{wang2024yolov9,
          title={{YOLOv9}: Learning What You Want to Learn Using Programmable Gradient Information},
          author={Wang, Chien-Yao  and Liao, Hong-Yuan Mark},
          booktitle={arXiv preprint arXiv:2402.13616},
          year={2024}
        }
        ```

The original YOLOv9 paper can be found on [arXiv](https://arxiv.org/pdf/2402.13616.pdf). The authors have made their work publicly available, and the codebase can be accessed on [GitHub](https://github.com/WongKinYiu/yolov9). We appreciate their efforts in advancing the field and making their work accessible to the broader community.
