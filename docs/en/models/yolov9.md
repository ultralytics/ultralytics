---
comments: true
description: Explore YOLOv9, the latest leap in real-time object detection, featuring innovations like PGI and GELAN, and achieving new benchmarks in efficiency and accuracy.
keywords: YOLOv9, object detection, real-time, PGI, GELAN, deep learning, MS COCO, AI, neural networks, model efficiency, accuracy, Ultralytics
---

# YOLOv9: A Leap Forward in Object Detection Technology

YOLOv9 marks a significant advancement in real-time object detection, introducing groundbreaking techniques such as Programmable Gradient Information (PGI) and the Generalized Efficient Layer Aggregation Network (GELAN). This model demonstrates remarkable improvements in efficiency, accuracy, and adaptability, setting new benchmarks on the MS COCO dataset. The YOLOv9 project, while developed by a separate open-source team, builds upon the robust codebase provided by [Ultralytics](https://ultralytics.com) [YOLOv5](yolov5.md), showcasing the collaborative spirit of the AI research community.

<p align="center">
  <br>
  <iframe loading="lazy" width="720" height="405" src="https://www.youtube.com/embed/ZF7EAodHn1U"
    title="YouTube video player" frameborder="0"
    allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share"
    allowfullscreen>
  </iframe>
  <br>
  <strong>Watch:</strong> YOLOv9 Training on Custom Data using Ultralytics | Industrial Package Dataset
</p>

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

!!! tip "Performance"

    === "Detection (COCO)"

        | Model                                                                                 | size<br><sup>(pixels) | mAP<sup>val<br>50-95 | mAP<sup>val<br>50 | params<br><sup>(M) | FLOPs<br><sup>(B) |
        |---------------------------------------------------------------------------------------|-----------------------|----------------------|-------------------|--------------------|-------------------|
        | [YOLOv9t](https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov9t.pt)  | 640                   | 38.3                 | 53.1              | 2.0                | 7.7               |
        | [YOLOv9s](https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov9s.pt)  | 640                   | 46.8                 | 63.4              | 7.2                | 26.7              |
        | [YOLOv9m](https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov9m.pt)  | 640                   | 51.4                 | 68.1              | 20.1               | 76.8              |
        | [YOLOv9c](https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov9c.pt)  | 640                   | 53.0                 | 70.2              | 25.5               | 102.8             |
        | [YOLOv9e](https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov9e.pt)  | 640                   | 55.6                 | 72.8              | 58.1               | 192.5             |

    === "Segmentation (COCO)"

        | Model                                                                                         | size<br><sup>(pixels) | mAP<sup>box<br>50-95 | mAP<sup>mask<br>50-95 | params<br><sup>(M) | FLOPs<br><sup>(B) |
        |-----------------------------------------------------------------------------------------------|-----------------------|----------------------|-----------------------|--------------------|-------------------|
        | [YOLOv9c-seg](https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov9c-seg.pt)  | 640                   | 52.4                 | 42.2                  | 27.9               | 159.4             |
        | [YOLOv9e-seg](https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov9e-seg.pt)  | 640                   | 55.1                 | 44.3                  | 60.5               | 248.4             |

YOLOv9's iterations, ranging from the tiny `t` variant to the extensive `e` model, demonstrate improvements not only in accuracy (mAP metrics) but also in efficiency with a reduced number of parameters and computational needs (FLOPs). This table underscores YOLOv9's ability to deliver high precision while maintaining or reducing the computational overhead compared to prior versions and competing models.

Comparatively, YOLOv9 exhibits remarkable gains:

- **Lightweight Models**: YOLOv9s surpasses the YOLO MS-S in parameter efficiency and computational load while achieving an improvement of 0.4∼0.6% in AP.
- **Medium to Large Models**: YOLOv9m and YOLOv9e show notable advancements in balancing the trade-off between model complexity and detection performance, offering significant reductions in parameters and computations against the backdrop of improved accuracy.

The YOLOv9c model, in particular, highlights the effectiveness of the architecture's optimizations. It operates with 42% fewer parameters and 21% less computational demand than YOLOv7 AF, yet it achieves comparable accuracy, demonstrating YOLOv9's significant efficiency improvements. Furthermore, the YOLOv9e model sets a new standard for large models, with 15% fewer parameters and 25% less computational need than [YOLOv8x](yolov8.md), alongside a incremental 1.7% improvement in AP.

These results showcase YOLOv9's strategic advancements in model design, emphasizing its enhanced efficiency without compromising on the precision essential for real-time object detection tasks. The model not only pushes the boundaries of performance metrics but also emphasizes the importance of computational efficiency, making it a pivotal development in the field of computer vision.

## Conclusion

YOLOv9 represents a pivotal development in real-time object detection, offering significant improvements in terms of efficiency, accuracy, and adaptability. By addressing critical challenges through innovative solutions like PGI and GELAN, YOLOv9 sets a new precedent for future research and application in the field. As the AI community continues to evolve, YOLOv9 stands as a testament to the power of collaboration and innovation in driving technological progress.

## Usage Examples

This example provides simple YOLOv9 training and inference examples. For full documentation on these and other [modes](../modes/index.md) see the [Predict](../modes/predict.md), [Train](../modes/train.md), [Val](../modes/val.md) and [Export](../modes/export.md) docs pages.

!!! Example

    === "Python"

        PyTorch pretrained `*.pt` models as well as configuration `*.yaml` files can be passed to the `YOLO()` class to create a model instance in python:

        ```python
        from ultralytics import YOLO

        # Build a YOLOv9c model from scratch
        model = YOLO("yolov9c.yaml")

        # Build a YOLOv9c model from pretrained weight
        model = YOLO("yolov9c.pt")

        # Display model information (optional)
        model.info()

        # Train the model on the COCO8 example dataset for 100 epochs
        results = model.train(data="coco8.yaml", epochs=100, imgsz=640)

        # Run inference with the YOLOv9c model on the 'bus.jpg' image
        results = model("path/to/bus.jpg")
        ```

    === "CLI"

        CLI commands are available to directly run the models:

        ```bash
        # Build a YOLOv9c model from scratch and train it on the COCO8 example dataset for 100 epochs
        yolo train model=yolov9c.yaml data=coco8.yaml epochs=100 imgsz=640

        # Build a YOLOv9c model from scratch and run inference on the 'bus.jpg' image
        yolo predict model=yolov9c.yaml source=path/to/bus.jpg
        ```

## Supported Tasks and Modes

The YOLOv9 series offers a range of models, each optimized for high-performance [Object Detection](../tasks/detect.md). These models cater to varying computational needs and accuracy requirements, making them versatile for a wide array of applications.

| Model      | Filenames                                               | Tasks                                        | Inference | Validation | Training | Export |
| ---------- | ------------------------------------------------------- | -------------------------------------------- | --------- | ---------- | -------- | ------ |
| YOLOv9     | `yolov9t` `yolov9s` `yolov9m` `yolov9c.pt` `yolov9e.pt` | [Object Detection](../tasks/detect.md)       | ✅        | ✅         | ✅       | ✅     |
| YOLOv9-seg | `yolov9c-seg.pt` `yolov9e-seg.pt`                       | [Instance Segmentation](../tasks/segment.md) | ✅        | ✅         | ✅       | ✅     |

This table provides a detailed overview of the YOLOv9 model variants, highlighting their capabilities in object detection tasks and their compatibility with various operational modes such as [Inference](../modes/predict.md), [Validation](../modes/val.md), [Training](../modes/train.md), and [Export](../modes/export.md). This comprehensive support ensures that users can fully leverage the capabilities of YOLOv9 models in a broad range of object detection scenarios.

!!! note

    Training YOLOv9 models will require _more_ resources **and** take longer than the equivalent sized [YOLOv8 model](yolov8.md).

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

## FAQ

### What is YOLOv9 and how does it improve object detection?

YOLOv9 is the latest advancement in real-time object detection, introducing cutting-edge techniques like Programmable Gradient Information (PGI) and the Generalized Efficient Layer Aggregation Network (GELAN). These innovations help YOLOv9 achieve remarkable improvements in efficiency, accuracy, and adaptability, setting new benchmarks on the MS COCO dataset. For more details, see our [Introduction to YOLOv9](#introduction-to-yolov9).

### How does Programmable Gradient Information (PGI) enhance YOLOv9 performance?

PGI is a novel concept in YOLOv9 designed to mitigate the information bottleneck inherent in deep neural networks. It ensures the preservation of essential data across deep network layers, leading to more reliable gradient generation and improved model convergence and performance. This innovation is crucial for retaining critical information throughout the detection process. Learn more about PGI in our [Core Innovations of YOLOv9](#programmable-gradient-information-pgi) section.

### What makes the Generalized Efficient Layer Aggregation Network (GELAN) unique in YOLOv9?

GELAN allows YOLOv9 to achieve superior parameter utilization and computational efficiency. Its flexible design enables the integration of various computational blocks, making YOLOv9 highly adaptable to different applications without compromising speed or accuracy. This architectural advancement is crucial for the model's efficiency. Explore more in the [Generalized Efficient Layer Aggregation Network (GELAN)](#generalized-efficient-layer-aggregation-network-gelan) section.

### How does YOLOv9 compare to previous versions and other state-of-the-art models?

YOLOv9 demonstrates significant improvements in both accuracy and efficiency compared to previous YOLO versions and other state-of-the-art models. For instance, the YOLOv9c model operates with 42% fewer parameters and 21% less computational demand than YOLOv7 AF, yet it achieves comparable accuracy. Check the [Performance on MS COCO Dataset](#performance-on-ms-coco-dataset) section for a detailed comparison.

### Can I train my own models using YOLOv9, and how?

Yes, you can train your own models using YOLOv9 with ease. You can use Ultralytics YOLO's simple Python API or CLI commands for model training and inference. Below is a Python example to get you started:

```python
from ultralytics import YOLO

# Build a YOLOv9c model from pretrained weights
model = YOLO("yolov9c.pt")

# Train the model on the COCO8 example dataset for 100 epochs
results = model.train(data="coco8.yaml", epochs=100, imgsz=640)

# Run inference with the YOLOv9c model on an image
results = model("path/to/bus.jpg")
```

For more details on how to train and use YOLOv9 models, visit the [Usage Examples](#usage-examples) section.
