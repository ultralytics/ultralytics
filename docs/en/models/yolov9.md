---
title: YOLOv9 Object Detection with PGI and GELAN
comments: true
description: "YOLOv9 real-time object detector with Programmable Gradient Information (PGI) and GELAN: architecture, COCO detection and segmentation results, and usage."
keywords: YOLOv9, object detection, real-time, PGI, GELAN, deep learning, MS COCO, AI, neural networks, model efficiency, accuracy, Ultralytics
---

# YOLOv9: A Leap Forward in Object Detection Technology

YOLOv9 advanced real-time [object detection](https://www.ultralytics.com/glossary/object-detection) by introducing Programmable Gradient Information (PGI) and the Generalized Efficient Layer Aggregation Network (GELAN). At release, it improved efficiency and accuracy over previous real-time detectors on the MS COCO dataset. The YOLOv9 project was developed by a separate open-source team and builds upon the [Ultralytics](https://www.ultralytics.com) [YOLOv5](yolov5.md) codebase.

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

![YOLOv9 performance comparison](https://cdn.ul.run/i/8aed32329a039a65a2e5134cb536cbd4.avif)

## Introduction to YOLOv9

YOLOv9 addresses the information loss inherent in deep [neural networks](https://www.ultralytics.com/glossary/neural-network-nn). By integrating PGI and the GELAN architecture, YOLOv9 increases the model's learning capacity and retains crucial information throughout the detection process, improving both accuracy and efficiency.

## Core Innovations of YOLOv9

YOLOv9's design targets information loss in deep neural networks. The Information Bottleneck Principle and the use of Reversible Functions are central to its design, helping YOLOv9 maintain high efficiency and accuracy.

### Information Bottleneck Principle

The Information Bottleneck Principle reveals a fundamental challenge in deep learning: as data passes through successive layers of a network, the potential for information loss increases. This phenomenon is mathematically represented as:

```text
I(X, X) >= I(X, f_theta(X)) >= I(X, g_phi(f_theta(X)))
```

where `I` denotes mutual information, and `f` and `g` represent transformation functions with parameters `theta` and `phi`, respectively. YOLOv9 counters this challenge by implementing Programmable Gradient Information (PGI), which aids in preserving essential data across the network's depth, ensuring more reliable gradient generation and, consequently, better model convergence and performance.

### Reversible Functions

The concept of Reversible Functions is another cornerstone of YOLOv9's design. A function is deemed reversible if it can be inverted without any loss of information, as expressed by:

```text
X = v_zeta(r_psi(X))
```

with `psi` and `zeta` as parameters for the reversible function and its inverse, respectively. This property is crucial for [deep learning](https://www.ultralytics.com/glossary/deep-learning-dl) architectures, as it allows the network to retain a complete information flow, thereby enabling more accurate updates to the model's parameters. YOLOv9 incorporates reversible functions within its architecture to mitigate the risk of information degradation, especially in deeper layers, ensuring the preservation of critical data for object detection tasks.

### Impact on Lightweight Models

Addressing information loss is particularly vital for lightweight models, which are often under-parameterized and prone to losing significant information during the feedforward process. YOLOv9's architecture, through the use of PGI and reversible functions, ensures that even with a streamlined model, the essential information required for accurate object detection is retained and effectively utilized.

### Programmable Gradient Information (PGI)

PGI is a novel concept introduced in YOLOv9 to combat the information bottleneck problem, ensuring the preservation of essential data across deep network layers. This allows for the generation of reliable gradients, facilitating accurate model updates and improving the overall detection performance.

### Generalized Efficient Layer Aggregation Network (GELAN)

GELAN is an architecture that improves YOLOv9's parameter utilization and computational efficiency. Its design allows for flexible integration of various computational blocks, making YOLOv9 adaptable to a wide range of applications without sacrificing speed or accuracy.

![YOLOv9 architecture comparison](https://cdn.ul.run/i/6483991ea76b36341988f6b06074b1da.avif)

## YOLOv9 Benchmarks

Benchmarking YOLOv9 with the Ultralytics [Benchmark mode](../modes/benchmark.md) involves evaluating the performance of your trained and validated model in real-world scenarios. This process includes:

- **Performance Evaluation:** Assessing the model's speed and accuracy.
- **Export Formats:** Testing the model across different export formats to ensure it meets the necessary standards and performs well in various environments.
- **Framework Support:** Running these assessments within the Ultralytics package for consistent and reliable results.

By benchmarking, you can ensure that your model not only performs well in controlled testing environments but also maintains high performance in practical, real-world applications.

<p align="center">
  <br>
  <iframe loading="lazy" width="720" height="405" src="https://www.youtube.com/embed/ziJR01lKnio"
    title="YouTube video player" frameborder="0"
    allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share"
    allowfullscreen>
  </iframe>
  <br>
  <strong>Watch:</strong> How to Benchmark the YOLOv9 Model Using the Ultralytics Python Package
</p>

## Performance on MS COCO Dataset

<canvas id="modelComparisonChart" width="1024" height="400" active-models='["YOLOv9"]'></canvas>

At release, YOLOv9 set new benchmarks on the [COCO dataset](../datasets/detect/coco.md) across various model sizes. The table below lists the [accuracy](https://www.ultralytics.com/glossary/accuracy), parameters, and FLOPs of each YOLOv9 model available in Ultralytics.

!!! tip "Performance"

    === "Detection (COCO)"

        | Model                                                                                | size<br><sup>(pixels)</sup> | mAP<sup>val<br>50-95</sup> | mAP<sup>val<br>50</sup> | params<br><sup>(M)</sup> | FLOPs<br><sup>(B)</sup> |
        | ------------------------------------------------------------------------------------ | --------------------------- | -------------------------- | ----------------------- | ------------------------ | ----------------------- |
        | [YOLOv9t](https://github.com/ultralytics/assets/releases/download/v8.4.0/yolov9t.pt) | 640                         | 38.3                       | 53.1                    | **2.0**                  | **7.7**                 |
        | [YOLOv9s](https://github.com/ultralytics/assets/releases/download/v8.4.0/yolov9s.pt) | 640                         | 46.8                       | 63.4                    | 7.2                      | 26.7                    |
        | [YOLOv9m](https://github.com/ultralytics/assets/releases/download/v8.4.0/yolov9m.pt) | 640                         | 51.4                       | 68.1                    | 20.1                     | 76.8                    |
        | [YOLOv9c](https://github.com/ultralytics/assets/releases/download/v8.4.0/yolov9c.pt) | 640                         | 53.0                       | 70.2                    | 25.5                     | 102.8                   |
        | [YOLOv9e](https://github.com/ultralytics/assets/releases/download/v8.4.0/yolov9e.pt) | 640                         | **55.6**                   | **72.8**                | 58.1                     | 192.5                   |

    === "Segmentation (COCO)"

        | Model                                                                                        | size<br><sup>(pixels)</sup> | mAP<sup>box<br>50-95</sup> | mAP<sup>mask<br>50-95</sup> | params<br><sup>(M)</sup> | FLOPs<br><sup>(B)</sup> |
        | -------------------------------------------------------------------------------------------- | --------------------------- | -------------------------- | --------------------------- | ------------------------ | ----------------------- |
        | [YOLOv9c-seg](https://github.com/ultralytics/assets/releases/download/v8.4.0/yolov9c-seg.pt) | 640                         | 52.4                       | 42.2                        | **27.9**                 | **159.4**               |
        | [YOLOv9e-seg](https://github.com/ultralytics/assets/releases/download/v8.4.0/yolov9e-seg.pt) | 640                         | **55.1**                   | **44.3**                    | 60.5                     | 248.4                   |

The YOLOv9 variants, ranging from the tiny `t` model to the large `e` model, improve accuracy ([mAP](https://www.ultralytics.com/glossary/mean-average-precision-map)) while maintaining or reducing parameters and computation (FLOPs) compared to prior versions and competing models.

Compared with other real-time detectors, YOLOv9 shows the following gains:

- **Lightweight Models**: YOLOv9s surpasses the YOLO MS-S in parameter efficiency and computational load while achieving an improvement of 0.4∼0.6% in AP.
- **Medium to Large Models**: YOLOv9m and YOLOv9e improve the trade-off between model complexity and detection performance, reducing parameters and computation while improving accuracy.

The YOLOv9c model operates with 42% fewer parameters and 21% less computation than YOLOv7 AF while achieving comparable accuracy. The YOLOv9e model uses 15% fewer parameters and 25% less computation than [YOLOv8x](yolov8.md), with a 1.7% improvement in AP.

These results show that YOLOv9 improved computational efficiency without compromising the accuracy needed for real-time object detection and other [computer vision](https://www.ultralytics.com/glossary/computer-vision-cv) tasks.

## Conclusion

YOLOv9, released in February 2024, improved the efficiency and accuracy of real-time object detection. By addressing information loss with PGI and GELAN, YOLOv9 set new benchmarks at its time of release. Newer models like [YOLO26](yolo26.md) and [YOLO11](yolo11.md) have since added further improvements (see [how YOLO11 compares with YOLOv8, YOLOv9, and YOLOv10](https://www.ultralytics.com/blog/comparing-ultralytics-yolo11-vs-previous-yolo-models)), but YOLOv9's architectural innovations continue to influence the field.

## Usage Examples

This example provides simple YOLOv9 training and inference examples. For full documentation on these and other [modes](../modes/index.md), see the [Train](../modes/train.md), [Val](../modes/val.md), [Predict](../modes/predict.md), and [Export](../modes/export.md) docs pages.

!!! example

    === "Python"

        [PyTorch](https://www.ultralytics.com/glossary/pytorch) pretrained `*.pt` models as well as configuration `*.yaml` files can be passed to the `YOLO()` class to create a model instance in Python:

        ```python
        from ultralytics import YOLO

        # Build a YOLOv9c model from scratch
        model = YOLO("yolov9c.yaml")

        # Build a YOLOv9c model from pretrained weights
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

The YOLOv9 series offers a range of models for [object detection](../tasks/detect.md) and [instance segmentation](../tasks/segment.md), covering a range of computational budgets and accuracy requirements.

| Model      | Filenames                                                        | Tasks                                        | Training | Validation | Inference | Export |
| ---------- | ---------------------------------------------------------------- | -------------------------------------------- | -------- | ---------- | --------- | ------ |
| YOLOv9     | `yolov9t.pt` `yolov9s.pt` `yolov9m.pt` `yolov9c.pt` `yolov9e.pt` | [Object Detection](../tasks/detect.md)       | ✅       | ✅         | ✅        | ✅     |
| YOLOv9-seg | `yolov9c-seg.pt` `yolov9e-seg.pt`                                | [Instance Segmentation](../tasks/segment.md) | ✅       | ✅         | ✅        | ✅     |

This table summarizes the YOLOv9 model variants and their support for the Training, Validation, Inference, and Export modes.

!!! note

    Training YOLOv9 models will require _more_ resources **and** take longer than the equivalent sized [YOLOv8 model](yolov8.md).

## Citations and Acknowledgments

We would like to acknowledge the YOLOv9 authors for their significant contributions in the field of real-time object detection:

!!! quote "Citation"

    === "BibTeX"

        ```bibtex
        @inproceedings{wang2024yolov9,
          title={YOLOv9: Learning What You Want to Learn Using Programmable Gradient Information},
          author={Wang, Chien-Yao and Yeh, I-Hau and Liao, Hong-Yuan Mark},
          booktitle={Computer Vision -- ECCV 2024},
          pages={1--21},
          year={2024},
          organization={Springer Nature Switzerland}
        }
        ```

The official YOLOv9 paper was published in the [Springer ECCV 2024 proceedings](https://link.springer.com/chapter/10.1007/978-3-031-72751-1_1), with a preprint on [arXiv](https://arxiv.org/pdf/2402.13616). The authors have made their work publicly available, and the codebase can be accessed on [GitHub](https://github.com/WongKinYiu/yolov9). We appreciate their efforts in advancing the field and making their work accessible to the broader community.

## FAQ

### What innovations does YOLOv9 introduce for real-time object detection?

YOLOv9 introduces two main techniques: Programmable Gradient Information (PGI) and the Generalized Efficient Layer Aggregation Network (GELAN). These innovations address information loss challenges in deep neural networks, ensuring high efficiency, accuracy, and adaptability. PGI preserves essential data across network layers, while GELAN optimizes parameter utilization and computational efficiency. Learn more about [YOLOv9's core innovations](#core-innovations-of-yolov9) that set new benchmarks on the MS COCO dataset.

### How does YOLOv9 perform on the MS COCO dataset compared to other models?

At release, YOLOv9 outperformed other state-of-the-art real-time object detectors in accuracy and efficiency. On the [COCO dataset](../datasets/detect/coco.md), YOLOv9 models achieve higher mAP scores across various sizes while maintaining or reducing computational overhead. For instance, YOLOv9c achieves comparable accuracy with 42% fewer parameters and 21% less computational demand than YOLOv7 AF. Explore [performance comparisons](#performance-on-ms-coco-dataset) for detailed metrics.

### How can I train a YOLOv9 model using Python and CLI?

You can train a YOLOv9 model using both Python and CLI commands. For Python, instantiate a model using the `YOLO` class and call the `train` method:

```python
from ultralytics import YOLO

# Build a YOLOv9c model from pretrained weights and train
model = YOLO("yolov9c.pt")
results = model.train(data="coco8.yaml", epochs=100, imgsz=640)
```

For CLI training, execute:

```bash
yolo train model=yolov9c.yaml data=coco8.yaml epochs=100 imgsz=640
```

Learn more about [usage examples](#usage-examples) for training and inference.

### What are the advantages of using Ultralytics YOLOv9 for lightweight models?

YOLOv9 is designed to mitigate information loss, which is particularly important for lightweight models often prone to losing significant information. By integrating Programmable Gradient Information (PGI) and reversible functions, YOLOv9 ensures essential data retention, enhancing the model's accuracy and efficiency. This makes it highly suitable for applications requiring compact models with high performance. For more details, explore the section on [YOLOv9's impact on lightweight models](#impact-on-lightweight-models).

### What tasks and modes does YOLOv9 support?

YOLOv9 supports object detection and [instance segmentation](https://www.ultralytics.com/glossary/instance-segmentation) in the training, validation, inference, and export modes. Refer to the [supported tasks and modes](#supported-tasks-and-modes) section for more information.
