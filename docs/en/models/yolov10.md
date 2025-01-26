---
comments: true
description: Discover YOLOv10, the latest in real-time object detection, eliminating NMS and boosting efficiency. Achieve top performance with a low computational cost.
keywords: YOLOv10, real-time object detection, NMS-free, deep learning, Tsinghua University, Ultralytics, machine learning, neural networks, performance optimization
---

# YOLOv10: Real-Time End-to-End [Object Detection](https://www.ultralytics.com/glossary/object-detection)

YOLOv10, built on the [Ultralytics](https://www.ultralytics.com/) [Python package](https://pypi.org/project/ultralytics/) by researchers at [Tsinghua University](https://www.tsinghua.edu.cn/en/), introduces a new approach to real-time object detection, addressing both the post-processing and model architecture deficiencies found in previous YOLO versions. By eliminating non-maximum suppression (NMS) and optimizing various model components, YOLOv10 achieves state-of-the-art performance with significantly reduced computational overhead. Extensive experiments demonstrate its superior accuracy-latency trade-offs across multiple model scales.

![YOLOv10 consistent dual assignment for NMS-free training](https://github.com/ultralytics/docs/releases/download/0/yolov10-consistent-dual-assignment.avif)

<p align="center">
  <br>
  <iframe loading="lazy" width="720" height="405" src="https://www.youtube.com/embed/_gRqR-miFPE"
    title="YouTube video player" frameborder="0"
    allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share"
    allowfullscreen>
  </iframe>
  <br>
  <strong>Watch:</strong> How to Train YOLOv10 on SKU-110k Dataset using Ultralytics | Retail Dataset
</p>

## Overview

Real-time object detection aims to accurately predict object categories and positions in images with low latency. The YOLO series has been at the forefront of this research due to its balance between performance and efficiency. However, reliance on NMS and architectural inefficiencies have hindered optimal performance. YOLOv10 addresses these issues by introducing consistent dual assignments for NMS-free training and a holistic efficiency-accuracy driven model design strategy.

### Architecture

The architecture of YOLOv10 builds upon the strengths of previous YOLO models while introducing several key innovations. The model architecture consists of the following components:

1. **Backbone**: Responsible for [feature extraction](https://www.ultralytics.com/glossary/feature-extraction), the backbone in YOLOv10 uses an enhanced version of CSPNet (Cross Stage Partial Network) to improve gradient flow and reduce computational redundancy.
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

<script async src="https://cdn.jsdelivr.net/npm/chart.js@3.9.1/dist/chart.min.js"></script>
<script defer src="../../javascript/benchmark.js"></script>

<canvas id="modelComparisonChart" width="1024" height="400" active-models='["YOLOv10"]'></canvas>

YOLOv10 outperforms previous YOLO versions and other state-of-the-art models in terms of accuracy and efficiency. For example, YOLOv10-S is 1.8x faster than RT-DETR-R18 with similar AP on the COCO dataset, and YOLOv10-B has 46% less latency and 25% fewer parameters than YOLOv9-C with the same performance.

| Model          | Input Size | AP<sup>val</sup> | FLOPs (G) | Latency (ms) |
| -------------- | ---------- | ---------------- | --------- | ------------ |
| [YOLOv10-N][1] | 640        | 38.5             | **6.7**   | **1.84**     |
| [YOLOv10-S][2] | 640        | 46.3             | 21.6      | 2.49         |
| [YOLOv10-M][3] | 640        | 51.1             | 59.1      | 4.74         |
| [YOLOv10-B][4] | 640        | 52.5             | 92.0      | 5.74         |
| [YOLOv10-L][5] | 640        | 53.2             | 120.3     | 7.28         |
| [YOLOv10-X][6] | 640        | **54.4**         | 160.4     | 10.70        |

Latency measured with TensorRT FP16 on T4 GPU.

## Methodology

### Consistent Dual Assignments for NMS-Free Training

YOLOv10 employs dual label assignments, combining one-to-many and one-to-one strategies during training to ensure rich supervision and efficient end-to-end deployment. The consistent matching metric aligns the supervision between both strategies, enhancing the quality of predictions during inference.

### Holistic Efficiency-[Accuracy](https://www.ultralytics.com/glossary/accuracy) Driven Model Design

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

![YOLOv10 comparison with SOTA object detectors](https://github.com/ultralytics/docs/releases/download/0/yolov10-comparison-sota-detectors.avif)

Compared to other state-of-the-art detectors:

- YOLOv10-S / X are 1.8× / 1.3× faster than RT-DETR-R18 / R101 with similar accuracy
- YOLOv10-B has 25% fewer parameters and 46% lower latency than YOLOv9-C at same accuracy
- YOLOv10-L / X outperform YOLOv8-L / X by 0.3 AP / 0.5 AP with 1.8× / 2.3× fewer parameters

Here is a detailed comparison of YOLOv10 variants with other state-of-the-art models:

| Model              | Params<br><sup>(M) | FLOPs<br><sup>(G) | mAP<sup>val<br>50-95 | Latency<br><sup>(ms) | Latency-forward<br><sup>(ms) |
| ------------------ | ------------------ | ----------------- | -------------------- | -------------------- | ---------------------------- |
| YOLOv6-3.0-N       | 4.7                | 11.4              | 37.0                 | 2.69                 | **1.76**                     |
| Gold-YOLO-N        | 5.6                | 12.1              | **39.6**             | 2.92                 | 1.82                         |
| YOLOv8-N           | 3.2                | 8.7               | 37.3                 | 6.16                 | 1.77                         |
| **[YOLOv10-N][1]** | **2.3**            | **6.7**           | 39.5                 | **1.84**             | 1.79                         |
|                    |                    |                   |                      |                      |                              |
| YOLOv6-3.0-S       | 18.5               | 45.3              | 44.3                 | 3.42                 | 2.35                         |
| Gold-YOLO-S        | 21.5               | 46.0              | 45.4                 | 3.82                 | 2.73                         |
| YOLOv8-S           | 11.2               | 28.6              | 44.9                 | 7.07                 | **2.33**                     |
| **[YOLOv10-S][2]** | **7.2**            | **21.6**          | **46.8**             | **2.49**             | 2.39                         |
|                    |                    |                   |                      |                      |                              |
| RT-DETR-R18        | 20.0               | 60.0              | 46.5                 | **4.58**             | **4.49**                     |
| YOLOv6-3.0-M       | 34.9               | 85.8              | 49.1                 | 5.63                 | 4.56                         |
| Gold-YOLO-M        | 41.3               | 87.5              | 49.8                 | 6.38                 | 5.45                         |
| YOLOv8-M           | 25.9               | 78.9              | 50.6                 | 9.50                 | 5.09                         |
| **[YOLOv10-M][3]** | **15.4**           | **59.1**          | **51.3**             | 4.74                 | 4.63                         |
|                    |                    |                   |                      |                      |                              |
| YOLOv6-3.0-L       | 59.6               | 150.7             | 51.8                 | 9.02                 | 7.90                         |
| Gold-YOLO-L        | 75.1               | 151.7             | 51.8                 | 10.65                | 9.78                         |
| YOLOv8-L           | 43.7               | 165.2             | 52.9                 | 12.39                | 8.06                         |
| RT-DETR-R50        | 42.0               | 136.0             | 53.1                 | 9.20                 | 9.07                         |
| **[YOLOv10-L][5]** | **24.4**           | **120.3**         | **53.4**             | **7.28**             | **7.21**                     |
|                    |                    |                   |                      |                      |                              |
| YOLOv8-X           | 68.2               | 257.8             | 53.9                 | 16.86                | 12.83                        |
| RT-DETR-R101       | 76.0               | 259.0             | 54.3                 | 13.71                | 13.58                        |
| **[YOLOv10-X][6]** | **29.5**           | **160.4**         | **54.4**             | **10.70**            | **10.60**                    |

[1]: https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov10n.pt
[2]: https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov10s.pt
[3]: https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov10m.pt
[4]: https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov10b.pt
[5]: https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov10l.pt
[6]: https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov10x.pt

## Usage Examples

For predicting new images with YOLOv10:

!!! example

    === "Python"

        ```python
        from ultralytics import YOLO

        # Load a pre-trained YOLOv10n model
        model = YOLO("yolov10n.pt")

        # Perform object detection on an image
        results = model("image.jpg")

        # Display the results
        results[0].show()
        ```

    === "CLI"

        ```bash
        # Load a COCO-pretrained YOLOv10n model and run inference on the 'bus.jpg' image
        yolo detect predict model=yolov10n.pt source=path/to/bus.jpg
        ```

For training YOLOv10 on a custom dataset:

!!! example

    === "Python"

        ```python
        from ultralytics import YOLO

        # Load YOLOv10n model from scratch
        model = YOLO("yolov10n.yaml")

        # Train the model
        model.train(data="coco8.yaml", epochs=100, imgsz=640)
        ```

    === "CLI"

        ```bash
        # Build a YOLOv10n model from scratch and train it on the COCO8 example dataset for 100 epochs
        yolo train model=yolov10n.yaml data=coco8.yaml epochs=100 imgsz=640

        # Build a YOLOv10n model from scratch and run inference on the 'bus.jpg' image
        yolo predict model=yolov10n.yaml source=path/to/bus.jpg
        ```

## Supported Tasks and Modes

The YOLOv10 models series offers a range of models, each optimized for high-performance [Object Detection](../tasks/detect.md). These models cater to varying computational needs and accuracy requirements, making them versatile for a wide array of applications.

| Model   | Filenames                                                             | Tasks                                  | Inference | Validation | Training | Export |
| ------- | --------------------------------------------------------------------- | -------------------------------------- | --------- | ---------- | -------- | ------ |
| YOLOv10 | `yolov10n.pt` `yolov10s.pt` `yolov10m.pt` `yolov10l.pt` `yolov10x.pt` | [Object Detection](../tasks/detect.md) | ✅        | ✅         | ✅       | ✅     |

## Exporting YOLOv10

Due to the new operations introduced with YOLOv10, not all export formats provided by Ultralytics are currently supported. The following table outlines which formats have been successfully converted using Ultralytics for YOLOv10. Feel free to open a pull request if you're able to [provide a contribution change](../help/contributing.md) for adding export support of additional formats for YOLOv10.

| Export Format                                     | Export Support | Exported Model Inference | Notes                                                                                  |
| ------------------------------------------------- | -------------- | ------------------------ | -------------------------------------------------------------------------------------- |
| [TorchScript](../integrations/torchscript.md)     | ✅             | ✅                       | Standard [PyTorch](https://www.ultralytics.com/glossary/pytorch) model format.         |
| [ONNX](../integrations/onnx.md)                   | ✅             | ✅                       | Widely supported for deployment.                                                       |
| [OpenVINO](../integrations/openvino.md)           | ✅             | ✅                       | Optimized for Intel hardware.                                                          |
| [TensorRT](../integrations/tensorrt.md)           | ✅             | ✅                       | Optimized for NVIDIA GPUs.                                                             |
| [CoreML](../integrations/coreml.md)               | ✅             | ✅                       | Limited to Apple devices.                                                              |
| [TF SavedModel](../integrations/tf-savedmodel.md) | ✅             | ✅                       | [TensorFlow](https://www.ultralytics.com/glossary/tensorflow)'s standard model format. |
| [TF GraphDef](../integrations/tf-graphdef.md)     | ✅             | ✅                       | Legacy TensorFlow format.                                                              |
| [TF Lite](../integrations/tflite.md)              | ✅             | ✅                       | Optimized for mobile and embedded.                                                     |
| [TF Edge TPU](../integrations/edge-tpu.md)        | ✅             | ✅                       | Specific to Google's Edge TPU devices.                                                 |
| [TF.js](../integrations/tfjs.md)                  | ✅             | ✅                       | JavaScript environment for browser use.                                                |
| [PaddlePaddle](../integrations/paddlepaddle.md)   | ❌             | ❌                       | Popular in China; less global support.                                                 |
| [NCNN](../integrations/ncnn.md)                   | ✅             | ❌                       | Layer `torch.topk` not exists or registered                                            |

## Conclusion

YOLOv10 sets a new standard in real-time object detection by addressing the shortcomings of previous YOLO versions and incorporating innovative design strategies. Its ability to deliver high accuracy with low computational cost makes it an ideal choice for a wide range of real-world applications.

## Citations and Acknowledgements

We would like to acknowledge the YOLOv10 authors from [Tsinghua University](https://www.tsinghua.edu.cn/en/) for their extensive research and significant contributions to the [Ultralytics](https://www.ultralytics.com/) framework:

!!! quote ""

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

## FAQ

### What is YOLOv10 and how does it differ from previous YOLO versions?

YOLOv10, developed by researchers at [Tsinghua University](https://www.tsinghua.edu.cn/en/), introduces several key innovations to real-time object detection. It eliminates the need for non-maximum suppression (NMS) by employing consistent dual assignments during training and optimized model components for superior performance with reduced computational overhead. For more details on its architecture and key features, check out the [YOLOv10 overview](#overview) section.

### How can I get started with running inference using YOLOv10?

For easy inference, you can use the Ultralytics YOLO Python library or the command line interface (CLI). Below are examples of predicting new images using YOLOv10:

!!! example

    === "Python"

        ```python
        from ultralytics import YOLO

        # Load the pre-trained YOLOv10-N model
        model = YOLO("yolov10n.pt")
        results = model("image.jpg")
        results[0].show()
        ```

    === "CLI"

        ```bash
        yolo detect predict model=yolov10n.pt source=path/to/image.jpg
        ```

For more usage examples, visit our [Usage Examples](#usage-examples) section.

### Which model variants does YOLOv10 offer and what are their use cases?

YOLOv10 offers several model variants to cater to different use cases:

- **YOLOv10-N**: Suitable for extremely resource-constrained environments
- **YOLOv10-S**: Balances speed and accuracy
- **YOLOv10-M**: General-purpose use
- **YOLOv10-B**: Higher accuracy with increased width
- **YOLOv10-L**: High accuracy at the cost of computational resources
- **YOLOv10-X**: Maximum accuracy and performance

Each variant is designed for different computational needs and accuracy requirements, making them versatile for a variety of applications. Explore the [Model Variants](#model-variants) section for more information.

### How does the NMS-free approach in YOLOv10 improve performance?

YOLOv10 eliminates the need for non-maximum suppression (NMS) during inference by employing consistent dual assignments for training. This approach reduces inference latency and enhances prediction efficiency. The architecture also includes a one-to-one head for inference, ensuring that each object gets a single best prediction. For a detailed explanation, see the [Consistent Dual Assignments for NMS-Free Training](#consistent-dual-assignments-for-nms-free-training) section.

### Where can I find the export options for YOLOv10 models?

YOLOv10 supports several export formats, including TorchScript, ONNX, OpenVINO, and TensorRT. However, not all export formats provided by Ultralytics are currently supported for YOLOv10 due to its new operations. For details on the supported formats and instructions on exporting, visit the [Exporting YOLOv10](#exporting-yolov10) section.

### What are the performance benchmarks for YOLOv10 models?

YOLOv10 outperforms previous YOLO versions and other state-of-the-art models in both accuracy and efficiency. For example, YOLOv10-S is 1.8x faster than RT-DETR-R18 with a similar AP on the COCO dataset. YOLOv10-B shows 46% less latency and 25% fewer parameters than YOLOv9-C with the same performance. Detailed benchmarks can be found in the [Comparisons](#comparisons) section.
