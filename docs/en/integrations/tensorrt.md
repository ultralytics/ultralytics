---
comments: true
description: Learn to convert YOLOv8 models to TensorRT for high-speed NVIDIA GPU inference. Boost efficiency and deploy optimized models with our step-by-step guide.
keywords: YOLOv8, TensorRT, NVIDIA, GPU, deep learning, model optimization, high-speed inference, model export
---

# TensorRT Export for YOLOv8 Models

Deploying computer vision models in high-performance environments can require a format that maximizes speed and efficiency. This is especially true when you are deploying your model on NVIDIA GPUs.

By using the TensorRT export format, you can enhance your [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics) models for swift and efficient inference on NVIDIA hardware. This guide will give you easy-to-follow steps for the conversion process and help you make the most of NVIDIA's advanced technology in your deep learning projects.

## TensorRT

<p align="center">
  <img width="100%" src="https://github.com/ultralytics/ultralytics/assets/26833433/7fea48c2-9709-4deb-8d04-eaf95d12a91d" alt="TensorRT Overview">
</p>

[TensorRT](https://developer.nvidia.com/tensorrt), developed by NVIDIA, is an advanced software development kit (SDK) designed for high-speed deep learning inference. It's well-suited for real-time applications like object detection.

This toolkit optimizes deep learning models for NVIDIA GPUs and results in faster and more efficient operations. TensorRT models undergo TensorRT optimization, which includes techniques like layer fusion, precision calibration (INT8 and FP16), dynamic tensor memory management, and kernel auto-tuning. Converting deep learning models into the TensorRT format allows developers to realize the potential of NVIDIA GPUs fully.

TensorRT is known for its compatibility with various model formats, including TensorFlow, PyTorch, and ONNX, providing developers with a flexible solution for integrating and optimizing models from different frameworks. This versatility enables efficient model deployment across diverse hardware and software environments.

## Key Features of TensorRT Models

TensorRT models offer a range of key features that contribute to their efficiency and effectiveness in high-speed deep learning inference:

- **Precision Calibration**: TensorRT supports precision calibration, allowing models to be fine-tuned for specific accuracy requirements. This includes support for reduced precision formats like INT8 and FP16, which can further boost inference speed while maintaining acceptable accuracy levels.

- **Layer Fusion**: The TensorRT optimization process includes layer fusion, where multiple layers of a neural network are combined into a single operation. This reduces computational overhead and improves inference speed by minimizing memory access and computation.

<p align="center">
  <img width="100%" src="https://developer-blogs.nvidia.com/wp-content/uploads/2017/12/pasted-image-0-3.png" alt="TensorRT Layer Fusion">
</p>

- **Dynamic Tensor Memory Management**: TensorRT efficiently manages tensor memory usage during inference, reducing memory overhead and optimizing memory allocation. This results in more efficient GPU memory utilization.

- **Automatic Kernel Tuning**: TensorRT applies automatic kernel tuning to select the most optimized GPU kernel for each layer of the model. This adaptive approach ensures that the model takes full advantage of the GPU's computational power.

## Deployment Options in TensorRT

Before we look at the code for exporting YOLOv8 models to the TensorRT format, let's understand where TensorRT models are normally used.

TensorRT offers several deployment options, and each option balances ease of integration, performance optimization, and flexibility differently:

- **Deploying within TensorFlow**: This method integrates TensorRT into TensorFlow, allowing optimized models to run in a familiar TensorFlow environment. It's useful for models with a mix of supported and unsupported layers, as TF-TRT can handle these efficiently.

<p align="center">
  <img width="100%" src="https://docs.nvidia.com/deeplearning/tensorrt/quick-start-guide/graphics/tf-trt-workflow.png" alt="TensorRT Overview">
</p>

- **Standalone TensorRT Runtime API**: Offers granular control, ideal for performance-critical applications. It's more complex but allows for custom implementation of unsupported operators.

- **NVIDIA Triton Inference Server**: An option that supports models from various frameworks. Particularly suited for cloud or edge inference, it provides features like concurrent model execution and model analysis.

## Exporting YOLOv8 Models to TensorRT

You can improve execution efficiency and optimize performance by converting YOLOv8 models to TensorRT format.

### Installation

To install the required package, run:

!!! Tip "Installation"

    === "CLI"

        ```bash
        # Install the required package for YOLOv8
        pip install ultralytics
        ```

For detailed instructions and best practices related to the installation process, check our [YOLOv8 Installation guide](../quickstart.md). While installing the required packages for YOLOv8, if you encounter any difficulties, consult our [Common Issues guide](../guides/yolo-common-issues.md) for solutions and tips.

### Usage

Before diving into the usage instructions, be sure to check out the range of [YOLOv8 models offered by Ultralytics](../models/index.md). This will help you choose the most appropriate model for your project requirements.

!!! Example "Usage"

    === "Python"

        ```python
        from ultralytics import YOLO

        # Load the YOLOv8 model
        model = YOLO("yolov8n.pt")

        # Export the model to TensorRT format
        model.export(format="engine")  # creates 'yolov8n.engine'

        # Load the exported TensorRT model
        tensorrt_model = YOLO("yolov8n.engine")

        # Run inference
        results = tensorrt_model("https://ultralytics.com/images/bus.jpg")
        ```

    === "CLI"

        ```bash
        # Export a YOLOv8n PyTorch model to TensorRT format
        yolo export model=yolov8n.pt format=engine  # creates 'yolov8n.engine''

        # Run inference with the exported model
        yolo predict model=yolov8n.engine source='https://ultralytics.com/images/bus.jpg'
        ```

For more details about the export process, visit the [Ultralytics documentation page on exporting](../modes/export.md).

### Exporting TensorRT with INT8 Quantization

Exporting Ultralytics YOLO models using TensorRT with INT8 precision executes post-training quantization (PTQ). TensorRT uses calibration for PTQ, which measures the distribution of activations within each activation tensor as the YOLO model processes inference on representative input data, and then uses that distribution to estimate scale values for each tensor. Each activation tensor that is a candidate for quantization has an associated scale that is deduced by a calibration process.

When processing implicitly quantized networks TensorRT uses INT8 opportunistically to optimize layer execution time. If a layer runs faster in INT8 and has assigned quantization scales on its data inputs and outputs, then a kernel with INT8 precision is assigned to that layer, otherwise TensorRT selects a precision of either FP32 or FP16 for the kernel based on whichever results in faster execution time for that layer.

!!! tip

    It is **critical** to ensure that the same device that will use the TensorRT model weights for deployment is used for exporting with INT8 precision, as the calibration results can vary across devices.

#### Configuring INT8 Export

The arguments provided when using [export](../modes/export.md) for an Ultralytics YOLO model will **greatly** influence the performance of the exported model. They will also need to be selected based on the device resources available, however the default arguments _should_ work for most [Ampere (or newer) NVIDIA discrete GPUs](https://developer.nvidia.com/blog/nvidia-ampere-architecture-in-depth/). The calibration algorithm used is `"ENTROPY_CALIBRATION_2"` and you can read more details about the options available [in the TensorRT Developer Guide](https://docs.nvidia.com/deeplearning/tensorrt/developer-guide/index.html#enable_int8_c). Ultralytics tests found that `"ENTROPY_CALIBRATION_2"` was the best choice and exports are fixed to using this algorithm.

- `workspace` : Controls the size (in GiB) of the device memory allocation while converting the model weights.

    - Adjust the `workspace` value according to your calibration needs and resource availability. While a larger `workspace` may increase calibration time, it allows TensorRT to explore a wider range of optimization tactics, potentially enhancing model performance and accuracy. Conversely, a smaller `workspace` can reduce calibration time but may limit the optimization strategies, affecting the quality of the quantized model.

    - Default is `workspace=4` (GiB), this value may need to be increased if calibration crashes (exits without warning).

    - TensorRT will report `UNSUPPORTED_STATE` during export if the value for `workspace` is larger than the memory available to the device, which means the value for `workspace` should be lowered.

    - If `workspace` is set to max value and calibration fails/crashes, consider reducing the values for `imgsz` and `batch` to reduce memory requirements.

    - <u><b>Remember</b> calibration for INT8 is specific to each device</u>, borrowing a "high-end" GPU for calibration, might result in poor performance when inference is run on another device.

- `batch` : The maximum batch-size that will be used for inference. During inference smaller batches can be used, but inference will not accept batches any larger than what is specified.

!!! note

    During calibration, twice the `batch` size provided will be used. Using small batches can lead to inaccurate scaling during calibration. This is because the process adjusts based on the data it sees. Small batches might not capture the full range of values, leading to issues with the final calibration, so the `batch` size is doubled automatically. If no batch size is specified `batch=1`, calibration will be run at `batch=1 * 2` to reduce calibration scaling errors.

Experimentation by NVIDIA led them to recommend using at least 500 calibration images that are representative of the data for your model, with INT8 quantization calibration. This is a guideline and not a _hard_ requirement, and <u>**you will need to experiment with what is required to perform well for your dataset**.</u> Since the calibration data is required for INT8 calibration with TensorRT, make certain to use the `data` argument when `int8=True` for TensorRT and use `data="my_dataset.yaml"`, which will use the images from [validation](../modes/val.md) to calibrate with. When no value is passed for `data` with export to TensorRT with INT8 quantization, the default will be to use one of the ["small" example datasets based on the model task](../datasets/index.md) instead of throwing an error.

!!! example

    === "Python"

        ```{ .py .annotate }
        from ultralytics import YOLO

        model = YOLO("yolov8n.pt")
        model.export(
            format="engine",
            dynamic=True,  # (1)!
            batch=8,  # (2)!
            workspace=4,  # (3)!
            int8=True,
            data="coco.yaml",  # (4)!
        )

        # Load the exported TensorRT INT8 model
        model = YOLO("yolov8n.engine", task="detect")

        # Run inference
        result = model.predict("https://ultralytics.com/images/bus.jpg")
        ```

        1. Exports with dynamic axes, this will be enabled by default when exporting with `int8=True` even when not explicitly set. See [export arguments](../modes/export.md#arguments) for additional information.
        2. Sets max batch size of 8 for exported model, which calibrates with `batch = 2 * 8` to avoid scaling errors during calibration.
        3. Allocates 4 GiB of memory instead of allocating the entire device for conversion process.
        4. Uses [COCO dataset](../datasets/detect/coco.md) for calibration, specifically the images used for [validation](../modes/val.md) (5,000 total).


    === "CLI"

        ```bash
        # Export a YOLOv8n PyTorch model to TensorRT format with INT8 quantization
        yolo export model=yolov8n.pt format=engine batch=8 workspace=4 int8=True data=coco.yaml  # creates 'yolov8n.engine''

        # Run inference with the exported TensorRT quantized model
        yolo predict model=yolov8n.engine source='https://ultralytics.com/images/bus.jpg'
        ```

???+ warning "Calibration Cache"

    TensorRT will generate a calibration `.cache` which can be re-used to speed up export of future model weights using the same data, but this may result in poor calibration when the data is vastly different or if the `batch` value is changed drastically. In these circumstances, the existing `.cache` should be renamed and moved to a different directory or deleted entirely.

#### Advantages of using YOLO with TensorRT INT8

- **Reduced model size:** Quantization from FP32 to INT8 can reduce the model size by 4x (on disk or in memory), leading to faster download times. lower storage requirements, and reduced memory footprint when deploying a model.

- **Lower power consumption:** Reduced precision operations for INT8 exported YOLO models can consume less power compared to FP32 models, especially for battery-powered devices.

- **Improved inference speeds:** TensorRT optimizes the model for the target hardware, potentially leading to faster inference speeds on GPUs, embedded devices, and accelerators.

??? note "Note on Inference Speeds"

    The first few inference calls with a model exported to TensorRT INT8 can be expected to have longer than usual preprocessing, inference, and/or postprocessing times. This may also occur when changing `imgsz` during inference, especially when `imgsz` is not the same as what was specified during export (export `imgsz` is set as TensorRT "optimal" profile).

#### Drawbacks of using YOLO with TensorRT INT8

- **Decreases in evaluation metrics:** Using a lower precision will mean that `mAP`, `Precision`, `Recall` or any [other metric used to evaluate model performance](../guides/yolo-performance-metrics.md) is likely to be somewhat worse. See the [Performance results section](#ultralytics-yolo-tensorrt-export-performance) to compare the differences in `mAP50` and `mAP50-95` when exporting with INT8 on small sample of various devices.

- **Increased development times:** Finding the "optimal" settings for INT8 calibration for dataset and device can take a significant amount of testing.

- **Hardware dependency:** Calibration and performance gains could be highly hardware dependent and model weights are less transferable.

## Ultralytics YOLO TensorRT Export Performance

### NVIDIA A100

!!! tip "Performance"

    Tested with Ubuntu 22.04.3 LTS, `python 3.10.12`, `ultralytics==8.2.4`, `tensorrt==8.6.1.post1`

    === "Detection (COCO)"

        See [Detection Docs](../tasks/detect.md) for usage examples with these models trained on [COCO](../datasets/detect/coco.md), which include 80 pre-trained classes.

        !!! note
            Inference times shown for `mean`, `min` (fastest), and `max` (slowest) for each test using pre-trained weights `yolov8n.engine`

        | Precision | Eval test    | mean<br>(ms) | min \| max<br>(ms) | mAP<sup>val<br>50(B) | mAP<sup>val<br>50-95(B) | `batch` | size<br><sup>(pixels) |
        |-----------|--------------|--------------|--------------------|----------------------|-------------------------|---------|-----------------------|
        | FP32      | Predict      | 0.52         | 0.51 \| 0.56       |                      |                         | 8       | 640                   |
        | FP32      | COCO<sup>val | 0.52         |                    | 0.52                 | 0.37                    | 1       | 640                   |
        | FP16      | Predict      | 0.34         | 0.34 \| 0.41       |                      |                         | 8       | 640                   |
        | FP16      | COCO<sup>val | 0.33         |                    | 0.52                 | 0.37                    | 1       | 640                   |
        | INT8      | Predict      | 0.28         | 0.27 \| 0.31       |                      |                         | 8       | 640                   |
        | INT8      | COCO<sup>val | 0.29         |                    | 0.47                 | 0.33                    | 1       | 640                   |

    === "Segmentation (COCO)"

        See [Segmentation Docs](../tasks/segment.md) for usage examples with these models trained on [COCO](../datasets/segment/coco.md), which include 80 pre-trained classes.

        !!! note
            Inference times shown for `mean`, `min` (fastest), and `max` (slowest) for each test using pre-trained weights `yolov8n-seg.engine`

        | Precision | Eval test    | mean<br>(ms) | min \| max<br>(ms) | mAP<sup>val<br>50(B) | mAP<sup>val<br>50-95(B) | mAP<sup>val<br>50(M) | mAP<sup>val<br>50-95(M) | `batch` | size<br><sup>(pixels) |
        |-----------|--------------|--------------|--------------------|----------------------|-------------------------|----------------------|-------------------------|---------|-----------------------|
        | FP32      | Predict      | 0.62         | 0.61 \| 0.68       |                      |                         |                      |                         | 8       | 640                   |
        | FP32      | COCO<sup>val | 0.63         |                    | 0.52                 | 0.36                    | 0.49                 | 0.31                    | 1       | 640                   |
        | FP16      | Predict      | 0.40         | 0.39 \| 0.44       |                      |                         |                      |                         | 8       | 640                   |
        | FP16      | COCO<sup>val | 0.43         |                    | 0.52                 | 0.36                    | 0.49                 | 0.30                    | 1       | 640                   |
        | INT8      | Predict      | 0.34         | 0.33 \| 0.37       |                      |                         |                      |                         | 8       | 640                   |
        | INT8      | COCO<sup>val | 0.36         |                    | 0.46                 | 0.32                    | 0.43                 | 0.27                    | 1       | 640                   |

    === "Classification (ImageNet)"

        See [Classification Docs](../tasks/classify.md) for usage examples with these models trained on [ImageNet](../datasets/classify/imagenet.md), which include 1000 pre-trained classes.

        !!! note
            Inference times shown for `mean`, `min` (fastest), and `max` (slowest) for each test using pre-trained weights `yolov8n-cls.engine`

        | Precision | Eval test        | mean<br>(ms) | min \| max<br>(ms) | top-1 | top-5 | `batch` | size<br><sup>(pixels) |
        |-----------|------------------|--------------|--------------------|-------|-------|---------|-----------------------|
        | FP32      | Predict          | 0.26         | 0.25 \| 0.28       |       |       | 8       | 640                   |
        | FP32      | ImageNet<sup>val | 0.26         |                    | 0.35  | 0.61  | 1       | 640                   |
        | FP16      | Predict          | 0.18         | 0.17 \| 0.19       |       |       | 8       | 640                   |
        | FP16      | ImageNet<sup>val | 0.18         |                    | 0.35  | 0.61  | 1       | 640                   |
        | INT8      | Predict          | 0.16         | 0.15 \| 0.57       |       |       | 8       | 640                   |
        | INT8      | ImageNet<sup>val | 0.15         |                    | 0.32  | 0.59  | 1       | 640                   |

    === "Pose (COCO)"

        See [Pose Estimation Docs](../tasks/pose.md) for usage examples with these models trained on [COCO](../datasets/pose/coco.md), which include 1 pre-trained class, "person".

        !!! note
            Inference times shown for `mean`, `min` (fastest), and `max` (slowest) for each test using pre-trained weights `yolov8n-pose.engine`

        | Precision | Eval test    | mean<br>(ms) | min \| max<br>(ms) | mAP<sup>val<br>50(B) | mAP<sup>val<br>50-95(B) | mAP<sup>val<br>50(P) | mAP<sup>val<br>50-95(P) | `batch` | size<br><sup>(pixels) |
        |-----------|--------------|--------------|--------------------|----------------------|-------------------------|----------------------|-------------------------|---------|-----------------------|
        | FP32      | Predict      | 0.54         | 0.53 \| 0.58       |                      |                         |                      |                         | 8       | 640                   |
        | FP32      | COCO<sup>val | 0.55         |                    | 0.91                 | 0.69                    | 0.80                 | 0.51                    | 1       | 640                   |
        | FP16      | Predict      | 0.37         | 0.35 \| 0.41       |                      |                         |                      |                         | 8       | 640                   |
        | FP16      | COCO<sup>val | 0.36         |                    | 0.91                 | 0.69                    | 0.80                 | 0.51                    | 1       | 640                   |
        | INT8      | Predict      | 0.29         | 0.28 \| 0.33       |                      |                         |                      |                         | 8       | 640                   |
        | INT8      | COCO<sup>val | 0.30         |                    | 0.90                 | 0.68                    | 0.78                 | 0.47                    | 1       | 640                   |

    === "OBB (DOTAv1)"

        See [Oriented Detection Docs](../tasks/obb.md) for usage examples with these models trained on [DOTAv1](../datasets/obb/dota-v2.md#dota-v10), which include 15 pre-trained classes.

        !!! note
            Inference times shown for `mean`, `min` (fastest), and `max` (slowest) for each test using pre-trained weights `yolov8n-obb.engine`

        | Precision | Eval test      | mean<br>(ms) | min \| max<br>(ms) | mAP<sup>val<br>50(B) | mAP<sup>val<br>50-95(B) | `batch` | size<br><sup>(pixels) |
        |-----------|----------------|--------------|--------------------|----------------------|-------------------------|---------|-----------------------|
        | FP32      | Predict        | 0.52         | 0.51 \| 0.59       |                      |                         | 8       | 640                   |
        | FP32      | DOTAv1<sup>val | 0.76         |                    | 0.50                 | 0.36                    | 1       | 640                   |
        | FP16      | Predict        | 0.34         | 0.33 \| 0.42       |                      |                         | 8       | 640                   |
        | FP16      | DOTAv1<sup>val | 0.59         |                    | 0.50                 | 0.36                    | 1       | 640                   |
        | INT8      | Predict        | 0.29         | 0.28 \| 0.33       |                      |                         | 8       | 640                   |
        | INT8      | DOTAv1<sup>val | 0.32         |                    | 0.45                 | 0.32                    | 1       | 640                   |

### Consumer GPUs

!!! tip "Detection Performance (COCO)"

    === "RTX 3080 12 GB"

        Tested with Windows 10.0.19045, `python 3.10.9`, `ultralytics==8.2.4`, `tensorrt==10.0.0b6`

        !!! note
            Inference times shown for `mean`, `min` (fastest), and `max` (slowest) for each test using pre-trained weights `yolov8n.engine`

        | Precision | Eval test    | mean<br>(ms) | min \| max<br>(ms) | mAP<sup>val<br>50(B) | mAP<sup>val<br>50-95(B) | `batch` | size<br><sup>(pixels) |
        |-----------|--------------|--------------|--------------------|----------------------|-------------------------|---------|-----------------------|
        | FP32      | Predict      | 1.06         | 0.75 \| 1.88       |                      |                         | 8       | 640                   |
        | FP32      | COCO<sup>val | 1.37         |                    | 0.52                 | 0.37                    | 1       | 640                   |
        | FP16      | Predict      | 0.62         | 0.75 \| 1.13       |                      |                         | 8       | 640                   |
        | FP16      | COCO<sup>val | 0.85         |                    | 0.52                 | 0.37                    | 1       | 640                   |
        | INT8      | Predict      | 0.52         | 0.38 \| 1.00       |                      |                         | 8       | 640                   |
        | INT8      | COCO<sup>val | 0.74         |                    | 0.47                 | 0.33                    | 1       | 640                   |

    === "RTX 3060 12 GB"

        Tested with Windows 10.0.22631, `python 3.11.9`, `ultralytics==8.2.4`, `tensorrt==10.0.1`

        !!! note
            Inference times shown for `mean`, `min` (fastest), and `max` (slowest) for each test using pre-trained weights `yolov8n.engine`


        | Precision | Eval test    | mean<br>(ms) | min \| max<br>(ms) | mAP<sup>val<br>50(B) | mAP<sup>val<br>50-95(B) | `batch` | size<br><sup>(pixels) |
        |-----------|--------------|--------------|--------------------|----------------------|-------------------------|---------|-----------------------|
        | FP32      | Predict      | 1.76         | 1.69 \| 1.87       |                      |                         | 8       | 640                   |
        | FP32      | COCO<sup>val | 1.94         |                    | 0.52                 | 0.37                    | 1       | 640                   |
        | FP16      | Predict      | 0.86         | 0.75 \| 1.00       |                      |                         | 8       | 640                   |
        | FP16      | COCO<sup>val | 1.43         |                    | 0.52                 | 0.37                    | 1       | 640                   |
        | INT8      | Predict      | 0.80         | 0.75 \| 1.00       |                      |                         | 8       | 640                   |
        | INT8      | COCO<sup>val | 1.35         |                    | 0.47                 | 0.33                    | 1       | 640                   |

    === "RTX 2060 6 GB"

        Tested with Pop!_OS 22.04 LTS, `python 3.10.12`, `ultralytics==8.2.4`, `tensorrt==8.6.1.post1`

        !!! note
            Inference times shown for `mean`, `min` (fastest), and `max` (slowest) for each test using pre-trained weights `yolov8n.engine`

        | Precision | Eval test    | mean<br>(ms) | min \| max<br>(ms) | mAP<sup>val<br>50(B) | mAP<sup>val<br>50-95(B) | `batch` | size<br><sup>(pixels) |
        |-----------|--------------|--------------|--------------------|----------------------|-------------------------|---------|-----------------------|
        | FP32      | Predict      | 2.84         | 2.84 \| 2.85       |                      |                         | 8       | 640                   |
        | FP32      | COCO<sup>val | 2.94         |                    | 0.52                 | 0.37                    | 1       | 640                   |
        | FP16      | Predict      | 1.09         | 1.09 \| 1.10       |                      |                         | 8       | 640                   |
        | FP16      | COCO<sup>val | 1.20         |                    | 0.52                 | 0.37                    | 1       | 640                   |
        | INT8      | Predict      | 0.75         | 0.74 \| 0.75       |                      |                         | 8       | 640                   |
        | INT8      | COCO<sup>val | 0.76         |                    | 0.47                 | 0.33                    | 1       | 640                   |

### Embedded Devices

!!! tip "Detection Performance (COCO)"

    === "Jetson Orin NX 16GB"

        Tested with JetPack 6.0 (L4T 36.3) Ubuntu 22.04.4 LTS, `python 3.10.12`, `ultralytics==8.2.16`, `tensorrt==10.0.1`

        !!! note
            Inference times shown for `mean`, `min` (fastest), and `max` (slowest) for each test using pre-trained weights `yolov8n.engine`

        | Precision | Eval test    | mean<br>(ms) | min \| max<br>(ms) | mAP<sup>val<br>50(B) | mAP<sup>val<br>50-95(B) | `batch` | size<br><sup>(pixels) |
        |-----------|--------------|--------------|--------------------|----------------------|-------------------------|---------|-----------------------|
        | FP32      | Predict      | 6.11         | 6.10 \| 6.29       |                      |                         | 8       | 640                   |
        | FP32      | COCO<sup>val | 6.17         |                    | 0.52                 | 0.37                    | 1       | 640                   |
        | FP16      | Predict      | 3.18         | 3.18 \| 3.20       |                      |                         | 8       | 640                   |
        | FP16      | COCO<sup>val | 3.19         |                    | 0.52                 | 0.37                    | 1       | 640                   |
        | INT8      | Predict      | 2.30         | 2.29 \| 2.35       |                      |                         | 8       | 640                   |
        | INT8      | COCO<sup>val | 2.32         |                    | 0.46                 | 0.32                    | 1       | 640                   |

!!! info

    See our [quickstart guide on NVIDIA Jetson with Ultralytics YOLO](../guides/nvidia-jetson.md) to learn more about setup and configuration.

#### Evaluation methods

Expand sections below for information on how these models were exported and tested.

??? example "Export configurations"

    See [export mode](../modes/export.md) for details regarding export configuration arguments.

    ```py
    from ultralytics import YOLO

    model = YOLO("yolov8n.pt")

    # TensorRT FP32
    out = model.export(format="engine", imgsz=640, dynamic=True, verbose=False, batch=8, workspace=2)

    # TensorRT FP16
    out = model.export(format="engine", imgsz=640, dynamic=True, verbose=False, batch=8, workspace=2, half=True)

    # TensorRT INT8 with calibration `data` (i.e. COCO, ImageNet, or DOTAv1 for appropriate model task)
    out = model.export(
        format="engine", imgsz=640, dynamic=True, verbose=False, batch=8, workspace=2, int8=True, data="coco8.yaml"
    )
    ```

??? example "Predict loop"

    See [predict mode](../modes/predict.md) for additional information.

    ```py
    import cv2

    from ultralytics import YOLO

    model = YOLO("yolov8n.engine")
    img = cv2.imread("path/to/image.jpg")

    for _ in range(100):
        result = model.predict(
            [img] * 8,  # batch=8 of the same image
            verbose=False,
            device="cuda",
        )
    ```

??? example "Validation configuration"

    See [`val` mode](../modes/val.md) to learn more about validation configuration arguments.

    ```py
    from ultralytics import YOLO

    model = YOLO("yolov8n.engine")
    results = model.val(
        data="data.yaml",  # COCO, ImageNet, or DOTAv1 for appropriate model task
        batch=1,
        imgsz=640,
        verbose=False,
        device="cuda",
    )
    ```

## Deploying Exported YOLOv8 TensorRT Models

Having successfully exported your Ultralytics YOLOv8 models to TensorRT format, you're now ready to deploy them. For in-depth instructions on deploying your TensorRT models in various settings, take a look at the following resources:

- **[Deploy Ultralytics with a Triton Server](../guides/triton-inference-server.md)**: Our guide on how to use NVIDIA's Triton Inference (formerly TensorRT Inference) Server specifically for use with Ultralytics YOLO models.

- **[Deploying Deep Neural Networks with NVIDIA TensorRT](https://developer.nvidia.com/blog/deploying-deep-learning-nvidia-tensorrt/)**: This article explains how to use NVIDIA TensorRT to deploy deep neural networks on GPU-based deployment platforms efficiently.

- **[End-to-End AI for NVIDIA-Based PCs: NVIDIA TensorRT Deployment](https://developer.nvidia.com/blog/end-to-end-ai-for-nvidia-based-pcs-nvidia-tensorrt-deployment/)**: This blog post explains the use of NVIDIA TensorRT for optimizing and deploying AI models on NVIDIA-based PCs.

- **[GitHub Repository for NVIDIA TensorRT:](https://github.com/NVIDIA/TensorRT)**: This is the official GitHub repository that contains the source code and documentation for NVIDIA TensorRT.

## Summary

In this guide, we focused on converting Ultralytics YOLOv8 models to NVIDIA's TensorRT model format. This conversion step is crucial for improving the efficiency and speed of YOLOv8 models, making them more effective and suitable for diverse deployment environments.

For more information on usage details, take a look at the [TensorRT official documentation](https://docs.nvidia.com/deeplearning/tensorrt/).

If you're curious about additional Ultralytics YOLOv8 integrations, our [integration guide page](../integrations/index.md) provides an extensive selection of informative resources and insights.

## FAQ

### How do I convert YOLOv8 models to TensorRT format?

To convert your Ultralytics YOLOv8 models to TensorRT format for optimized NVIDIA GPU inference, follow these steps:

1. **Install the required package**:

    ```bash
    pip install ultralytics
    ```

2. **Export your YOLOv8 model**:

    ```python
    from ultralytics import YOLO

    model = YOLO("yolov8n.pt")
    model.export(format="engine")  # creates 'yolov8n.engine'

    # Run inference
    model = YOLO("yolov8n.engine")
    results = model("https://ultralytics.com/images/bus.jpg")
    ```

For more details, visit the [YOLOv8 Installation guide](../quickstart.md) and the [export documentation](../modes/export.md).

### What are the benefits of using TensorRT for YOLOv8 models?

Using TensorRT to optimize YOLOv8 models offers several benefits:

- **Faster Inference Speed**: TensorRT optimizes the model layers and uses precision calibration (INT8 and FP16) to speed up inference without significantly sacrificing accuracy.
- **Memory Efficiency**: TensorRT manages tensor memory dynamically, reducing overhead and improving GPU memory utilization.
- **Layer Fusion**: Combines multiple layers into single operations, reducing computational complexity.
- **Kernel Auto-Tuning**: Automatically selects optimized GPU kernels for each model layer, ensuring maximum performance.

For more information, explore the detailed features of TensorRT [here](https://developer.nvidia.com/tensorrt) and read our [TensorRT overview section](#tensorrt).

### Can I use INT8 quantization with TensorRT for YOLOv8 models?

Yes, you can export YOLOv8 models using TensorRT with INT8 quantization. This process involves post-training quantization (PTQ) and calibration:

1. **Export with INT8**:

    ```python
    from ultralytics import YOLO

    model = YOLO("yolov8n.pt")
    model.export(format="engine", batch=8, workspace=4, int8=True, data="coco.yaml")
    ```

2. **Run inference**:

    ```python
    from ultralytics import YOLO

    model = YOLO("yolov8n.engine", task="detect")
    result = model.predict("https://ultralytics.com/images/bus.jpg")
    ```

For more details, refer to the [exporting TensorRT with INT8 quantization section](#exporting-tensorrt-with-int8-quantization).

### How do I deploy YOLOv8 TensorRT models on an NVIDIA Triton Inference Server?

Deploying YOLOv8 TensorRT models on an NVIDIA Triton Inference Server can be done using the following resources:

- **[Deploy Ultralytics YOLOv8 with Triton Server](../guides/triton-inference-server.md)**: Step-by-step guidance on setting up and using Triton Inference Server.
- **[NVIDIA Triton Inference Server Documentation](https://developer.nvidia.com/blog/deploying-deep-learning-nvidia-tensorrt/)**: Official NVIDIA documentation for detailed deployment options and configurations.

These guides will help you integrate YOLOv8 models efficiently in various deployment environments.

### What are the performance improvements observed with YOLOv8 models exported to TensorRT?

Performance improvements with TensorRT can vary based on the hardware used. Here are some typical benchmarks:

- **NVIDIA A100**:

    - **FP32** Inference: ~0.52 ms / image
    - **FP16** Inference: ~0.34 ms / image
    - **INT8** Inference: ~0.28 ms / image
    - Slight reduction in mAP with INT8 precision, but significant improvement in speed.

- **Consumer GPUs (e.g., RTX 3080)**:
    - **FP32** Inference: ~1.06 ms / image
    - **FP16** Inference: ~0.62 ms / image
    - **INT8** Inference: ~0.52 ms / image

Detailed performance benchmarks for different hardware configurations can be found in the [performance section](#ultralytics-yolo-tensorrt-export-performance).

For more comprehensive insights into TensorRT performance, refer to the [Ultralytics documentation](../modes/export.md) and our performance analysis reports.
