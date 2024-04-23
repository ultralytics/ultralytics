---
comments: true
description: Discover the power and flexibility of exporting Ultralytics YOLOv8 models to TensorRT format for enhanced performance and efficiency on NVIDIA GPUs.
keywords: Ultralytics, YOLOv8, TensorRT Export, Model Deployment, GPU Acceleration, NVIDIA Support, CUDA Deployment
---

# TensorRT Export for YOLOv8 Models

Deploying computer vision models in high-performance environments can require a format that maximizes speed and efficiency. This is especially true when you are deploying your model on NVIDIA GPUs.

By using the TensorRT export format, you can enhance your [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics) models for swift and efficient inference on NVIDIA hardware. This guide will give you easy-to-follow steps for the conversion process and help you make the most of NVIDIA's advanced technology in your deep learning projects.

## TensorRT

<p align="center">
  <img width="100%" src="https://docs.nvidia.com/deeplearning/tensorrt/archives/tensorrt-601/tensorrt-developer-guide/graphics/whatistrt2.png" alt="TensorRT Overview">
</p>

[TensorRT](https://developer.nvidia.com/tensorrt), developed by NVIDIA, is an advanced software development kit (SDK) designed for high-speed deep learning inference. It’s well-suited for real-time applications like object detection.

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

Before we look at the code for exporting YOLOv8 models to the TensorRT format, let’s understand where TensorRT models are normally used.

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
        model = YOLO('yolov8n.pt')

        # Export the model to TensorRT format
        model.export(format='engine')  # creates 'yolov8n.engine'

        # Load the exported TensorRT model
        tensorrt_model = YOLO('yolov8n.engine')

        # Run inference
        results = tensorrt_model('https://ultralytics.com/images/bus.jpg')
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

Exporting Ultralytics YOLO models using TensorRT with `INT8` precision executes post-training quantization (PTQ). TensorRT uses calibration for PTQ, which measures the distribution of activations within each activation tensor as the YOLO model processes inference on representative input data, and then uses that distribution to estimate scale values for each tensor. Each activation tensor that is a candidate for quantization has an associated scale that is deduced by a calibration process. 

When processing implicitly quantized networks TensorRT uses `INT8` opportunistically to optimize layer execution time. If a layer runs faster in `INT8` and has assigned quantization scales on its data inputs and outputs, then a kernel with `INT8` precision is assigned to that layer, otherwise TensorRT selects a precision of either `FP32` or `FP16` for the kernel based on whichever results in faster execution time for that layer.

!!! tip

    It is **critical** to ensure that the same device that will use the TensorRT model weights for deployment is used for exporting with `INT8` precision, as the calibration results can vary across devices.

#### Configuring INT8 Export

The arguments provided when using [export](../modes/export.md) for an Ultralytics YOLO model will **greatly** influence the performance of the exported model. They will also need to be selected based on the device resources available, however the default arguments _should_ work for most [Ampere (or newer) Nvidia discrete GPUs](https://developer.nvidia.com/blog/nvidia-ampere-architecture-in-depth/).

  - `workspace` : Controls the size (in GiB) of the device memory allocation while converting the model weights.
    
    - Aim to use the <u>minimum</u> `workspace` value required as this prevents testing algorithms that require more `workspace` from being considered by the TensorRT builder. Setting a higher value for `workspace` may take **considerably longer** to calibrate and export.

    - Default is `workspace=4` (GiB), this value may need to be increased if calibration crashes (exits without warning).
    
    - TensorRT will report `UNSUPPORTED_STATE` during export if the value for `workspace` is larger than the memory available to the device, which means the value for `workspace` should be lowered.
    
    - If `workspace` is set to max value and calibration fails/crashes, consider reducing the values for `imgsz` and `batch` to reduce memory requirements.

    - <u><b>Remember</b> calibration for `INT8` is specific to each device</u>, borrowing a "high-end" GPU for calibration, might result in poor performance when inference is run on another device.

  - `trt_quant_algo` : Specifies the algorithm to use for `INT8` quantization. The default is value is `trt_quant_algo="ENTROPY_CALIBRATION_2"` and you can read more details about the options available [in the TensorRT Developer Guide](https://docs.nvidia.com/deeplearning/tensorrt/developer-guide/index.html#enable_int8_c).

    ??? info "Available Algorithms for TensorRT INT8 Calibration"

        - `LEGACY_CALIBRATION`
  
        - `ENTROPY_CALIBRATION`
  
        - `ENTROPY_CALIBRATION_2`
  
        - `MINMAX_CALIBRATION`

  - `batch` : The maximum batch-size that will be used for inference. During inference smaller batches can be used but will not accept batches any larger than what is specified.
    
    !!! note

        During calibration, twice the `batch` size provided will be used. Using small batches can lead to inaccurate scaling during calibration. This is because the process adjusts based on the data it sees. Small batches might not capture the full range of values, leading to issues with the final calibration, so the `batch` size is doubled automatically. If no batch size is specified `batch=1`, `batch` will instead be set to `batch=4` and calibration will be run at `4 × 2` to reduce calibration scaling errors.

Experimentation by Nvidia led them to recommend using at least 500 calibration images that are representative of the data for your model, with `INT8` quantization calibration. This is a guideline and not a _hard_ requirement, and <u>**you will need to experiment with what is required to perform well for your dataset**.</u> Since the calibration data is required for `INT8` calibration with TensorRT, make certain to use the `data` argument when `int8=True` for TensorRT and use `data="my_dataset.yaml"`, which will use the images from [validation](../modes/val.md) to calibrate with. When no value is passed for `data` with export to TensorRT with `INT8` quantization, the default will be to use `coco128.yaml` instead of throwing an error.

!!! example

    ```{ .py .annotate }
    from ultralytics import YOLO

    model = YOLO("yolov8n.pt")
    model.export(
        format="engine",
        dynamic=True, #(1)!
        batch=8, #(2)!
        workspace=4, #(3)!
        int8=True,
        data="coco.yaml", #(4)!
    )

    model = YOLO("yolov8n.engine", task="detect") # load the model
    
    ```
    
    1. Exports with dynamic axes, this will be enabled by default when exporting with `int8=True` even when not explicitly set. See [export arguments](../modes/export.md#arguments) for additional information.
    2. Sets max batch size of 8 for exported model, with calibrate with `2 × 8` to avoid scaling errors during calibration
    3. Allocates 4 GiB of memory instead of allocating the entire device for conversion process.
    4. Uses [COCO dataset](../datasets/detect/coco.md) for calibration, specifically the images used for [validation](../modes/val.md) (5,000 total).

???+ warning "Calibration Cache"

    TensorRT will generate a calibration `.cache` which can be re-used to speed up export of future model weights using the same data, but this may result in poor calibration when the data is vastly different or if the `batch` value is changed drastically. In these circumstances, the existing `.cache` should be renamed and moved to a different directory or deleted entirely.

#### Advantages of using YOLO with TensorRT INT8

- **Reduced model size:** Quantization from `FP32` to `INT8` can reduce the model size by 4x (on disk or in memory), leading to faster download times. lower storage requirements, and reduced memory footprint when deploying a model.

- **Lower power consumption:** Reduced precision operations (INT8) can consume less power compared to FP32 calculations, especially on battery-powered devices.

- **Improved inference speeds:** TensorRT optimizes the model for the target hardware, potentially leading to faster inference speeds on GPUs, embedded devices, and accelerators.

??? note "Note on Inference Speeds"

    The first few inference calls with a model exported to TensorRT INT8 can be expected to have longer than usual preprocessing, inference, and/or postprocessing times. This may also occur when changing `imgsz` during inference, especially when `imgsz` is not the same as what was specified during export (export `imgsz` is set as TensorRT "optimal" profile).

#### Advantages of using YOLO with TensorRT INT8

- **Decreases in evaluation metrics:** Using a lower precision will mean that `mAP`, `Precision`, `Recall` or any [other metric used to evaluate model performance](../guides/yolo-performance-metrics.md) is likely to be somewhat worse.

- **Increased development times:** Finding the "optimal" settings for `INT8` calibration for dataset and device can take a significant amount of testing.

- Calibration and performance gains could be highly hardware dependent and model weights are less transferrable.

## Deploying Exported YOLOv8 TensorRT Models

Having successfully exported your Ultralytics YOLOv8 models to TensorRT format, you're now ready to deploy them. For in-depth instructions on deploying your TensorRT models in various settings, take a look at the following resources:

- **[Deploying Deep Neural Networks with NVIDIA TensorRT](https://developer.nvidia.com/blog/deploying-deep-learning-nvidia-tensorrt/)**: This article explains how to use NVIDIA TensorRT to deploy deep neural networks on GPU-based deployment platforms efficiently.

- **[End-to-End AI for NVIDIA-Based PCs: NVIDIA TensorRT Deployment](https://developer.nvidia.com/blog/end-to-end-ai-for-nvidia-based-pcs-nvidia-tensorrt-deployment/)**: This blog post explains the use of NVIDIA TensorRT for optimizing and deploying AI models on NVIDIA-based PCs.

- **[GitHub Repository for NVIDIA TensorRT:](https://github.com/NVIDIA/TensorRT)**: This is the official GitHub repository that contains the source code and documentation for NVIDIA TensorRT.

## Summary

In this guide, we focused on converting Ultralytics YOLOv8 models to NVIDIA's TensorRT model format. This conversion step is crucial for improving the efficiency and speed of YOLOv8 models, making them more effective and suitable for diverse deployment environments.

For more information on usage details, take a look at the [TensorRT official documentation](https://docs.nvidia.com/deeplearning/tensorrt/).

If you're curious about additional Ultralytics YOLOv8 integrations, our [integration guide page](../integrations/index.md) provides an extensive selection of informative resources and insights.
