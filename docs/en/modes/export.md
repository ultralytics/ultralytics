---
comments: true
description: Learn how to export your YOLO26 model to various formats like ONNX, TensorRT, and CoreML. Achieve maximum compatibility and performance.
keywords: YOLO26, Model Export, ONNX, TensorRT, CoreML, Ultralytics, AI, Machine Learning, Inference, Deployment
---

# Model Export with Ultralytics YOLO

<img width="1024" src="https://cdn.jsdelivr.net/gh/ultralytics/assets@main/docs/ultralytics-yolov8-ecosystem-integrations.avif" alt="Ultralytics YOLO ecosystem and integrations">

## Introduction

The ultimate goal of training a model is to deploy it for real-world applications. Export mode in Ultralytics YOLO26 offers a versatile range of options for exporting your trained model to different formats, making it deployable across various platforms and devices. This comprehensive guide aims to walk you through the nuances of model exporting, showcasing how to achieve maximum compatibility and performance.

<p align="center">
  <br>
  <iframe loading="lazy" width="720" height="405" src="https://www.youtube.com/embed/KGHYU-MKYeE"
    title="YouTube video player" frameborder="0"
    allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share"
    allowfullscreen>
  </iframe>
  <br>
  <strong>Watch:</strong> How to Export Ultralytics YOLO26 in different formats for Deployment | ONNX, TensorRT, CoreML 🚀
</p>

## Why Choose YOLO26's Export Mode?

- **Versatility:** Export to multiple formats including [ONNX](../integrations/onnx.md), [TensorRT](../integrations/tensorrt.md), [CoreML](../integrations/coreml.md), and more.
- **Performance:** Gain up to 5x GPU speedup with TensorRT and 3x CPU speedup with ONNX or [OpenVINO](../integrations/openvino.md).
- **Compatibility:** Make your model universally deployable across numerous hardware and software environments.
- **Ease of Use:** Simple CLI and Python API for quick and straightforward model exporting.

### Key Features of Export Mode

Here are some of the standout functionalities:

- **One-Click Export:** Simple commands for exporting to different formats.
- **Batch Export:** Export batched-inference capable models.
- **Optimized Inference:** Exported models are optimized for quicker inference times.
- **Tutorial Videos:** In-depth guides and tutorials for a smooth exporting experience.

!!! tip

    * Export to [ONNX](../integrations/onnx.md) or [OpenVINO](../integrations/openvino.md) for up to 3x CPU speedup.
    * Export to [TensorRT](../integrations/tensorrt.md) for up to 5x GPU speedup.

## Usage Examples

Export a YOLO26n model to a different format like ONNX or TensorRT. See the Arguments section below for a full list of export arguments.

!!! example

    === "Python"

        ```python
        from ultralytics import YOLO

        # Load a model
        model = YOLO("yolo26n.pt")  # load an official model
        model = YOLO("path/to/best.pt")  # load a custom-trained model

        # Export the model
        model.export(format="onnx")
        ```

    === "CLI"

        ```bash
        yolo export model=yolo26n.pt format=onnx      # export official model
        yolo export model=path/to/best.pt format=onnx # export custom-trained model
        ```

## Arguments

This table details the configurations and options available for exporting YOLO models to different formats. These settings are critical for optimizing the exported model's performance, size, and compatibility across various platforms and environments. Proper configuration ensures that the model is ready for deployment in the intended application with optimal efficiency.

{% include "macros/export-args.md" %}

Adjusting these parameters allows for customization of the export process to fit specific requirements, such as deployment environment, hardware constraints, and performance targets. Selecting the appropriate format and settings is essential for achieving the best balance between model size, speed, and [accuracy](https://www.ultralytics.com/glossary/accuracy).

## Export Formats

Available YOLO26 export formats are in the table below. You can export to any format using the `format` argument, i.e., `format='onnx'` or `format='engine'`. You can predict or validate directly on exported models, i.e., `yolo predict model=yolo26n.onnx`. Usage examples are shown for your model after export completes.

{% include "macros/export-table.md" %}

## FAQ

### How do I export a YOLO26 model to ONNX format?

Exporting a YOLO26 model to ONNX format is straightforward with Ultralytics. It provides both Python and CLI methods for exporting models.

!!! example

    === "Python"

        ```python
        from ultralytics import YOLO

        # Load a model
        model = YOLO("yolo26n.pt")  # load an official model
        model = YOLO("path/to/best.pt")  # load a custom-trained model

        # Export the model
        model.export(format="onnx")
        ```

    === "CLI"

        ```bash
        yolo export model=yolo26n.pt format=onnx      # export official model
        yolo export model=path/to/best.pt format=onnx # export custom-trained model
        ```

For more details on the process, including advanced options like handling different input sizes, refer to the [ONNX integration guide](../integrations/onnx.md).

### What are the benefits of using TensorRT for model export?

Using TensorRT for model export offers significant performance improvements. YOLO26 models exported to TensorRT can achieve up to a 5x GPU speedup, making it ideal for real-time inference applications.

- **Versatility:** Optimize models for a specific hardware setup.
- **Speed:** Achieve faster inference through advanced optimizations.
- **Compatibility:** Integrate smoothly with NVIDIA hardware.

To learn more about integrating TensorRT, see the [TensorRT integration guide](../integrations/tensorrt.md).

### How do I enable INT8 quantization when exporting my YOLO26 model?

INT8 quantization is an excellent way to compress the model and speed up inference, especially on edge devices. Here's how you can enable INT8 quantization:

!!! example

    === "Python"

        ```python
        from ultralytics import YOLO

        model = YOLO("yolo26n.pt")  # Load a model
        model.export(format="engine", int8=True)
        ```

    === "CLI"

        ```bash
        yolo export model=yolo26n.pt format=engine int8=True # export TensorRT model with INT8 quantization
        ```

INT8 quantization can be applied to various formats, such as [TensorRT](../integrations/tensorrt.md), [OpenVINO](../integrations/openvino.md), and [CoreML](../integrations/coreml.md). For optimal quantization results, provide a representative [dataset](https://docs.ultralytics.com/datasets/) using the `data` parameter.

### Why is dynamic input size important when exporting models?

Dynamic input size allows the exported model to handle varying image dimensions, providing flexibility and optimizing processing efficiency for different use cases. When exporting to formats like [ONNX](../integrations/onnx.md) or [TensorRT](../integrations/tensorrt.md), enabling dynamic input size ensures that the model can adapt to different input shapes seamlessly.

To enable this feature, use the `dynamic=True` flag during export:

!!! example

    === "Python"

        ```python
        from ultralytics import YOLO

        model = YOLO("yolo26n.pt")
        model.export(format="onnx", dynamic=True)
        ```

    === "CLI"

        ```bash
        yolo export model=yolo26n.pt format=onnx dynamic=True
        ```

Dynamic input sizing is particularly useful for applications where input dimensions may vary, such as video processing or when handling images from different sources.

### What are the key export arguments to consider for optimizing model performance?

Understanding and configuring export arguments is crucial for optimizing model performance:

- **`format:`** The target format for the exported model (e.g., `onnx`, `torchscript`, `tensorflow`).
- **`imgsz:`** Desired image size for the model input (e.g., `640` or `(height, width)`).
- **`half:`** Enables FP16 quantization, reducing model size and potentially speeding up inference.
- **`optimize:`** Applies specific optimizations for mobile or constrained environments.
- **`int8:`** Enables INT8 quantization, highly beneficial for [edge AI](https://www.ultralytics.com/blog/deploying-computer-vision-applications-on-edge-ai-devices) deployments.

For deployment on specific hardware platforms, consider using specialized export formats like [TensorRT](../integrations/tensorrt.md) for NVIDIA GPUs, [CoreML](../integrations/coreml.md) for Apple devices, or [Edge TPU](../integrations/edge-tpu.md) for Google Coral devices.

### What do the output tensors represent in exported YOLO models?

When you export a YOLO model to formats like ONNX or TensorRT, the output tensor structure depends on the model task. Understanding these outputs is important for custom inference implementations.

For **detection models** (e.g., `yolo26n.pt`), the output is typically a single tensor shaped like `(batch_size, 4 + num_classes, num_predictions)` where the channels represent box coordinates plus per-class scores, and `num_predictions` depends on the export input resolution (and can be dynamic).

For **segmentation models** (e.g., `yolo26n-seg.pt`), you'll typically get two outputs: the first tensor shaped like `(batch_size, 4 + num_classes + mask_dim, num_predictions)` (boxes, class scores, and mask coefficients), and the second tensor shaped like `(batch_size, mask_dim, proto_h, proto_w)` containing mask prototypes used with the coefficients to generate instance masks. Sizes depend on the export input resolution (and can be dynamic).

For **pose models** (e.g., `yolo26n-pose.pt`), the output tensor is typically shaped like `(batch_size, 4 + num_classes + keypoint_dims, num_predictions)`, where `keypoint_dims` depends on the pose specification (e.g., number of keypoints and whether confidence is included), and `num_predictions` depends on the export input resolution (and can be dynamic).

The examples in the [ONNX inference examples](https://github.com/ultralytics/ultralytics/tree/main/examples) demonstrate how to process these outputs for each model type.
