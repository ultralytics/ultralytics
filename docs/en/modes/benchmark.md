---
comments: true
description: Learn how to evaluate your YOLOv8 model's performance in real-world scenarios using benchmark mode. Optimize speed, accuracy, and resource allocation across export formats.
keywords: model benchmarking, YOLOv8, Ultralytics, performance evaluation, export formats, ONNX, TensorRT, OpenVINO, CoreML, TensorFlow, optimization, mAP50-95, inference time
---

# Model Benchmarking with Ultralytics YOLO

<img width="1024" src="https://github.com/ultralytics/assets/raw/main/yolov8/banner-integrations.png" alt="Ultralytics YOLO ecosystem and integrations">

## Introduction

Once your model is trained and validated, the next logical step is to evaluate its performance in various real-world scenarios. Benchmark mode in Ultralytics YOLOv8 serves this purpose by providing a robust framework for assessing the speed and accuracy of your model across a range of export formats.

<p align="center">
  <br>
  <iframe loading="lazy" width="720" height="405" src="https://www.youtube.com/embed/j8uQc0qB91s?start=105"
    title="YouTube video player" frameborder="0"
    allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share"
    allowfullscreen>
  </iframe>
  <br>
  <strong>Watch:</strong> Ultralytics Modes Tutorial: Benchmark
</p>

## Why Is Benchmarking Crucial?

- **Informed Decisions:** Gain insights into the trade-offs between speed and accuracy.
- **Resource Allocation:** Understand how different export formats perform on different hardware.
- **Optimization:** Learn which export format offers the best performance for your specific use case.
- **Cost Efficiency:** Make more efficient use of hardware resources based on benchmark results.

### Key Metrics in Benchmark Mode

- **mAP50-95:** For object detection, segmentation, and pose estimation.
- **accuracy_top5:** For image classification.
- **Inference Time:** Time taken for each image in milliseconds.

### Supported Export Formats

- **ONNX:** For optimal CPU performance
- **TensorRT:** For maximal GPU efficiency
- **OpenVINO:** For Intel hardware optimization
- **CoreML, TensorFlow SavedModel, and More:** For diverse deployment needs.

!!! Tip "Tip"

    * Export to ONNX or OpenVINO for up to 3x CPU speedup.
    * Export to TensorRT for up to 5x GPU speedup.

## Usage Examples

Run YOLOv8n benchmarks on all supported export formats including ONNX, TensorRT etc. See Arguments section below for a full list of export arguments.

!!! Example

    === "Python"

        ```python
        from ultralytics.utils.benchmarks import benchmark

        # Benchmark on GPU
        benchmark(model="yolov8n.pt", data="coco8.yaml", imgsz=640, half=False, device=0)
        ```

    === "CLI"

        ```bash
        yolo benchmark model=yolov8n.pt data='coco8.yaml' imgsz=640 half=False device=0
        ```

## Arguments

Arguments such as `model`, `data`, `imgsz`, `half`, `device`, and `verbose` provide users with the flexibility to fine-tune the benchmarks to their specific needs and compare the performance of different export formats with ease.

| Key       | Default Value | Description                                                                                                                                       |
| --------- | ------------- | ------------------------------------------------------------------------------------------------------------------------------------------------- |
| `model`   | `None`        | Specifies the path to the model file. Accepts both `.pt` and `.yaml` formats, e.g., `"yolov8n.pt"` for pre-trained models or configuration files. |
| `data`    | `None`        | Path to a YAML file defining the dataset for benchmarking, typically including paths and settings for validation data. Example: `"coco8.yaml"`.   |
| `imgsz`   | `640`         | The input image size for the model. Can be a single integer for square images or a tuple `(width, height)` for non-square, e.g., `(640, 480)`.    |
| `half`    | `False`       | Enables FP16 (half-precision) inference, reducing memory usage and possibly increasing speed on compatible hardware. Use `half=True` to enable.   |
| `int8`    | `False`       | Activates INT8 quantization for further optimized performance on supported devices, especially useful for edge devices. Set `int8=True` to use.   |
| `device`  | `None`        | Defines the computation device(s) for benchmarking, such as `"cpu"`, `"cuda:0"`, or a list of devices like `"cuda:0,1"` for multi-GPU setups.     |
| `verbose` | `False`       | Controls the level of detail in logging output. A boolean value; set `verbose=True` for detailed logs or a float for thresholding errors.         |

## Export Formats

Benchmarks will attempt to run automatically on all possible export formats below.

| Format                                            | `format` Argument | Model                     | Metadata | Arguments                                                            |
| ------------------------------------------------- | ----------------- | ------------------------- | -------- | -------------------------------------------------------------------- |
| [PyTorch](https://pytorch.org/)                   | -                 | `yolov8n.pt`              | ✅       | -                                                                    |
| [TorchScript](../integrations/torchscript.md)     | `torchscript`     | `yolov8n.torchscript`     | ✅       | `imgsz`, `optimize`, `batch`                                         |
| [ONNX](../integrations/onnx.md)                   | `onnx`            | `yolov8n.onnx`            | ✅       | `imgsz`, `half`, `dynamic`, `simplify`, `opset`, `batch`             |
| [OpenVINO](../integrations/openvino.md)           | `openvino`        | `yolov8n_openvino_model/` | ✅       | `imgsz`, `half`, `int8`, `batch`                                     |
| [TensorRT](../integrations/tensorrt.md)           | `engine`          | `yolov8n.engine`          | ✅       | `imgsz`, `half`, `dynamic`, `simplify`, `workspace`, `int8`, `batch` |
| [CoreML](../integrations/coreml.md)               | `coreml`          | `yolov8n.mlpackage`       | ✅       | `imgsz`, `half`, `int8`, `nms`, `batch`                              |
| [TF SavedModel](../integrations/tf-savedmodel.md) | `saved_model`     | `yolov8n_saved_model/`    | ✅       | `imgsz`, `keras`, `int8`, `batch`                                    |
| [TF GraphDef](../integrations/tf-graphdef.md)     | `pb`              | `yolov8n.pb`              | ❌       | `imgsz`, `batch`                                                     |
| [TF Lite](../integrations/tflite.md)              | `tflite`          | `yolov8n.tflite`          | ✅       | `imgsz`, `half`, `int8`, `batch`                                     |
| [TF Edge TPU](../integrations/edge-tpu.md)        | `edgetpu`         | `yolov8n_edgetpu.tflite`  | ✅       | `imgsz`                                                              |
| [TF.js](../integrations/tfjs.md)                  | `tfjs`            | `yolov8n_web_model/`      | ✅       | `imgsz`, `half`, `int8`, `batch`                                     |
| [PaddlePaddle](../integrations/paddlepaddle.md)   | `paddle`          | `yolov8n_paddle_model/`   | ✅       | `imgsz`, `batch`                                                     |
| [NCNN](../integrations/ncnn.md)                   | `ncnn`            | `yolov8n_ncnn_model/`     | ✅       | `imgsz`, `half`, `batch`                                             |

See full `export` details in the [Export](../modes/export.md) page.


## FAQ

### How do I benchmark my YOLOv8 model's performance on different export formats?

To benchmark your YOLOv8 model's performance across different export formats, you can use the `benchmark` function available in Ultralytics YOLOv8. Here’s a quick example:

#### Python
```python
from ultralytics.utils.benchmarks import benchmark

# Benchmark on GPU
benchmark(model="yolov8n.pt", data="coco8.yaml", imgsz=640, half=False, device=0)
```

#### CLI
```bash
yolo benchmark model=yolov8n.pt data='coco8.yaml' imgsz=640 half=False device=0
```
This function will measure key metrics like mAP50-95 and inference time across formats including ONNX, TensorRT, and others. Check the [Export](../modes/export.md) documentation for detailed arguments and usage.

### What are the key metrics used in YOLOv8 benchmarking, and why are they important?

Key metrics in YOLOv8 benchmarking include mAP50-95, accuracy_top5, and inference time:
- **mAP50-95**: Measures mean Average Precision across different IoU thresholds for object detection, segmentation, and pose estimation.
- **accuracy_top5**: For image classification, indicates the model's ability to correctly predict one of its top-5 categories.
- **Inference Time**: Measures the time taken for each image inference in milliseconds.

These metrics are crucial as they help evaluate the trade-offs between model speed and accuracy, enabling informed decisions on which export format best suits your use case.

### Why should I use TensorRT for exporting YOLOv8 models, and what benefits does it offer?

Exporting YOLOv8 models to TensorRT provides maximal GPU efficiency, significantly increasing inference speeds. TensorRT optimizes and compiles the model to leverage NVIDIA GPUs' full potential, which can offer up to 5x speedup. This is especially beneficial for real-time applications and large-scale deployment. For detailed export instructions, see the [TensorRT documentation](../integrations/tensorrt.md).

### How can I enable FP16 inference in YOLOv8 benchmarking and what are its advantages?

To enable FP16 (half-precision) inference in YOLOv8 benchmarking, set the `half` argument to `True`. For example:
```python
benchmark(model="yolov8n.pt", data="coco8.yaml", imgsz=640, half=True, device=0)
```
Using FP16 reduces memory usage and can increase inference speed, particularly on compatible hardware like NVIDIA GPUs. This is useful for optimizing performance on edge devices. Learn more about benchmark arguments in the [Arguments](#arguments) section.

### What export formats are supported by YOLOv8, and how do they differ?

YOLOv8 supports multiple export formats including ONNX, TensorRT, OpenVINO, CoreML, and TensorFlow SavedModel. Each format has unique advantages:
- **ONNX**: Optimal for cross-platform inference, particularly on CPUs.
- **TensorRT**: Best for NVIDIA GPU performance.
- **OpenVINO**: Ideal for Intel hardware acceleration.
- **CoreML**: Tailored for iOS/macOS devices.
- **TensorFlow SavedModel**: Flexible for various TensorFlow-based deployments.

Understanding these differences helps select the best format for your deployment environment. For a full list and export guidelines, refer to [Export Formats](../modes/export.md).