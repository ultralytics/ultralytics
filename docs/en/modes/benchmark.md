---
comments: true
description: Learn how to profile speed and accuracy of YOLOv8 across various export formats; get insights on mAP50-95, accuracy_top5 metrics, and more.
keywords: Ultralytics, YOLOv8, benchmarking, speed profiling, accuracy profiling, mAP50-95, accuracy_top5, ONNX, OpenVINO, TensorRT, YOLO export formats
---

# Model Benchmarking with Ultralytics YOLO

<img width="1024" src="https://github.com/ultralytics/assets/raw/main/yolov8/banner-integrations.png" alt="Ultralytics YOLO ecosystem and integrations">

## Introduction

Once your model is trained and validated, the next logical step is to evaluate its performance in various real-world scenarios. Benchmark mode in Ultralytics YOLOv8 serves this purpose by providing a robust framework for assessing the speed and accuracy of your model across a range of export formats.

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

!!! Example ""

    === "Python"

        ```python
        from ultralytics.utils.benchmarks import benchmark

        # Benchmark on GPU
        benchmark(model='yolov8n.pt', data='coco8.yaml', imgsz=640, half=False, device=0)
        ```
    === "CLI"

        ```bash
        yolo benchmark model=yolov8n.pt data='coco8.yaml' imgsz=640 half=False device=0
        ```

## Arguments

Arguments such as `model`, `data`, `imgsz`, `half`, `device`, and `verbose` provide users with the flexibility to fine-tune the benchmarks to their specific needs and compare the performance of different export formats with ease.

| Key       | Value   | Description                                                           |
|-----------|---------|-----------------------------------------------------------------------|
| `model`   | `None`  | path to model file, i.e. yolov8n.pt, yolov8n.yaml                     |
| `data`    | `None`  | path to YAML referencing the benchmarking dataset (under `val` label) |
| `imgsz`   | `640`   | image size as scalar or (h, w) list, i.e. (640, 480)                  |
| `half`    | `False` | FP16 quantization                                                     |
| `int8`    | `False` | INT8 quantization                                                     |
| `device`  | `None`  | device to run on, i.e. cuda device=0 or device=0,1,2,3 or device=cpu  |
| `verbose` | `False` | do not continue on error (bool), or val floor threshold (float)       |

## Export Formats

Benchmarks will attempt to run automatically on all possible export formats below.

| Format                                                             | `format` Argument | Model                     | Metadata | Arguments                                           |
|--------------------------------------------------------------------|-------------------|---------------------------|----------|-----------------------------------------------------|
| [PyTorch](https://pytorch.org/)                                    | -                 | `yolov8n.pt`              | ✅        | -                                                   |
| [TorchScript](https://pytorch.org/docs/stable/jit.html)            | `torchscript`     | `yolov8n.torchscript`     | ✅        | `imgsz`, `optimize`                                 |
| [ONNX](https://onnx.ai/)                                           | `onnx`            | `yolov8n.onnx`            | ✅        | `imgsz`, `half`, `dynamic`, `simplify`, `opset`     |
| [OpenVINO](https://docs.openvino.ai/latest/index.html)             | `openvino`        | `yolov8n_openvino_model/` | ✅        | `imgsz`, `half`, `int8`                             |
| [TensorRT](https://developer.nvidia.com/tensorrt)                  | `engine`          | `yolov8n.engine`          | ✅        | `imgsz`, `half`, `dynamic`, `simplify`, `workspace` |
| [CoreML](https://github.com/apple/coremltools)                     | `coreml`          | `yolov8n.mlpackage`       | ✅        | `imgsz`, `half`, `int8`, `nms`                      |
| [TF SavedModel](https://www.tensorflow.org/guide/saved_model)      | `saved_model`     | `yolov8n_saved_model/`    | ✅        | `imgsz`, `keras`, `int8`                            |
| [TF GraphDef](https://www.tensorflow.org/api_docs/python/tf/Graph) | `pb`              | `yolov8n.pb`              | ❌        | `imgsz`                                             |
| [TF Lite](https://www.tensorflow.org/lite)                         | `tflite`          | `yolov8n.tflite`          | ✅        | `imgsz`, `half`, `int8`                             |
| [TF Edge TPU](https://coral.ai/docs/edgetpu/models-intro/)         | `edgetpu`         | `yolov8n_edgetpu.tflite`  | ✅        | `imgsz`                                             |
| [TF.js](https://www.tensorflow.org/js)                             | `tfjs`            | `yolov8n_web_model/`      | ✅        | `imgsz`                                             |
| [PaddlePaddle](https://github.com/PaddlePaddle)                    | `paddle`          | `yolov8n_paddle_model/`   | ✅        | `imgsz`                                             |
| [ncnn](https://github.com/Tencent/ncnn)                            | `ncnn`            | `yolov8n_ncnn_model/`     | ✅        | `imgsz`, `half`                                     |

See full `export` details in the [Export](https://docs.ultralytics.com/modes/export/) page.
