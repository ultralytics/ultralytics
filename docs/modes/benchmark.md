---
comments: true
description: Benchmark mode compares speed and accuracy of various YOLOv8 export formats like ONNX or OpenVINO. Optimize formats for speed or accuracy.
keywords: YOLOv8, Benchmark Mode, Export Formats, ONNX, OpenVINO, TensorRT, Ultralytics Docs
---

<img width="1024" src="https://github.com/ultralytics/assets/raw/main/yolov8/banner-integrations.png">

**Benchmark mode** is used to profile the speed and accuracy of various export formats for YOLOv8. The benchmarks
provide information on the size of the exported format, its `mAP50-95` metrics (for object detection, segmentation and pose)
or `accuracy_top5` metrics (for classification), and the inference time in milliseconds per image across various export
formats like ONNX, OpenVINO, TensorRT and others. This information can help users choose the optimal export format for
their specific use case based on their requirements for speed and accuracy.

!!! tip "Tip"

    * Export to ONNX or OpenVINO for up to 3x CPU speedup.
    * Export to TensorRT for up to 5x GPU speedup.

## Usage Examples

Run YOLOv8n benchmarks on all supported export formats including ONNX, TensorRT etc. See Arguments section below for a
full list of export arguments.

!!! example ""

    === "Python"
    
        ```python
        from ultralytics.yolo.utils.benchmarks import benchmark
        
        # Benchmark on GPU
        benchmark(model='yolov8n.pt', imgsz=640, half=False, device=0)
        ```
    === "CLI"
    
        ```bash
        yolo benchmark model=yolov8n.pt imgsz=640 half=False device=0
        ```

## Arguments

Arguments such as `model`, `imgsz`, `half`, `device`, and `hard_fail` provide users with the flexibility to fine-tune
the benchmarks to their specific needs and compare the performance of different export formats with ease.

| Key         | Value   | Description                                                          |
|-------------|---------|----------------------------------------------------------------------|
| `model`     | `None`  | path to model file, i.e. yolov8n.pt, yolov8n.yaml                    |
| `imgsz`     | `640`   | image size as scalar or (h, w) list, i.e. (640, 480)                 |
| `half`      | `False` | FP16 quantization                                                    |
| `int8`      | `False` | INT8 quantization                                                    |
| `device`    | `None`  | device to run on, i.e. cuda device=0 or device=0,1,2,3 or device=cpu |
| `hard_fail` | `False` | do not continue on error (bool), or val floor threshold (float)      |

## Export Formats

Benchmarks will attempt to run automatically on all possible export formats below.

| Format                                                             | `format` Argument | Model                     | Metadata |
|--------------------------------------------------------------------|-------------------|---------------------------|----------|
| [PyTorch](https://pytorch.org/)                                    | -                 | `yolov8n.pt`              | ✅        |
| [TorchScript](https://pytorch.org/docs/stable/jit.html)            | `torchscript`     | `yolov8n.torchscript`     | ✅        |
| [ONNX](https://onnx.ai/)                                           | `onnx`            | `yolov8n.onnx`            | ✅        |
| [OpenVINO](https://docs.openvino.ai/latest/index.html)             | `openvino`        | `yolov8n_openvino_model/` | ✅        |
| [TensorRT](https://developer.nvidia.com/tensorrt)                  | `engine`          | `yolov8n.engine`          | ✅        |
| [CoreML](https://github.com/apple/coremltools)                     | `coreml`          | `yolov8n.mlmodel`         | ✅        |
| [TF SavedModel](https://www.tensorflow.org/guide/saved_model)      | `saved_model`     | `yolov8n_saved_model/`    | ✅        |
| [TF GraphDef](https://www.tensorflow.org/api_docs/python/tf/Graph) | `pb`              | `yolov8n.pb`              | ❌        |
| [TF Lite](https://www.tensorflow.org/lite)                         | `tflite`          | `yolov8n.tflite`          | ✅        |
| [TF Edge TPU](https://coral.ai/docs/edgetpu/models-intro/)         | `edgetpu`         | `yolov8n_edgetpu.tflite`  | ✅        |
| [TF.js](https://www.tensorflow.org/js)                             | `tfjs`            | `yolov8n_web_model/`      | ✅        |
| [PaddlePaddle](https://github.com/PaddlePaddle)                    | `paddle`          | `yolov8n_paddle_model/`   | ✅        |

See full `export` details in the [Export](https://docs.ultralytics.com/modes/export/) page.