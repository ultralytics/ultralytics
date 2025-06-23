---
comments: true
description: Learn to export YOLO11 models to OpenVINO format for up to 3x CPU speedup and hardware acceleration on Intel GPU and NPU.
keywords: YOLO11, OpenVINO, model export, Intel, AI inference, CPU speedup, GPU acceleration, NPU, deep learning
---

# Intel OpenVINO Export

<img width="1024" src="https://github.com/ultralytics/docs/releases/download/0/openvino-ecosystem.avif" alt="OpenVINO Ecosystem">

In this guide, we cover exporting YOLO11 models to the [OpenVINO](https://docs.openvino.ai/) format, which can provide up to 3x [CPU](https://docs.openvino.ai/2024/openvino-workflow/running-inference/inference-devices-and-modes/cpu-device.html) speedup, as well as accelerating YOLO inference on Intel [GPU](https://docs.openvino.ai/2024/openvino-workflow/running-inference/inference-devices-and-modes/gpu-device.html) and [NPU](https://docs.openvino.ai/2024/openvino-workflow/running-inference/inference-devices-and-modes/npu-device.html) hardware.

OpenVINO, short for Open Visual Inference & [Neural Network](https://www.ultralytics.com/glossary/neural-network-nn) Optimization toolkit, is a comprehensive toolkit for optimizing and deploying AI inference models. Even though the name contains Visual, OpenVINO also supports various additional tasks including language, audio, time series, etc.

<p align="center">
  <br>
  <iframe loading="lazy" width="720" height="405" src="https://www.youtube.com/embed/kONm9nE5_Fk?si=kzquuBrxjSbntHoU"
    title="YouTube video player" frameborder="0"
    allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share"
    allowfullscreen>
  </iframe>
  <br>
  <strong>Watch:</strong> How To Export and Optimize an Ultralytics YOLOv8 Model for Inference with OpenVINO.
</p>

## Usage Examples

Export a YOLO11n model to OpenVINO format and run inference with the exported model.

!!! example

    === "Python"

        ```python
        from ultralytics import YOLO

        # Load a YOLO11n PyTorch model
        model = YOLO("yolo11n.pt")

        # Export the model
        model.export(format="openvino")  # creates 'yolo11n_openvino_model/'

        # Load the exported OpenVINO model
        ov_model = YOLO("yolo11n_openvino_model/")

        # Run inference
        results = ov_model("https://ultralytics.com/images/bus.jpg")

        # Run inference with specified device, available devices: ["intel:gpu", "intel:npu", "intel:cpu"]
        results = ov_model("https://ultralytics.com/images/bus.jpg", device="intel:gpu")
        ```

    === "CLI"

        ```bash
        # Export a YOLO11n PyTorch model to OpenVINO format
        yolo export model=yolo11n.pt format=openvino # creates 'yolo11n_openvino_model/'

        # Run inference with the exported model
        yolo predict model=yolo11n_openvino_model source='https://ultralytics.com/images/bus.jpg'

        # Run inference with specified device, available devices: ["intel:gpu", "intel:npu", "intel:cpu"]
        yolo predict model=yolo11n_openvino_model source='https://ultralytics.com/images/bus.jpg' device="intel:gpu"
        ```

## Export Arguments

| Argument   | Type             | Default        | Description                                                                                                                                                                                                                                                      |
| ---------- | ---------------- | -------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `format`   | `str`            | `'openvino'`   | Target format for the exported model, defining compatibility with various deployment environments.                                                                                                                                                               |
| `imgsz`    | `int` or `tuple` | `640`          | Desired image size for the model input. Can be an integer for square images or a tuple `(height, width)` for specific dimensions.                                                                                                                                |
| `half`     | `bool`           | `False`        | Enables FP16 (half-precision) quantization, reducing model size and potentially speeding up inference on supported hardware.                                                                                                                                     |
| `int8`     | `bool`           | `False`        | Activates INT8 quantization, further compressing the model and speeding up inference with minimal [accuracy](https://www.ultralytics.com/glossary/accuracy) loss, primarily for edge devices.                                                                    |
| `dynamic`  | `bool`           | `False`        | Allows dynamic input sizes, enhancing flexibility in handling varying image dimensions.                                                                                                                                                                          |
| `nms`      | `bool`           | `False`        | Adds Non-Maximum Suppression (NMS), essential for accurate and efficient detection post-processing.                                                                                                                                                              |
| `batch`    | `int`            | `1`            | Specifies export model batch inference size or the max number of images the exported model will process concurrently in `predict` mode.                                                                                                                          |
| `data`     | `str`            | `'coco8.yaml'` | Path to the [dataset](https://docs.ultralytics.com/datasets/) configuration file (default: `coco8.yaml`), essential for quantization.                                                                                                                            |
| `fraction` | `float`          | `1.0`          | Specifies the fraction of the dataset to use for INT8 quantization calibration. Allows for calibrating on a subset of the full dataset, useful for experiments or when resources are limited. If not specified with INT8 enabled, the full dataset will be used. |

For more details about the export process, visit the [Ultralytics documentation page on exporting](../modes/export.md).

!!! warning

    OpenVINO™ is compatible with most Intel® processors but to ensure optimal performance:

    1. Verify OpenVINO™ support
        Check whether your Intel® chip is officially supported by OpenVINO™ using [Intel's compatibility list](https://docs.openvino.ai/2025/about-openvino/release-notes-openvino/system-requirements.html).

    2. Identify your accelerator
        Determine if your processor includes an integrated NPU (Neural Processing Unit) or GPU (integrated GPU) by consulting [Intel's hardware guide](https://www.intel.com/content/www/us/en/support/articles/000097597/processors.html).

    3. Install the latest drivers
        If your chip supports an NPU or GPU but OpenVINO™ isn't detecting it, you may need to install or update the associated drivers. Follow the [driver‑installation instructions](https://medium.com/openvino-toolkit/how-to-run-openvino-on-a-linux-ai-pc-52083ce14a98) to enable full acceleration.

    By following these three steps, you can ensure OpenVINO™ runs optimally on your Intel® hardware.

## Benefits of OpenVINO

1. **Performance**: OpenVINO delivers high-performance inference by utilizing the power of Intel CPUs, integrated and discrete GPUs, and FPGAs.
2. **Support for Heterogeneous Execution**: OpenVINO provides an API to write once and deploy on any supported Intel hardware (CPU, GPU, FPGA, VPU, etc.).
3. **Model Optimizer**: OpenVINO provides a Model Optimizer that imports, converts, and optimizes models from popular [deep learning](https://www.ultralytics.com/glossary/deep-learning-dl) frameworks such as PyTorch, [TensorFlow](https://www.ultralytics.com/glossary/tensorflow), TensorFlow Lite, Keras, ONNX, PaddlePaddle, and Caffe.
4. **Ease of Use**: The toolkit comes with more than [80 tutorial notebooks](https://github.com/openvinotoolkit/openvino_notebooks) (including [YOLOv8 optimization](https://github.com/openvinotoolkit/openvino_notebooks/tree/latest/notebooks/yolov8-optimization)) teaching different aspects of the toolkit.

## OpenVINO Export Structure

When you export a model to OpenVINO format, it results in a directory containing the following:

1. **XML file**: Describes the network topology.
2. **BIN file**: Contains the weights and biases binary data.
3. **Mapping file**: Holds mapping of original model output tensors to OpenVINO tensor names.

You can use these files to run inference with the OpenVINO Inference Engine.

## Using OpenVINO Export in Deployment

Once your model is successfully exported to the OpenVINO format, you have two primary options for running inference:

1. Use the `ultralytics` package, which provides a high-level API and wraps the OpenVINO Runtime.

2. Use the native `openvino` package for more advanced or customized control over inference behavior.

### Inference with Ultralytics

The ultralytics package allows you to easily run inference using the exported OpenVINO model via the predict method. You can also specify the target device (e.g., `intel:gpu`, `intel:npu`, `intel:cpu`) using the device argument.

```python
from ultralytics import YOLO

# Load the exported OpenVINO model
ov_model = YOLO("yolo11n_openvino_model/")  # the path of your exported OpenVINO model
# Run inference with the exported model
ov_model.predict(device="intel:gpu")  # specify the device you want to run inference on
```

This approach is ideal for fast prototyping or deployment when you don't need full control over the inference pipeline.

### Inference with OpenVINO Runtime

The openvino Runtime provides a unified API to inference across all supported Intel hardware. It also provides advanced capabilities like load balancing across Intel hardware and asynchronous execution. For more information on running the inference, refer to the [YOLO11 notebooks](https://github.com/openvinotoolkit/openvino_notebooks/tree/latest/notebooks/yolov11-optimization).

Remember, you'll need the XML and BIN files as well as any application-specific settings like input size, scale factor for normalization, etc., to correctly set up and use the model with the Runtime.

In your deployment application, you would typically do the following steps:

1. Initialize OpenVINO by creating `core = Core()`.
2. Load the model using the `core.read_model()` method.
3. Compile the model using the `core.compile_model()` function.
4. Prepare the input (image, text, audio, etc.).
5. Run inference using `compiled_model(input_data)`.

For more detailed steps and code snippets, refer to the [OpenVINO documentation](https://docs.openvino.ai/) or [API tutorial](https://github.com/openvinotoolkit/openvino_notebooks/blob/latest/notebooks/openvino-api/openvino-api.ipynb).

## OpenVINO YOLO11 Benchmarks

The Ultralytics team benchmarked YOLO11 across various model formats and [precision](https://www.ultralytics.com/glossary/precision), evaluating speed and accuracy on different Intel devices compatible with OpenVINO.

!!! note

    The benchmarking results below are for reference and might vary based on the exact hardware and software configuration of a system, as well as the current workload of the system at the time the benchmarks are run.

    All benchmarks run with `openvino` Python package version [2025.1.0](https://pypi.org/project/openvino/2025.1.0/).

### Intel Core CPU

The Intel® Core® series is a range of high-performance processors by Intel. The lineup includes Core i3 (entry-level), Core i5 (mid-range), Core i7 (high-end), and Core i9 (extreme performance). Each series caters to different computing needs and budgets, from everyday tasks to demanding professional workloads. With each new generation, improvements are made to performance, energy efficiency, and features.

Benchmarks below run on 12th Gen Intel® Core® i9-12900KS CPU at FP32 precision.

<div align="center">
<img width="800" src="https://github.com/ultralytics/docs/releases/download/0/openvino-corei9.avif" alt="Core CPU benchmarks">
</div>

??? abstract "Detailed Benchmark Results"

    | Model   | Format      | Status | Size (MB) | metrics/mAP50-95(B) | Inference time (ms/im) |
    | ------- | ----------- | ------ | --------- | ------------------- | ---------------------- |
    | YOLO11n | PyTorch     | ✅     | 5.4       | 0.5071              | 21.00                  |
    | YOLO11n | TorchScript | ✅     | 10.5      | 0.5077              | 21.39                  |
    | YOLO11n | ONNX        | ✅     | 10.2      | 0.5077              | 15.55                  |
    | YOLO11n | OpenVINO    | ✅     | 10.4      | 0.5077              | 11.49                  |
    | YOLO11s | PyTorch     | ✅     | 18.4      | 0.5770              | 43.16                  |
    | YOLO11s | TorchScript | ✅     | 36.6      | 0.5781              | 50.06                  |
    | YOLO11s | ONNX        | ✅     | 36.3      | 0.5781              | 31.53                  |
    | YOLO11s | OpenVINO    | ✅     | 36.4      | 0.5781              | 30.82                  |
    | YOLO11m | PyTorch     | ✅     | 38.8      | 0.6257              | 110.60                 |
    | YOLO11m | TorchScript | ✅     | 77.3      | 0.6306              | 128.09                 |
    | YOLO11m | ONNX        | ✅     | 76.9      | 0.6306              | 76.06                  |
    | YOLO11m | OpenVINO    | ✅     | 77.1      | 0.6306              | 79.38                  |
    | YOLO11l | PyTorch     | ✅     | 49.0      | 0.6367              | 150.38                 |
    | YOLO11l | TorchScript | ✅     | 97.7      | 0.6408              | 172.57                 |
    | YOLO11l | ONNX        | ✅     | 97.0      | 0.6408              | 108.91                 |
    | YOLO11l | OpenVINO    | ✅     | 97.3      | 0.6408              | 102.30                 |
    | YOLO11x | PyTorch     | ✅     | 109.3     | 0.6989              | 272.72                 |
    | YOLO11x | TorchScript | ✅     | 218.1     | 0.6900              | 320.86                 |
    | YOLO11x | ONNX        | ✅     | 217.5     | 0.6900              | 196.20                 |
    | YOLO11x | OpenVINO    | ✅     | 217.8     | 0.6900              | 195.32                 |

### Intel® Core™ Ultra

The Intel® Core™ Ultra™ series represents a new benchmark in high-performance computing, engineered to meet the evolving demands of modern users—from gamers and creators to professionals leveraging AI. This next-generation lineup is more than a traditional CPU series; it combines powerful CPU cores, integrated high-performance GPU capabilities, and a dedicated Neural Processing Unit (NPU) within a single chip, offering a unified solution for diverse and intensive computing workloads.

At the heart of the Intel® Core Ultra™ architecture is a hybrid design that enables exceptional performance across traditional processing tasks, GPU-accelerated workloads, and AI-driven operations. The inclusion of the NPU enhances on-device AI inference, enabling faster, more efficient machine learning and data processing across a wide range of applications.

The Core Ultra™ family includes various models tailored for different performance needs, with options ranging from energy-efficient designs to high-power variants marked by the "H" designation—ideal for laptops and compact form factors that demand serious computing power. Across the lineup, users benefit from the synergy of CPU, GPU, and NPU integration, delivering remarkable efficiency, responsiveness, and multitasking capabilities.

As part of Intel's ongoing innovation, the Core Ultra™ series sets a new standard for future-ready computing. With multiple models available and more on the horizon, this series underscores Intel's commitment to delivering cutting-edge solutions for the next generation of intelligent, AI-enhanced devices.

Benchmarks below run on Intel® Core™ Ultra™ 7 258V and Intel® Core™ Ultra™ 7 265K at FP32 and INT8 precision.

#### Intel® Core™ Ultra™ 7 258V

!!! tip "Benchmarks"

    === "Integrated Intel® Arc™ GPU"

        <div align="center">
        <img width="800" src="https://github.com/ultralytics/docs/releases/download/0/openvino-ultra7-258V-gpu.avif" alt="Intel Core Ultra GPU benchmarks">
        </div>

        ??? abstract "Detailed Benchmark Results"

            | Model   | Format   | Precision | Status | Size (MB) | metrics/mAP50-95(B) | Inference time (ms/im) |
            | ------- | -------- | --------- | ------ | --------- | ------------------- | ---------------------- |
            | YOLO11n | PyTorch  | FP32      | ✅     | 5.4       | 0.5052              | 32.27                  |
            | YOLO11n | OpenVINO | FP32      | ✅     | 10.4      | 0.5068              | 11.84                  |
            | YOLO11n | OpenVINO | INT8      | ✅     | 3.3       | 0.4969              | 11.24                  |
            | YOLO11s | PyTorch  | FP32      | ✅     | 18.4      | 0.5776              | 92.09                  |
            | YOLO11s | OpenVINO | FP32      | ✅     | 36.4      | 0.5797              | 14.82                  |
            | YOLO11s | OpenVINO | INT8      | ✅     | 9.8       | 0.5751              | 12.88                  |
            | YOLO11m | PyTorch  | FP32      | ✅     | 38.8      | 0.6262              | 277.24                 |
            | YOLO11m | OpenVINO | FP32      | ✅     | 77.1      | 0.6306              | 22.94                  |
            | YOLO11m | OpenVINO | INT8      | ✅     | 20.2      | 0.6126              | 17.85                  |
            | YOLO11l | PyTorch  | FP32      | ✅     | 49.0      | 0.6361              | 348.97                 |
            | YOLO11l | OpenVINO | FP32      | ✅     | 97.3      | 0.6365              | 27.34                  |
            | YOLO11l | OpenVINO | INT8      | ✅     | 25.7      | 0.6242              | 20.83                  |
            | YOLO11x | PyTorch  | FP32      | ✅     | 109.3     | 0.6984              | 666.07                 |
            | YOLO11x | OpenVINO | FP32      | ✅     | 217.8     | 0.6890              | 39.09                  |
            | YOLO11x | OpenVINO | INT8      | ✅     | 55.9      | 0.6856              | 30.60                  |

    === "Intel® Lunar Lake CPU"

        <div align="center">
        <img width="800" src="https://github.com/ultralytics/docs/releases/download/0/openvino-ultra7-258V-cpu.avif" alt="Intel Core Ultra CPU benchmarks">
        </div>

        ??? abstract "Detailed Benchmark Results"

            | Model   | Format   | Precision | Status | Size (MB) | metrics/mAP50-95(B) | Inference time (ms/im) |
            | ------- | -------- | --------- | ------ | --------- | ------------------- | ---------------------- |
            | YOLO11n | PyTorch  | FP32      | ✅     | 5.4       | 0.5052              | 32.27                  |
            | YOLO11n | OpenVINO | FP32      | ✅     | 10.4      | 0.5077              | 32.55                  |
            | YOLO11n | OpenVINO | INT8      | ✅     | 3.3       | 0.4980              | 22.98                  |
            | YOLO11s | PyTorch  | FP32      | ✅     | 18.4      | 0.5776              | 92.09                  |
            | YOLO11s | OpenVINO | FP32      | ✅     | 36.4      | 0.5782              | 98.38                  |
            | YOLO11s | OpenVINO | INT8      | ✅     | 9.8       | 0.5745              | 52.84                  |
            | YOLO11m | PyTorch  | FP32      | ✅     | 38.8      | 0.6262              | 277.24                 |
            | YOLO11m | OpenVINO | FP32      | ✅     | 77.1      | 0.6307              | 275.74                 |
            | YOLO11m | OpenVINO | INT8      | ✅     | 20.2      | 0.6172              | 132.63                 |
            | YOLO11l | PyTorch  | FP32      | ✅     | 49.0      | 0.6361              | 348.97                 |
            | YOLO11l | OpenVINO | FP32      | ✅     | 97.3      | 0.6361              | 348.97                 |
            | YOLO11l | OpenVINO | INT8      | ✅     | 25.7      | 0.6240              | 171.36                 |
            | YOLO11x | PyTorch  | FP32      | ✅     | 109.3     | 0.6984              | 666.07                 |
            | YOLO11x | OpenVINO | FP32      | ✅     | 217.8     | 0.6900              | 783.16                 |
            | YOLO11x | OpenVINO | INT8      | ✅     | 55.9      | 0.6890              | 346.82                 |


    === "Integrated Intel® AI Boost NPU"

        <div align="center">
        <img width="800" src="https://github.com/ultralytics/docs/releases/download/0/openvino-ultra7-258V-npu.avif" alt="Intel Core Ultra NPU benchmarks">
        </div>

        ??? abstract "Detailed Benchmark Results"

            | Model   | Format   | Precision | Status | Size (MB) | metrics/mAP50-95(B) | Inference time (ms/im) |
            | ------- | -------- | --------- | ------ | --------- | ------------------- | ---------------------- |
            | YOLO11n | PyTorch  | FP32      | ✅     | 5.4       | 0.5052              | 32.27                  |
            | YOLO11n | OpenVINO | FP32      | ✅     | 10.4      | 0.5085              | 8.33                   |
            | YOLO11n | OpenVINO | INT8      | ✅     | 3.3       | 0.5019              | 8.91                   |
            | YOLO11s | PyTorch  | FP32      | ✅     | 18.4      | 0.5776              | 92.09                  |
            | YOLO11s | OpenVINO | FP32      | ✅     | 36.4      | 0.5788              | 9.72                   |
            | YOLO11s | OpenVINO | INT8      | ✅     | 9.8       | 0.5710              | 10.58                  |
            | YOLO11m | PyTorch  | FP32      | ✅     | 38.8      | 0.6262              | 277.24                 |
            | YOLO11m | OpenVINO | FP32      | ✅     | 77.1      | 0.6301              | 19.41                  |
            | YOLO11m | OpenVINO | INT8      | ✅     | 20.2      | 0.6124              | 18.26                  |
            | YOLO11l | PyTorch  | FP32      | ✅     | 49.0      | 0.6361              | 348.97                 |
            | YOLO11l | OpenVINO | FP32      | ✅     | 97.3      | 0.6362              | 23.70                  |
            | YOLO11l | OpenVINO | INT8      | ✅     | 25.7      | 0.6240              | 21.40                  |
            | YOLO11x | PyTorch  | FP32      | ✅     | 109.3     | 0.6984              | 666.07                 |
            | YOLO11x | OpenVINO | FP32      | ✅     | 217.8     | 0.6892              | 43.91                  |
            | YOLO11x | OpenVINO | INT8      | ✅     | 55.9      | 0.6890              | 34.04                  |

#### Intel® Core™ Ultra™ 7 265K

!!! tip "Benchmarks"

    === "Integrated Intel® Arc™ GPU"

        <div align="center">
        <img width="800" src="https://github.com/ultralytics/docs/releases/download/0/openvino-ultra7-265K-gpu.avif" alt="Intel Core Ultra GPU benchmarks">
        </div>

        ??? abstract "Detailed Benchmark Results"

            | Model   | Format   | Precision | Status | Size (MB) | metrics/mAP50-95(B) | Inference time (ms/im) |
            | ------- | -------- | --------- | ------ | --------- | ------------------- | ---------------------- |
            | YOLO11n | PyTorch  | FP32      | ✅     | 5.4       | 0.5072              | 16.29                  |
            | YOLO11n | OpenVINO | FP32      | ✅     | 10.4      | 0.5079              | 13.13                  |
            | YOLO11n | OpenVINO | INT8      | ✅     | 3.3       | 0.4976              | 8.86                   |
            | YOLO11s | PyTorch  | FP32      | ✅     | 18.4      | 0.5771              | 39.61                  |
            | YOLO11s | OpenVINO | FP32      | ✅     | 36.4      | 0.5808              | 18.26                  |
            | YOLO11s | OpenVINO | INT8      | ✅     | 9.8       | 0.5726              | 13.24                  |
            | YOLO11m | PyTorch  | FP32      | ✅     | 38.8      | 0.6258              | 100.65                 |
            | YOLO11m | OpenVINO | FP32      | ✅     | 77.1      | 0.6310              | 43.50                  |
            | YOLO11m | OpenVINO | INT8      | ✅     | 20.2      | 0.6137              | 20.90                  |
            | YOLO11l | PyTorch  | FP32      | ✅     | 49.0      | 0.6367              | 131.37                 |
            | YOLO11l | OpenVINO | FP32      | ✅     | 97.3      | 0.6371              | 54.52                  |
            | YOLO11l | OpenVINO | INT8      | ✅     | 25.7      | 0.6226              | 27.36                  |
            | YOLO11x | PyTorch  | FP32      | ✅     | 109.3     | 0.6990              | 212.45                 |
            | YOLO11x | OpenVINO | FP32      | ✅     | 217.8     | 0.6884              | 112.76                 |
            | YOLO11x | OpenVINO | INT8      | ✅     | 55.9      | 0.6900              | 52.06                  |


    === "Intel® Arrow Lake CPU"

        <div align="center">
        <img width="800" src="https://github.com/ultralytics/docs/releases/download/0/openvino-ultra7-265K-cpu.avif" alt="Intel Core Ultra CPU benchmarks">
        </div>

        ??? abstract "Detailed Benchmark Results"

            | Model   | Format   | Precision | Status | Size (MB) | metrics/mAP50-95(B) | Inference time (ms/im) |
            | ------- | -------- | --------- | ------ | --------- | ------------------- | ---------------------- |
            | YOLO11n | PyTorch  | FP32      | ✅     | 5.4       | 0.5072              | 16.29                  |
            | YOLO11n | OpenVINO | FP32      | ✅     | 10.4      | 0.5077              | 15.04                  |
            | YOLO11n | OpenVINO | INT8      | ✅     | 3.3       | 0.4980              | 11.60                  |
            | YOLO11s | PyTorch  | FP32      | ✅     | 18.4      | 0.5771              | 39.61                  |
            | YOLO11s | OpenVINO | FP32      | ✅     | 36.4      | 0.5782              | 33.45                  |
            | YOLO11s | OpenVINO | INT8      | ✅     | 9.8       | 0.5745              | 20.64                  |
            | YOLO11m | PyTorch  | FP32      | ✅     | 38.8      | 0.6258              | 100.65                 |
            | YOLO11m | OpenVINO | FP32      | ✅     | 77.1      | 0.6307              | 81.15                  |
            | YOLO11m | OpenVINO | INT8      | ✅     | 20.2      | 0.6172              | 44.63                  |
            | YOLO11l | PyTorch  | FP32      | ✅     | 49.0      | 0.6367              | 131.37                 |
            | YOLO11l | OpenVINO | FP32      | ✅     | 97.3      | 0.6409              | 103.77                 |
            | YOLO11l | OpenVINO | INT8      | ✅     | 25.7      | 0.6240              | 58.00                  |
            | YOLO11x | PyTorch  | FP32      | ✅     | 109.3     | 0.6990              | 212.45                 |
            | YOLO11x | OpenVINO | FP32      | ✅     | 217.8     | 0.6900              | 208.37                 |
            | YOLO11x | OpenVINO | INT8      | ✅     | 55.9      | 0.6897              | 113.04                 |


    === "Integrated Intel® AI Boost NPU"

        <div align="center">
        <img width="800" src="https://github.com/ultralytics/docs/releases/download/0/openvino-ultra7-265K-npu.avif" alt="Intel Core Ultra NPU benchmarks">
        </div>

        ??? abstract "Detailed Benchmark Results"

            | Model   | Format   | Precision | Status | Size (MB) | metrics/mAP50-95(B) | Inference time (ms/im) |
            | ------- | -------- | --------- | ------ | --------- | ------------------- | ---------------------- |
            | YOLO11n | PyTorch  | FP32      | ✅     | 5.4       | 0.5072              | 16.29                  |
            | YOLO11n | OpenVINO | FP32      | ✅     | 10.4      | 0.5075              | 8.02                   |
            | YOLO11n | OpenVINO | INT8      | ✅     | 3.3       | 0.3656              | 9.28                   |
            | YOLO11s | PyTorch  | FP32      | ✅     | 18.4      | 0.5771              | 39.61                  |
            | YOLO11s | OpenVINO | FP32      | ✅     | 36.4      | 0.5801              | 13.12                  |
            | YOLO11s | OpenVINO | INT8      | ✅     | 9.8       | 0.5686              | 13.12                  |
            | YOLO11m | PyTorch  | FP32      | ✅     | 38.8      | 0.6258              | 100.65                 |
            | YOLO11m | OpenVINO | FP32      | ✅     | 77.1      | 0.6310              | 29.88                  |
            | YOLO11m | OpenVINO | INT8      | ✅     | 20.2      | 0.6111              | 26.32                  |
            | YOLO11l | PyTorch  | FP32      | ✅     | 49.0      | 0.6367              | 131.37                 |
            | YOLO11l | OpenVINO | FP32      | ✅     | 97.3      | 0.6356              | 37.08                  |
            | YOLO11l | OpenVINO | INT8      | ✅     | 25.7      | 0.6245              | 30.81                  |
            | YOLO11x | PyTorch  | FP32      | ✅     | 109.3     | 0.6990              | 212.45                 |
            | YOLO11x | OpenVINO | FP32      | ✅     | 217.8     | 0.6894              | 68.48                  |
            | YOLO11x | OpenVINO | INT8      | ✅     | 55.9      | 0.6417              | 49.76                  |

## Intel® Arc GPU

Intel® Arc™ is Intel's line of discrete graphics cards designed for high-performance gaming, content creation, and AI workloads. The Arc series features advanced GPU architectures that support real-time ray tracing, AI-enhanced graphics, and high-resolution gaming. With a focus on performance and efficiency, Intel® Arc™ aims to compete with other leading GPU brands while providing unique features like hardware-accelerated AV1 encoding and support for the latest graphics APIs.

Benchmarks below run on Intel Arc A770 and Intel Arc B580 at FP32 and INT8 precision.

### Intel Arc A770

<div align="center">
<img width="800" src="https://github.com/ultralytics/docs/releases/download/0/openvino-arc-a770-gpu.avif" alt="Intel Core Ultra CPU benchmarks">
</div>

??? abstract "Detailed Benchmark Results"

    | Model   | Format   | Precision | Status | Size (MB) | metrics/mAP50-95(B) | Inference time (ms/im) |
    | ------- | -------- | --------- | ------ | --------- | ------------------- | ---------------------- |
    | YOLO11n | PyTorch  | FP32      | ✅     | 5.4       | 0.5072              | 16.29                  |
    | YOLO11n | OpenVINO | FP32      | ✅     | 10.4      | 0.5073              | 6.98                   |
    | YOLO11n | OpenVINO | INT8      | ✅     | 3.3       | 0.4978              | 7.24                   |
    | YOLO11s | PyTorch  | FP32      | ✅     | 18.4      | 0.5771              | 39.61                  |
    | YOLO11s | OpenVINO | FP32      | ✅     | 36.4      | 0.5798              | 9.41                   |
    | YOLO11s | OpenVINO | INT8      | ✅     | 9.8       | 0.5751              | 8.72                   |
    | YOLO11m | PyTorch  | FP32      | ✅     | 38.8      | 0.6258              | 100.65                 |
    | YOLO11m | OpenVINO | FP32      | ✅     | 77.1      | 0.6311              | 14.88                  |
    | YOLO11m | OpenVINO | INT8      | ✅     | 20.2      | 0.6126              | 11.97                  |
    | YOLO11l | PyTorch  | FP32      | ✅     | 49.0      | 0.6367              | 131.37                 |
    | YOLO11l | OpenVINO | FP32      | ✅     | 97.3      | 0.6364              | 19.17                  |
    | YOLO11l | OpenVINO | INT8      | ✅     | 25.7      | 0.6241              | 15.75                  |
    | YOLO11x | PyTorch  | FP32      | ✅     | 109.3     | 0.6990              | 212.45                 |
    | YOLO11x | OpenVINO | FP32      | ✅     | 217.8     | 0.6888              | 18.13                  |
    | YOLO11x | OpenVINO | INT8      | ✅     | 55.9      | 0.6930              | 18.91                  |

### Intel Arc B580

<div align="center">
<img width="800" src="https://github.com/ultralytics/docs/releases/download/0/openvino-arc-b580-gpu.avif" alt="Intel Core Ultra CPU benchmarks">
</div>

??? abstract "Detailed Benchmark Results"

    | Model   | Format   | Precision | Status | Size (MB) | metrics/mAP50-95(B) | Inference time (ms/im) |
    | ------- | -------- | --------- | ------ | --------- | ------------------- | ---------------------- |
    | YOLO11n | PyTorch  | FP32      | ✅     | 5.4       | 0.5072              | 16.29                  |
    | YOLO11n | OpenVINO | FP32      | ✅     | 10.4      | 0.5072              | 4.27                   |
    | YOLO11n | OpenVINO | INT8      | ✅     | 3.3       | 0.4981              | 4.33                   |
    | YOLO11s | PyTorch  | FP32      | ✅     | 18.4      | 0.5771              | 39.61                  |
    | YOLO11s | OpenVINO | FP32      | ✅     | 36.4      | 0.5789              | 5.04                   |
    | YOLO11s | OpenVINO | INT8      | ✅     | 9.8       | 0.5746              | 4.97                   |
    | YOLO11m | PyTorch  | FP32      | ✅     | 38.8      | 0.6258              | 100.65                 |
    | YOLO11m | OpenVINO | FP32      | ✅     | 77.1      | 0.6306              | 6.45                   |
    | YOLO11m | OpenVINO | INT8      | ✅     | 20.2      | 0.6125              | 6.28                   |
    | YOLO11l | PyTorch  | FP32      | ✅     | 49.0      | 0.6367              | 131.37                 |
    | YOLO11l | OpenVINO | FP32      | ✅     | 97.3      | 0.6360              | 8.23                   |
    | YOLO11l | OpenVINO | INT8      | ✅     | 25.7      | 0.6236              | 8.49                   |
    | YOLO11x | PyTorch  | FP32      | ✅     | 109.3     | 0.6990              | 212.45                 |
    | YOLO11x | OpenVINO | FP32      | ✅     | 217.8     | 0.6889              | 11.10                  |
    | YOLO11x | OpenVINO | INT8      | ✅     | 55.9      | 0.6924              | 10.30                  |

## Reproduce Our Results

To reproduce the Ultralytics benchmarks above on all export [formats](../modes/export.md) run this code:

!!! example

    === "Python"

        ```python
        from ultralytics import YOLO

        # Load a YOLO11n PyTorch model
        model = YOLO("yolo11n.pt")

        # Benchmark YOLO11n speed and accuracy on the COCO128 dataset for all export formats
        results = model.benchmark(data="coco128.yaml")
        ```

    === "CLI"

        ```bash
        # Benchmark YOLO11n speed and accuracy on the COCO128 dataset for all export formats
        yolo benchmark model=yolo11n.pt data=coco128.yaml
        ```

    Note that benchmarking results might vary based on the exact hardware and software configuration of a system, as well as the current workload of the system at the time the benchmarks are run. For the most reliable results use a dataset with a large number of images, i.e. `data='coco.yaml'` (5000 val images).

## Conclusion

The benchmarking results clearly demonstrate the benefits of exporting the YOLO11 model to the OpenVINO format. Across different models and hardware platforms, the OpenVINO format consistently outperforms other formats in terms of inference speed while maintaining comparable accuracy.

The benchmarks underline the effectiveness of OpenVINO as a tool for deploying deep learning models. By converting models to the OpenVINO format, developers can achieve significant performance improvements, making it easier to deploy these models in real-world applications.

For more detailed information and instructions on using OpenVINO, refer to the [official OpenVINO documentation](https://docs.openvino.ai/).

## FAQ

### How do I export YOLO11 models to OpenVINO format?

Exporting YOLO11 models to the OpenVINO format can significantly enhance CPU speed and enable GPU and NPU accelerations on Intel hardware. To export, you can use either Python or CLI as shown below:

!!! example

    === "Python"

        ```python
        from ultralytics import YOLO

        # Load a YOLO11n PyTorch model
        model = YOLO("yolo11n.pt")

        # Export the model
        model.export(format="openvino")  # creates 'yolo11n_openvino_model/'
        ```

    === "CLI"

        ```bash
        # Export a YOLO11n PyTorch model to OpenVINO format
        yolo export model=yolo11n.pt format=openvino # creates 'yolo11n_openvino_model/'
        ```

For more information, refer to the [export formats documentation](../modes/export.md).

### What are the benefits of using OpenVINO with YOLO11 models?

Using Intel's OpenVINO toolkit with YOLO11 models offers several benefits:

1. **Performance**: Achieve up to 3x speedup on CPU inference and leverage Intel GPUs and NPUs for acceleration.
2. **Model Optimizer**: Convert, optimize, and execute models from popular frameworks like PyTorch, TensorFlow, and ONNX.
3. **Ease of Use**: Over 80 tutorial notebooks are available to help users get started, including ones for YOLO11.
4. **Heterogeneous Execution**: Deploy models on various Intel hardware with a unified API.

For detailed performance comparisons, visit our [benchmarks section](#openvino-yolo11-benchmarks).

### How can I run inference using a YOLO11 model exported to OpenVINO?

After exporting a YOLO11n model to OpenVINO format, you can run inference using Python or CLI:

!!! example

    === "Python"

        ```python
        from ultralytics import YOLO

        # Load the exported OpenVINO model
        ov_model = YOLO("yolo11n_openvino_model/")

        # Run inference
        results = ov_model("https://ultralytics.com/images/bus.jpg")
        ```

    === "CLI"

        ```bash
        # Run inference with the exported model
        yolo predict model=yolo11n_openvino_model source='https://ultralytics.com/images/bus.jpg'
        ```

Refer to our [predict mode documentation](../modes/predict.md) for more details.

### Why should I choose Ultralytics YOLO11 over other models for OpenVINO export?

Ultralytics YOLO11 is optimized for real-time object detection with high accuracy and speed. Specifically, when combined with OpenVINO, YOLO11 provides:

- Up to 3x speedup on Intel CPUs
- Seamless deployment on Intel GPUs and NPUs
- Consistent and comparable accuracy across various export formats

For in-depth performance analysis, check our detailed [YOLO11 benchmarks](#openvino-yolo11-benchmarks) on different hardware.

### Can I benchmark YOLO11 models on different formats such as PyTorch, ONNX, and OpenVINO?

Yes, you can benchmark YOLO11 models in various formats including PyTorch, TorchScript, ONNX, and OpenVINO. Use the following code snippet to run benchmarks on your chosen dataset:

!!! example

    === "Python"

        ```python
        from ultralytics import YOLO

        # Load a YOLO11n PyTorch model
        model = YOLO("yolo11n.pt")

        # Benchmark YOLO11n speed and [accuracy](https://www.ultralytics.com/glossary/accuracy) on the COCO8 dataset for all export formats
        results = model.benchmark(data="coco8.yaml")
        ```

    === "CLI"

        ```bash
        # Benchmark YOLO11n speed and accuracy on the COCO8 dataset for all export formats
        yolo benchmark model=yolo11n.pt data=coco8.yaml
        ```

For detailed benchmark results, refer to our [benchmarks section](#openvino-yolo11-benchmarks) and [export formats](../modes/export.md) documentation.
