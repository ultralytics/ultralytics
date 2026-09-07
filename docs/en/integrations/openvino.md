---
comments: true
description: Learn to export YOLO26 models to OpenVINO format for up to 3x CPU speedup and hardware acceleration on Intel GPU and NPU.
keywords: YOLO26, OpenVINO, model export, Intel, AI inference, CPU speedup, GPU acceleration, NPU, deep learning
---

# Intel OpenVINO Export

<img width="1024" src="https://cdn.ul.run/i/c3120a6b5d08e902d20cc1447249c1ac.avif" alt="OpenVINO Intel AI inference toolkit">

In this guide, we cover exporting YOLO26 models to the [OpenVINO](https://docs.openvino.ai/) format, which can provide up to 3x [CPU](https://docs.openvino.ai/2025/openvino-workflow/running-inference/inference-devices-and-modes/cpu-device.html) speedup, as well as accelerating YOLO inference on Intel [GPU](https://docs.openvino.ai/2025/openvino-workflow/running-inference/inference-devices-and-modes/gpu-device.html) and [NPU](https://docs.openvino.ai/2025/openvino-workflow/running-inference/inference-devices-and-modes/npu-device.html) hardware.

OpenVINO, short for Open Visual Inference & [Neural Network](https://www.ultralytics.com/glossary/neural-network-nn) Optimization toolkit, is a comprehensive toolkit for optimizing and deploying AI inference models. Even though the name contains Visual, OpenVINO also supports various additional tasks including language, audio, time series, etc.

<p align="center">
  <br>
  <iframe loading="lazy" width="720" height="405" src="https://www.youtube.com/embed/rMllxg8ZLs8"
    title="YouTube video player" frameborder="0"
    allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share"
    allowfullscreen>
  </iframe>
  <br>
  <strong>Watch:</strong> How to Export Ultralytics YOLO26 to Intel OpenVINO Format for Faster Inference 🚀
</p>

## Supported Tasks

OpenVINO export supports all seven Ultralytics tasks. Semantic segmentation and depth estimation are available only with YOLO26, the only family that ships those heads.

{% include "macros/supported-tasks.md" %}

## Installation

Install the OpenVINO export dependencies with:

```bash
pip install "ultralytics[export-openvino]"
```

These dependencies are also included in `ultralytics[export]` and installed automatically when required by an export.

## Usage Examples

The OpenVINO format supports the [Export](../modes/export.md), [Predict](../modes/predict.md), and [Validate](../modes/val.md) modes. Export your model, then load the exported model to run inference or validate its accuracy on Intel CPU, integrated/discrete GPU, or NPU.

!!! example "Export"

    === "Python"

        ```python
        from ultralytics import YOLO

        # Load a YOLO26 model
        model = YOLO("yolo26n.pt")

        # Export the model to OpenVINO format
        model.export(format="openvino")  # creates 'yolo26n_openvino_model/'
        ```

    === "CLI"

        ```bash
        # Export a YOLO26n PyTorch model to OpenVINO format
        yolo export model=yolo26n.pt format=openvino # creates 'yolo26n_openvino_model/'
        ```

!!! example "Predict"

    === "Python"

        ```python
        from ultralytics import YOLO

        # Load the exported OpenVINO model
        model = YOLO("yolo26n_openvino_model/")

        # Run inference
        results = model("https://ultralytics.com/images/bus.jpg")

        # Run inference on a specific device, available devices: ["intel:gpu", "intel:npu", "intel:cpu"]
        results = model("https://ultralytics.com/images/bus.jpg", device="intel:gpu")
        ```

    === "CLI"

        ```bash
        # Run inference with the exported OpenVINO model
        yolo predict model=yolo26n_openvino_model source='https://ultralytics.com/images/bus.jpg'

        # Run inference on a specific device, available devices: ["intel:gpu", "intel:npu", "intel:cpu"]
        yolo predict model=yolo26n_openvino_model source='https://ultralytics.com/images/bus.jpg' device="intel:gpu"
        ```

!!! example "Validate"

    === "Python"

        ```python
        from ultralytics import YOLO

        # Load the exported OpenVINO model
        model = YOLO("yolo26n_openvino_model/")

        # Validate accuracy on the COCO8 dataset
        metrics = model.val(data="coco8.yaml")
        ```

    === "CLI"

        ```bash
        # Validate the exported OpenVINO model
        yolo val model=yolo26n_openvino_model data=coco8.yaml
        ```

## Export Arguments

| Argument   | Type                      | Default      | Description                                                                                                                                                                                                                                                      |
| ---------- | ------------------------- | ------------ | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `format`   | `str`                     | `'openvino'` | Target format for the exported model, defining compatibility with various deployment environments.                                                                                                                                                               |
| `imgsz`    | `int` or `tuple`          | `640`        | Desired image size for the model input. Can be an integer for square images or a tuple `(height, width)` for specific dimensions.                                                                                                                                |
| `quantize` | `int` or `str`            | `None`       | Quantization precision: `16` (FP16) or `8` (INT8/PTQ; needs calibration `data`/`fraction`); `32`/unset is FP32. Replaces the deprecated `half`/`int8` flags.                                                                                                     |
| `dynamic`  | `bool`                    | `False`      | Allows dynamic input sizes, enhancing flexibility in handling varying image dimensions.                                                                                                                                                                          |
| `nms`      | `bool`, optional          | `None`       | Select raw output (`None`, default), embedded NMS (`True`), or the NMS-free head (`False`).                                                                                                                                                                      |
| `batch`    | `int`                     | `1`          | Specifies export model batch inference size or the max number of images the exported model will process concurrently in `predict` mode.                                                                                                                          |
| `data`     | `str`                     | `None`       | Path to the [dataset](../datasets/index.md) YAML, essential for quantization; classification instead takes a dataset directory or a built-in dataset name. If omitted with `quantize=8`, Ultralytics selects the default calibration dataset for the model task. |
| `fraction` | `float`, `int`, or `list` | `1.0`        | Calibration subset as a ratio, image count, or `[train, val, test]` ratios/counts. Two-item lists leave `test` full, while `0` skips it.                                                                                                                         |

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
4. **Ease of Use**: The toolkit comes with a large collection of [tutorial notebooks](https://github.com/openvinotoolkit/openvino_notebooks) (including [YOLO26 optimization](https://github.com/openvinotoolkit/openvino_notebooks/tree/latest/notebooks/yolov26-optimization)) teaching different aspects of the toolkit.

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
ov_model = YOLO("yolo26n_openvino_model/")  # the path of your exported OpenVINO model
# Run inference with the exported model
ov_model.predict(device="intel:gpu")  # specify the device you want to run inference on
```

This approach is ideal for fast prototyping or deployment when you don't need full control over the inference pipeline.

### Inference with OpenVINO Runtime

The OpenVINO Runtime provides a unified API for inference across all supported Intel hardware. It also provides advanced capabilities like load balancing across Intel hardware and asynchronous execution. For more information on running inference, refer to the [YOLO26 notebooks](https://github.com/openvinotoolkit/openvino_notebooks/tree/latest/notebooks/yolov26-optimization).

Remember, you'll need the XML and BIN files as well as any application-specific settings like input size, scale factor for normalization, etc., to correctly set up and use the model with the Runtime.

In your deployment application, you would typically do the following steps:

1. Initialize OpenVINO by creating `core = Core()`.
2. Load the model using the `core.read_model()` method.
3. Compile the model using the `core.compile_model()` function.
4. Prepare the input (image, text, audio, etc.).
5. Run inference using `compiled_model(input_data)`.

For more detailed steps and code snippets, refer to the [OpenVINO documentation](https://docs.openvino.ai/) or [API tutorial](https://github.com/openvinotoolkit/openvino_notebooks/blob/latest/notebooks/openvino-api/openvino-api.ipynb).

## OpenVINO YOLO26 Benchmarks

The Ultralytics team benchmarked YOLO26 across various model formats and [precision](https://www.ultralytics.com/glossary/precision), evaluating speed and accuracy on different Intel devices compatible with OpenVINO.

!!! note

    - The benchmarking results below are for reference and might vary based on the exact hardware and software configuration of a system, as well as the current workload of the system at the time the benchmarks are run.

    - All benchmarks were run with `openvino` Python package version [2026.2.1](https://pypi.org/project/openvino/2026.2.1).

    - YOLO26 models on NPU are only supported on Intel® Core™ Ultra™ systems with 2xxV series and 3xx series and above.

### Intel® Core™ Ultra

The Intel® Core™ Ultra™ series represents a new benchmark in high-performance computing, engineered to meet the evolving demands of modern users—from gamers and creators to professionals leveraging AI. This next-generation lineup is more than a traditional CPU series; it combines powerful CPU cores, integrated high-performance GPU capabilities, and a dedicated Neural Processing Unit (NPU) within a single chip, offering a unified solution for diverse and intensive computing workloads.

At the heart of the Intel® Core Ultra™ architecture is a hybrid design that enables exceptional performance across traditional processing tasks, GPU-accelerated workloads, and AI-driven operations. The inclusion of the NPU enhances on-device AI inference, enabling faster, more efficient machine learning and data processing across a wide range of applications.

The Core Ultra™ family includes various models tailored for different performance needs, with options ranging from energy-efficient designs to high-power variants marked by the "H" designation—ideal for laptops and compact form factors that demand serious computing power. Across the lineup, users benefit from the synergy of CPU, GPU, and NPU integration, delivering remarkable efficiency, responsiveness, and multitasking capabilities.

As part of Intel's ongoing innovation, the Core Ultra™ series sets a new standard for future-ready computing. With multiple models available and more on the horizon, this series underscores Intel's commitment to delivering cutting-edge solutions for the next generation of intelligent, AI-enhanced devices.

Benchmarks below run on Intel® Core™ Ultra™ X7 358H, Intel® Core™ Ultra™ 7 258V and Intel® Core™ Ultra™ 7 155H at FP32, FP16 and INT8 precision.

#### Intel® Core™ Ultra™ X7 358H

!!! tip "Benchmarks"

    === "Integrated Intel® Arc™ GPU"

        <div align="center">
        <img width="800" src="https://cdn.ul.run/i/77186d2ff07d47c8d926468ebdecf8e1.avif" alt="Intel Core Ultra GPU benchmarks">
        </div>

        ??? abstract "Detailed Benchmark Results"

            | Model   | Format        | Precision | Status | Size (MB) | metrics/mAP50-95(B) | Inference time (ms/im) |
            | ------- | ------------- | --------- | ------ | --------- | ------------------- | ---------------------- |
            | YOLO26n | PyTorch (CPU) | FP32      | ✅     | 5.3       | 0.4765              | 29.28                  |
            | YOLO26n | OpenVINO      | FP32      | ✅     | 9.7       | 0.4762              | 4.09                   |
            | YOLO26n | OpenVINO      | FP16      | ✅     | 5.1       | 0.4762              | 4.03                   |
            | YOLO26n | OpenVINO      | INT8      | ✅     | 3.2       | 0.4634              | 4.24                   |
            | YOLO26s | PyTorch (CPU) | FP32      | ✅     | 19.5      | 0.5703              | 59.32                  |
            | YOLO26s | OpenVINO      | FP32      | ✅     | 36.7      | 0.5616              | 5.08                   |
            | YOLO26s | OpenVINO      | FP16      | ✅     | 18.7      | 0.5616              | 5.14                   |
            | YOLO26s | OpenVINO      | INT8      | ✅     | 10.0      | 0.5462              | 4.47                   |
            | YOLO26m | PyTorch (CPU) | FP32      | ✅     | 42.2      | 0.6196              | 143.21                 |
            | YOLO26m | OpenVINO      | FP32      | ✅     | 78.4      | 0.6166              | 7.1                    |
            | YOLO26m | OpenVINO      | FP16      | ✅     | 39.5      | 0.6166              | 7.06                   |
            | YOLO26m | OpenVINO      | INT8      | ✅     | 20.6      | 0.6055              | 5.61                   |
            | YOLO26l | PyTorch (CPU) | FP32      | ✅     | 50.7      | 0.6215              | 181.16                 |
            | YOLO26l | OpenVINO      | FP32      | ✅     | 95.3      | 0.6205              | 8.45                   |
            | YOLO26l | OpenVINO      | FP16      | ✅     | 48.1      | 0.6205              | 8.49                   |
            | YOLO26l | OpenVINO      | INT8      | ✅     | 25.2      | 0.5938              | 6.25                   |
            | YOLO26x | PyTorch (CPU) | FP32      | ✅     | 113.2     | 0.6512              | 379.75                 |
            | YOLO26x | OpenVINO      | FP32      | ✅     | 213.3     | 0.6568              | 13.03                  |
            | YOLO26x | OpenVINO      | FP16      | ✅     | 107.1     | 0.6568              | 12.98                  |
            | YOLO26x | OpenVINO      | INT8      | ✅     | 54.8      | 0.6385              | 8.74                   |

    === "Intel® Panther Lake CPU"

        <div align="center">
        <img width="800" src="https://cdn.ul.run/i/0e59bd9390255fd2bca31c34193b1442.avif" alt="Intel Core Ultra CPU benchmarks">
        </div>

        ??? abstract "Detailed Benchmark Results"

            | Model   | Format   | Precision | Status | Size (MB) | metrics/mAP50-95(B) | Inference time (ms/im) |
            | ------- | -------- | --------- | ------ | --------- | ------------------- | ---------------------- |
            | YOLO26n | PyTorch  | FP32      | ✅     | 5.3       | 0.4765              | 29.28                  |
            | YOLO26n | OpenVINO | FP32      | ✅     | 9.7       | 0.4734              | 14.35                  |
            | YOLO26n | OpenVINO | FP16      | ✅     | 5.1       | 0.4771              | 14.23                  |
            | YOLO26n | OpenVINO | INT8      | ✅     | 3.2       | 0.472               | 9.99                   |
            | YOLO26s | PyTorch  | FP32      | ✅     | 19.5      | 0.5703              | 59.32                  |
            | YOLO26s | OpenVINO | FP32      | ✅     | 36.7      | 0.5632              | 35.03                  |
            | YOLO26s | OpenVINO | FP16      | ✅     | 18.7      | 0.563               | 33.55                  |
            | YOLO26s | OpenVINO | INT8      | ✅     | 10.0      | 0.5513              | 16.68                  |
            | YOLO26m | PyTorch  | FP32      | ✅     | 42.2      | 0.6196              | 143.21                 |
            | YOLO26m | OpenVINO | FP32      | ✅     | 78.4      | 0.6191              | 90.53                  |
            | YOLO26m | OpenVINO | FP16      | ✅     | 39.5      | 0.618               | 88.82                  |
            | YOLO26m | OpenVINO | INT8      | ✅     | 20.6      | 0.6046              | 33.33                  |
            | YOLO26l | PyTorch  | FP32      | ✅     | 50.7      | 0.6215              | 181.16                 |
            | YOLO26l | OpenVINO | FP32      | ✅     | 95.3      | 0.6206              | 112.83                 |
            | YOLO26l | OpenVINO | FP16      | ✅     | 48.1      | 0.621               | 111.31                 |
            | YOLO26l | OpenVINO | INT8      | ✅     | 25.2      | 0.5974              | 43.27                  |
            | YOLO26x | PyTorch  | FP32      | ✅     | 113.2     | 0.6512              | 379.75                 |
            | YOLO26x | OpenVINO | FP32      | ✅     | 213.3     | 0.6552              | 241.21                 |
            | YOLO26x | OpenVINO | FP16      | ✅     | 107.1     | 0.6552              | 236.87                 |
            | YOLO26x | OpenVINO | INT8      | ✅     | 54.8      | 0.6365              | 78.92                  |

    === "Integrated Intel® AI Boost NPU"

        <div align="center">
        <img width="800" src="https://cdn.ul.run/i/755a0515d5283c2e1227fe177d2d1981.avif" alt="Intel Core Ultra NPU benchmarks">
        </div>

        ??? abstract "Detailed Benchmark Results"

            | Model   | Format        | Precision | Status | Size (MB) | metrics/mAP50-95(B) | Inference time (ms/im) |
            | ------- | ------------- | --------- | ------ | --------- | ------------------- | ---------------------- |
            | YOLO26n | PyTorch (CPU) | FP32      | ✅     | 5.3       | 0.4765              | 29.28                  |
            | YOLO26n | OpenVINO      | FP32      | ✅     | 9.7       | 0.4751              | 7.68                   |
            | YOLO26n | OpenVINO      | FP16      | ✅     | 5.1       | 0.4762              | 7.71                   |
            | YOLO26n | OpenVINO      | INT8      | ✅     | 3.2       | 0.4731              | 8.6                    |
            | YOLO26s | PyTorch (CPU) | FP32      | ✅     | 19.5      | 0.5703              | 59.32                  |
            | YOLO26s | OpenVINO      | FP32      | ✅     | 36.7      | 0.5615              | 9.89                   |
            | YOLO26s | OpenVINO      | FP16      | ✅     | 18.7      | 0.5615              | 9.94                   |
            | YOLO26s | OpenVINO      | INT8      | ✅     | 10.0      | 0.5469              | 10.69                  |
            | YOLO26m | PyTorch (CPU) | FP32      | ✅     | 42.2      | 0.6196              | 143.21                 |
            | YOLO26m | OpenVINO      | FP32      | ✅     | 78.4      | 0.6168              | 13.02                  |
            | YOLO26m | OpenVINO      | FP16      | ✅     | 39.5      | 0.6166              | 13.13                  |
            | YOLO26m | OpenVINO      | INT8      | ✅     | 20.6      | 0.6039              | 14.84                  |
            | YOLO26l | PyTorch (CPU) | FP32      | ✅     | 50.7      | 0.6215              | 181.16                 |
            | YOLO26l | OpenVINO      | FP32      | ✅     | 95.3      | 0.6201              | 14.53                  |
            | YOLO26l | OpenVINO      | FP16      | ✅     | 48.1      | 0.6205              | 14.55                  |
            | YOLO26l | OpenVINO      | INT8      | ✅     | 25.2      | 0.5976              | 15.64                  |
            | YOLO26x | PyTorch (CPU) | FP32      | ✅     | 113.2     | 0.6512              | 379.75                 |
            | YOLO26x | OpenVINO      | FP32      | ✅     | 213.3     | 0.6563              | 23.48                  |
            | YOLO26x | OpenVINO      | FP16      | ✅     | 107.1     | 0.6561              | 23.41                  |
            | YOLO26x | OpenVINO      | INT8      | ✅     | 54.8      | 0.6413              | 21.56                  |

#### Intel® Core™ Ultra™ 7 258V

!!! tip "Benchmarks"

    === "Integrated Intel® Arc™ GPU"

        <div align="center">
        <img width="800" src="https://cdn.ul.run/i/49405e2a81ee5cedcb04c0cb1751bed9.avif" alt="Intel Core Ultra GPU benchmarks">
        </div>

        ??? abstract "Detailed Benchmark Results"

            | Model   | Format        | Precision | Status | Size (MB) | metrics/mAP50-95(B) | Inference time (ms/im) |
            | ------- | ------------- | --------- | ------ | --------- | ------------------- | ---------------------- |
            | YOLO26n | PyTorch (CPU) | FP32      | ✅     | 5.3       | 0.4765              | 27.33                  |
            | YOLO26n | OpenVINO      | FP32      | ✅     | 9.6       | 0.4764              | 3.33                   |
            | YOLO26n | OpenVINO      | FP16      | ✅     | 5.1       | 0.4764              | 3.34                   |
            | YOLO26n | OpenVINO      | INT8      | ✅     | 3.2       | 0.4634              | 3.46                   |
            | YOLO26s | PyTorch (CPU) | FP32      | ✅     | 19.5      | 0.5703              | 60.2                   |
            | YOLO26s | OpenVINO      | FP32      | ✅     | 36.7      | 0.5616              | 4.66                   |
            | YOLO26s | OpenVINO      | FP16      | ✅     | 18.6      | 0.5616              | 4.65                   |
            | YOLO26s | OpenVINO      | INT8      | ✅     | 10.0      | 0.5462              | 3.99                   |
            | YOLO26m | PyTorch (CPU) | FP32      | ✅     | 42.2      | 0.6196              | 152.87                 |
            | YOLO26m | OpenVINO      | FP32      | ✅     | 78.4      | 0.6168              | 9.03                   |
            | YOLO26m | OpenVINO      | FP16      | ✅     | 39.5      | 0.6168              | 8.98                   |
            | YOLO26m | OpenVINO      | INT8      | ✅     | 20.6      | 0.6055              | 5.7                    |
            | YOLO26l | PyTorch (CPU) | FP32      | ✅     | 50.7      | 0.6215              | 201.57                 |
            | YOLO26l | OpenVINO      | FP32      | ✅     | 95.3      | 0.6203              | 11.32                  |
            | YOLO26l | OpenVINO      | FP16      | ✅     | 48.1      | 0.6203              | 11.36                  |
            | YOLO26l | OpenVINO      | INT8      | ✅     | 25.2      | 0.5938              | 7.45                   |
            | YOLO26x | PyTorch (CPU) | FP32      | ✅     | 113.2     | 0.6512              | 431.04                 |
            | YOLO26x | OpenVINO      | FP32      | ✅     | 213.2     | 0.6568              | 19.08                  |
            | YOLO26x | OpenVINO      | FP16      | ✅     | 107.1     | 0.6568              | 19.63                  |
            | YOLO26x | OpenVINO      | INT8      | ✅     | 54.8      | 0.6385              | 13.69                  |

    === "Intel® Lunar Lake CPU"

        <div align="center">
        <img width="800" src="https://cdn.ul.run/i/070db9675dfee5a2662aa09f0ba42c33.avif" alt="Intel Core Ultra CPU benchmarks">
        </div>

        ??? abstract "Detailed Benchmark Results"

            | Model   | Format   | Precision | Status | Size (MB) | metrics/mAP50-95(B) | Inference time (ms/im) |
            | ------- | -------- | --------- | ------ | --------- | ------------------- | ---------------------- |
            | YOLO26n | PyTorch  | FP32      | ✅     | 5.3       | 0.4765              | 27.33                  |
            | YOLO26n | OpenVINO | FP32      | ✅     | 9.6       | 0.4734              | 16.84                  |
            | YOLO26n | OpenVINO | FP16      | ✅     | 5.1       | 0.4771              | 16.73                  |
            | YOLO26n | OpenVINO | INT8      | ✅     | 3.2       | 0.472               | 8.62                   |
            | YOLO26s | PyTorch  | FP32      | ✅     | 19.5      | 0.5703              | 60.2                   |
            | YOLO26s | OpenVINO | FP32      | ✅     | 36.7      | 0.5632              | 54.71                  |
            | YOLO26s | OpenVINO | FP16      | ✅     | 18.6      | 0.563               | 54.45                  |
            | YOLO26s | OpenVINO | INT8      | ✅     | 10.0      | 0.5513              | 20.19                  |
            | YOLO26m | PyTorch  | FP32      | ✅     | 42.2      | 0.6196              | 152.87                 |
            | YOLO26m | OpenVINO | FP32      | ✅     | 78.4      | 0.6191              | 178.59                 |
            | YOLO26m | OpenVINO | FP16      | ✅     | 39.5      | 0.618               | 179.25                 |
            | YOLO26m | OpenVINO | INT8      | ✅     | 20.6      | 0.6046              | 53.07                  |
            | YOLO26l | PyTorch  | FP32      | ✅     | 50.7      | 0.6215              | 201.57                 |
            | YOLO26l | OpenVINO | FP32      | ✅     | 95.3      | 0.6206              | 238.15                 |
            | YOLO26l | OpenVINO | FP16      | ✅     | 48.1      | 0.621               | 233.7                  |
            | YOLO26l | OpenVINO | INT8      | ✅     | 25.2      | 0.5974              | 67.73                  |
            | YOLO26x | PyTorch  | FP32      | ✅     | 113.2     | 0.6512              | 431.04                 |
            | YOLO26x | OpenVINO | FP32      | ✅     | 213.2     | 0.6552              | 567.22                 |
            | YOLO26x | OpenVINO | FP16      | ✅     | 107.1     | 0.6552              | 561.39                 |
            | YOLO26x | OpenVINO | INT8      | ✅     | 54.8      | 0.6365              | 151.11                 |

    === "Integrated Intel® AI Boost NPU"

        <div align="center">
        <img width="800" src="https://cdn.ul.run/i/ed172cc0d4eff0c333c3cdd584600c3d.avif" alt="Intel Core Ultra NPU benchmarks">
        </div>

        ??? abstract "Detailed Benchmark Results"

            | Model   | Format        | Precision | Status | Size (MB) | metrics/mAP50-95(B) | Inference time (ms/im) |
            | ------- | ------------- | --------- | ------ | --------- | ------------------- | ---------------------- |
            | YOLO26n | PyTorch (CPU) | FP32      | ✅     | 5.3       | 0.4765              | 27.33                  |
            | YOLO26n | OpenVINO      | FP32      | ✅     | 9.6       | 0.4724              | 23.71                  |
            | YOLO26n | OpenVINO      | FP16      | ✅     | 5.1       | 0.476               | 23.7                   |
            | YOLO26n | OpenVINO      | INT8      | ✅     | 3.2       | 0.4615              | 23.96                  |
            | YOLO26s | PyTorch (CPU) | FP32      | ✅     | 19.5      | 0.5703              | 60.2                   |
            | YOLO26s | OpenVINO      | FP32      | ✅     | 36.7      | 0.5618              | 26.62                  |
            | YOLO26s | OpenVINO      | FP16      | ✅     | 18.6      | 0.5616              | 26.62                  |
            | YOLO26s | OpenVINO      | INT8      | ✅     | 10.0      | 0.5497              | 25.89                  |
            | YOLO26m | PyTorch (CPU) | FP32      | ✅     | 42.2      | 0.6196              | 152.87                 |
            | YOLO26m | OpenVINO      | FP32      | ✅     | 78.4      | 0.6165              | 32.94                  |
            | YOLO26m | OpenVINO      | FP16      | ✅     | 39.5      | 0.6165              | 33.01                  |
            | YOLO26m | OpenVINO      | INT8      | ✅     | 20.6      | 0.6042              | 30.95                  |
            | YOLO26l | PyTorch (CPU) | FP32      | ✅     | 50.7      | 0.6215              | 201.57                 |
            | YOLO26l | OpenVINO      | FP32      | ✅     | 95.3      | 0.62                | 36.02                  |
            | YOLO26l | OpenVINO      | FP16      | ✅     | 48.1      | 0.6198              | 35.77                  |
            | YOLO26l | OpenVINO      | INT8      | ✅     | 25.2      | 0.6009              | 32.64                  |
            | YOLO26x | PyTorch (CPU) | FP32      | ✅     | 113.2     | 0.6512              | 431.04                 |
            | YOLO26x | OpenVINO      | FP32      | ✅     | 213.2     | 0.6563              | 49.64                  |
            | YOLO26x | OpenVINO      | FP16      | ✅     | 107.1     | 0.6563              | 49.69                  |
            | YOLO26x | OpenVINO      | INT8      | ✅     | 54.8      | 0.6399              | 41.75                  |

#### Intel® Core™ Ultra™ 7 155H

!!! tip "Benchmarks"

    === "Integrated Intel® Arc™ GPU"

        <div align="center">
        <img width="800" src="https://cdn.ul.run/i/532d3bb375fcadaab3e1d13526f4638e.avif" alt="Intel Core Ultra GPU benchmarks">
        </div>

        ??? abstract "Detailed Benchmark Results"

            | Model   | Format        | Precision | Status | Size (MB) | metrics/mAP50-95(B) | Inference time (ms/im) |
            | ------- | ------------- | --------- | ------ | --------- | ------------------- | ---------------------- |
            | YOLO26n | PyTorch (CPU) | FP32      | ✅     | 5.3       | 0.4765              | 33.19                  |
            | YOLO26n | OpenVINO      | FP32      | ✅     | 9.6       | 0.4758              | 9.13                   |
            | YOLO26n | OpenVINO      | FP16      | ✅     | 5.1       | 0.4758              | 9.07                   |
            | YOLO26n | OpenVINO      | INT8      | ✅     | 3.2       | 0.4573              | 5.79                   |
            | YOLO26s | PyTorch (CPU) | FP32      | ✅     | 19.5      | 0.5703              | 70.92                  |
            | YOLO26s | OpenVINO      | FP32      | ✅     | 36.7      | 0.5616              | 16.37                  |
            | YOLO26s | OpenVINO      | FP16      | ✅     | 18.6      | 0.5616              | 16.65                  |
            | YOLO26s | OpenVINO      | INT8      | ✅     | 10.0      | 0.545               | 9.27                   |
            | YOLO26m | PyTorch (CPU) | FP32      | ✅     | 42.2      | 0.6196              | 192.45                 |
            | YOLO26m | OpenVINO      | FP32      | ✅     | 78.4      | 0.6167              | 34.47                  |
            | YOLO26m | OpenVINO      | FP16      | ✅     | 39.5      | 0.6167              | 35.71                  |
            | YOLO26m | OpenVINO      | INT8      | ✅     | 20.6      | 0.6039              | 15.46                  |
            | YOLO26l | PyTorch (CPU) | FP32      | ✅     | 50.7      | 0.6215              | 239.19                 |
            | YOLO26l | OpenVINO      | FP32      | ✅     | 95.3      | 0.621               | 43.24                  |
            | YOLO26l | OpenVINO      | FP16      | ✅     | 48.1      | 0.621               | 43.4                   |
            | YOLO26l | OpenVINO      | INT8      | ✅     | 25.2      | 0.5999              | 19.72                  |
            | YOLO26x | PyTorch (CPU) | FP32      | ✅     | 113.2     | 0.6512              | 511.89                 |
            | YOLO26x | OpenVINO      | FP32      | ✅     | 213.2     | 0.6552              | 79.49                  |
            | YOLO26x | OpenVINO      | FP16      | ✅     | 107.1     | 0.6552              | 79.45                  |
            | YOLO26x | OpenVINO      | INT8      | ✅     | 54.8      | 0.642               | 34.66                  |

    === "Intel® Meteor Lake CPU"

        <div align="center">
        <img width="800" src="https://cdn.ul.run/i/9fc9ef5ec0d44f4405819edff62c1672.avif" alt="Intel Core Ultra CPU benchmarks">
        </div>

        ??? abstract "Detailed Benchmark Results"

            | Model   | Format   | Precision | Status | Size (MB) | metrics/mAP50-95(B) | Inference time (ms/im) |
            | ------- | -------- | --------- | ------ | --------- | ------------------- | ---------------------- |
            | YOLO26n | PyTorch  | FP32      | ✅     | 5.3       | 0.4765              | 33.19                  |
            | YOLO26n | OpenVINO | FP32      | ✅     | 9.6       | 0.4734              | 13.43                  |
            | YOLO26n | OpenVINO | FP16      | ✅     | 5.1       | 0.4771              | 13.2                   |
            | YOLO26n | OpenVINO | INT8      | ✅     | 3.2       | 0.4611              | 11.4                   |
            | YOLO26s | PyTorch  | FP32      | ✅     | 19.5      | 0.5703              | 70.92                  |
            | YOLO26s | OpenVINO | FP32      | ✅     | 36.7      | 0.5632              | 36.16                  |
            | YOLO26s | OpenVINO | FP16      | ✅     | 18.6      | 0.563               | 36.07                  |
            | YOLO26s | OpenVINO | INT8      | ✅     | 10.0      | 0.5471              | 15.54                  |
            | YOLO26m | PyTorch  | FP32      | ✅     | 42.2      | 0.6196              | 192.45                 |
            | YOLO26m | OpenVINO | FP32      | ✅     | 78.4      | 0.6191              | 102.52                 |
            | YOLO26m | OpenVINO | FP16      | ✅     | 39.5      | 0.618               | 101.88                 |
            | YOLO26m | OpenVINO | INT8      | ✅     | 20.6      | 0.6025              | 35.72                  |
            | YOLO26l | PyTorch  | FP32      | ✅     | 50.7      | 0.6215              | 239.19                 |
            | YOLO26l | OpenVINO | FP32      | ✅     | 95.3      | 0.6206              | 129.24                 |
            | YOLO26l | OpenVINO | FP16      | ✅     | 48.1      | 0.621               | 128.5                  |
            | YOLO26l | OpenVINO | INT8      | ✅     | 25.2      | 0.5984              | 45.38                  |
            | YOLO26x | PyTorch  | FP32      | ✅     | 113.2     | 0.6512              | 511.89                 |
            | YOLO26x | OpenVINO | FP32      | ✅     | 213.2     | 0.6552              | 293.4                  |
            | YOLO26x | OpenVINO | FP16      | ✅     | 107.1     | 0.6552              | 296.48                 |
            | YOLO26x | OpenVINO | INT8      | ✅     | 54.8      | 0.6418              | 85.24                  |

## Reproduce Our Results

To reproduce the Ultralytics benchmarks above on all export [formats](../modes/export.md) run this code:

!!! example

    === "Python"

        ```python
        from ultralytics import YOLO

        # Load a YOLO26n PyTorch model
        model = YOLO("yolo26n.pt")

        # Benchmark YOLO26n speed and accuracy on the COCO128 dataset for all export formats
        results = model.benchmark(data="coco128.yaml")
        ```

    === "CLI"

        ```bash
        # Benchmark YOLO26n speed and accuracy on the COCO128 dataset for all export formats
        yolo benchmark model=yolo26n.pt data=coco128.yaml
        ```

    Note that benchmarking results might vary based on the exact hardware and software configuration of a system, as well as the current workload of the system at the time the benchmarks are run. For the most reliable results use a dataset with a large number of images, i.e. `data='coco.yaml'` (5000 val images).

## Conclusion

The benchmarking results clearly demonstrate the benefits of exporting the YOLO26 model to the OpenVINO format. Across different models and hardware platforms, the OpenVINO format consistently outperforms other formats in terms of inference speed while maintaining comparable accuracy.

The benchmarks underline the effectiveness of OpenVINO as a tool for deploying deep learning models. By converting models to the OpenVINO format, developers can achieve significant performance improvements, making it easier to deploy these models in real-world applications.

For more detailed information and instructions on using OpenVINO, refer to the [official OpenVINO documentation](https://docs.openvino.ai/).

## FAQ

### How do I export YOLO26 models to OpenVINO format?

Exporting YOLO26 models to the OpenVINO format can significantly enhance CPU speed and enable GPU and NPU accelerations on Intel hardware. To export, you can use either Python or CLI as shown below:

!!! example

    === "Python"

        ```python
        from ultralytics import YOLO

        # Load a YOLO26n PyTorch model
        model = YOLO("yolo26n.pt")

        # Export the model
        model.export(format="openvino")  # creates 'yolo26n_openvino_model/'
        ```

    === "CLI"

        ```bash
        # Export a YOLO26n PyTorch model to OpenVINO format
        yolo export model=yolo26n.pt format=openvino # creates 'yolo26n_openvino_model/'
        ```

For more information, refer to the [export formats documentation](../modes/export.md).

### What are the benefits of using OpenVINO with YOLO26 models?

Using Intel's OpenVINO toolkit with YOLO26 models offers several benefits:

1. **Performance**: Achieve up to 3x speedup on CPU inference and leverage Intel GPUs and NPUs for acceleration.
2. **Model Optimizer**: Convert, optimize, and execute models from popular frameworks like PyTorch, TensorFlow, and ONNX.
3. **Ease of Use**: A large collection of tutorial notebooks is available to help users get started, including ones for YOLO26.
4. **Heterogeneous Execution**: Deploy models on various Intel hardware with a unified API.

For detailed performance comparisons, visit our [benchmarks section](#openvino-yolo26-benchmarks).

### How can I run inference using a YOLO26 model exported to OpenVINO?

After exporting a YOLO26n model to OpenVINO format, you can run inference using Python or CLI:

!!! example

    === "Python"

        ```python
        from ultralytics import YOLO

        # Load the exported OpenVINO model
        ov_model = YOLO("yolo26n_openvino_model/")

        # Run inference
        results = ov_model("https://ultralytics.com/images/bus.jpg")
        ```

    === "CLI"

        ```bash
        # Run inference with the exported model
        yolo predict model=yolo26n_openvino_model source='https://ultralytics.com/images/bus.jpg'
        ```

Refer to our [predict mode documentation](../modes/predict.md) for more details.

### Why should I choose Ultralytics YOLO26 over other models for OpenVINO export?

Ultralytics YOLO26 is optimized for real-time object detection with high accuracy and speed. Specifically, when combined with OpenVINO, YOLO26 provides:

- Up to 3x speedup on Intel CPUs
- Seamless deployment on Intel GPUs and NPUs
- Consistent and comparable accuracy across various export formats

For in-depth performance analysis, check our detailed [YOLO26 benchmarks](#openvino-yolo26-benchmarks) on different hardware.

### Can I benchmark YOLO26 models on different formats such as PyTorch, ONNX, and OpenVINO?

Yes, you can benchmark YOLO26 models in various formats including PyTorch, TorchScript, ONNX, and OpenVINO. Use the following code snippet to run benchmarks on your chosen dataset:

!!! example

    === "Python"

        ```python
        from ultralytics import YOLO

        # Load a YOLO26n PyTorch model
        model = YOLO("yolo26n.pt")

        # Benchmark YOLO26n speed and [accuracy](https://www.ultralytics.com/glossary/accuracy) on the COCO8 dataset for all export formats
        results = model.benchmark(data="coco8.yaml")
        ```

    === "CLI"

        ```bash
        # Benchmark YOLO26n speed and accuracy on the COCO8 dataset for all export formats
        yolo benchmark model=yolo26n.pt data=coco8.yaml
        ```

For detailed benchmark results, refer to our [benchmarks section](#openvino-yolo26-benchmarks) and [export formats](../modes/export.md) documentation.
