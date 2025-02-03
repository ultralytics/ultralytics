---
comments: true
description: Learn to export YOLOv8 models to OpenVINO format for up to 3x CPU speedup and hardware acceleration on Intel GPU and NPU.
keywords: YOLOv8, OpenVINO, model export, Intel, AI inference, CPU speedup, GPU acceleration, NPU, deep learning
---

# Intel OpenVINO Export

<img width="1024" src="https://github.com/ultralytics/docs/releases/download/0/openvino-ecosystem.avif" alt="OpenVINO Ecosystem">

In this guide, we cover exporting YOLOv8 models to the [OpenVINO](https://docs.openvino.ai/) format, which can provide up to 3x [CPU](https://docs.openvino.ai/2024/openvino-workflow/running-inference/inference-devices-and-modes/cpu-device.html) speedup, as well as accelerating YOLO inference on Intel [GPU](https://docs.openvino.ai/2024/openvino-workflow/running-inference/inference-devices-and-modes/gpu-device.html) and [NPU](https://docs.openvino.ai/2024/openvino-workflow/running-inference/inference-devices-and-modes/npu-device.html) hardware.

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

Export a YOLOv8n model to OpenVINO format and run inference with the exported model.

!!! example

    === "Python"

        ```python
        from ultralytics import YOLO

        # Load a YOLOv8n PyTorch model
        model = YOLO("yolov8n.pt")

        # Export the model
        model.export(format="openvino")  # creates 'yolov8n_openvino_model/'

        # Load the exported OpenVINO model
        ov_model = YOLO("yolov8n_openvino_model/")

        # Run inference
        results = ov_model("https://ultralytics.com/images/bus.jpg")
        ```

    === "CLI"

        ```bash
        # Export a YOLOv8n PyTorch model to OpenVINO format
        yolo export model=yolov8n.pt format=openvino  # creates 'yolov8n_openvino_model/'

        # Run inference with the exported model
        yolo predict model=yolov8n_openvino_model source='https://ultralytics.com/images/bus.jpg'
        ```

## Export Arguments

| Argument  | Type             | Default      | Description                                                                                                                                                                                   |
| --------- | ---------------- | ------------ | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `format`  | `str`            | `openvino`   | Target format for the exported model, defining compatibility with various deployment environments.                                                                                            |
| `imgsz`   | `int` or `tuple` | `640`        | Desired image size for the model input. Can be an integer for square images or a tuple `(height, width)` for specific dimensions.                                                             |
| `half`    | `bool`           | `False`      | Enables FP16 (half-precision) quantization, reducing model size and potentially speeding up inference on supported hardware.                                                                  |
| `int8`    | `bool`           | `False`      | Activates INT8 quantization, further compressing the model and speeding up inference with minimal [accuracy](https://www.ultralytics.com/glossary/accuracy) loss, primarily for edge devices. |
| `dynamic` | `bool`           | `False`      | Allows dynamic input sizes, enhancing flexibility in handling varying image dimensions.                                                                                                       |
| `nms`     | `bool`           | `False`      | Adds Non-Maximum Suppression (NMS), essential for accurate and efficient detection post-processing.                                                                                           |
| `batch`   | `int`            | `1`          | Specifies export model batch inference size or the max number of images the exported model will process concurrently in `predict` mode.                                                       |
| `data`    | `str`            | `coco8.yaml` | Path to the [dataset](https://docs.ultralytics.com/datasets) configuration file (default: `coco8.yaml`), essential for quantization.                                                          |

For more details about the export process, visit the [Ultralytics documentation page on exporting](../modes/export.md).

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

Once you have the OpenVINO files, you can use the OpenVINO Runtime to run the model. The Runtime provides a unified API to inference across all supported Intel hardware. It also provides advanced capabilities like load balancing across Intel hardware and asynchronous execution. For more information on running the inference, refer to the [Inference with OpenVINO Runtime Guide](https://docs.openvino.ai/2024/openvino-workflow/running-inference.html).

Remember, you'll need the XML and BIN files as well as any application-specific settings like input size, scale factor for normalization, etc., to correctly set up and use the model with the Runtime.

In your deployment application, you would typically do the following steps:

1. Initialize OpenVINO by creating `core = Core()`.
2. Load the model using the `core.read_model()` method.
3. Compile the model using the `core.compile_model()` function.
4. Prepare the input (image, text, audio, etc.).
5. Run inference using `compiled_model(input_data)`.

For more detailed steps and code snippets, refer to the [OpenVINO documentation](https://docs.openvino.ai/) or [API tutorial](https://github.com/openvinotoolkit/openvino_notebooks/blob/latest/notebooks/openvino-api/openvino-api.ipynb).

## OpenVINO YOLOv8 Benchmarks

YOLOv8 benchmarks below were run by the Ultralytics team on 4 different model formats measuring speed and accuracy: PyTorch, TorchScript, ONNX and OpenVINO. Benchmarks were run on Intel Flex and Arc GPUs, and on Intel Xeon CPUs at FP32 [precision](https://www.ultralytics.com/glossary/precision) (with the `half=False` argument).

!!! note

    The benchmarking results below are for reference and might vary based on the exact hardware and software configuration of a system, as well as the current workload of the system at the time the benchmarks are run.

    All benchmarks run with `openvino` Python package version [2023.0.1](https://pypi.org/project/openvino/2023.0.1/).

### Intel Flex GPU

The Intel® Data Center GPU Flex Series is a versatile and robust solution designed for the intelligent visual cloud. This GPU supports a wide array of workloads including media streaming, cloud gaming, AI visual inference, and virtual desktop Infrastructure workloads. It stands out for its open architecture and built-in support for the AV1 encode, providing a standards-based software stack for high-performance, cross-architecture applications. The Flex Series GPU is optimized for density and quality, offering high reliability, availability, and scalability.

Benchmarks below run on Intel® Data Center GPU Flex 170 at FP32 precision.

<div align="center">
<img width="800" src="https://github.com/ultralytics/docs/releases/download/0/flex-gpu-benchmarks.avif" alt="Flex GPU benchmarks">
</div>

| Model   | Format                                                  | Status | Size (MB) | mAP50-95(B) | Inference time (ms/im) |
| ------- | ------------------------------------------------------- | ------ | --------- | ----------- | ---------------------- |
| YOLOv8n | [PyTorch](https://www.ultralytics.com/glossary/pytorch) | ✅     | 6.2       | 0.3709      | 21.79                  |
| YOLOv8n | TorchScript                                             | ✅     | 12.4      | 0.3704      | 23.24                  |
| YOLOv8n | ONNX                                                    | ✅     | 12.2      | 0.3704      | 37.22                  |
| YOLOv8n | OpenVINO                                                | ✅     | 12.3      | 0.3703      | 3.29                   |
| YOLOv8s | PyTorch                                                 | ✅     | 21.5      | 0.4471      | 31.89                  |
| YOLOv8s | TorchScript                                             | ✅     | 42.9      | 0.4472      | 32.71                  |
| YOLOv8s | ONNX                                                    | ✅     | 42.8      | 0.4472      | 43.42                  |
| YOLOv8s | OpenVINO                                                | ✅     | 42.9      | 0.4470      | 3.92                   |
| YOLOv8m | PyTorch                                                 | ✅     | 49.7      | 0.5013      | 50.75                  |
| YOLOv8m | TorchScript                                             | ✅     | 99.2      | 0.4999      | 47.90                  |
| YOLOv8m | ONNX                                                    | ✅     | 99.0      | 0.4999      | 63.16                  |
| YOLOv8m | OpenVINO                                                | ✅     | 49.8      | 0.4997      | 7.11                   |
| YOLOv8l | PyTorch                                                 | ✅     | 83.7      | 0.5293      | 77.45                  |
| YOLOv8l | TorchScript                                             | ✅     | 167.2     | 0.5268      | 85.71                  |
| YOLOv8l | ONNX                                                    | ✅     | 166.8     | 0.5268      | 88.94                  |
| YOLOv8l | OpenVINO                                                | ✅     | 167.0     | 0.5264      | 9.37                   |
| YOLOv8x | PyTorch                                                 | ✅     | 130.5     | 0.5404      | 100.09                 |
| YOLOv8x | TorchScript                                             | ✅     | 260.7     | 0.5371      | 114.64                 |
| YOLOv8x | ONNX                                                    | ✅     | 260.4     | 0.5371      | 110.32                 |
| YOLOv8x | OpenVINO                                                | ✅     | 260.6     | 0.5367      | 15.02                  |

This table represents the benchmark results for five different models (YOLOv8n, YOLOv8s, YOLOv8m, YOLOv8l, YOLOv8x) across four different formats (PyTorch, TorchScript, ONNX, OpenVINO), giving us the status, size, mAP50-95(B) metric, and inference time for each combination.

### Intel Arc GPU

Intel® Arc™ represents Intel's foray into the dedicated GPU market. The Arc™ series, designed to compete with leading GPU manufacturers like AMD and NVIDIA, caters to both the laptop and desktop markets. The series includes mobile versions for compact devices like laptops, and larger, more powerful versions for desktop computers.

The Arc™ series is divided into three categories: Arc™ 3, Arc™ 5, and Arc™ 7, with each number indicating the performance level. Each category includes several models, and the 'M' in the GPU model name signifies a mobile, integrated variant.

Early reviews have praised the Arc™ series, particularly the integrated A770M GPU, for its impressive graphics performance. The availability of the Arc™ series varies by region, and additional models are expected to be released soon. Intel® Arc™ GPUs offer high-performance solutions for a range of computing needs, from gaming to content creation.

Benchmarks below run on Intel® Arc 770 GPU at FP32 precision.

<div align="center">
<img width="800" src="https://github.com/ultralytics/docs/releases/download/0/arc-gpu-benchmarks.avif" alt="Arc GPU benchmarks">
</div>

| Model   | Format      | Status | Size (MB) | metrics/mAP50-95(B) | Inference time (ms/im) |
| ------- | ----------- | ------ | --------- | ------------------- | ---------------------- |
| YOLOv8n | PyTorch     | ✅     | 6.2       | 0.3709              | 88.79                  |
| YOLOv8n | TorchScript | ✅     | 12.4      | 0.3704              | 102.66                 |
| YOLOv8n | ONNX        | ✅     | 12.2      | 0.3704              | 57.98                  |
| YOLOv8n | OpenVINO    | ✅     | 12.3      | 0.3703              | 8.52                   |
| YOLOv8s | PyTorch     | ✅     | 21.5      | 0.4471              | 189.83                 |
| YOLOv8s | TorchScript | ✅     | 42.9      | 0.4472              | 227.58                 |
| YOLOv8s | ONNX        | ✅     | 42.7      | 0.4472              | 142.03                 |
| YOLOv8s | OpenVINO    | ✅     | 42.9      | 0.4469              | 9.19                   |
| YOLOv8m | PyTorch     | ✅     | 49.7      | 0.5013              | 411.64                 |
| YOLOv8m | TorchScript | ✅     | 99.2      | 0.4999              | 517.12                 |
| YOLOv8m | ONNX        | ✅     | 98.9      | 0.4999              | 298.68                 |
| YOLOv8m | OpenVINO    | ✅     | 99.1      | 0.4996              | 12.55                  |
| YOLOv8l | PyTorch     | ✅     | 83.7      | 0.5293              | 725.73                 |
| YOLOv8l | TorchScript | ✅     | 167.1     | 0.5268              | 892.83                 |
| YOLOv8l | ONNX        | ✅     | 166.8     | 0.5268              | 576.11                 |
| YOLOv8l | OpenVINO    | ✅     | 167.0     | 0.5262              | 17.62                  |
| YOLOv8x | PyTorch     | ✅     | 130.5     | 0.5404              | 988.92                 |
| YOLOv8x | TorchScript | ✅     | 260.7     | 0.5371              | 1186.42                |
| YOLOv8x | ONNX        | ✅     | 260.4     | 0.5371              | 768.90                 |
| YOLOv8x | OpenVINO    | ✅     | 260.6     | 0.5367              | 19                     |

### Intel Xeon CPU

The Intel® Xeon® CPU is a high-performance, server-grade processor designed for complex and demanding workloads. From high-end [cloud computing](https://www.ultralytics.com/glossary/cloud-computing) and virtualization to [artificial intelligence](https://www.ultralytics.com/glossary/artificial-intelligence-ai) and machine learning applications, Xeon® CPUs provide the power, reliability, and flexibility required for today's data centers.

Notably, Xeon® CPUs deliver high compute density and scalability, making them ideal for both small businesses and large enterprises. By choosing Intel® Xeon® CPUs, organizations can confidently handle their most demanding computing tasks and foster innovation while maintaining cost-effectiveness and operational efficiency.

Benchmarks below run on 4th Gen Intel® Xeon® Scalable CPU at FP32 precision.

<div align="center">
<img width="800" src="https://github.com/ultralytics/docs/releases/download/0/xeon-cpu-benchmarks.avif" alt="Xeon CPU benchmarks">
</div>

| Model   | Format      | Status | Size (MB) | metrics/mAP50-95(B) | Inference time (ms/im) |
| ------- | ----------- | ------ | --------- | ------------------- | ---------------------- |
| YOLOv8n | PyTorch     | ✅     | 6.2       | 0.3709              | 24.36                  |
| YOLOv8n | TorchScript | ✅     | 12.4      | 0.3704              | 23.93                  |
| YOLOv8n | ONNX        | ✅     | 12.2      | 0.3704              | 39.86                  |
| YOLOv8n | OpenVINO    | ✅     | 12.3      | 0.3704              | 11.34                  |
| YOLOv8s | PyTorch     | ✅     | 21.5      | 0.4471              | 33.77                  |
| YOLOv8s | TorchScript | ✅     | 42.9      | 0.4472              | 34.84                  |
| YOLOv8s | ONNX        | ✅     | 42.8      | 0.4472              | 43.23                  |
| YOLOv8s | OpenVINO    | ✅     | 42.9      | 0.4471              | 13.86                  |
| YOLOv8m | PyTorch     | ✅     | 49.7      | 0.5013              | 53.91                  |
| YOLOv8m | TorchScript | ✅     | 99.2      | 0.4999              | 53.51                  |
| YOLOv8m | ONNX        | ✅     | 99.0      | 0.4999              | 64.16                  |
| YOLOv8m | OpenVINO    | ✅     | 99.1      | 0.4996              | 28.79                  |
| YOLOv8l | PyTorch     | ✅     | 83.7      | 0.5293              | 75.78                  |
| YOLOv8l | TorchScript | ✅     | 167.2     | 0.5268              | 79.13                  |
| YOLOv8l | ONNX        | ✅     | 166.8     | 0.5268              | 88.45                  |
| YOLOv8l | OpenVINO    | ✅     | 167.0     | 0.5263              | 56.23                  |
| YOLOv8x | PyTorch     | ✅     | 130.5     | 0.5404              | 96.60                  |
| YOLOv8x | TorchScript | ✅     | 260.7     | 0.5371              | 114.28                 |
| YOLOv8x | ONNX        | ✅     | 260.4     | 0.5371              | 111.02                 |
| YOLOv8x | OpenVINO    | ✅     | 260.6     | 0.5371              | 83.28                  |

### Intel Core CPU

The Intel® Core® series is a range of high-performance processors by Intel. The lineup includes Core i3 (entry-level), Core i5 (mid-range), Core i7 (high-end), and Core i9 (extreme performance). Each series caters to different computing needs and budgets, from everyday tasks to demanding professional workloads. With each new generation, improvements are made to performance, energy efficiency, and features.

Benchmarks below run on 13th Gen Intel® Core® i7-13700H CPU at FP32 precision.

<div align="center">
<img width="800" src="https://github.com/ultralytics/docs/releases/download/0/core-cpu-benchmarks.avif" alt="Core CPU benchmarks">
</div>

| Model   | Format      | Status | Size (MB) | metrics/mAP50-95(B) | Inference time (ms/im) |
| ------- | ----------- | ------ | --------- | ------------------- | ---------------------- |
| YOLOv8n | PyTorch     | ✅     | 6.2       | 0.4478              | 104.61                 |
| YOLOv8n | TorchScript | ✅     | 12.4      | 0.4525              | 112.39                 |
| YOLOv8n | ONNX        | ✅     | 12.2      | 0.4525              | 28.02                  |
| YOLOv8n | OpenVINO    | ✅     | 12.3      | 0.4504              | 23.53                  |
| YOLOv8s | PyTorch     | ✅     | 21.5      | 0.5885              | 194.83                 |
| YOLOv8s | TorchScript | ✅     | 43.0      | 0.5962              | 202.01                 |
| YOLOv8s | ONNX        | ✅     | 42.8      | 0.5962              | 65.74                  |
| YOLOv8s | OpenVINO    | ✅     | 42.9      | 0.5966              | 38.66                  |
| YOLOv8m | PyTorch     | ✅     | 49.7      | 0.6101              | 355.23                 |
| YOLOv8m | TorchScript | ✅     | 99.2      | 0.6120              | 424.78                 |
| YOLOv8m | ONNX        | ✅     | 99.0      | 0.6120              | 173.39                 |
| YOLOv8m | OpenVINO    | ✅     | 99.1      | 0.6091              | 69.80                  |
| YOLOv8l | PyTorch     | ✅     | 83.7      | 0.6591              | 593.00                 |
| YOLOv8l | TorchScript | ✅     | 167.2     | 0.6580              | 697.54                 |
| YOLOv8l | ONNX        | ✅     | 166.8     | 0.6580              | 342.15                 |
| YOLOv8l | OpenVINO    | ✅     | 167.0     | 0.0708              | 117.69                 |
| YOLOv8x | PyTorch     | ✅     | 130.5     | 0.6651              | 804.65                 |
| YOLOv8x | TorchScript | ✅     | 260.8     | 0.6650              | 921.46                 |
| YOLOv8x | ONNX        | ✅     | 260.4     | 0.6650              | 526.66                 |
| YOLOv8x | OpenVINO    | ✅     | 260.6     | 0.6619              | 158.73                 |

### Intel Ultra 7 155H Meteor Lake CPU

The Intel® Ultra™ 7 155H represents a new benchmark in high-performance computing, designed to cater to the most demanding users, from gamers to content creators. The Ultra™ 7 155H is not just a CPU; it integrates a powerful GPU and an advanced NPU (Neural Processing Unit) within a single chip, offering a comprehensive solution for diverse computing needs.

This hybrid architecture allows the Ultra™ 7 155H to excel in both traditional CPU tasks and GPU-accelerated workloads, while the NPU enhances AI-driven processes, enabling faster and more efficient [machine learning](https://www.ultralytics.com/glossary/machine-learning-ml) operations. This makes the Ultra™ 7 155H a versatile choice for applications requiring high-performance graphics, complex computations, and AI inference.

The Ultra™ 7 series includes multiple models, each offering different levels of performance, with the 'H' designation indicating a high-power variant suitable for laptops and compact devices. Early benchmarks have highlighted the exceptional performance of the Ultra™ 7 155H, particularly in multitasking environments, where the combined power of the CPU, GPU, and NPU leads to remarkable efficiency and speed.

As part of Intel's commitment to cutting-edge technology, the Ultra™ 7 155H is designed to meet the needs of future computing, with more models expected to be released. The availability of the Ultra™ 7 155H varies by region, and it continues to receive praise for its integration of three powerful processing units in a single chip, setting new standards in computing performance.

Benchmarks below run on Intel® Ultra™ 7 155H at FP32 and INT8 precision.

!!! tip "Benchmarks"

    === "Integrated Intel® Arc™ GPU"

        | Model   | Format      | Precision | Status | Size (MB) | metrics/mAP50-95(B) | Inference time (ms/im) |
        | ------- | ----------- | --------- | ------ | --------- | ------------------- | ---------------------- |
        | YOLOv8n | PyTorch     | FP32      |  ✅    | 6.2       | 0.6381              | 35.95                  |
        | YOLOv8n | OpenVINO    | FP32      |  ✅    | 12.3      | 0.6117              | 8.32                   |
        | YOLOv8n | OpenVINO    | INT8      |  ✅    | 3.6       | 0.5791              | 9.88                   |
        | YOLOv8s | PyTorch     | FP32      |  ✅    | 21.5      | 0.6967              | 79.72                  |
        | YOLOv8s | OpenVINO    | FP32      |  ✅    | 42.9      | 0.7136              | 13.37                  |
        | YOLOv8s | OpenVINO    | INT8      |  ✅    | 11.2      | 0.7086              | 9.96                   |
        | YOLOv8m | PyTorch     | FP32      |  ✅    | 49.7      | 0.737               | 202.05                 |
        | YOLOv8m | OpenVINO    | FP32      |  ✅    | 99.1      | 0.7331              | 28.07                  |
        | YOLOv8m | OpenVINO    | INT8      |  ✅    | 25.5      | 0.7259              | 21.11                  |
        | YOLOv8l | PyTorch     | FP32      |  ✅    | 83.7      | 0.7769              | 393.37                 |
        | YOLOv8l | OpenVINO    | FP32      |  ✅    | 167.0     | 0.0                 | 52.73                  |
        | YOLOv8l | OpenVINO    | INT8      |  ✅    | 42.6      | 0.7861              | 28.11                  |
        | YOLOv8x | PyTorch     | FP32      |  ✅    | 130.5     | 0.7759              | 610.71                 |
        | YOLOv8x | OpenVINO    | FP32      |  ✅    | 260.6     | 0.748               | 73.51                  |
        | YOLOv8x | OpenVINO    | INT8      |  ✅    | 66.0      | 0.8085              | 51.71                  |

        <div align="center">
        <img width="800" src="https://github.com/ultralytics/docs/releases/download/0/intel-ultra-gpu.avif" alt="Intel Core Ultra GPU benchmarks">
        </div>

    === "Intel® Meteor Lake CPU"

        | Model   | Format      | Precision | Status | Size (MB) | metrics/mAP50-95(B) | Inference time (ms/im) |
        | ------- | ----------- | --------- | ------ | --------- | ------------------- | ---------------------- |
        | YOLOv8n | PyTorch     | FP32      |  ✅    | 6.2       | 0.6381              | 34.69                  |
        | YOLOv8n | OpenVINO    | FP32      |  ✅    | 12.3      | 0.6092              | 39.06                  |
        | YOLOv8n | OpenVINO    | INT8      |  ✅    | 3.6       | 0.5968              | 18.37                  |
        | YOLOv8s | PyTorch     | FP32      |  ✅    | 21.5      | 0.6967              | 79.9                   |
        | YOLOv8s | OpenVINO    | FP32      |  ✅    | 42.9      | 0.7136              | 82.6                   |
        | YOLOv8s | OpenVINO    | INT8      |  ✅    | 11.2      | 0.7083              | 29.51                  |
        | YOLOv8m | PyTorch     | FP32      |  ✅    | 49.7      | 0.737               | 202.43                 |
        | YOLOv8m | OpenVINO    | FP32      |  ✅    | 99.1      | 0.728               | 181.27                 |
        | YOLOv8m | OpenVINO    | INT8      |  ✅    | 25.5      | 0.7285              | 51.25                  |
        | YOLOv8l | PyTorch     | FP32      |  ✅    | 83.7      | 0.7769              | 385.87                 |
        | YOLOv8l | OpenVINO    | FP32      |  ✅    | 167.0     | 0.7551              | 347.75                 |
        | YOLOv8l | OpenVINO    | INT8      |  ✅    | 42.6      | 0.7675              | 91.66                  |
        | YOLOv8x | PyTorch     | FP32      |  ✅    | 130.5     | 0.7759              | 603.63                 |
        | YOLOv8x | OpenVINO    | FP32      |  ✅    | 260.6     | 0.7479              | 516.39                 |
        | YOLOv8x | OpenVINO    | INT8      |  ✅    | 66.0      | 0.8119              | 142.42                 |

        <div align="center">
        <img width="800" src="https://github.com/ultralytics/docs/releases/download/0/intel-ultra-cpu.avif" alt="Intel Core Ultra CPU benchmarks">
        </div>

    === "Integrated Intel® AI Boost NPU"

        | Model   | Format      | Precision | Status | Size (MB) | metrics/mAP50-95(B) | Inference time (ms/im) |
        | ------- | ----------- | --------- | ------ | --------- | ------------------- | ---------------------- |
        | YOLOv8n | PyTorch     | FP32      |  ✅    | 6.2       | 0.6381              | 36.98                  |
        | YOLOv8n | OpenVINO    | FP32      |  ✅    | 12.3      | 0.6103              | 16.68                  |
        | YOLOv8n | OpenVINO    | INT8      |  ✅    | 3.6       | 0.5941              | 14.6                   |
        | YOLOv8s | PyTorch     | FP32      |  ✅    | 21.5      | 0.6967              | 79.76                  |
        | YOLOv8s | OpenVINO    | FP32      |  ✅    | 42.9      | 0.7144              | 32.89                  |
        | YOLOv8s | OpenVINO    | INT8      |  ✅    | 11.2      | 0.7062              | 26.13                  |
        | YOLOv8m | PyTorch     | FP32      |  ✅    | 49.7      | 0.737               | 201.44                 |
        | YOLOv8m | OpenVINO    | FP32      |  ✅    | 99.1      | 0.7284              | 54.4                   |
        | YOLOv8m | OpenVINO    | INT8      |  ✅    | 25.5      | 0.7268              | 30.76                  |
        | YOLOv8l | PyTorch     | FP32      |  ✅    | 83.7      | 0.7769              | 385.46                 |
        | YOLOv8l | OpenVINO    | FP32      |  ✅    | 167.0     | 0.7539              | 80.1                   |
        | YOLOv8l | OpenVINO    | INT8      |  ✅    | 42.6      | 0.7508              | 52.25                  |
        | YOLOv8x | PyTorch     | FP32      |  ✅    | 130.5     | 0.7759              | 609.4                  |
        | YOLOv8x | OpenVINO    | FP32      |  ✅    | 260.6     | 0.7637              | 104.79                 |
        | YOLOv8x | OpenVINO    | INT8      |  ✅    | 66.0      | 0.8077              | 64.96                  |

        <div align="center">
        <img width="800" src="https://github.com/ultralytics/docs/releases/download/0/intel-ultra-npu.avif" alt="Intel Core Ultra NPU benchmarks">
        </div>

## Reproduce Our Results

To reproduce the Ultralytics benchmarks above on all export [formats](../modes/export.md) run this code:

!!! example

    === "Python"

        ```python
        from ultralytics import YOLO

        # Load a YOLOv8n PyTorch model
        model = YOLO("yolov8n.pt")

        # Benchmark YOLOv8n speed and accuracy on the COCO8 dataset for all export formats
        results = model.benchmark(data="coco8.yaml")
        ```

    === "CLI"

        ```bash
        # Benchmark YOLOv8n speed and accuracy on the COCO8 dataset for all export formats
        yolo benchmark model=yolov8n.pt data=coco8.yaml
        ```

    Note that benchmarking results might vary based on the exact hardware and software configuration of a system, as well as the current workload of the system at the time the benchmarks are run. For the most reliable results use a dataset with a large number of images, i.e. `data='coco128.yaml' (128 val images), or `data='coco.yaml'` (5000 val images).

## Conclusion

The benchmarking results clearly demonstrate the benefits of exporting the YOLOv8 model to the OpenVINO format. Across different models and hardware platforms, the OpenVINO format consistently outperforms other formats in terms of inference speed while maintaining comparable accuracy.

For the Intel® Data Center GPU Flex Series, the OpenVINO format was able to deliver inference speeds almost 10 times faster than the original PyTorch format. On the Xeon CPU, the OpenVINO format was twice as fast as the PyTorch format. The accuracy of the models remained nearly identical across the different formats.

The benchmarks underline the effectiveness of OpenVINO as a tool for deploying deep learning models. By converting models to the OpenVINO format, developers can achieve significant performance improvements, making it easier to deploy these models in real-world applications.

For more detailed information and instructions on using OpenVINO, refer to the [official OpenVINO documentation](https://docs.openvino.ai/).

## FAQ

### How do I export YOLOv8 models to OpenVINO format?

Exporting YOLOv8 models to the OpenVINO format can significantly enhance CPU speed and enable GPU and NPU accelerations on Intel hardware. To export, you can use either Python or CLI as shown below:

!!! example

    === "Python"

        ```python
        from ultralytics import YOLO

        # Load a YOLOv8n PyTorch model
        model = YOLO("yolov8n.pt")

        # Export the model
        model.export(format="openvino")  # creates 'yolov8n_openvino_model/'
        ```

    === "CLI"

        ```bash
        # Export a YOLOv8n PyTorch model to OpenVINO format
        yolo export model=yolov8n.pt format=openvino  # creates 'yolov8n_openvino_model/'
        ```

For more information, refer to the [export formats documentation](../modes/export.md).

### What are the benefits of using OpenVINO with YOLOv8 models?

Using Intel's OpenVINO toolkit with YOLOv8 models offers several benefits:

1. **Performance**: Achieve up to 3x speedup on CPU inference and leverage Intel GPUs and NPUs for acceleration.
2. **Model Optimizer**: Convert, optimize, and execute models from popular frameworks like PyTorch, TensorFlow, and ONNX.
3. **Ease of Use**: Over 80 tutorial notebooks are available to help users get started, including ones for YOLOv8.
4. **Heterogeneous Execution**: Deploy models on various Intel hardware with a unified API.

For detailed performance comparisons, visit our [benchmarks section](#openvino-yolov8-benchmarks).

### How can I run inference using a YOLOv8 model exported to OpenVINO?

After exporting a YOLOv8 model to OpenVINO format, you can run inference using Python or CLI:

!!! example

    === "Python"

        ```python
        from ultralytics import YOLO

        # Load the exported OpenVINO model
        ov_model = YOLO("yolov8n_openvino_model/")

        # Run inference
        results = ov_model("https://ultralytics.com/images/bus.jpg")
        ```

    === "CLI"

        ```bash
        # Run inference with the exported model
        yolo predict model=yolov8n_openvino_model source='https://ultralytics.com/images/bus.jpg'
        ```

Refer to our [predict mode documentation](../modes/predict.md) for more details.

### Why should I choose Ultralytics YOLOv8 over other models for OpenVINO export?

Ultralytics YOLOv8 is optimized for real-time object detection with high accuracy and speed. Specifically, when combined with OpenVINO, YOLOv8 provides:

- Up to 3x speedup on Intel CPUs
- Seamless deployment on Intel GPUs and NPUs
- Consistent and comparable accuracy across various export formats

For in-depth performance analysis, check our detailed [YOLOv8 benchmarks](#openvino-yolov8-benchmarks) on different hardware.

### Can I benchmark YOLOv8 models on different formats such as PyTorch, ONNX, and OpenVINO?

Yes, you can benchmark YOLOv8 models in various formats including PyTorch, TorchScript, ONNX, and OpenVINO. Use the following code snippet to run benchmarks on your chosen dataset:

!!! example

    === "Python"

        ```python
        from ultralytics import YOLO

        # Load a YOLOv8n PyTorch model
        model = YOLO("yolov8n.pt")

        # Benchmark YOLOv8n speed and [accuracy](https://www.ultralytics.com/glossary/accuracy) on the COCO8 dataset for all export formats
        results = model.benchmark(data="coco8.yaml")
        ```

    === "CLI"

        ```bash
        # Benchmark YOLOv8n speed and accuracy on the COCO8 dataset for all export formats
        yolo benchmark model=yolov8n.pt data=coco8.yaml
        ```

For detailed benchmark results, refer to our [benchmarks section](#openvino-yolov8-benchmarks) and [export formats](../modes/export.md) documentation.
