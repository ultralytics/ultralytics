---
comments: true
description: 'Export mode: Create a deployment-ready YOLOv8 model by converting it to OpenVINO format. Export to OpenVINO for up to 3x CPU speedup.'
keywords: ultralytics docs, YOLOv8, export YOLOv8, YOLOv8 model deployment, exporting YOLOv8, OpenVINO, OpenVINO format
---

<img width="1024" src="https://user-images.githubusercontent.com/26833433/252345644-0cf84257-4b34-404c-b7ce-eb73dfbcaff1.png">

**Export mode** is used for exporting a YOLOv8 model to a format that can be used for deployment. In this guide, we specifically cover exporting to OpenVINO, which can provide up to 3x CPU speedup.

OpenVINO, short for Open Visual Inference & Neural Network Optimization toolkit, is a comprehensive toolkit for quickly developing applications and solutions that emulate human vision. It includes optimized calls for CV standards, including OpenCV, OpenCL kernels, and more. OpenVINO is particularly useful for neural network inference and is compatible with many different types of pre-trained models from the open model zoo.

## Usage Examples

Export a YOLOv8n model to OpenVINO format.

!!! example ""

    === "Python"
    
        ```python
        from ultralytics import YOLO
        
        # Load a model
        model = YOLO('yolov8n.pt')  # load an official model
        model = YOLO('path/to/best.pt')  # load a custom trained
        
        # Export the model
        model.export(format='openvino')
        ```
    === "CLI"
    
        ```bash
        yolo export model=yolov8n.pt format=openvino  # export official model
        yolo export model=path/to/best.pt format=openvino  # export custom trained model
        ```

## Arguments

| Key      | Value        | Description                                          |
|----------|--------------|------------------------------------------------------|
| `format` | `'openvino'` | format to export to                                  |
| `imgsz`  | `640`        | image size as scalar or (h, w) list, i.e. (640, 480) |
| `half`   | `False`      | FP16 quantization                                    |

## Benefits of OpenVINO

1. **Performance**: OpenVINO delivers high-performance inference by utilizing the power of Intel CPUs, integrated GPUs, and FPGAs.
2. **Support for Heterogeneous Execution**: OpenVINO provides an API to write once and deploy on any supported Intel hardware (CPU, GPU, FPGA, VPU, etc.).
3. **Model Optimizer**: OpenVINO provides a Model Optimizer that imports, converts, and optimizes models from popular deep learning frameworks such as TensorFlow, Keras, ONNX, PyTorch, and Caffe.
4. **Pre-optimized Libraries**: OpenVINO includes optimized calls for computer vision (CV) standards, including OpenCV, OpenCL kernels, and more.
5. **Flexibility**: OpenVINO is compatible with pre-trained deep learning models and algorithms from the Open Model Zoo.
6. **Ease of Use**: The toolkit comes with more than 20 pre-trained models, and supports another 20+ public and custom models (including YOLOv8).

## OpenVINO Export Structure

When you export a model to OpenVINO format, it results in a directory containing the following:

1. **XML file**: Describes the network topology.
2. **BIN file**: Contains the weights and biases binary data.
3. **Mapping file**: Holds mapping of original model output tensors to OpenVINO tensor names.

You can use these files to run inference with the OpenVINO Inference Engine.

## Using OpenVINO Export in Deployment

Once you have the OpenVINO files, you can use the OpenVINO Inference Engine to run the model. The Inference Engine provides a unified API to inference across all supported Intel hardware. It also provides advanced capabilities like load balancing across Intel hardware and asynchronous execution. For more information on using the Inference Engine, refer to the [Inference with OpenVINO Runtime Guide](https://docs.openvino.ai/2023.0/openvino_docs_OV_UG_OV_Runtime_User_Guide.html).

Remember, you'll need the XML and BIN files as well as any application-specific settings like input size, scale factor for normalization, etc., to correctly setup and use the model with the Inference Engine.

In your deployment application, you would typically do the following steps:

1. Load the model using the `InferenceEngine::Core::ReadNetwork` method.
2. Prepare the input and output blobs.
3. Run inference using `InferenceEngine::ExecutableNetwork::Infer`.

For more detailed steps and code snippets, refer to the [OpenVINO documentation](https://docs.openvino.ai/).

## OpenVINO YOLOv8 Benchmarks

### Flex GPU

The Intel® Data Center GPU Flex Series is a versatile and robust solution designed for the intelligent visual cloud. This GPU supports a wide array of workloads including media streaming, cloud gaming, AI visual inference, and virtual desktop Infrastructure workloads. It stands out for its open architecture and built-in support for the AV1 encode, providing a standards-based software stack for high-performance, cross-architecture applications. The Flex Series GPU is optimized for density and quality, offering high reliability, availability, and scalability.

#### Flex GPU (FP32 Precision)

Benchmark results for Flex GPU at FP32 precision with `half=False`:

| Model   | Format      | Status | Size (MB) | mAP50-95(B) | Inference time (ms/im) |
|---------|-------------|--------|-----------|-------------|------------------------|
| yolov8n | PyTorch     | ✅      | 6.2       | 0.3709      | 21.79                  |
| yolov8n | TorchScript | ✅      | 12.4      | 0.3704      | 23.24                  |
| yolov8n | ONNX        | ✅      | 12.2      | 0.3704      | 37.22                  |
| yolov8n | OpenVINO    | ✅      | 12.3      | 0.3703      | 3.29                   |
| yolov8s | PyTorch     | ✅      | 21.5      | 0.4471      | 31.89                  |
| yolov8s | TorchScript | ✅      | 42.9      | 0.4472      | 32.71                  |
| yolov8s | ONNX        | ✅      | 42.8      | 0.4472      | 43.42                  |
| yolov8s | OpenVINO    | ✅      | 42.9      | 0.4470      | 3.92                   |
| yolov8m | PyTorch     | ✅      | 49.7      | 0.5013      | 50.75                  |
| yolov8m | TorchScript | ✅      | 99.2      | 0.4999      | 47.90                  |
| yolov8m | ONNX        | ✅      | 99.0      | 0.4999      | 63.16                  |
| yolov8m | OpenVINO    | ✅      | 49.8      | 0.4997      | 7.11                   |
| yolov8l | PyTorch     | ✅      | 83.7      | 0.5293      | 77.45                  |
| yolov8l | TorchScript | ✅      | 167.2     | 0.5268      | 85.71                  |
| yolov8l | ONNX        | ✅      | 166.8     | 0.5268      | 88.94                  |
| yolov8l | OpenVINO    | ✅      | 167.0     | 0.5264      | 9.37                   |
| yolov8x | PyTorch     | ✅      | 130.5     | 0.5404      | 100.09                 |
| yolov8x | TorchScript | ✅      | 260.7     | 0.5371      | 114.64                 |
| yolov8x | ONNX        | ✅      | 260.4     | 0.5371      | 110.32                 |
| yolov8x | OpenVINO    | ✅      | 260.6     | 0.5367      | 15.02                  |

#### Flex GPU (FP16 Precision)

Benchmark results for Flex GPU at FP16 precision with `half=True`:

| Model   | Format      | Status | Size (MB) | mAP50-95(B) | Inference time (ms/im) |
|---------|-------------|--------|-----------|-------------|------------------------|
| yolov8n | PyTorch     | ✅      | 6.2       | 0.3709      | 24.39                  |
| yolov8n | TorchScript | ✅      | 12.4      | 0.3704      | 23.76                  |
| yolov8n | ONNX        | ✅      | 12.2      | 0.3704      | 37.37                  |
| yolov8n | OpenVINO    | ✅      | 6.3       | 0.3703      | 3.29                   |
| yolov8s | PyTorch     | ✅      | 21.5      | 0.4471      | 30.60                  |
| yolov8s | TorchScript | ✅      | 42.9      | 0.4472      | 34.40                  |
| yolov8s | ONNX        | ✅      | 42.8      | 0.4472      | 43.53                  |
| yolov8s | OpenVINO    | ✅      | 21.6      | 0.4470      | 3.91                   |
| yolov8m | PyTorch     | ✅      | 49.7      | 0.5013      | 50.75                  |
| yolov8m | TorchScript | ✅      | 99.2      | 0.4999      | 47.90                  |
| yolov8m | ONNX        | ✅      | 99.0      | 0.4999      | 63.16                  |
| yolov8m | OpenVINO    | ✅      | 49.8      | 0.4997      | 7.11                   |
| yolov8l | PyTorch     | ✅      | 83.7      | 0.5293      | 78.55                  |
| yolov8l | TorchScript | ✅      | 167.2     | 0.5268      | 84.75                  |
| yolov8l | ONNX        | ✅      | 166.8     | 0.5268      | 87.71                  |
| yolov8l | OpenVINO    | ✅      | 83.8      | 0.5264      | 9.38                   |
| yolov8x | PyTorch     | ✅      | 130.5     | 0.5404      | 94.85                  |
| yolov8x | TorchScript | ✅      | 260.7     | 0.5371      | 113.11                 |
| yolov8x | ONNX        | ✅      | 260.4     | 0.5371      | 111.00                 |
| yolov8x | OpenVINO    | ✅      | 130.6     | 0.5367      | 15.02                  |

### Xeon CPU

The Intel® Xeon® CPU is a high-performance, server-grade processor designed for complex and demanding workloads. From high-end cloud computing and virtualization to artificial intelligence and machine learning applications, Xeon® CPUs provide the power, reliability, and flexibility required for today's data centers.

Notably, Xeon® CPUs deliver high compute density and scalability, making them ideal for both small businesses and large enterprises. By choosing Intel® Xeon® CPUs, organizations can confidently handle their most demanding computing tasks and foster innovation while maintaining cost-effectiveness and operational efficiency.

### Arc GPU

Intel® Arc™ represents Intel's foray into the dedicated GPU market. The Arc™ series, designed to compete with leading GPU manufacturers like AMD and Nvidia, caters to both the laptop and desktop markets. The series includes mobile versions for compact devices like laptops, and larger, more powerful versions for desktop computers.

The Arc™ series is divided into three categories: Arc™ 3, Arc™ 5, and Arc™ 7, with each number indicating the performance level. Each category includes several models, and the 'M' in the GPU model name signifies a mobile, integrated variant.

Early reviews have praised the Arc™ series, particularly the integrated A770M GPU, for its impressive graphics performance. The availability of the Arc™ series varies by region, and additional models are expected to be released soon. Intel® Arc™ GPUs offer high-performance solutions for a range of computing needs, from gaming to content creation.