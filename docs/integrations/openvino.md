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