---
comments: true
description: Learn how to integrate Ultralytics YOLO26 with NVIDIA Triton Inference Server for scalable, high-performance AI model deployment.
keywords: Triton Inference Server, YOLO26, Ultralytics, NVIDIA, deep learning, AI model deployment, ONNX, scalable inference
---

# Triton Inference Server with Ultralytics YOLO26

The [Triton Inference Server](https://developer.nvidia.com/dynamo-triton) (formerly known as TensorRT Inference Server) is an open-source software solution developed by NVIDIA. It provides a cloud inference solution optimized for NVIDIA GPUs. Triton simplifies the deployment of AI models at scale in production. Integrating [Ultralytics YOLO26](../models/yolo26.md) with Triton Inference Server allows you to deploy scalable, high-performance [deep learning](https://www.ultralytics.com/glossary/deep-learning-dl) inference workloads. This guide provides steps to set up and test the integration.

<p align="center">
  <br>
  <iframe loading="lazy" width="720" height="405" src="https://www.youtube.com/embed/NQDtfSi5QF4"
    title="Getting Started with NVIDIA Triton Inference Server" frameborder="0"
    allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share"
    allowfullscreen>
  </iframe>
  <br>
  <strong>Watch:</strong> Getting Started with NVIDIA Triton Inference Server.
</p>

## What is Triton Inference Server?

Triton Inference Server is designed to deploy a variety of AI models in production. It supports a wide range of deep learning and [machine learning](https://www.ultralytics.com/glossary/machine-learning-ml) frameworks, including [PyTorch](https://www.ultralytics.com/glossary/pytorch), [TensorFlow](../integrations/tf-savedmodel.md), [ONNX](../integrations/onnx.md), [OpenVINO](../integrations/openvino.md), [TensorRT](../integrations/tensorrt.md) and many others. Its primary use cases are:

- Serving multiple models from a single server instance
- Dynamic model loading and unloading without server restart
- Ensemble inference, allowing multiple models to be used together to achieve results
- Model versioning for A/B testing and rolling updates

## Key Benefits of Triton Inference Server

Using Triton Inference Server with [Ultralytics YOLO26](../models/yolo26.md) provides several advantages:

- **Automatic batching**: Groups multiple AI requests together before processing them, reducing latency and improving inference speed
- **Kubernetes integration**: Cloud-native design works seamlessly with Kubernetes for managing and scaling AI applications
- **Hardware-specific optimizations**: Takes full advantage of NVIDIA GPUs for maximum performance
- **Framework flexibility**: Supports multiple AI frameworks including [PyTorch](https://www.ultralytics.com/glossary/pytorch), [TensorFlow](../integrations/tf-savedmodel.md), [ONNX](../integrations/onnx.md), [OpenVINO](../integrations/openvino.md) and [TensorRT](../integrations/tensorrt.md)
- **Open-source and customizable**: Can be modified to fit specific needs, ensuring flexibility for various AI applications

## Prerequisites

Ensure you have the following prerequisites before proceeding:

- Docker or Podman installed on your machine
- Install `ultralytics`:
    ```bash
    pip install ultralytics
    ```
- Install `tritonclient`:
    ```bash
    pip install tritonclient[all]
    ```

## Setting Up Triton Inference Server

Run this full setup block to export [Ultralytics YOLO26](../models/yolo26.md) to [ONNX](../integrations/onnx.md), build the Triton model repository, and start Triton Inference Server:

!!! note

    Use the `runtime` switch in the script to choose your container engine:

    - Set `runtime = "docker"` for Docker
    - Set `runtime = "podman"` for Podman

```python
import contextlib
import subprocess
import time
from pathlib import Path

from tritonclient.http import InferenceServerClient

from ultralytics import YOLO

runtime = "docker"  # set to "podman" to use Podman

# 1) Exporting YOLO26 to ONNX Format

# Load a model
model = YOLO("yolo26n.pt")  # load an official model

# Retrieve metadata during export. Metadata needs to be added to config.pbtxt. See next section.
metadata = []


def export_cb(exporter):
    metadata.append(exporter.metadata)


model.add_callback("on_export_end", export_cb)

# Export the model
onnx_file = model.export(format="onnx", dynamic=True)


# 2) Setting Up Triton Model Repository

# Define paths
model_name = "yolo"
triton_repo_path = Path("tmp") / "triton_repo"
triton_model_path = triton_repo_path / model_name

# Create directories
(triton_model_path / "1").mkdir(parents=True, exist_ok=True)

# Move ONNX model to Triton Model path
Path(onnx_file).rename(triton_model_path / "1" / "model.onnx")

# Create config file
(triton_model_path / "config.pbtxt").touch()

data = """
# Add metadata
parameters {
  key: "metadata"
  value {
    string_value: "%s"
  }
}

# (Optional) Enable TensorRT for GPU inference
# First run will be slow due to TensorRT engine conversion
optimization {
  execution_accelerators {
    gpu_execution_accelerator {
      name: "tensorrt"
      parameters {
        key: "precision_mode"
        value: "FP16"
      }
      parameters {
        key: "max_workspace_size_bytes"
        value: "3221225472"
      }
      parameters {
        key: "trt_engine_cache_enable"
        value: "1"
      }
      parameters {
        key: "trt_engine_cache_path"
        value: "/models/yolo/1"
      }
    }
  }
}
""" % metadata[0]  # noqa

with open(triton_model_path / "config.pbtxt", "w") as f:
    f.write(data)

# 3) Running Triton Inference Server

# Define image https://catalog.ngc.nvidia.com/orgs/nvidia/containers/tritonserver
tag = "nvcr.io/nvidia/tritonserver:26.02-py3"  # 16.17 GB (Compressed Size)

subprocess.call(f"{runtime} pull {tag}", shell=True)

# GPU flags differ between Docker and Podman
gpu_flags = "--device nvidia.com/gpu=all" if runtime == "podman" else "--runtime=nvidia --gpus all"

container_name = "triton_server"

# Note: The :z flag on the volume mount is necessary for systems with SELinux (like Fedora/RHEL)
subprocess.call(
    f"{runtime} run -d --rm --name {container_name} {gpu_flags} -v {triton_repo_path.absolute()}:/models:z -p 8000:8000 {tag} tritonserver --model-repository=/models",
    shell=True,
)

# Wait for the Triton server to start
triton_client = InferenceServerClient(url="127.0.0.1:8000", verbose=False, ssl=False)

# Wait until model is ready
for _ in range(10):
    with contextlib.suppress(Exception):
        assert triton_client.is_model_ready(model_name)
        break
    time.sleep(1)
```

## Running Inference

Run inference using the Triton Server model:

```python
from ultralytics import YOLO

# Load the Triton Server model
model = YOLO("http://127.0.0.1:8000/yolo", task="detect")

# Run inference on the server
results = model("path/to/image.jpg")
```

Cleanup the container (`runtime` and `container_name` are defined in the setup block above):

```python
import subprocess

runtime = "docker"  # set to "podman" to use Podman
container_name = "triton_server"  # Kill the named container
subprocess.call(f"{runtime} kill {container_name}", shell=True)
```

## TensorRT Optimization (Optional)

For even greater performance, you can use [TensorRT](../integrations/tensorrt.md) with Triton Inference Server. TensorRT is a high-performance deep learning optimizer built specifically for NVIDIA GPUs that can significantly increase inference speed.

Key benefits of using [TensorRT](../integrations/tensorrt.md) with Triton include:

- Up to 36x faster inference compared to unoptimized models
- Hardware-specific optimizations for maximum GPU utilization
- Support for reduced precision formats (INT8, FP16) while maintaining accuracy
- Layer fusion to reduce computational overhead

To use TensorRT directly, you can export your [Ultralytics YOLO26](../models/yolo26.md) model to TensorRT format:

```python
from ultralytics import YOLO

# Load the YOLO26 model
model = YOLO("yolo26n.pt")

# Export the model to TensorRT format
model.export(format="engine")  # creates 'yolo26n.engine'
```

For more information on TensorRT optimization, see the [TensorRT integration guide](../integrations/tensorrt.md).

---

By following the above steps, you can deploy and run [Ultralytics YOLO26](../models/yolo26.md) models efficiently on Triton Inference Server, providing a scalable and high-performance solution for deep learning inference tasks. If you face any issues or have further queries, refer to the [official Triton documentation](https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/index.html) or reach out to the Ultralytics community for support.

## FAQ

### How do I set up Ultralytics YOLO26 with NVIDIA Triton Inference Server?

Setting up [Ultralytics YOLO26](../models/yolo26.md) with [NVIDIA Triton Inference Server](https://developer.nvidia.com/dynamo-triton) involves a few key steps:

1. **Export YOLO26 to ONNX format**:

    ```python
    from ultralytics import YOLO

    # Load a model
    model = YOLO("yolo26n.pt")  # load an official model

    # Export the model to ONNX format
    onnx_file = model.export(format="onnx", dynamic=True)
    ```

2. **Set up Triton Model Repository**:

    ```python
    from pathlib import Path

    # Define paths
    model_name = "yolo"
    triton_repo_path = Path("tmp") / "triton_repo"
    triton_model_path = triton_repo_path / model_name

    # Create directories
    (triton_model_path / "1").mkdir(parents=True, exist_ok=True)
    Path(onnx_file).rename(triton_model_path / "1" / "model.onnx")
    (triton_model_path / "config.pbtxt").touch()
    ```

3. **Run the Triton Server**:

    ```python
    import contextlib
    import subprocess
    import time

    from tritonclient.http import InferenceServerClient

    # Define image https://catalog.ngc.nvidia.com/orgs/nvidia/containers/tritonserver
    tag = "nvcr.io/nvidia/tritonserver:26.02-py3"

    runtime = "docker"  # set to "podman" to use Podman
    subprocess.call(f"{runtime} pull {tag}", shell=True)

    # GPU flags differ between Docker and Podman
    gpu_flags = "--device nvidia.com/gpu=all" if runtime == "podman" else "--runtime=nvidia --gpus all"

    container_name = "triton_server"
    subprocess.call(
        f"{runtime} run -d --rm --name {container_name} {gpu_flags} -v {triton_repo_path.absolute()}:/models:z -p 8000:8000 {tag} tritonserver --model-repository=/models",
        shell=True,
    )

    triton_client = InferenceServerClient(url="127.0.0.1:8000", verbose=False, ssl=False)

    for _ in range(10):
        with contextlib.suppress(Exception):
            assert triton_client.is_model_ready(model_name)
            break
        time.sleep(1)
    ```

This setup can help you efficiently deploy [Ultralytics YOLO26](../models/yolo26.md) models at scale on Triton Inference Server for high-performance AI model inference.

### What benefits does using Ultralytics YOLO26 with NVIDIA Triton Inference Server offer?

Integrating [Ultralytics YOLO26](../models/yolo26.md) with [NVIDIA Triton Inference Server](https://developer.nvidia.com/dynamo-triton) provides several advantages:

- **Scalable AI Inference**: Triton allows serving multiple models from a single server instance, supporting dynamic model loading and unloading, making it highly scalable for diverse AI workloads.
- **High Performance**: Optimized for NVIDIA GPUs, Triton Inference Server ensures high-speed inference operations, perfect for real-time applications such as [object detection](https://www.ultralytics.com/glossary/object-detection).
- **Ensemble and Model Versioning**: Triton's ensemble mode enables combining multiple models to improve results, and its model versioning supports A/B testing and rolling updates.
- **Automatic Batching**: Triton automatically groups multiple inference requests together, significantly improving throughput and reducing latency.
- **Simplified Deployment**: Gradual optimization of AI workflows without requiring complete system overhauls, making it easier to scale efficiently.

For detailed instructions on setting up and running [Ultralytics YOLO26](../models/yolo26.md) with Triton, refer to **Setting Up Triton Inference Server** and **Running Inference**.

### Why should I export my YOLO26 model to ONNX format before using Triton Inference Server?

Using ONNX (Open Neural Network Exchange) format for your [Ultralytics YOLO26](../models/yolo26.md) model before deploying it on [NVIDIA Triton Inference Server](https://developer.nvidia.com/dynamo-triton) offers several key benefits:

- **Interoperability**: ONNX format supports transfer between different deep learning frameworks (such as PyTorch, TensorFlow), ensuring broader compatibility.
- **Optimization**: Many deployment environments, including Triton, optimize for ONNX, enabling faster inference and better performance.
- **Ease of Deployment**: ONNX is widely supported across frameworks and platforms, simplifying the deployment process in various operating systems and hardware configurations.
- **Framework Independence**: Once converted to ONNX, your model is no longer tied to its original framework, making it more portable.
- **Standardization**: ONNX provides a standardized representation that helps overcome compatibility issues between different AI frameworks.

To export your model, use:

```python
from ultralytics import YOLO

model = YOLO("yolo26n.pt")
onnx_file = model.export(format="onnx", dynamic=True)
```

You can follow the steps in the [ONNX integration guide](https://docs.ultralytics.com/integrations/onnx/) to complete the process.

### Can I run inference using the Ultralytics YOLO26 model on Triton Inference Server?

Yes, you can run inference using the [Ultralytics YOLO26](../models/yolo26.md) model on [NVIDIA Triton Inference Server](https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/index.html). Once your model is set up in the Triton Model Repository and the server is running, you can load and run inference on your model as follows:

```python
from ultralytics import YOLO

# Load the Triton Server model
model = YOLO("http://127.0.0.1:8000/yolo", task="detect")

# Run inference on the server
results = model("path/to/image.jpg")
```

This approach allows you to leverage Triton's optimizations while using the familiar Ultralytics YOLO interface.

### How does Ultralytics YOLO26 compare to TensorFlow and PyTorch models for deployment?

[Ultralytics YOLO26](../models/yolo26.md) offers several unique advantages compared to [TensorFlow](https://www.ultralytics.com/glossary/tensorflow) and PyTorch models for deployment:

- **Real-time Performance**: Optimized for real-time object detection tasks, [Ultralytics YOLO26](../models/yolo26.md) provides state-of-the-art [accuracy](https://www.ultralytics.com/glossary/accuracy) and speed, making it ideal for applications requiring live video analytics.
- **Ease of Use**: [Ultralytics YOLO26](../models/yolo26.md) integrates seamlessly with Triton Inference Server and supports diverse export formats ([ONNX](../integrations/onnx.md), [TensorRT](../integrations/tensorrt.md)), making it flexible for various deployment scenarios.
- **Advanced Features**: [Ultralytics YOLO26](../models/yolo26.md) includes features like dynamic model loading, model versioning, and ensemble inference, which are crucial for scalable and reliable AI deployments.
- **Simplified API**: The Ultralytics API provides a consistent interface across different deployment targets, reducing the learning curve and development time.
- **Edge Optimization**: [Ultralytics YOLO26](../models/yolo26.md) models are designed with edge deployment in mind, offering excellent performance even on resource-constrained devices.

For more details, compare the deployment options in the [model export guide](../modes/export.md).
