---
comments: true
description: Learn how to integrate Ultralytics YOLO11 with NVIDIA Triton Inference Server for scalable, high-performance AI model deployment.
keywords: Triton Inference Server, YOLO11, Ultralytics, NVIDIA, deep learning, AI model deployment, ONNX, scalable inference
---

# Triton Inference Server with Ultralytics YOLO11

The [Triton Inference Server](https://developer.nvidia.com/triton-inference-server) (formerly known as TensorRT Inference Server) is an open-source software solution developed by NVIDIA. It provides a cloud inference solution optimized for NVIDIA GPUs. Triton simplifies the deployment of AI models at scale in production. Integrating Ultralytics YOLO11 with Triton Inference Server allows you to deploy scalable, high-performance [deep learning](https://www.ultralytics.com/glossary/deep-learning-dl) inference workloads. This guide provides steps to set up and test the integration.

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

Triton Inference Server is designed to deploy a variety of AI models in production. It supports a wide range of deep learning and [machine learning](https://www.ultralytics.com/glossary/machine-learning-ml) frameworks, including TensorFlow, [PyTorch](https://www.ultralytics.com/glossary/pytorch), ONNX Runtime, and many others. Its primary use cases are:

- Serving multiple models from a single server instance.
- Dynamic model loading and unloading without server restart.
- Ensemble inference, allowing multiple models to be used together to achieve results.
- Model versioning for A/B testing and rolling updates.

## Prerequisites

Ensure you have the following prerequisites before proceeding:

- Docker installed on your machine.
- Install `tritonclient`:
    ```bash
    pip install tritonclient[all]
    ```

## Exporting YOLO11 to ONNX Format

Before deploying the model on Triton, it must be exported to the ONNX format. ONNX (Open Neural Network Exchange) is a format that allows models to be transferred between different deep learning frameworks. Use the `export` function from the `YOLO` class:

```python
from ultralytics import YOLO

# Load a model
model = YOLO("yolo11n.pt")  # load an official model

# Export the model
onnx_file = model.export(format="onnx", dynamic=True)
```

## Setting Up Triton Model Repository

The Triton Model Repository is a storage location where Triton can access and load models.

1. Create the necessary directory structure:

    ```python
    from pathlib import Path

    # Define paths
    model_name = "yolo"
    triton_repo_path = Path("tmp") / "triton_repo"
    triton_model_path = triton_repo_path / model_name

    # Create directories
    (triton_model_path / "1").mkdir(parents=True, exist_ok=True)
    ```

2. Move the exported ONNX model to the Triton repository:

    ```python
    from pathlib import Path

    # Move ONNX model to Triton Model path
    Path(onnx_file).rename(triton_model_path / "1" / "model.onnx")

    # Create config file
    (triton_model_path / "config.pbtxt").touch()

    # (Optional) Enable TensorRT for GPU inference
    # First run will be slow due to TensorRT engine conversion
    data = """
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
    """

    with open(triton_model_path / "config.pbtxt", "w") as f:
        f.write(data)
    ```

## Running Triton Inference Server

Run the Triton Inference Server using Docker:

```python
import contextlib
import subprocess
import time

from tritonclient.http import InferenceServerClient

# Define image https://catalog.ngc.nvidia.com/orgs/nvidia/containers/tritonserver
tag = "nvcr.io/nvidia/tritonserver:24.09-py3"  # 8.57 GB

# Pull the image
subprocess.call(f"docker pull {tag}", shell=True)

# Run the Triton server and capture the container ID
container_id = (
    subprocess.check_output(
        f"docker run -d --rm --gpus 0 -v {triton_repo_path}:/models -p 8000:8000 {tag} tritonserver --model-repository=/models",
        shell=True,
    )
    .decode("utf-8")
    .strip()
)

# Wait for the Triton server to start
triton_client = InferenceServerClient(url="localhost:8000", verbose=False, ssl=False)

# Wait until model is ready
for _ in range(10):
    with contextlib.suppress(Exception):
        assert triton_client.is_model_ready(model_name)
        break
    time.sleep(1)
```

Then run inference using the Triton Server model:

```python
from ultralytics import YOLO

# Load the Triton Server model
model = YOLO("http://localhost:8000/yolo", task="detect")

# Run inference on the server
results = model("path/to/image.jpg")
```

Cleanup the container:

```python
# Kill and remove the container at the end of the test
subprocess.call(f"docker kill {container_id}", shell=True)
```

---

By following the above steps, you can deploy and run Ultralytics YOLO11 models efficiently on Triton Inference Server, providing a scalable and high-performance solution for deep learning inference tasks. If you face any issues or have further queries, refer to the [official Triton documentation](https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/index.html) or reach out to the Ultralytics community for support.

## FAQ

### How do I set up Ultralytics YOLO11 with NVIDIA Triton Inference Server?

Setting up [Ultralytics YOLO11](https://docs.ultralytics.com/models/yolov8/) with [NVIDIA Triton Inference Server](https://developer.nvidia.com/triton-inference-server) involves a few key steps:

1. **Export YOLO11 to ONNX format**:

    ```python
    from ultralytics import YOLO

    # Load a model
    model = YOLO("yolo11n.pt")  # load an official model

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
    tag = "nvcr.io/nvidia/tritonserver:24.09-py3"

    subprocess.call(f"docker pull {tag}", shell=True)

    container_id = (
        subprocess.check_output(
            f"docker run -d --rm --gpus 0 -v {triton_repo_path}/models -p 8000:8000 {tag} tritonserver --model-repository=/models",
            shell=True,
        )
        .decode("utf-8")
        .strip()
    )

    triton_client = InferenceServerClient(url="localhost:8000", verbose=False, ssl=False)

    for _ in range(10):
        with contextlib.suppress(Exception):
            assert triton_client.is_model_ready(model_name)
            break
        time.sleep(1)
    ```

This setup can help you efficiently deploy YOLO11 models at scale on Triton Inference Server for high-performance AI model inference.

### What benefits does using Ultralytics YOLO11 with NVIDIA Triton Inference Server offer?

Integrating [Ultralytics YOLO11](../models/yolov8.md) with [NVIDIA Triton Inference Server](https://developer.nvidia.com/triton-inference-server) provides several advantages:

- **Scalable AI Inference**: Triton allows serving multiple models from a single server instance, supporting dynamic model loading and unloading, making it highly scalable for diverse AI workloads.
- **High Performance**: Optimized for NVIDIA GPUs, Triton Inference Server ensures high-speed inference operations, perfect for real-time applications such as [object detection](https://www.ultralytics.com/glossary/object-detection).
- **Ensemble and Model Versioning**: Triton's ensemble mode enables combining multiple models to improve results, and its model versioning supports A/B testing and rolling updates.

For detailed instructions on setting up and running YOLO11 with Triton, you can refer to the [setup guide](#setting-up-triton-model-repository).

### Why should I export my YOLO11 model to ONNX format before using Triton Inference Server?

Using ONNX (Open Neural Network Exchange) format for your [Ultralytics YOLO11](../models/yolov8.md) model before deploying it on [NVIDIA Triton Inference Server](https://developer.nvidia.com/triton-inference-server) offers several key benefits:

- **Interoperability**: ONNX format supports transfer between different deep learning frameworks (such as PyTorch, TensorFlow), ensuring broader compatibility.
- **Optimization**: Many deployment environments, including Triton, optimize for ONNX, enabling faster inference and better performance.
- **Ease of Deployment**: ONNX is widely supported across frameworks and platforms, simplifying the deployment process in various operating systems and hardware configurations.

To export your model, use:

```python
from ultralytics import YOLO

model = YOLO("yolo11n.pt")
onnx_file = model.export(format="onnx", dynamic=True)
```

You can follow the steps in the [exporting guide](../modes/export.md) to complete the process.

### Can I run inference using the Ultralytics YOLO11 model on Triton Inference Server?

Yes, you can run inference using the [Ultralytics YOLO11](../models/yolov8.md) model on [NVIDIA Triton Inference Server](https://developer.nvidia.com/triton-inference-server). Once your model is set up in the Triton Model Repository and the server is running, you can load and run inference on your model as follows:

```python
from ultralytics import YOLO

# Load the Triton Server model
model = YOLO("http://localhost:8000/yolo", task="detect")

# Run inference on the server
results = model("path/to/image.jpg")
```

For an in-depth guide on setting up and running Triton Server with YOLO11, refer to the [running triton inference server](#running-triton-inference-server) section.

### How does Ultralytics YOLO11 compare to [TensorFlow](https://www.ultralytics.com/glossary/tensorflow) and PyTorch models for deployment?

[Ultralytics YOLO11](https://docs.ultralytics.com/models/yolov8/) offers several unique advantages compared to TensorFlow and PyTorch models for deployment:

- **Real-time Performance**: Optimized for real-time object detection tasks, YOLO11 provides state-of-the-art [accuracy](https://www.ultralytics.com/glossary/accuracy) and speed, making it ideal for applications requiring live video analytics.
- **Ease of Use**: YOLO11 integrates seamlessly with Triton Inference Server and supports diverse export formats (ONNX, TensorRT, CoreML), making it flexible for various deployment scenarios.
- **Advanced Features**: YOLO11 includes features like dynamic model loading, model versioning, and ensemble inference, which are crucial for scalable and reliable AI deployments.

For more details, compare the deployment options in the [model deployment guide](../modes/export.md).
