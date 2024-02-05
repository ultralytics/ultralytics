---
comments: true
description: A step-by-step guide on integrating Ultralytics YOLOv8 with Triton Inference Server for scalable and high-performance deep learning inference deployments.
keywords: YOLOv8, Triton Inference Server, ONNX, Deep Learning Deployment, Scalable Inference, Ultralytics, NVIDIA, Object Detection, Cloud Inference
---

# Triton Inference Server with Ultralytics YOLOv8

The [Triton Inference Server](https://developer.nvidia.com/nvidia-triton-inference-server) (formerly known as TensorRT Inference Server) is an open-source software solution developed by NVIDIA. It provides a cloud inference solution optimized for NVIDIA GPUs. Triton simplifies the deployment of AI models at scale in production. Integrating Ultralytics YOLOv8 with Triton Inference Server allows you to deploy scalable, high-performance deep learning inference workloads. This guide provides steps to set up and test the integration.

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

Triton Inference Server is designed to deploy a variety of AI models in production. It supports a wide range of deep learning and machine learning frameworks, including TensorFlow, PyTorch, ONNX Runtime, and many others. Its primary use cases are:

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

## Exporting YOLOv8 to ONNX Format

Before deploying the model on Triton, it must be exported to the ONNX format. ONNX (Open Neural Network Exchange) is a format that allows models to be transferred between different deep learning frameworks. Use the `export` function from the `YOLO` class:

```python
from ultralytics import YOLO

# Load a model
model = YOLO('yolov8n.pt')  # load an official model

# Export the model
onnx_file = model.export(format='onnx', dynamic=True)
```

## Setting Up Triton Model Repository

The Triton Model Repository is a storage location where Triton can access and load models.

1. Create the necessary directory structure:

    ```python
    from pathlib import Path

    # Define paths
    triton_repo_path = Path('tmp') / 'triton_repo'
    triton_model_path = triton_repo_path / 'yolo'

    # Create directories
    (triton_model_path / '1').mkdir(parents=True, exist_ok=True)
    ```

2. Move the exported ONNX model to the Triton repository:

    ```python
    from pathlib import Path

    # Move ONNX model to Triton Model path
    Path(onnx_file).rename(triton_model_path / '1' / 'model.onnx')

    # Create config file
    (triton_model_path / 'config.pbtxt').touch()
    ```

## Running Triton Inference Server

Run the Triton Inference Server using Docker:

```python
import subprocess
import time

from tritonclient.http import InferenceServerClient

# Define image https://catalog.ngc.nvidia.com/orgs/nvidia/containers/tritonserver
tag = 'nvcr.io/nvidia/tritonserver:23.09-py3'  # 6.4 GB

# Pull the image
subprocess.call(f'docker pull {tag}', shell=True)

# Run the Triton server and capture the container ID
container_id = subprocess.check_output(
    f'docker run -d --rm -v {triton_repo_path}:/models -p 8000:8000 {tag} tritonserver --model-repository=/models',
    shell=True).decode('utf-8').strip()

# Wait for the Triton server to start
triton_client = InferenceServerClient(url='localhost:8000', verbose=False, ssl=False)

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
model = YOLO(f'http://localhost:8000/yolo', task='detect')

# Run inference on the server
results = model('path/to/image.jpg')
```

Cleanup the container:

```python
# Kill and remove the container at the end of the test
subprocess.call(f'docker kill {container_id}', shell=True)
```

---

By following the above steps, you can deploy and run Ultralytics YOLOv8 models efficiently on Triton Inference Server, providing a scalable and high-performance solution for deep learning inference tasks. If you face any issues or have further queries, refer to the [official Triton documentation](https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/index.html) or reach out to the Ultralytics community for support.
