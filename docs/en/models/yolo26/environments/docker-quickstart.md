---
comments: true
description: Run Ultralytics YOLO26 in Docker. Setup guide for CPU and GPU containers, volume mounting, and NMS-free object detection in an isolated environment.
keywords: YOLO26, Docker, Ultralytics, container, GPU, NVIDIA, object detection, machine learning, deep learning, NMS-free, deployment, containerization
canonical: https://docs.ultralytics.com/models/yolo26/environments/docker-quickstart/
---

<script type="application/ld+json">
{
  "@context": "https://schema.org",
  "@type": "TechArticle",
  "headline": "YOLO26 Docker Quickstart",
  "description": "Run Ultralytics YOLO26 in Docker. Setup guide for CPU and GPU containers, volume mounting, and NMS-free object detection in an isolated environment.",
  "url": "https://docs.ultralytics.com/models/yolo26/environments/docker-quickstart/",
  "image": "https://cdn.jsdelivr.net/gh/ultralytics/assets@main/docs/gcp-running-docker.avif",
  "datePublished": "2026-06-04",
  "dateModified": "2026-06-04",
  "author": {"@type": "Organization", "name": "Ultralytics", "url": "https://www.ultralytics.com"},
  "publisher": {"@type": "Organization", "name": "Ultralytics", "url": "https://www.ultralytics.com"},
  "mainEntityOfPage": "https://docs.ultralytics.com/models/yolo26/environments/docker-quickstart/"
}
</script>

# YOLO26 Docker Quickstart

<!-- NOTE FOR MURAT: Please verify the Docker Hub image tag for YOLO26 (is it ultralytics/ultralytics:latest or a dedicated yolo26 tag?). Confirm whether there is a dedicated YOLO26 image on Docker Hub and add the correct Docker pull badge. Also add screenshots where relevant. -->

This guide shows you how to run [Ultralytics YOLO26](../../../models/yolo26.md) inside a [Docker](https://www.ultralytics.com/glossary/docker) container. Containerisation gives you a clean, reproducible environment — no dependency conflicts, consistent behaviour across machines, and easy deployment.

For alternative cloud environments, see the [YOLO26 GCP Quickstart](./google-cloud-quickstart.md) or [AWS Quickstart](./aws-quickstart.md). For a general Docker guide covering all Ultralytics models, see the [Docker Quickstart Guide](../../../guides/docker-quickstart.md).

## Prerequisites

Before starting, install the following:

1. **Docker** — Download from the [official Docker website](https://docs.docker.com/get-started/get-docker/).
2. **NVIDIA Drivers** (GPU only) — Version 455.23 or higher from [NVIDIA's site](https://www.nvidia.com/Download/index.aspx).
3. **NVIDIA Container Toolkit** (GPU only) — Follow the [official install guide](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html).

### Verify GPU Access

```bash
nvidia-smi
```

Then install the NVIDIA Container Toolkit:

=== "Ubuntu/Debian"

    ```bash
    curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey \
      | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg \
      && curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list \
      | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' \
      | sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
    sudo apt-get update && sudo apt-get install -y nvidia-container-toolkit
    sudo nvidia-ctk runtime configure --runtime=docker
    sudo systemctl restart docker
    ```

=== "RHEL/CentOS/Fedora"

    ```bash
    curl -s -L https://nvidia.github.io/libnvidia-container/stable/rpm/nvidia-container-toolkit.repo \
      | sudo tee /etc/yum.repos.d/nvidia-container-toolkit.repo
    sudo dnf install -y nvidia-container-toolkit
    sudo nvidia-ctk runtime configure --runtime=docker
    sudo systemctl restart docker
    ```

Confirm the `nvidia` runtime is registered:

```bash
docker info | grep -i runtime
```

## Step 1: Pull the Ultralytics Image

Ultralytics publishes an official image on [Docker Hub](https://hub.docker.com/r/ultralytics/ultralytics). It includes the latest `ultralytics` package, PyTorch, and CUDA dependencies:

```bash
# Pull the latest Ultralytics image
sudo docker pull ultralytics/ultralytics:latest
```

## Step 2: Run the Container

### CPU Only

```bash
sudo docker run -it --ipc=host ultralytics/ultralytics:latest
```

### GPU

```bash
# All GPUs
sudo docker run -it --runtime=nvidia --ipc=host --gpus all ultralytics/ultralytics:latest

# Specific GPUs (e.g., 0 and 1)
sudo docker run -it --runtime=nvidia --ipc=host --gpus '"device=0,1"' ultralytics/ultralytics:latest
```

### Mount Local Files

To access datasets and save model weights locally, mount a host directory:

```bash
sudo docker run -it --runtime=nvidia --ipc=host --gpus all \
  -v /path/on/host:/path/in/container \
  ultralytics/ultralytics:latest
```

## Step 3: Use YOLO26 Inside the Container

Once inside the container, run YOLO26 commands immediately — the `ultralytics` package is pre-installed:

=== "CLI"

    ```bash
    # Inference on an image
    yolo predict model=yolo26n.pt source="https://ultralytics.com/images/bus.jpg"

    # Train YOLO26 small on COCO128
    yolo train model=yolo26s.pt data=coco128.yaml epochs=10 imgsz=640

    # Validate a checkpoint
    yolo val model=yolo26s.pt data=coco128.yaml

    # Export to ONNX
    yolo export model=yolo26n.pt format=onnx
    ```

=== "Python"

    ```python
    from ultralytics import YOLO

    model = YOLO("yolo26n.pt")

    # Inference
    results = model("https://ultralytics.com/images/bus.jpg")
    results[0].show()

    # Train
    model.train(data="coco128.yaml", epochs=10, imgsz=640)

    # Export
    model.export(format="onnx")
    ```

Available variants: `yolo26n`, `yolo26s`, `yolo26m`, `yolo26l`, `yolo26x`.

## What's Next

- See [YOLO26 model documentation](../../../models/yolo26.md) for benchmarks and architecture details.
- Explore [Export Mode](../../../modes/export.md) to convert YOLO26 to TensorRT, CoreML, or TFLite.
- Try [Ultralytics Platform](../../../platform/index.md) for a no-code training and deployment workflow.
- Try [GCP](./google-cloud-quickstart.md) or [AWS](./aws-quickstart.md) if you need a managed cloud environment.
