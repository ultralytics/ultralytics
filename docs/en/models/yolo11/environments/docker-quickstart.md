---
comments: true
description: Run Ultralytics YOLO11 in Docker. Setup guide for CPU and GPU containers, volume mounting, and running detection, segmentation, and pose estimation tasks.
keywords: YOLO11, Docker, Ultralytics, container, GPU, NVIDIA, object detection, segmentation, pose estimation, machine learning, containerization, deep learning
canonical: https://docs.ultralytics.com/models/yolo11/environments/docker-quickstart/
---

<script type="application/ld+json">
{
  "@context": "https://schema.org",
  "@type": "TechArticle",
  "headline": "YOLO11 Docker Quickstart",
  "description": "Run Ultralytics YOLO11 in Docker. Setup guide for CPU and GPU containers, volume mounting, and running detection, segmentation, and pose estimation tasks.",
  "url": "https://docs.ultralytics.com/models/yolo11/environments/docker-quickstart/",
  "image": "https://cdn.jsdelivr.net/gh/ultralytics/assets@main/docs/gcp-running-docker.avif",
  "datePublished": "2026-06-04",
  "dateModified": "2026-06-04",
  "author": {"@type": "Organization", "name": "Ultralytics", "url": "https://www.ultralytics.com"},
  "publisher": {"@type": "Organization", "name": "Ultralytics", "url": "https://www.ultralytics.com"},
  "mainEntityOfPage": "https://docs.ultralytics.com/models/yolo11/environments/docker-quickstart/"
}
</script>

# YOLO11 Docker Quickstart

<!-- NOTE FOR MURAT: Please confirm the correct Docker Hub image tag for YOLO11 (ultralytics/ultralytics:latest should work as it ships with the current ultralytics package). Add Docker Pulls badge and screenshots where useful. -->

This guide shows you how to run [Ultralytics YOLO11](../../../models/yolo11.md) inside a [Docker](https://www.ultralytics.com/glossary/docker) container. Containerisation provides a clean, reproducible environment with no dependency conflicts — ideal for both development and production deployments.

For alternative cloud setups, see the [YOLO11 GCP Quickstart](./google-cloud-quickstart.md) or [AWS Quickstart](./aws-quickstart.md). For a general Docker guide, see the [Ultralytics Docker Quickstart Guide](../../../guides/docker-quickstart.md).

## Prerequisites

1. **Docker** — Download from the [official Docker website](https://docs.docker.com/get-started/get-docker/).
2. **NVIDIA Drivers** (GPU only) — Version 455.23 or higher from [NVIDIA's site](https://www.nvidia.com/Download/index.aspx).
3. **NVIDIA Container Toolkit** (GPU only) — Follow the [official install guide](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html).

### Set Up NVIDIA Container Toolkit

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

Verify the runtime is registered:

```bash
docker info | grep -i runtime
```

## Step 1: Pull the Ultralytics Image

```bash
sudo docker pull ultralytics/ultralytics:latest
```

The image includes the `ultralytics` package, PyTorch, and CUDA — everything needed to run YOLO11.

## Step 2: Run the Container

### CPU Only

```bash
sudo docker run -it --ipc=host ultralytics/ultralytics:latest
```

### GPU

```bash
# All GPUs
sudo docker run -it --runtime=nvidia --ipc=host --gpus all ultralytics/ultralytics:latest

# Specific GPUs
sudo docker run -it --runtime=nvidia --ipc=host --gpus '"device=0,1"' ultralytics/ultralytics:latest
```

### Mount Local Files

```bash
sudo docker run -it --runtime=nvidia --ipc=host --gpus all \
  -v /path/on/host:/path/in/container \
  ultralytics/ultralytics:latest
```

## Step 3: Use YOLO11 Inside the Container

=== "CLI"

    ```bash
    # Inference on a sample image
    yolo predict model=yolo11n.pt source="https://ultralytics.com/images/bus.jpg"

    # Train on COCO128
    yolo train model=yolo11s.pt data=coco128.yaml epochs=10 imgsz=640

    # Validate
    yolo val model=yolo11s.pt data=coco128.yaml

    # Export to ONNX
    yolo export model=yolo11n.pt format=onnx
    ```

=== "Python"

    ```python
    from ultralytics import YOLO

    model = YOLO("yolo11n.pt")

    results = model("https://ultralytics.com/images/bus.jpg")
    results[0].show()

    model.train(data="coco128.yaml", epochs=10, imgsz=640)
    model.export(format="onnx")
    ```

### Task-Specific Models

```bash
# Instance segmentation
yolo predict model=yolo11n-seg.pt source="https://ultralytics.com/images/bus.jpg"

# Pose estimation
yolo predict model=yolo11n-pose.pt source="https://ultralytics.com/images/bus.jpg"

# Classification
yolo predict model=yolo11n-cls.pt source="https://ultralytics.com/images/bus.jpg"
```

Available variants: `yolo11n`, `yolo11s`, `yolo11m`, `yolo11l`, `yolo11x`.

## What's Next

- Read the [YOLO11 model documentation](../../../models/yolo11.md) for benchmarks and task guides.
- Explore [Export Mode](../../../modes/export.md) to convert YOLO11 to TensorRT, CoreML, or TFLite.
- Try [Ultralytics Platform](../../../platform/index.md) for no-code training and deployment.
- For the latest Ultralytics model with NMS-free inference, upgrade to [YOLO26](../../../models/yolo26.md).
