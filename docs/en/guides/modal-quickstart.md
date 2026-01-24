---
comments: true
description: Learn how to run Ultralytics YOLO26 on Modal's serverless cloud platform. Quickstart guide for scalable GPU inference and training without managing infrastructure.
keywords: YOLO26, Modal, serverless, cloud computing, machine learning, GPU, inference, training, Ultralytics, Python, object detection, deep learning
---

# Quick Start Guide: Modal with Ultralytics YOLO26

This guide provides a comprehensive introduction to deploying [Ultralytics YOLO26](https://docs.ultralytics.com/models/yolo26/) models on [Modal](https://modal.com/), a serverless [cloud computing](https://www.ultralytics.com/glossary/cloud-computing) platform designed for AI and [machine learning](https://www.ultralytics.com/glossary/machine-learning-ml) workloads.

## What is Modal?

[Modal](https://modal.com/) is a serverless cloud platform built specifically for AI and machine learning applications. Unlike traditional cloud providers that require managing virtual machines, containers, and infrastructure, Modal lets you define your environment entirely in Python code. The platform handles provisioning, scaling, and execution automatically, making it ideal for running [deep learning](https://www.ultralytics.com/glossary/deep-learning-dl) models like YOLO26.

## How Does Modal Benefit YOLO Users?

Modal offers several compelling advantages for running Ultralytics YOLO26 models:

- **Zero Infrastructure Management**: Define your environment in Python and run—no VMs, drivers, or container orchestration needed.
- **Instant GPU Access**: Access NVIDIA GPUs (T4, A10G, A100, H100) on-demand without reservations or long-term commitments.
- **Automatic Scaling**: Scale from zero to hundreds of containers automatically based on demand, perfect for variable [inference](https://www.ultralytics.com/glossary/inference) workloads.
- **Pay-Per-Use Pricing**: Only pay for actual compute time down to the second, with no idle costs.
- **Fast Cold Starts**: Optimized container caching enables YOLO26 models to start running in seconds.
- **Simple Python Interface**: Deploy models using familiar Python decorators without learning new frameworks.

## Prerequisites

Before you begin, ensure you have:

- A Modal account (sign up for free at [modal.com](https://modal.com/))
- Python 3.9 or later installed on your local machine
- Basic familiarity with [Python](https://www.ultralytics.com/glossary/python) and command-line tools

## Installation

Install the Modal Python package and authenticate:

```bash
# Install Modal
pip install modal

# Authenticate with Modal (opens browser)
modal token new
```

!!! tip "Authentication"

    The `modal token new` command will open a browser window to authenticate your Modal account. After authentication, you can run Modal commands from the terminal.

## Quickstart: Running YOLO26 Inference

Create a new Python file called `modal_yolo.py` with the following code:

!!! example "YOLO26 Inference on Modal"

    ```python
    """
    Modal + Ultralytics YOLO26 Quickstart
    Run: modal run modal_yolo.py.
    """

    import modal

    # Create Modal app
    app = modal.App("ultralytics-yolo")

    # Define container image with Ultralytics
    image = (
        modal.Image.debian_slim(python_version="3.11")
        .apt_install("libgl1", "libglib2.0-0")  # Required for OpenCV
        .pip_install("ultralytics")
    )


    @app.function(image=image)
    def predict(image_url: str):
        """Run YOLO26 inference on an image URL."""
        from ultralytics import YOLO

        # Load model (downloads automatically on first run)
        model = YOLO("yolo26n.pt")

        # Run inference
        results = model(image_url)

        # Extract detection results
        detections = []
        for r in results:
            for box in r.boxes:
                detections.append(
                    {
                        "class": model.names[int(box.cls)],
                        "confidence": float(box.conf),
                        "bbox": box.xyxy[0].tolist(),
                    }
                )
        return detections


    @app.local_entrypoint()
    def main():
        """Test inference with sample image."""
        image_url = "https://ultralytics.com/images/bus.jpg"
        print(f"Running YOLO26 inference on: {image_url}")

        results = predict.remote(image_url)

        print(f"\nDetected {len(results)} objects:")
        for det in results:
            print(f"  - {det['class']}: {det['confidence']:.2f}")
    ```

Run the inference:

```bash
modal run modal_yolo.py
```

Expected output:

```
Running YOLO26 inference on: https://ultralytics.com/images/bus.jpg

Detected 5 objects:
  - bus: 0.94
  - person: 0.89
  - person: 0.88
  - person: 0.86
  - person: 0.62
```

!!! note "OpenCV Dependencies"

    The `apt_install("libgl1", "libglib2.0-0")` line is required because Ultralytics uses [OpenCV](https://opencv.org/) for image processing, which needs these system libraries in headless environments.

## Using GPU for Faster Inference

For faster inference, add a GPU to your function by specifying the `gpu` parameter:

!!! example "GPU-Accelerated Inference"

    ```python
    @app.function(image=image, gpu="T4")  # Options: "T4", "A10G", "A100-40GB", "H100"
    def predict_gpu(image_url: str):
        """Run YOLO26 inference on GPU."""
        from ultralytics import YOLO

        model = YOLO("yolo26n.pt")
        results = model(image_url)

        detections = []
        for r in results:
            for box in r.boxes:
                detections.append(
                    {
                        "class": model.names[int(box.cls)],
                        "confidence": float(box.conf),
                        "bbox": box.xyxy[0].tolist(),
                    }
                )
        return detections
    ```

The available GPU options are:

| GPU       | Memory | Best For                                                                                    |
| --------- | ------ | ------------------------------------------------------------------------------------------- |
| T4        | 16 GB  | Inference, small model training                                                             |
| A10G      | 24 GB  | Medium training jobs, larger [batch sizes](https://www.ultralytics.com/glossary/batch-size) |
| A100-40GB | 40 GB  | Large-scale training, fine-tuning large models                                              |
| A100-80GB | 80 GB  | Large models, bigger batch sizes                                                            |
| H100      | 80 GB  | Maximum performance for intensive training                                                  |

## Training YOLO26 on Modal

For [model training](https://www.ultralytics.com/glossary/model-training), you'll want persistent storage for datasets and checkpoints. Modal provides [Volumes](https://modal.com/docs/guide/volumes) for this purpose:

!!! example "YOLO26 Training with Persistent Storage"

    ```python
    import modal

    app = modal.App("ultralytics-training")

    # Create a volume for storing datasets and checkpoints
    volume = modal.Volume.from_name("yolo-training-vol", create_if_missing=True)

    image = modal.Image.debian_slim(python_version="3.11").apt_install("libgl1", "libglib2.0-0").pip_install("ultralytics")


    @app.function(
        image=image,
        gpu="A10G",  # Use A10G or A100 for training
        timeout=3600,  # 1 hour timeout
        volumes={"/data": volume},
    )
    def train_yolo():
        """Train YOLO26 model on Modal with GPU."""
        from ultralytics import YOLO

        # Load model
        model = YOLO("yolo26n.pt")

        # Train using built-in COCO8 dataset
        # Weights are auto-saved to /data/runs/train/weights/best.pt
        results = model.train(
            data="coco8.yaml",
            epochs=10,
            imgsz=640,
            project="/data/runs",
        )

        return "Training complete! Weights saved to /data/runs/train/weights/"


    @app.local_entrypoint()
    def main():
        result = train_yolo.remote()
        print(result)
    ```

Run the training:

```bash
modal run train_yolo.py
```

!!! tip "Volume Persistence"

    Modal Volumes persist data between function runs. Your trained models and checkpoints will be available for subsequent runs, making it easy to resume training or deploy trained models.

## Deploying as a Web Endpoint

Modal can expose your YOLO26 model as a persistent HTTP endpoint for real-time [object detection](https://www.ultralytics.com/glossary/object-detection):

!!! example "YOLO26 Web API"

    ```python
    import modal

    app = modal.App("yolo-api")

    image = (
        modal.Image.debian_slim(python_version="3.11")
        .apt_install("libgl1", "libglib2.0-0")
        .pip_install("ultralytics", "fastapi")
    )


    @app.function(image=image, gpu="T4")
    @modal.fastapi_endpoint(method="POST")
    def detect(image_url: str):
        """YOLO26 detection API endpoint."""
        from ultralytics import YOLO

        model = YOLO("yolo26n.pt")
        results = model(image_url)

        detections = []
        for r in results:
            for box in r.boxes:
                detections.append(
                    {
                        "class": model.names[int(box.cls)],
                        "confidence": float(box.conf),
                        "bbox": box.xyxy[0].tolist(),
                    }
                )

        return {"detections": detections, "count": len(detections)}
    ```

Deploy the endpoint:

```bash
modal deploy yolo_api.py
```

This creates a persistent URL like `https://your-username--yolo-api-detect.modal.run` that you can call from anywhere.

## Best Practices

### Caching Model Downloads

To avoid downloading the model on every cold start, cache it in the container image:

```python
def download_model():
    """Download and cache YOLO26 model during image build."""
    from ultralytics import YOLO

    YOLO("yolo26n.pt")  # Downloads and caches the model


image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("libgl1", "libglib2.0-0")
    .pip_install("ultralytics")
    .run_function(download_model)  # Cache model in image
)
```

### Batch Processing

For processing multiple images efficiently in a single function call:

```python
@app.function(image=image, gpu="T4")
def predict_batch(image_urls: list[str]):
    """Process multiple images in one function call."""
    from ultralytics import YOLO

    model = YOLO("yolo26n.pt")

    all_results = []
    for url in image_urls:
        results = model(url)
        detections = []
        for r in results:
            for box in r.boxes:
                detections.append(
                    {
                        "class": model.names[int(box.cls)],
                        "confidence": float(box.conf),
                        "bbox": box.xyxy[0].tolist(),
                    }
                )
        all_results.append({"url": url, "detections": detections})

    return all_results
```

!!! tip "Parallel Processing"

    For even higher throughput, use Modal's `.map()` method to process images in parallel across multiple containers:

    ```python
    results = list(predict.map(image_urls))
    ```

## Next Steps

Congratulations! You have successfully set up Ultralytics YOLO26 on Modal. For further learning and support:

- Explore the [Ultralytics YOLO26 documentation](../models/yolo26.md) for advanced model features
- Learn about [training custom models](../modes/train.md) with your own datasets
- Check out [export options](../modes/export.md) for deploying models in different formats
- Visit the [Modal documentation](https://modal.com/docs) for advanced platform features

## FAQ

### How do I choose the right GPU for my YOLO26 workload?

For inference with YOLO26n through YOLO26l models, an NVIDIA T4 (16 GB) is typically sufficient and cost-effective. For training or running larger models like YOLO26x, consider using A10G (24 GB) or A100-40GB GPUs. The H100 (80 GB) is recommended only for intensive training jobs with large [batch sizes](https://www.ultralytics.com/glossary/batch-size) or very large custom datasets.

### How much does it cost to run YOLO26 on Modal?

Modal uses pay-per-second pricing with no idle costs. Approximate rates (check [Modal pricing](https://modal.com/pricing) for current rates):

| Resource | Cost per Hour |
| -------- | ------------- |
| CPU      | ~$0.05        |
| T4 GPU   | ~$0.59        |
| A10G GPU | ~$1.10        |
| A100 GPU | ~$2.10        |
| H100 GPU | ~$3.95        |

### Can I use my own custom-trained YOLO model?

Yes! You can load custom models from a Modal Volume or download them at runtime:

```python
# From a Modal Volume
model = YOLO("/data/my_custom_model.pt")

# From a URL
model = YOLO("https://your-storage.com/my_model.pt")
```

For more information on training custom models, see the [training guide](../modes/train.md).

### How do I handle large datasets for training?

Use Modal Volumes to store and access large datasets:

1. Create a volume: `modal volume create my-dataset`
2. Upload data: `modal volume put my-dataset ./local_data /remote_path`
3. Mount in your function using the `volumes` parameter

For very large datasets, consider using Modal's [CloudBucketMount](https://modal.com/docs/guide/cloud-bucket-mounts) to stream data directly from Amazon S3 or Google Cloud Storage.

### What other YOLO tasks can I run on Modal?

Modal supports all YOLO26 tasks including:

- **Object Detection**: Detect objects in images and videos
- **[Instance Segmentation](https://www.ultralytics.com/glossary/instance-segmentation)**: Segment individual objects with pixel-level masks
- **[Pose Estimation](https://www.ultralytics.com/glossary/pose-estimation)**: Detect human body keypoints
- **Oriented Bounding Boxes (OBB)**: Detect rotated objects
- **Classification**: Classify entire images

See the [tasks documentation](../tasks/index.md) for more details on each task type.
