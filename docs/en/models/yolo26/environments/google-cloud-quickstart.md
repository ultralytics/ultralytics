---
comments: true
description: Deploy Ultralytics YOLO26 on GCP Deep Learning VM. GPU-accelerated setup guide for real-time object detection with YOLO26's end-to-end NMS-free inference.
keywords: YOLO26, Google Cloud Platform, GCP, Deep Learning VM, object detection, Ultralytics, computer vision, GPU, cloud computing, machine learning, NMS-free, edge AI
canonical: https://docs.ultralytics.com/models/yolo26/environments/google-cloud-quickstart/
---

<script type="application/ld+json">
{
  "@context": "https://schema.org",
  "@type": "TechArticle",
  "headline": "YOLO26 on Google Cloud Platform (GCP) Deep Learning VM",
  "description": "Deploy Ultralytics YOLO26 on GCP Deep Learning VM. GPU-accelerated setup guide for real-time object detection with YOLO26's end-to-end NMS-free inference.",
  "url": "https://docs.ultralytics.com/models/yolo26/environments/google-cloud-quickstart/",
  "image": "https://cdn.jsdelivr.net/gh/ultralytics/assets@main/docs/gcp-deep-learning-vm-setup.avif",
  "datePublished": "2026-06-04",
  "dateModified": "2026-06-04",
  "author": {"@type": "Organization", "name": "Ultralytics", "url": "https://www.ultralytics.com"},
  "publisher": {"@type": "Organization", "name": "Ultralytics", "url": "https://www.ultralytics.com"},
  "mainEntityOfPage": "https://docs.ultralytics.com/models/yolo26/environments/google-cloud-quickstart/"
}
</script>

# YOLO26 on Google Cloud Platform (GCP) Deep Learning VM

<!-- NOTE FOR MURAT: Please verify the recommended GCP instance type (n1-standard-8 still correct?), GPU recommendations for YOLO26 model sizes, and whether GCP Deep Learning VM images ship with ultralytics>=26.0.0 or require a manual pip upgrade. Also please add screenshots/images for each step. -->

This guide walks you through deploying [Ultralytics YOLO26](../../../models/yolo26.md) on a [Google Cloud Platform (GCP)](https://cloud.google.com/) Deep Learning VM. YOLO26 is the latest Ultralytics model family, delivering **40.9–57.5 mAP on COCO** with native end-to-end NMS-free inference, making it well-suited to GPU-accelerated cloud environments.

🆓 New GCP users receive a [$300 free credit](https://cloud.google.com/free/docs/free-cloud-features) to get started.

For other deployment options, see the [YOLO26 Docker Quickstart](./docker-quickstart.md), [AWS Quickstart](./aws-quickstart.md), or the [Ultralytics Docker Quickstart Guide](../../../guides/docker-quickstart.md).

## Step 1: Create a Deep Learning VM

1. Open the [GCP Marketplace](https://cloud.google.com/marketplace) and select **Deep Learning VM**.
2. Choose an **n1-standard-8** instance (8 vCPUs, 30 GB RAM) — a balanced starting point for most YOLO26 workloads.
3. Select a GPU. An **NVIDIA T4** is cost-effective for inference and lighter training; an **A100** or **V100** is preferred for full-scale YOLO26 training runs.
4. Enable **"Install NVIDIA GPU driver automatically on first startup"** for a seamless CUDA setup.
5. Allocate at least **300 GB SSD Persistent Disk** to avoid I/O bottlenecks with large datasets.
6. Click **Deploy** and wait for GCP to provision the VM.

The Deep Learning VM ships with [PyTorch](https://pytorch.org/), CUDA drivers, and the Anaconda distribution pre-installed.

## Step 2: Install YOLO26

Connect to your VM via SSH and install the Ultralytics package:

```bash
# Install or upgrade the Ultralytics package (requires >=26.0.0 for YOLO26)
pip install "ultralytics>=26.0.0"

# Verify the installation
yolo checks
```

## Step 3: Run YOLO26

With the environment ready, you can train, validate, predict, and export using the standard Ultralytics CLI or Python API.

=== "CLI"

    ```bash
    # Run inference on an image using YOLO26 nano
    yolo predict model=yolo26n.pt source="https://ultralytics.com/images/bus.jpg"

    # Train YOLO26 small on COCO128 for 10 epochs
    yolo train model=yolo26s.pt data=coco128.yaml epochs=10 imgsz=640

    # Validate a trained checkpoint
    yolo val model=yolo26s.pt data=coco128.yaml

    # Export to ONNX for deployment
    yolo export model=yolo26n.pt format=onnx
    ```

=== "Python"

    ```python
    from ultralytics import YOLO

    # Load a YOLO26 model
    model = YOLO("yolo26n.pt")

    # Run inference
    results = model("https://ultralytics.com/images/bus.jpg")
    results[0].show()

    # Train on a custom dataset
    model.train(data="coco128.yaml", epochs=10, imgsz=640)

    # Export to ONNX
    model.export(format="onnx")
    ```

Available YOLO26 model variants: `yolo26n`, `yolo26s`, `yolo26m`, `yolo26l`, `yolo26x`.

## Allocate Swap Space (Optional)

For very large datasets that may exceed VM RAM, add swap space to prevent out-of-memory errors:

```bash
sudo fallocate -l 64G /swapfile
sudo chmod 600 /swapfile
sudo mkswap /swapfile
sudo swapon /swapfile
free -h
```

## Training on a Custom Dataset

To train YOLO26 on your own dataset:

1. Prepare your dataset in [YOLO format](../../../datasets/index.md) (images + label `.txt` files + a dataset YAML).
2. Upload to your GCP VM using `gcloud compute scp` or the Cloud Console SSH browser file upload.
3. Start training:

```bash
yolo train model=yolo26s.pt data=custom_dataset.yaml epochs=100 imgsz=640
```

## Using Google Cloud Storage (GCS)

For large datasets, store data in a GCS bucket and stream directly to your VM:

```bash
# Copy a dataset from GCS to the VM
gsutil cp -r gs://your-bucket/my_dataset ./datasets/

# Copy trained weights back to GCS
gsutil cp -r ./runs/train/exp/weights gs://your-bucket/yolo26_weights/
```

## What's Next

- Read the [YOLO26 model docs](../../../models/yolo26.md) for architecture details and benchmarks.
- Explore [Ultralytics Platform](../../../platform/index.md) for a no-code training and deployment workflow.
- Continue to [Train Mode](../../../modes/train.md), [Val Mode](../../../modes/val.md), and [Export Mode](../../../modes/export.md) documentation.
- Try the [YOLO26 Docker Quickstart](./docker-quickstart.md) or [AWS Quickstart](./aws-quickstart.md) for alternative environments.
