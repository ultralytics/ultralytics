---
comments: true
description: Deploy Ultralytics YOLO11 on GCP Deep Learning VM. GPU-accelerated setup guide for object detection, segmentation, pose estimation, and classification.
keywords: YOLO11, Google Cloud Platform, GCP, Deep Learning VM, object detection, Ultralytics, computer vision, GPU, cloud computing, machine learning, pose estimation, segmentation
canonical: https://docs.ultralytics.com/models/yolo11/environments/google-cloud-quickstart/
---

<script type="application/ld+json">
{
  "@context": "https://schema.org",
  "@type": "TechArticle",
  "headline": "YOLO11 on Google Cloud Platform (GCP) Deep Learning VM",
  "description": "Deploy Ultralytics YOLO11 on GCP Deep Learning VM. GPU-accelerated setup guide for object detection, segmentation, pose estimation, and classification.",
  "url": "https://docs.ultralytics.com/models/yolo11/environments/google-cloud-quickstart/",
  "image": "https://cdn.jsdelivr.net/gh/ultralytics/assets@main/docs/gcp-deep-learning-vm-setup.avif",
  "datePublished": "2026-06-04",
  "dateModified": "2026-06-04",
  "author": {"@type": "Organization", "name": "Ultralytics", "url": "https://www.ultralytics.com"},
  "publisher": {"@type": "Organization", "name": "Ultralytics", "url": "https://www.ultralytics.com"},
  "mainEntityOfPage": "https://docs.ultralytics.com/models/yolo11/environments/google-cloud-quickstart/"
}
</script>

# YOLO11 on Google Cloud Platform (GCP) Deep Learning VM

<!-- NOTE FOR MURAT: Please verify the recommended GCP instance type and GPU for YOLO11 workloads, and confirm whether Deep Learning VM images ship with ultralytics>=8.3 (YOLO11 was released Sept 2024). Add screenshots for each step. -->

This guide walks you through deploying [Ultralytics YOLO11](../../../models/yolo11.md) on a [Google Cloud Platform (GCP)](https://cloud.google.com/) Deep Learning VM. YOLO11 delivers state-of-the-art performance across detection, segmentation, pose estimation, classification, and oriented object detection (OBB) tasks.

🆓 New GCP users receive a [$300 free credit](https://cloud.google.com/free/docs/free-cloud-features) to get started.

For other environments, see the [YOLO11 Docker Quickstart](./docker-quickstart.md) or [YOLO11 AWS Quickstart](./aws-quickstart.md). For the latest Ultralytics model with NMS-free inference, see [YOLO26](../../yolo26/environments/google-cloud-quickstart.md).

## Step 1: Create a Deep Learning VM

1. Open the [GCP Marketplace](https://cloud.google.com/marketplace) and select **Deep Learning VM**.
2. Choose an **n1-standard-8** instance (8 vCPUs, 30 GB RAM) as a balanced starting point.
3. Select a GPU. An **NVIDIA T4** is well-suited for YOLO11 inference and lighter training runs.
4. Enable **"Install NVIDIA GPU driver automatically on first startup"**.
5. Allocate at least **300 GB SSD Persistent Disk** for datasets and checkpoints.
6. Click **Deploy** and wait for the VM to provision.

The Deep Learning VM comes pre-loaded with PyTorch, CUDA, and Anaconda.

## Step 2: Install YOLO11

Connect via SSH and install the Ultralytics package:

```bash
# Install the Ultralytics package (YOLO11 requires ultralytics>=8.3.0)
pip install ultralytics

# Verify
yolo checks
```

## Step 3: Run YOLO11

=== "CLI"

    ```bash
    # Inference on a sample image with YOLO11 nano
    yolo predict model=yolo11n.pt source="https://ultralytics.com/images/bus.jpg"

    # Train YOLO11 small on COCO128
    yolo train model=yolo11s.pt data=coco128.yaml epochs=10 imgsz=640

    # Validate
    yolo val model=yolo11s.pt data=coco128.yaml

    # Export to ONNX
    yolo export model=yolo11n.pt format=onnx
    ```

=== "Python"

    ```python
    from ultralytics import YOLO

    # Load a YOLO11 model
    model = YOLO("yolo11n.pt")

    # Inference
    results = model("https://ultralytics.com/images/bus.jpg")
    results[0].show()

    # Train on a custom dataset
    model.train(data="coco128.yaml", epochs=10, imgsz=640)

    # Export
    model.export(format="onnx")
    ```

Available YOLO11 variants: `yolo11n`, `yolo11s`, `yolo11m`, `yolo11l`, `yolo11x`.

### Task-Specific Models

YOLO11 supports multiple vision tasks with dedicated model weights:

| Task | Model | Example |
|------|-------|---------|
| Detection | `yolo11n.pt` | `yolo predict model=yolo11n.pt` |
| Segmentation | `yolo11n-seg.pt` | `yolo predict model=yolo11n-seg.pt` |
| Pose | `yolo11n-pose.pt` | `yolo predict model=yolo11n-pose.pt` |
| Classification | `yolo11n-cls.pt` | `yolo predict model=yolo11n-cls.pt` |
| OBB | `yolo11n-obb.pt` | `yolo predict model=yolo11n-obb.pt` |

## Allocate Swap Space (Optional)

For large datasets that may exceed VM RAM:

```bash
sudo fallocate -l 64G /swapfile
sudo chmod 600 /swapfile
sudo mkswap /swapfile
sudo swapon /swapfile
free -h
```

## Training on a Custom Dataset

1. Prepare your dataset in [YOLO format](../../../datasets/index.md).
2. Upload via `gcloud compute scp` or the Cloud Console SSH file upload.
3. Start training:

```bash
yolo train model=yolo11s.pt data=custom_dataset.yaml epochs=100 imgsz=640
```

## Using Google Cloud Storage (GCS)

```bash
# Copy dataset from GCS to the VM
gsutil cp -r gs://your-bucket/my_dataset ./datasets/

# Copy trained weights back to GCS
gsutil cp -r ./runs/train/exp/weights gs://your-bucket/yolo11_weights/
```

## What's Next

- Read the [YOLO11 model docs](../../../models/yolo11.md) for architecture details and benchmarks.
- Explore [Ultralytics Platform](../../../platform/index.md) for a no-code training and deployment workflow.
- Try the [YOLO11 Docker Quickstart](./docker-quickstart.md) or [AWS Quickstart](./aws-quickstart.md) for alternative environments.
- For the latest model with NMS-free inference, upgrade to [YOLO26](../../../models/yolo26.md).
