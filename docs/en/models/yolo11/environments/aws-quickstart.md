---
comments: true
description: Deploy Ultralytics YOLO11 on AWS EC2. Launch a GPU-accelerated Deep Learning AMI for object detection, segmentation, pose estimation, and classification.
keywords: YOLO11, AWS, Amazon Web Services, EC2, Deep Learning AMI, object detection, segmentation, pose estimation, Ultralytics, machine learning, GPU, cloud computing
canonical: https://docs.ultralytics.com/models/yolo11/environments/aws-quickstart/
---

<script type="application/ld+json">
{
  "@context": "https://schema.org",
  "@type": "TechArticle",
  "headline": "YOLO11 on AWS Deep Learning Instance",
  "description": "Deploy Ultralytics YOLO11 on AWS EC2. Launch a GPU-accelerated Deep Learning AMI for object detection, segmentation, pose estimation, and classification.",
  "url": "https://docs.ultralytics.com/models/yolo11/environments/aws-quickstart/",
  "image": "https://cdn.jsdelivr.net/gh/ultralytics/assets@main/docs/aws-console-sign-in.avif",
  "datePublished": "2026-06-04",
  "dateModified": "2026-06-04",
  "author": {"@type": "Organization", "name": "Ultralytics", "url": "https://www.ultralytics.com"},
  "publisher": {"@type": "Organization", "name": "Ultralytics", "url": "https://www.ultralytics.com"},
  "mainEntityOfPage": "https://docs.ultralytics.com/models/yolo11/environments/aws-quickstart/"
}
</script>

# YOLO11 on AWS Deep Learning Instance

<!-- NOTE FOR MURAT: Please verify the recommended EC2 instance type for YOLO11 workloads at each model scale, and confirm the current AWS Deep Learning AMI name. Add screenshots for each numbered step. -->

This guide walks you through launching [Ultralytics YOLO11](../../../models/yolo11.md) on an [Amazon Web Services (AWS)](https://aws.amazon.com/) EC2 Deep Learning instance. AWS offers scalable GPU compute suited to YOLO11 training, validation, and production inference across detection, segmentation, pose, classification, and oriented object detection tasks.

For alternative environments, see the [YOLO11 GCP Quickstart](./google-cloud-quickstart.md) or [Docker Quickstart](./docker-quickstart.md).

## Step 1: Sign In and Open EC2

Log in to the [AWS Management Console](https://aws.amazon.com/console/) and go to the **EC2** dashboard.

## Step 2: Launch an Instance

Click **Launch Instance** to start the wizard.

### Choose an AMI

Search for **Deep Learning** and select the latest **AWS Deep Learning AMI (Ubuntu)**. This includes PyTorch, CUDA, and the core ML stack pre-installed.

### Choose an Instance Type

| Instance | GPUs | GPU RAM | Recommended for |
|----------|------|---------|-----------------|
| `g4dn.xlarge` | 1× T4 | 16 GB | YOLO11n / YOLO11s inference |
| `g4dn.2xlarge` | 1× T4 | 16 GB | YOLO11m training |
| `p3.2xlarge` | 1× V100 | 16 GB | YOLO11l / YOLO11x training |
| `p3.8xlarge` | 4× V100 | 64 GB | Multi-GPU training |

See [EC2 Instance Types](https://aws.amazon.com/ec2/instance-types/) and [On-Demand Pricing](https://aws.amazon.com/ec2/pricing/on-demand/).

### Configure Storage and Security

- Allocate at least **100 GB** EBS storage (more for large datasets).
- Open **SSH (port 22)** in the Security Group for your IP.

### Launch

Select or create an SSH key pair, then click **Launch Instance**.

## Step 3: Connect via SSH

```bash
ssh -i /path/to/your-key.pem ubuntu@<your-instance-public-ip>
```

## Step 4: Install YOLO11

```bash
pip install ultralytics

# Verify
yolo checks
```

## Step 5: Run YOLO11

=== "CLI"

    ```bash
    # Inference
    yolo predict model=yolo11n.pt source="https://ultralytics.com/images/bus.jpg"

    # Train on COCO128
    yolo train model=yolo11s.pt data=coco128.yaml epochs=10 imgsz=640

    # Validate
    yolo val model=yolo11s.pt data=coco128.yaml

    # Export to TensorRT
    yolo export model=yolo11n.pt format=engine
    ```

=== "Python"

    ```python
    from ultralytics import YOLO

    model = YOLO("yolo11n.pt")

    results = model("https://ultralytics.com/images/bus.jpg")
    results[0].show()

    model.train(data="coco128.yaml", epochs=10, imgsz=640)
    model.export(format="engine")
    ```

Available variants: `yolo11n`, `yolo11s`, `yolo11m`, `yolo11l`, `yolo11x`.

### Task-Specific Models

```bash
# Segmentation
yolo predict model=yolo11n-seg.pt source="https://ultralytics.com/images/bus.jpg"

# Pose estimation
yolo predict model=yolo11n-pose.pt source="https://ultralytics.com/images/bus.jpg"
```

## Using Spot Instances (Cost Saving)

Use a **persistent Spot request** to save up to 70% on EC2 costs for training jobs that can tolerate interruption. See [Spot Instance Pricing](https://aws.amazon.com/ec2/spot/pricing/).

## Using Amazon S3 for Data Storage

```bash
# Copy dataset from S3
aws s3 cp s3://your-bucket/my_dataset ./datasets/ --recursive

# Copy weights back to S3
aws s3 cp ./runs/train/exp/weights s3://your-bucket/yolo11_weights/ --recursive
```

## What's Next

- Read the [YOLO11 model documentation](../../../models/yolo11.md) for full benchmark tables and architecture details.
- Explore [Export Mode](../../../modes/export.md) to convert YOLO11 to TensorRT, ONNX, or CoreML.
- Try [Ultralytics Platform](../../../platform/index.md) for managed, no-code training and deployment.
- For the latest Ultralytics model with NMS-free inference, see [YOLO26](../../../models/yolo26.md).
