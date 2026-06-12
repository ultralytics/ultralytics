---
comments: true
description: Deploy Ultralytics YOLO26 on AWS EC2. Launch a GPU-accelerated Deep Learning AMI, install YOLO26, and run NMS-free object detection in the cloud.
keywords: YOLO26, AWS, Amazon Web Services, EC2, Deep Learning AMI, object detection, Ultralytics, machine learning, GPU, cloud computing, NMS-free, computer vision
canonical: https://docs.ultralytics.com/models/yolo26/environments/aws-quickstart/
---

<script type="application/ld+json">
{
  "@context": "https://schema.org",
  "@type": "TechArticle",
  "headline": "YOLO26 on AWS Deep Learning Instance",
  "description": "Deploy Ultralytics YOLO26 on AWS EC2. Launch a GPU-accelerated Deep Learning AMI, install YOLO26, and run NMS-free object detection in the cloud.",
  "url": "https://docs.ultralytics.com/models/yolo26/environments/aws-quickstart/",
  "image": "https://cdn.jsdelivr.net/gh/ultralytics/assets@main/docs/aws-console-sign-in.avif",
  "datePublished": "2026-06-04",
  "dateModified": "2026-06-04",
  "author": {"@type": "Organization", "name": "Ultralytics", "url": "https://www.ultralytics.com"},
  "publisher": {"@type": "Organization", "name": "Ultralytics", "url": "https://www.ultralytics.com"},
  "mainEntityOfPage": "https://docs.ultralytics.com/models/yolo26/environments/aws-quickstart/"
}
</script>

# YOLO26 on AWS Deep Learning Instance

<!-- NOTE FOR MURAT: Please verify the recommended EC2 instance type (g4dn.xlarge is cost-effective for YOLO26n/s; p3.2xlarge for YOLO26l/x). Confirm the latest AWS Deep Learning AMI name includes ultralytics>=26.0.0 or note the pip upgrade step. Add screenshots for each step. -->

This guide walks you through launching [Ultralytics YOLO26](../../../models/yolo26.md) on an [Amazon Web Services (AWS)](https://aws.amazon.com/) EC2 Deep Learning instance. AWS provides scalable GPU compute that is ideal for YOLO26 training, validation, and production inference.

For alternative environments, see the [YOLO26 GCP Quickstart](./google-cloud-quickstart.md) or [Docker Quickstart](./docker-quickstart.md).

## Step 1: Sign In to AWS and Open EC2

Log in to the [AWS Management Console](https://aws.amazon.com/console/) and navigate to the **EC2** dashboard.

## Step 2: Launch an Instance

Click **Launch Instance** to start the instance creation wizard.

### Choose an AMI

Search for **Deep Learning** and select the latest **AWS Deep Learning AMI (Ubuntu)**. This AMI comes pre-installed with PyTorch, CUDA drivers, and common ML frameworks, minimising setup time.

### Choose an Instance Type

Select a GPU instance from the **Accelerated Computing** family:

| Instance | GPUs | GPU RAM | Recommended for |
|----------|------|---------|-----------------|
| `g4dn.xlarge` | 1× T4 | 16 GB | YOLO26n / YOLO26s inference & light training |
| `g4dn.2xlarge` | 1× T4 | 16 GB | YOLO26m training |
| `p3.2xlarge` | 1× V100 | 16 GB | YOLO26l / YOLO26x training |
| `p3.8xlarge` | 4× V100 | 64 GB | Multi-GPU YOLO26 training |

See [EC2 Instance Types](https://aws.amazon.com/ec2/instance-types/) and [On-Demand Pricing](https://aws.amazon.com/ec2/pricing/on-demand/) for a full comparison.

### Configure Storage

Allocate at least **100 GB** of EBS storage for the AMI, datasets, and model checkpoints. For large dataset workflows, consider **500 GB** or more.

### Security Group

Ensure **SSH (port 22)** is open from your IP address so you can connect to the instance.

### Launch

Create or select an existing SSH key pair, then click **Launch Instance**.

## Step 3: Connect via SSH

```bash
ssh -i /path/to/your-key.pem ubuntu@<your-instance-public-ip>
```

## Step 4: Install YOLO26

The AWS Deep Learning AMI includes PyTorch and CUDA. Install or upgrade the Ultralytics package:

```bash
pip install "ultralytics>=26.0.0"

# Verify
yolo checks
```

## Step 5: Run YOLO26

=== "CLI"

    ```bash
    # Inference on a sample image
    yolo predict model=yolo26n.pt source="https://ultralytics.com/images/bus.jpg"

    # Train YOLO26 small on COCO128
    yolo train model=yolo26s.pt data=coco128.yaml epochs=10 imgsz=640

    # Validate
    yolo val model=yolo26s.pt data=coco128.yaml

    # Export to TensorRT for production inference
    yolo export model=yolo26n.pt format=engine
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

    # Export to TensorRT
    model.export(format="engine")
    ```

Available variants: `yolo26n`, `yolo26s`, `yolo26m`, `yolo26l`, `yolo26x`.

## Using Spot Instances (Cost Saving)

AWS Spot Instances can reduce EC2 costs by up to 70%. For training jobs that can be interrupted and resumed, configure a **persistent Spot request** so the EBS volume is retained if the Spot instance is reclaimed.

See [Spot Instance Pricing](https://aws.amazon.com/ec2/spot/pricing/) for current rates.

## Using Amazon S3 for Data Storage

For large datasets, store data in [Amazon S3](https://aws.amazon.com/s3/) and sync to your instance:

```bash
# Install AWS CLI if not already present
pip install awscli

# Copy dataset from S3 to the instance
aws s3 cp s3://your-bucket/my_dataset ./datasets/ --recursive

# Copy trained weights back to S3
aws s3 cp ./runs/train/exp/weights s3://your-bucket/yolo26_weights/ --recursive
```

## What's Next

- Read [YOLO26 model documentation](../../../models/yolo26.md) for architecture details and full benchmark tables.
- Explore [Export Mode](../../../modes/export.md) to convert YOLO26 to TensorRT, ONNX, or CoreML.
- Try [Ultralytics Platform](../../../platform/index.md) for a managed, no-code training and deployment workflow.
- See [GCP](./google-cloud-quickstart.md) or [Docker](./docker-quickstart.md) for alternative environment options.
