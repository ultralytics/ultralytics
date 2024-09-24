---
comments: true
description: Discover how to set up and run YOLOv5 on AWS Deep Learning Instances. Follow our comprehensive guide to get started quickly and cost-effectively.
keywords: YOLOv5, AWS, Deep Learning, Machine Learning, AWS EC2, YOLOv5 setup, Deep Learning Instances, AI, Object Detection
---

# YOLOv5 🚀 on AWS Deep Learning Instance: Your Complete Guide

Setting up a high-performance deep learning environment can be daunting for newcomers, but fear not! 🛠️ With this guide, we'll walk you through the process of getting YOLOv5 up and running on an AWS Deep Learning instance. By leveraging the power of Amazon Web Services (AWS), even those new to [machine learning](https://www.ultralytics.com/glossary/machine-learning-ml) can get started quickly and cost-effectively. The AWS platform's scalability is perfect for both experimentation and production deployment.

Other quickstart options for YOLOv5 include our [Colab Notebook](https://colab.research.google.com/github/ultralytics/yolov5/blob/master/tutorial.ipynb) <a href="https://colab.research.google.com/github/ultralytics/yolov5/blob/master/tutorial.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"></a> <a href="https://www.kaggle.com/ultralytics/yolov5"><img src="https://kaggle.com/static/images/open-in-kaggle.svg" alt="Open In Kaggle"></a>, [GCP Deep Learning VM](./google_cloud_quickstart_tutorial.md), and our Docker image at [Docker Hub](https://hub.docker.com/r/ultralytics/yolov5) <a href="https://hub.docker.com/r/ultralytics/yolov5"><img src="https://img.shields.io/docker/pulls/ultralytics/yolov5?logo=docker" alt="Docker Pulls"></a>.

## Step 1: AWS Console Sign-In

Start by creating an account or signing in to the AWS console at [https://aws.amazon.com/console/](https://aws.amazon.com/console/). Once logged in, select the **EC2** service to manage and set up your instances.

![Console](https://github.com/ultralytics/docs/releases/download/0/aws-console-sign-in.avif)

## Step 2: Launch Your Instance

In the EC2 dashboard, you'll find the **Launch Instance** button which is your gateway to creating a new virtual server.

![Launch](https://github.com/ultralytics/docs/releases/download/0/launch-instance-button.avif)

### Selecting the Right Amazon Machine Image (AMI)

Here's where you choose the operating system and software stack for your instance. Type '[Deep Learning](https://www.ultralytics.com/glossary/deep-learning-dl)' into the search field and select the latest Ubuntu-based Deep Learning AMI, unless your needs dictate otherwise. Amazon's Deep Learning AMIs come pre-installed with popular frameworks and GPU drivers to streamline your setup process.

![Choose AMI](https://github.com/ultralytics/docs/releases/download/0/choose-ami.avif)

### Picking an Instance Type

For deep learning tasks, selecting a GPU instance type is generally recommended as it can vastly accelerate model training. For instance size considerations, remember that the model's memory requirements should never exceed what your instance can provide.

**Note:** The size of your model should be a factor in selecting an instance. If your model exceeds an instance's available RAM, select a different instance type with enough memory for your application.

For a list of available GPU instance types, visit [EC2 Instance Types](https://aws.amazon.com/ec2/instance-types/), specifically under Accelerated Computing.

![Choose Type](https://github.com/ultralytics/docs/releases/download/0/choose-instance-type.avif)

For more information on GPU monitoring and optimization, see [GPU Monitoring and Optimization](https://docs.aws.amazon.com/dlami/latest/devguide/tutorial-gpu.html). For pricing, see [On-Demand Pricing](https://aws.amazon.com/ec2/pricing/on-demand/) and [Spot Pricing](https://aws.amazon.com/ec2/spot/pricing/).

### Configuring Your Instance

Amazon EC2 Spot Instances offer a cost-effective way to run applications as they allow you to bid for unused capacity at a fraction of the standard cost. For a persistent experience that retains data even when the Spot Instance goes down, opt for a persistent request.

![Spot Request](https://github.com/ultralytics/docs/releases/download/0/spot-request.avif)

Remember to adjust the rest of your instance settings and security configurations as needed in Steps 4-7 before launching.

## Step 3: Connect to Your Instance

Once your instance is running, select its checkbox and click Connect to access the SSH information. Use the displayed SSH command in your preferred terminal to establish a connection to your instance.

![Connect](https://github.com/ultralytics/docs/releases/download/0/connect-to-instance.avif)

## Step 4: Running YOLOv5

Logged into your instance, you're now ready to clone the YOLOv5 repository and install dependencies within a Python 3.8 or later environment. YOLOv5's models and datasets will automatically download from the latest [release](https://github.com/ultralytics/yolov5/releases).

```bash
git clone https://github.com/ultralytics/yolov5  # clone repository
cd yolov5
pip install -r requirements.txt  # install dependencies
```

With your environment set up, you can begin training, validating, performing inference, and exporting your YOLOv5 models:

```bash
# Train a model on your data
python train.py

# Validate the trained model for Precision, Recall, and mAP
python val.py --weights yolov5s.pt

# Run inference using the trained model on your images or videos
python detect.py --weights yolov5s.pt --source path/to/images

# Export the trained model to other formats for deployment
python export.py --weights yolov5s.pt --include onnx coreml tflite
```

## Optional Extras

To add more swap memory, which can be a savior for large datasets, run:

```bash
sudo fallocate -l 64G /swapfile  # allocate 64GB swap file
sudo chmod 600 /swapfile  # modify permissions
sudo mkswap /swapfile  # set up a Linux swap area
sudo swapon /swapfile  # activate swap file
free -h  # verify swap memory
```

And that's it! 🎉 You've successfully created an AWS Deep Learning instance and run YOLOv5. Whether you're just starting with [object detection](https://www.ultralytics.com/glossary/object-detection) or scaling up for production, this setup can help you achieve your machine learning goals. Happy training, validating, and deploying! If you encounter any hiccups along the way, the robust AWS documentation and the active Ultralytics community are here to support you.
