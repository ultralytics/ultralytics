---
comments: true
description: Get started with YOLOv5 on AWS. Our comprehensive guide provides everything you need to know to run YOLOv5 on an Amazon Deep Learning instance.
keywords: YOLOv5, AWS, Deep Learning, Instance, Guide, Quickstart
---

# YOLOv5 🚀 on AWS Deep Learning Instance: A Comprehensive Guide

This guide will help new users run YOLOv5 on an Amazon Web Services (AWS) Deep Learning instance. AWS offers a [Free Tier](https://aws.amazon.com/free/) and a [credit program](https://aws.amazon.com/activate/) for a quick and affordable start.

Other quickstart options for YOLOv5 include our [Colab Notebook](https://colab.research.google.com/github/ultralytics/yolov5/blob/master/tutorial.ipynb) <a href="https://colab.research.google.com/github/ultralytics/yolov5/blob/master/tutorial.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"></a> <a href="https://www.kaggle.com/ultralytics/yolov5"><img src="https://kaggle.com/static/images/open-in-kaggle.svg" alt="Open In Kaggle"></a>, [GCP Deep Learning VM](https://docs.ultralytics.com/yolov5/environments/google_cloud_quickstart_tutorial), and our Docker image at [Docker Hub](https://hub.docker.com/r/ultralytics/yolov5) <a href="https://hub.docker.com/r/ultralytics/yolov5"><img src="https://img.shields.io/docker/pulls/ultralytics/yolov5?logo=docker" alt="Docker Pulls"></a>. *Updated: 21 April 2023*.

## 1. AWS Console Sign-in

Create an account or sign in to the AWS console at [https://aws.amazon.com/console/](https://aws.amazon.com/console/) and select the **EC2** service.

![Console](https://user-images.githubusercontent.com/26833433/106323804-debddd00-622c-11eb-997f-b8217dc0e975.png)

## 2. Launch Instance

In the EC2 section of the AWS console, click the **Launch instance** button.

![Launch](https://user-images.githubusercontent.com/26833433/106323950-204e8800-622d-11eb-915d-5c90406973ea.png)

### Choose an Amazon Machine Image (AMI)

Enter 'Deep Learning' in the search field and select the most recent Ubuntu Deep Learning AMI (recommended), or an alternative Deep Learning AMI. For more information on selecting an AMI, see [Choosing Your DLAMI](https://docs.aws.amazon.com/dlami/latest/devguide/options.html).

![Choose AMI](https://user-images.githubusercontent.com/26833433/106326107-c9e34880-6230-11eb-97c9-3b5fc2f4e2ff.png)

### Select an Instance Type

A GPU instance is recommended for most deep learning purposes. Training new models will be faster on a GPU instance than a CPU instance. Multi-GPU instances or distributed training across multiple instances with GPUs can offer sub-linear scaling. To set up distributed training, see [Distributed Training](https://docs.aws.amazon.com/dlami/latest/devguide/distributed-training.html).

**Note:** The size of your model should be a factor in selecting an instance. If your model exceeds an instance's available RAM, select a different instance type with enough memory for your application.

Refer to [EC2 Instance Types](https://aws.amazon.com/ec2/instance-types/) and choose Accelerated Computing to see the different GPU instance options.

![Choose Type](https://user-images.githubusercontent.com/26833433/106324624-52141e80-622e-11eb-9662-1a376d9c887d.png)

For more information on GPU monitoring and optimization, see [GPU Monitoring and Optimization](https://docs.aws.amazon.com/dlami/latest/devguide/tutorial-gpu.html). For pricing, see [On-Demand Pricing](https://aws.amazon.com/ec2/pricing/on-demand/) and [Spot Pricing](https://aws.amazon.com/ec2/spot/pricing/).

### Configure Instance Details

Amazon EC2 Spot Instances let you take advantage of unused EC2 capacity in the AWS cloud. Spot Instances are available at up to a 70% discount compared to On-Demand prices. We recommend a persistent spot instance, which will save your data and restart automatically when spot instance availability returns after spot instance termination. For full-price On-Demand instances, leave these settings at their default values.

![Spot Request](https://user-images.githubusercontent.com/26833433/106324835-ac14e400-622e-11eb-8853-df5ec9b16dfc.png)

Complete Steps 4-7 to finalize your instance hardware and security settings, and then launch the instance.

## 3. Connect to Instance

Select the checkbox next to your running instance, and then click Connect. Copy and paste the SSH terminal command into a terminal of your choice to connect to your instance.

![Connect](https://user-images.githubusercontent.com/26833433/106325530-cf8c5e80-622f-11eb-9f64-5b313a9d57a1.png)

## 4. Run YOLOv5

Once you have logged in to your instance, clone the repository and install the dependencies in a [**Python>=3.7.0**](https://www.python.org/) environment, including [**PyTorch>=1.7**](https://pytorch.org/get-started/locally/). [Models](https://github.com/ultralytics/yolov5/tree/master/models) and [datasets](https://github.com/ultralytics/yolov5/tree/master/data) download automatically from the latest YOLOv5 [release](https://github.com/ultralytics/yolov5/releases).

```bash
git clone https://github.com/ultralytics/yolov5  # clone
cd yolov5
pip install -r requirements.txt  # install
```

Then, start training, testing, detecting, and exporting YOLOv5 models:

```bash
python train.py  # train a model
python val.py --weights yolov5s.pt  # validate a model for Precision, Recall, and mAP
python detect.py --weights yolov5s.pt --source path/to/images  # run inference on images and videos
python export.py --weights yolov5s.pt --include onnx coreml tflite  # export models to other formats
```

## Optional Extras

Add 64GB of swap memory (to `--cache` large datasets):

```bash
sudo fallocate -l 64G /swapfile
sudo chmod 600 /swapfile
sudo mkswap /swapfile
sudo swapon /swapfile
free -h  # check memory
```

Now you have successfully set up and run YOLOv5 on an AWS Deep Learning instance. Enjoy training, testing, and deploying your object detection models!