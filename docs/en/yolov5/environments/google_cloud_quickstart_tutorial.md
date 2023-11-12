---
comments: true
description: Step-by-step tutorial on how to set up and run YOLOv5 on Google Cloud Platform Deep Learning VM. Perfect guide for beginners and GCP new users!.
keywords: YOLOv5, Google Cloud Platform, GCP, Deep Learning VM, Ultralytics
---

# Run YOLOv5 🚀 on Google Cloud Platform (GCP) Deep Learning Virtual Machine (VM) ⭐

This tutorial will guide you through the process of setting up and running YOLOv5 on a GCP Deep Learning VM. New GCP users are eligible for a [$300 free credit offer](https://cloud.google.com/free/docs/gcp-free-tier#free-trial).

You can also explore other quickstart options for YOLOv5, such as our [Colab Notebook](https://colab.research.google.com/github/ultralytics/yolov5/blob/master/tutorial.ipynb) <a href="https://colab.research.google.com/github/ultralytics/yolov5/blob/master/tutorial.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"></a> <a href="https://www.kaggle.com/ultralytics/yolov5"><img src="https://kaggle.com/static/images/open-in-kaggle.svg" alt="Open In Kaggle"></a>, [Amazon AWS](https://docs.ultralytics.com/yolov5/environments/aws_quickstart_tutorial) and our Docker image at [Docker Hub](https://hub.docker.com/r/ultralytics/yolov5) <a href="https://hub.docker.com/r/ultralytics/yolov5"><img src="https://img.shields.io/docker/pulls/ultralytics/yolov5?logo=docker" alt="Docker Pulls"></a>. *Updated: 21 April 2023*.

**Last Updated**: 6 May 2022

## Step 1: Create a Deep Learning VM

1. Go to the [GCP marketplace](https://console.cloud.google.com/marketplace/details/click-to-deploy-images/deeplearning) and select a **Deep Learning VM**.
2. Choose an **n1-standard-8** instance (with 8 vCPUs and 30 GB memory).
3. Add a GPU of your choice.
4. Check 'Install NVIDIA GPU driver automatically on first startup?'
5. Select a 300 GB SSD Persistent Disk for sufficient I/O speed.
6. Click 'Deploy'.

The preinstalled [Anaconda](https://docs.anaconda.com/anaconda/packages/pkg-docs/) Python environment includes all dependencies.

<img width="1000" alt="GCP Marketplace" src="https://user-images.githubusercontent.com/26833433/105811495-95863880-5f61-11eb-841d-c2f2a5aa0ffe.png">

## Step 2: Set Up the VM

Clone the YOLOv5 repository and install the [requirements.txt](https://github.com/ultralytics/yolov5/blob/master/requirements.txt) in a [**Python>=3.8.0**](https://www.python.org/) environment, including [**PyTorch>=1.8**](https://pytorch.org/get-started/locally/). [Models](https://github.com/ultralytics/yolov5/tree/master/models) and [datasets](https://github.com/ultralytics/yolov5/tree/master/data) will be downloaded automatically from the latest YOLOv5 [release](https://github.com/ultralytics/yolov5/releases).

```bash
git clone https://github.com/ultralytics/yolov5  # clone
cd yolov5
pip install -r requirements.txt  # install
```

## Step 3: Run YOLOv5 🚀 on the VM

You can now train, test, detect, and export YOLOv5 models on your VM:

```bash
python train.py  # train a model
python val.py --weights yolov5s.pt  # validate a model for Precision, Recall, and mAP
python detect.py --weights yolov5s.pt --source path/to/images  # run inference on images and videos
python export.py --weights yolov5s.pt --include onnx coreml tflite  # export models to other formats
```

<img width="1000" alt="GCP terminal" src="https://user-images.githubusercontent.com/26833433/142223900-275e5c9e-e2b5-43f7-a21c-35c4ca7de87c.png">
