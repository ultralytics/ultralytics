---
comments: true
description: Learn how to set up and run YOLOv5 on AzureML. Follow this quickstart guide for easy configuration and model training on an AzureML compute instance.
keywords: YOLOv5, AzureML, machine learning, compute instance, quickstart, model training, virtual environment, Python, AI, deep learning
---

# YOLOv5 🚀 on AzureML

## What is Azure?

[Azure](https://azure.microsoft.com/) is Microsoft's [cloud computing](https://www.ultralytics.com/glossary/cloud-computing) platform, designed to help organizations move their workloads to the cloud from on-premises data centers. With a full spectrum of cloud services including computing, databases, analytics, [machine learning](https://www.ultralytics.com/glossary/machine-learning-ml), and networking, users can pick and choose from these services to develop and scale new applications, or run existing applications, in the public cloud.

## What is Azure Machine Learning (AzureML)?

Azure Machine Learning, commonly referred to as AzureML, is a fully managed cloud service that enables data scientists and developers to efficiently embed predictive analytics into their applications. AzureML offers a variety of services and capabilities aimed at making machine learning accessible, easy to use, and scalable, providing features like automated machine learning, drag-and-drop model training, and a robust Python SDK.

## Prerequisites

Before getting started, you need an [AzureML workspace](https://learn.microsoft.com/azure/machine-learning/concept-workspace?view=azureml-api-2). If you don't have one, you can create a new workspace by following Azure's official documentation.

## Create a compute instance

From your AzureML workspace, select Compute > Compute instances > New, and select the instance with the resources you need.

<img width="1741" alt="create-compute-arrow" src="https://github.com/ultralytics/docs/releases/download/0/create-compute-arrow.avif">

## Open a Terminal

From the Notebooks view, open a Terminal and select your compute.

![open-terminal-arrow](https://github.com/ultralytics/docs/releases/download/0/open-terminal-arrow.avif)

## Setup and run YOLOv5

### Create a virtual environment

Create a conda environment with your preferred Python version:

```bash
conda create --name yolov5env -y python=3.10
conda activate yolov5env
conda install pip -y
```

### Clone YOLOv5 repository

Clone the YOLOv5 repository with its submodules:

```bash
git clone https://github.com/ultralytics/yolov5
cd yolov5
git submodule update --init --recursive # You might see a message asking you to add your folder as a safe.directory
```

### Install dependencies

Install the required dependencies:

```bash
pip install -r requirements.txt
pip install onnx>=1.12.0
```

### Perform YOLOv5 tasks

Train the YOLOv5 model:

```bash
python train.py --data coco128.yaml --weights yolov5s.pt --img 640
```

Validate the model for [Precision](https://www.ultralytics.com/glossary/precision), [Recall](https://www.ultralytics.com/glossary/recall), and [mAP](https://www.ultralytics.com/glossary/mean-average-precision-map):

```bash
python val.py --weights yolov5s.pt --data coco128.yaml
```

Run inference on images:

```bash
python detect.py --weights yolov5s.pt --source path/to/images
```

Export models to other formats (like ONNX):

```bash
python export.py --weights yolov5s.pt --include onnx
```

## Using a Notebook

If you prefer using a notebook instead of the terminal, you'll need to [create a new Kernel](https://learn.microsoft.com/en-us/azure/machine-learning/how-to-access-terminal?view=azureml-api-2#add-new-kernels) and select it at the top of your notebook.

### Create a new IPython kernel

From your compute terminal:

```bash
conda create --name yolov5env -y python=3.10
conda activate yolov5env
conda install pip ipykernel -y
python -m ipykernel install --user --name yolov5env --display-name "yolov5env"
```

When creating Python cells in your notebook, they will automatically use your custom environment. For bash cells, you need to activate your environment in each cell:

```bash
%%bash
source activate yolov5env
python val.py --weights yolov5s.pt --data coco128.yaml
```
