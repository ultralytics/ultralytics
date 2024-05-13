---
comments: true
description: Step-by-step Quickstart Guide to Running YOLOv8 Object Detection Models on AzureML for Fast Prototyping and Testing
keywords: Ultralytics, YOLOv8, Object Detection, Azure Machine Learning, Quickstart Guide, Prototype, Compute Instance, Terminal, Notebook, IPython Kernel, CLI, Python SDK
---

# YOLOv8 🚀 on AzureML

## What is Azure?

[Azure](https://azure.microsoft.com/) is Microsoft's cloud computing platform, designed to help organizations move their workloads to the cloud from on-premises data centers. With the full spectrum of cloud services including those for computing, databases, analytics, machine learning, and networking, users can pick and choose from these services to develop and scale new applications, or run existing applications, in the public cloud.

## What is Azure Machine Learning (AzureML)?

Azure Machine Learning, commonly referred to as AzureML, is a fully managed cloud service that enables data scientists and developers to efficiently embed predictive analytics into their applications, helping organizations use massive data sets and bring all the benefits of the cloud to machine learning. AzureML offers a variety of services and capabilities aimed at making machine learning accessible, easy to use, and scalable. It provides capabilities like automated machine learning, drag-and-drop model training, as well as a robust Python SDK so that developers can make the most out of their machine learning models.

## How Does AzureML Benefit YOLO Users?

For users of YOLO (You Only Look Once), AzureML provides a robust, scalable, and efficient platform to both train and deploy machine learning models. Whether you are looking to run quick prototypes or scale up to handle more extensive data, AzureML's flexible and user-friendly environment offers various tools and services to fit your needs. You can leverage AzureML to:

- Easily manage large datasets and computational resources for training.
- Utilize built-in tools for data preprocessing, feature selection, and model training.
- Collaborate more efficiently with capabilities for MLOps (Machine Learning Operations), including but not limited to monitoring, auditing, and versioning of models and data.

In the subsequent sections, you will find a quickstart guide detailing how to run YOLOv8 object detection models using AzureML, either from a compute terminal or a notebook.

## Prerequisites

Before you can get started, make sure you have access to an AzureML workspace. If you don't have one, you can create a new [AzureML workspace](https://learn.microsoft.com/azure/machine-learning/concept-workspace?view=azureml-api-2) by following Azure's official documentation. This workspace acts as a centralized place to manage all AzureML resources.

## Create a compute instance

From your AzureML workspace, select Compute > Compute instances > New, select the instance with the resources you need.

<p align="center">
  <img width="1280" src="https://github.com/ouphi/ultralytics/assets/17216799/3e92fcc0-a08e-41a4-af81-d289cfe3b8f2" alt="Create Azure Compute Instance">
</p>

## Quickstart from Terminal

Start your compute and open a Terminal:

<p align="center">
  <img width="480" src="https://github.com/ouphi/ultralytics/assets/17216799/635152f1-f4a3-4261-b111-d416cb5ef357" alt="Open Terminal">
</p>

### Create virtualenv

Create your conda virtualenv and install pip in it:

```bash
conda create --name yolov8env -y
conda activate yolov8env
conda install pip -y
```

Install the required dependencies:

```bash
cd ultralytics
pip install -r requirements.txt
pip install ultralytics
pip install onnx>=1.12.0
```

### Perform YOLOv8 tasks

Predict:

```bash
yolo predict model=yolov8n.pt source='https://ultralytics.com/images/bus.jpg'
```

Train a detection model for 10 epochs with an initial learning_rate of 0.01:

```bash
yolo train data=coco8.yaml model=yolov8n.pt epochs=10 lr0=0.01
```

You can find more [instructions to use the Ultralytics CLI here](../quickstart.md#use-ultralytics-with-cli).

## Quickstart from a Notebook

### Create a new IPython kernel

Open the compute Terminal.

<p align="center">
  <img width="480" src="https://github.com/ouphi/ultralytics/assets/17216799/635152f1-f4a3-4261-b111-d416cb5ef357" alt="Open Terminal">
</p>

From your compute terminal, you need to create a new ipykernel that will be used by your notebook to manage your dependencies:

```bash
conda create --name yolov8env -y
conda activate yolov8env
conda install pip -y
conda install ipykernel -y
python -m ipykernel install --user --name yolov8env --display-name "yolov8env"
```

Close your terminal and create a new notebook. From your Notebook, you can select the new kernel.

Then you can open a Notebook cell and install the required dependencies:

```bash
%%bash
source activate yolov8env
cd ultralytics
pip install -r requirements.txt
pip install ultralytics
pip install onnx>=1.12.0
```

Note that we need to use the `source activate yolov8env` for all the %%bash cells, to make sure that the %%bash cell uses environment we want.

Run some predictions using the [Ultralytics CLI](../quickstart.md#use-ultralytics-with-cli):

```bash
%%bash
source activate yolov8env
yolo predict model=yolov8n.pt source='https://ultralytics.com/images/bus.jpg'
```

Or with the [Ultralytics Python interface](../quickstart.md#use-ultralytics-with-python), for example to train the model:

```python
from ultralytics import YOLO

# Load a model
model = YOLO("yolov8n.pt")  # load an official YOLOv8n model

# Use the model
model.train(data="coco8.yaml", epochs=3)  # train the model
metrics = model.val()  # evaluate model performance on the validation set
results = model("https://ultralytics.com/images/bus.jpg")  # predict on an image
path = model.export(format="onnx")  # export the model to ONNX format
```

You can use either the Ultralytics CLI or Python interface for running YOLOv8 tasks, as described in the terminal section above.

By following these steps, you should be able to get YOLOv8 running quickly on AzureML for quick trials. For more advanced uses, you may refer to the full AzureML documentation linked at the beginning of this guide.

## Explore More with AzureML

This guide serves as an introduction to get you up and running with YOLOv8 on AzureML. However, it only scratches the surface of what AzureML can offer. To delve deeper and unlock the full potential of AzureML for your machine learning projects, consider exploring the following resources:

- [Create a Data Asset](https://learn.microsoft.com/azure/machine-learning/how-to-create-data-assets): Learn how to set up and manage your data assets effectively within the AzureML environment.
- [Initiate an AzureML Job](https://learn.microsoft.com/azure/machine-learning/how-to-train-model): Get a comprehensive understanding of how to kickstart your machine learning training jobs on AzureML.
- [Register a Model](https://learn.microsoft.com/azure/machine-learning/how-to-manage-models): Familiarize yourself with model management practices including registration, versioning, and deployment.
- [Train YOLOv8 with AzureML Python SDK](https://medium.com/@ouphi/how-to-train-the-yolov8-model-with-azure-machine-learning-python-sdk-8268696be8ba): Explore a step-by-step guide on using the AzureML Python SDK to train your YOLOv8 models.
- [Train YOLOv8 with AzureML CLI](https://medium.com/@ouphi/how-to-train-the-yolov8-model-with-azureml-and-the-az-cli-73d3c870ba8e): Discover how to utilize the command-line interface for streamlined training and management of YOLOv8 models on AzureML.
