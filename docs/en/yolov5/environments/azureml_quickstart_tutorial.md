---
comments: true
description: Learn how to set up and run Ultralytics YOLOv5 on AzureML. Follow this quickstart guide for easy configuration and model training on an AzureML compute instance.
keywords: YOLOv5, AzureML, machine learning, compute instance, quickstart, model training, virtual environment, Python, AI, deep learning, Ultralytics
---

# Ultralytics YOLOv5 🚀 on AzureML Quickstart

Welcome to the Ultralytics [YOLOv5](../../models/yolov5.md) quickstart guide for Microsoft Azure Machine Learning (AzureML)! This guide will walk you through setting up YOLOv5 on an AzureML compute instance, covering everything from creating a virtual environment to training and running inference with the model.

## What is Azure?

[Azure](https://azure.microsoft.com/) is Microsoft's comprehensive [cloud computing](https://www.ultralytics.com/glossary/cloud-computing) platform. It offers a vast array of services, including computing power, databases, analytics tools, [machine learning](https://www.ultralytics.com/glossary/machine-learning-ml) capabilities, and networking solutions. Azure enables organizations to build, deploy, and manage applications and services through Microsoft-managed data centers, facilitating the migration of workloads from on-premises infrastructure to the cloud.

## What is Azure Machine Learning (AzureML)?

[Azure Machine Learning](https://azure.microsoft.com/products/machine-learning) (AzureML) is a specialized cloud service designed for developing, training, and deploying machine learning models. It provides a collaborative environment with tools suitable for data scientists and developers of all skill levels. Key features include [automated machine learning (AutoML)](https://www.ultralytics.com/glossary/automated-machine-learning-automl), a drag-and-drop interface for model creation, and a powerful [Python](https://www.python.org/) SDK for more granular control over the ML lifecycle. AzureML simplifies the process of embedding [predictive modeling](https://www.ultralytics.com/glossary/predictive-modeling) into applications.

## Prerequisites

To follow this guide, you'll need an active [Azure subscription](https://azure.microsoft.com/free/) and access to an [AzureML workspace](https://learn.microsoft.com/azure/machine-learning/concept-workspace?view=azureml-api-2). If you don't have a workspace set up, please refer to the official [Azure documentation](https://learn.microsoft.com/azure/machine-learning/quickstart-create-resources?view=azureml-api-2) to create one.

## Create a Compute Instance

A compute instance in AzureML provides a managed cloud-based workstation for data scientists.

1.  Navigate to your AzureML workspace.
2.  On the left pane, select **Compute**.
3.  Go to the **Compute instances** tab and click **New**.
4.  Configure your instance by selecting the appropriate CPU or [GPU](https://www.ultralytics.com/glossary/gpu-graphics-processing-unit) resources based on your needs for training or inference.

<img width="1741" alt="create-compute-arrow" src="https://github.com/ultralytics/docs/releases/download/0/create-compute-arrow.avif">

## Open a Terminal

Once your compute instance is running, you can access its terminal directly from the AzureML studio.

1.  Go to the **Notebooks** section in the left pane.
2.  Find your compute instance in the top dropdown menu.
3.  Click on the **Terminal** option below the file browser to open a command-line interface to your instance.

![open-terminal-arrow](https://github.com/ultralytics/docs/releases/download/0/open-terminal-arrow.avif)

## Setup and Run YOLOv5

Now, let's set up the environment and run Ultralytics YOLOv5.

### 1. Create a Virtual Environment

It's best practice to use a virtual environment to manage dependencies. We'll use [Conda](https://docs.conda.io/en/latest/), which is pre-installed on AzureML compute instances. For a detailed Conda setup guide, see the Ultralytics [Conda Quickstart Guide](../../guides/conda-quickstart.md).

Create a Conda environment (e.g., `yolov5env`) with a specific Python version and activate it:

```bash
conda create --name yolov5env -y python=3.10 # Create a new Conda environment
conda activate yolov5env                     # Activate the environment
conda install pip -y                         # Ensure pip is installed
```

### 2. Clone YOLOv5 Repository

Clone the official Ultralytics YOLOv5 repository from [GitHub](https://github.com/) using [Git](https://git-scm.com/):

```bash
git clone https://github.com/ultralytics/yolov5 # Clone the repository
cd yolov5                                       # Navigate into the directory
# Initialize submodules (if any, though YOLOv5 typically doesn't require this step)
# git submodule update --init --recursive
```

### 3. Install Dependencies

Install the necessary Python packages listed in the `requirements.txt` file. We also install [ONNX](https://www.ultralytics.com/glossary/onnx-open-neural-network-exchange) for model export capabilities.

```bash
pip install -r requirements.txt # Install core dependencies
pip install onnx > =1.12.0      # Install ONNX for exporting
```

### 4. Perform YOLOv5 Tasks

With the setup complete, you can now train, validate, perform inference, and export your YOLOv5 model.

- **Train** the model on a dataset like [COCO128](../../datasets/detect/coco128.md). Check the [Training Mode](../../modes/train.md) documentation for more details.

    ```bash
    # Start training using yolov5s pretrained weights on the COCO128 dataset
    python train.py --data coco128.yaml --weights yolov5s.pt --img 640 --epochs 10 --batch 16
    ```

- **Validate** the trained model's performance using metrics like [Precision](https://www.ultralytics.com/glossary/precision), [Recall](https://www.ultralytics.com/glossary/recall), and [mAP](https://www.ultralytics.com/glossary/mean-average-precision-map). See the [Validation Mode](../../modes/val.md) guide for options.

    ```bash
    # Validate the yolov5s model on the COCO128 validation set
    python val.py --weights yolov5s.pt --data coco128.yaml --img 640
    ```

- **Run Inference** on new images or videos. Explore the [Prediction Mode](../../modes/predict.md) documentation for various inference sources.

    ```bash
    # Run inference with yolov5s on sample images
    python detect.py --weights yolov5s.pt --source data/images --img 640
    ```

- **Export** the model to different formats like ONNX, [TensorRT](https://www.ultralytics.com/glossary/tensorrt), or [CoreML](https://docs.ultralytics.com/integrations/coreml/) for deployment. Refer to the [Export Mode](../../modes/export.md) guide and the [ONNX Integration](../../integrations/onnx.md) page.

    ```bash
    # Export yolov5s to ONNX format
    python export.py --weights yolov5s.pt --include onnx --img 640
    ```

## Using a Notebook

If you prefer an interactive experience, you can run these commands within an AzureML Notebook. You'll need to create a custom [IPython kernel](https://ipython.readthedocs.io/en/stable/install/kernel_install.html) linked to your Conda environment.

### Create a New IPython Kernel

Run the following commands in your compute instance terminal:

```bash
# Ensure your Conda environment is active
# conda activate yolov5env

# Install ipykernel if not already present
conda install ipykernel -y

# Create a new kernel linked to your environment
python -m ipykernel install --user --name yolov5env --display-name "Python (yolov5env)"
```

After creating the kernel, refresh your browser. When you open or create a `.ipynb` notebook file, select your new kernel ("Python (yolov5env)") from the kernel dropdown menu at the top right.

### Running Commands in Notebook Cells

- **Python Cells:** Code in Python cells will automatically execute using the selected `yolov5env` kernel.

- **Bash Cells:** To run shell commands, use the `%%bash` magic command at the beginning of the cell. Remember to activate your Conda environment within each bash cell, as they don't automatically inherit the notebook's kernel environment context.

    ```bash
    %%bash
    source activate yolov5env # Activate environment within the cell

    # Example: Run validation using the activated environment
    python val.py --weights yolov5s.pt --data coco128.yaml --img 640
    ```

Congratulations! You've successfully set up and run Ultralytics YOLOv5 on AzureML. For further exploration, consider checking out other [Ultralytics Integrations](../../integrations/index.md) or the detailed [YOLOv5 documentation](../index.md). You might also find the [AzureML documentation](https://learn.microsoft.com/en-us/azure/machine-learning/?view=azureml-api-2) useful for advanced scenarios like distributed training or model deployment as an endpoint.
