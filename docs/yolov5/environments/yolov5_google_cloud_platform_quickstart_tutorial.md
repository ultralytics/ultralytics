# Run YOLOv5 🚀 on Google Cloud Platform (GCP) Deep Learning Virtual Machine (VM) ⭐

This tutorial will guide you through the process of setting up and running YOLOv5 on a GCP Deep Learning VM. New GCP users are eligible for a [$300 free credit offer](https://cloud.google.com/free/docs/gcp-free-tier#free-trial). You can also explore other quickstart options for YOLOv5, such as the [Colab Notebook](https://colab.research.google.com/github/ultralytics/yolov5/blob/master/tutorial.ipynb) and the [Docker image](https://hub.docker.com/r/ultralytics/yolov5).

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

Clone the YOLOv5 repository and install the [requirements.txt](https://github.com/ultralytics/yolov5/blob/master/requirements.txt) in a [**Python>=3.7.0**](https://www.python.org/) environment, including [**PyTorch>=1.7**](https://pytorch.org/get-started/locally/). [Models](https://github.com/ultralytics/yolov5/tree/master/models) and [datasets](https://github.com/ultralytics/yolov5/tree/master/data) will be downloaded automatically from the latest YOLOv5 [release](https://github.com/ultralytics/yolov5/releases).

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

## USER
Update and improve this Docker tutorial for better content, readability, and value to the reader:


To get started with YOLOv5 🚀 in a **Docker image** follow the instructions below. Other quickstart options for YOLOv5 include our [Colab Notebook](https://colab.research.google.com/github/ultralytics/yolov5/blob/master/tutorial.ipynb) <a href="https://colab.research.google.com/github/ultralytics/yolov5/blob/master/tutorial.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"></a> <a href="https://www.kaggle.com/ultralytics/yolov5"><img src="https://kaggle.com/static/images/open-in-kaggle.svg" alt="Open In Kaggle"></a> and a [Google](https://github.com/ultralytics/yolov5/wiki/GCP-Quickstart) and [Amazon](https://github.com/ultralytics/yolov5/wiki/AWS-Quickstart) cloud instances. UPDATED 21 May 2022.


## 1. Install Docker and Nvidia-Docker

Docker images come with all dependencies preinstalled, however Docker itself requires installation, and relies of nvidia driver installations in order to interact properly with local GPU resources. The requirements are: 
- Nvidia Driver >= 455.23 https://www.nvidia.com/Download/index.aspx
- Nvidia-Docker https://github.com/NVIDIA/nvidia-docker
- Docker Engine - CE >= 19.03 https://docs.docker.com/install/


## 2. Pull Image
The Ultralytics YOLOv5 DockerHub is https://hub.docker.com/r/ultralytics/yolov5 ![Docker Pulls](https://img.shields.io/docker/pulls/ultralytics/yolov5?logo=docker). [Docker Autobuild](https://docs.docker.com/docker-hub/builds/) is used to automatically build images from the latest repository commits, so the `ultralytics/yolov5:latest` image hosted on the DockerHub **will always be in sync with the most recent repository commit**. To pull this image:
```bash
sudo docker pull ultralytics/yolov5:latest
```


## 3. Run Container
Run an interactive instance of this image (called a "container") using `-it`:
```bash
sudo docker run --ipc=host -it ultralytics/yolov5:latest
```

Run a container with **local file access** (like COCO training data in `/datasets`) using `-v`:
```bash
sudo docker run --ipc=host -it -v "$(pwd)"/datasets:/usr/src/datasets ultralytics/yolov5:latest
```

Run a container with **GPU access** using `--gpus all`:
```bash
sudo docker run --ipc=host -it --gpus all ultralytics/yolov5:latest
```


## 4. Run YOLOv5 🚀

Start training, testing, detecting and exporting YOLOv5 models within the running Docker container!
```bash
python train.py  # train a model
python val.py --weights yolov5s.pt  # validate a model for Precision, Recall and mAP
python detect.py --weights yolov5s.pt --source path/to/images  # run inference on images and videos
python export.py --weights yolov5s.pt --include onnx coreml tflite  # export models to other formats
```

<p align="center"><img width="1000" src="https://user-images.githubusercontent.com/26833433/142224770-6e57caaf-ac01-4719-987f-c37d1b6f401f.png"></p>





## ASSISTANT
# Get Started with YOLOv5 🚀 in Docker

This tutorial will guide you through the process of setting up and running YOLOv5 in a Docker container. You can also explore other quickstart options for YOLOv5, such as the [Colab Notebook](https://colab.research.google.com/github/ultralytics/yolov5/blob/master/tutorial.ipynb) and cloud instances on [Google](https://github.com/ultralytics/yolov5/wiki/GCP-Quickstart) and [Amazon](https://github.com/ultralytics/yolov5/wiki/AWS-Quickstart). 

**Last Updated**: 21 May 2022

## Prerequisites

1. **Nvidia Driver**: Version 455.23 or higher. Download from [Nvidia's website](https://www.nvidia.com/Download/index.aspx).
2. **Nvidia-Docker**: Allows Docker to interact with your local GPU. Installation instructions are available on the [Nvidia-Docker GitHub repository](https://github.com/NVIDIA/nvidia-docker).
3. **Docker Engine - CE**: Version 19.03 or higher. Download and installation instructions can be found on the [Docker website](https://docs.docker.com/install/).

## Step 1: Pull the YOLOv5 Docker Image

The Ultralytics YOLOv5 DockerHub repository is available at [https://hub.docker.com/r/ultralytics/yolov5](https://hub.docker.com/r/ultralytics/yolov5). Docker Autobuild ensures that the `ultralytics/yolov5:latest` image is always in sync with the most recent repository commit. To pull the latest image, run the following command:

```bash
sudo docker pull ultralytics/yolov5:latest
```

## Step 2: Run the Docker Container

### Basic container:

Run an interactive instance of the YOLOv5 Docker image (called a "container") using the `-it` flag:

```bash
sudo docker run --ipc=host -it ultralytics/yolov5:latest
```

### Container with local file access:

To run a container with access to local files (e.g., COCO training data in `/datasets`), use the `-v` flag:

```bash
sudo docker run --ipc=host -it -v "$(pwd)"/datasets:/usr/src/datasets ultralytics/yolov5:latest
```

### Container with GPU access:

To run a container with GPU access, use the `--gpus all` flag:

```bash
sudo docker run --ipc=host -it --gpus all ultralytics/yolov5:latest
```

## Step 3: Use YOLOv5 🚀 within the Docker Container

Now you can train, test, detect, and export YOLOv5 models within the running Docker container:

```bash
python train.py  # train a model
python val.py --weights yolov5s.pt  # validate a model for Precision, Recall, and mAP
python detect.py --weights yolov5s.pt --source path/to/images  # run inference on images and videos
python export.py --weights yolov5s.pt --include onnx coreml tflite  # export models to other formats
```

<p align="center"><img width="1000" src="https://user-images.githubusercontent.com/26833433/142224770-6e57caaf-ac01-4719-987f-c37d1b6f401f.png"></p>

## USER
Great! Do the same for the GCP tutorial below.




This quickstart guide 📚 helps new users run YOLOv5 🚀 on a Google Cloud Platform (GCP) Deep Learning Virtual Machine (VM) ⭐. New GCP users are eligible for a [$300 free credit offer](https://cloud.google.com/free/docs/gcp-free-tier#free-trial). Other quickstart options for YOLOv5 include our [Colab Notebook](https://colab.research.google.com/github/ultralytics/yolov5/blob/master/tutorial.ipynb) <a href="https://colab.research.google.com/github/ultralytics/yolov5/blob/master/tutorial.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"></a> <a href="https://www.kaggle.com/ultralytics/yolov5"><img src="https://kaggle.com/static/images/open-in-kaggle.svg" alt="Open In Kaggle"></a> and our Docker image at https://hub.docker.com/r/ultralytics/yolov5 ![Docker Pulls](https://img.shields.io/docker/pulls/ultralytics/yolov5?logo=docker). UPDATED 6 May 2022.

## 1. Create VM
Select a **Deep Learning VM** from the [GCP marketplace](https://console.cloud.google.com/marketplace/details/click-to-deploy-images/deeplearning), select an **n1-standard-8** instance (with 8 vCPUs and 30 GB memory), add a GPU of your choice, check 'Install NVIDIA GPU driver automatically on first startup?', and select a 300 GB SSD Persistent Disk for sufficient I/O speed, then click 'Deploy'. **All dependencies are included** in the preinstalled [Anaconda](https://docs.anaconda.com/anaconda/packages/pkg-docs/) Python environment.
<img width="1000" alt="GCP Marketplace" src="https://user-images.githubusercontent.com/26833433/105811495-95863880-5f61-11eb-841d-c2f2a5aa0ffe.png">

## 2. Setup VM

Clone repo and install [requirements.txt](https://github.com/ultralytics/yolov5/blob/master/requirements.txt) in a [**Python>=3.7.0**](https://www.python.org/) environment, including [**PyTorch>=1.7**](https://pytorch.org/get-started/locally/). [Models](https://github.com/ultralytics/yolov5/tree/master/models) and [datasets](https://github.com/ultralytics/yolov5/tree/master/data) download automatically from the latest YOLOv5 [release](https://github.com/ultralytics/yolov5/releases).

```bash
git clone https://github.com/ultralytics/yolov5  # clone
cd yolov5
pip install -r requirements.txt  # install
```

## 3. Run YOLOv5 🚀

Start training, testing, detecting and exporting YOLOv5 models on your VM!
```bash
python train.py  # train a model
python val.py --weights yolov5s.pt  # validate a model for Precision, Recall and mAP
python detect.py --weights yolov5s.pt --source path/to/images  # run inference on images and videos
python export.py --weights yolov5s.pt --include onnx coreml tflite  # export models to other formats
```

<img width="1000" alt="GCP terminal" src="https://user-images.githubusercontent.com/26833433/142223900-275e5c9e-e2b5-43f7-a21c-35c4ca7de87c.png">


## Optional Extras

Create 16GB of swap memory (to `--cache` large datasets).
```bash
sudo swapoff /swapfile  # (optional) clear existing
sudo fallocate -l 16G /swapfile
sudo chmod 600 /swapfile
sudo mkswap /swapfile
sudo swapon /swapfile
free -h  # check memory
```

Mount local SSD
```bash
lsblk
sudo mkfs.ext4 -F /dev/nvme0n1
sudo mkdir -p /mnt/disks/nvme0n1
sudo mount /dev/nvme0n1 /mnt/disks/nvme0n1
sudo chmod a+w /mnt/disks/nvme0n1
cp -r coco /mnt/disks/nvme0n1
```

## ASSISTANT
# Run YOLOv5 🚀 on Google Cloud Platform (GCP) Deep Learning Virtual Machine (VM) ⭐

This tutorial will guide you through the process of setting up and running YOLOv5 on a GCP Deep Learning VM. New GCP users are eligible for a [$300 free credit offer](https://cloud.google.com/free/docs/gcp-free-tier#free-trial). You can also explore other quickstart options for YOLOv5, such as the [Colab Notebook](https://colab.research.google.com/github/ultralytics/yolov5/blob/master/tutorial.ipynb) and the [Docker image](https://hub.docker.com/r/ultralytics/yolov5).

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

Clone the YOLOv5 repository and install the [requirements.txt](https://github.com/ultralytics/yolov5/blob/master/requirements.txt) in a [**Python>=3.7.0**](https://www.python.org/) environment, including [**PyTorch>=1.7**](https://pytorch.org/get-started/locally/). [Models](https://github.com/ultralytics/yolov5/tree/master/models) and [datasets](https://github.com/ultralytics/yolov5/tree/master/data) will be downloaded automatically from the latest YOLOv5 [release](https://github.com/ultralytics/yolov5/releases).

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

## Optional Extras

### Create 16 GB of Swap Memory

To `--cache` large datasets,

## USER
Redo this section please:

## Optional Extras

Create 16GB of swap memory (to `--cache` large datasets).
```bash
sudo swapoff /swapfile  # (optional) clear existing
sudo fallocate -l 16G /swapfile
sudo chmod 600 /swapfile
sudo mkswap /swapfile
sudo swapon /swapfile
free -h  # check memory
```

Mount local SSD
```bash
lsblk
sudo mkfs.ext4 -F /dev/nvme0n1
sudo mkdir -p /mnt/disks/nvme0n1
sudo mount /dev/nvme0n1 /mnt/disks/nvme0n1
sudo chmod a+w /mnt/disks/nvme0n1
cp -r coco /mnt/disks/nvme0n1
```

## Optional Extras

### Create 16 GB of Swap Memory

To cache large datasets, you can create 16 GB of swap memory with the following commands:

```bash
sudo swapoff /swapfile  # (optional) clear existing swap file
sudo fallocate -l 16G /swapfile  # allocate 16 GB for the swap file
sudo chmod 600 /swapfile  # set appropriate permissions
sudo mkswap /swapfile  # create the swap file
sudo swapon /swapfile  # enable the swap file
free -h  # check memory status
```

### Mount a Local SSD

To mount a local SSD for better I/O performance, follow these steps:

```bash
lsblk  # list available block devices
sudo mkfs.ext4 -F /dev/nvme0n1  # format the SSD as ext4
sudo mkdir -p /mnt/disks/nvme0n1  # create a mount point for the SSD
sudo mount /dev/nvme0n1 /mnt/disks/nvme0n1  # mount the SSD
sudo chmod a+w /mnt/disks/nvme0n1  # set appropriate permissions
cp -r coco /mnt/disks/nvme0n1  # copy the 'coco' dataset to the SSD
```