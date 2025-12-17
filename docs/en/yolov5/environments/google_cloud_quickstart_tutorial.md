---
comments: true
description: Master Ultralytics YOLOv5 deployment on Google Cloud Platform Deep Learning VM. Perfect for AI beginners and experts to achieve high-performance object detection.
keywords: YOLOv5, Google Cloud Platform, GCP, Deep Learning VM, object detection, AI, machine learning, tutorial, cloud computing, GPU acceleration, Ultralytics
---

# Mastering YOLOv5 Deployment on Google Cloud Platform (GCP) Deep Learning VM

Embarking on the journey of [artificial intelligence (AI)](https://www.ultralytics.com/glossary/artificial-intelligence-ai) and [machine learning (ML)](https://www.ultralytics.com/glossary/machine-learning-ml) can be exhilarating, especially when you leverage the power and flexibility of a [cloud computing](https://www.ultralytics.com/glossary/cloud-computing) platform. Google Cloud Platform (GCP) offers robust tools tailored for ML enthusiasts and professionals alike. One such tool is the Deep Learning VM, preconfigured for data science and ML tasks. In this tutorial, we will navigate the process of setting up [Ultralytics YOLOv5](../../models/yolov5.md) on a [GCP Deep Learning VM](https://docs.cloud.google.com/deep-learning-vm/docs). Whether you're taking your first steps in ML or you're a seasoned practitioner, this guide provides a clear pathway to implementing [object detection](https://www.ultralytics.com/glossary/object-detection) models powered by YOLOv5.

🆓 Plus, if you're a new GCP user, you're in luck with a [$300 free credit offer](https://cloud.google.com/free/docs/free-cloud-features#free-trial) to kickstart your projects.

In addition to GCP, explore other accessible quickstart options for YOLOv5, like our [Google Colab Notebook](https://colab.research.google.com/github/ultralytics/yolov5/blob/master/tutorial.ipynb) <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"> for a browser-based experience, or the scalability of [Amazon AWS](./aws_quickstart_tutorial.md). Furthermore, container aficionados can utilize our official Docker image available on [Docker Hub](https://hub.docker.com/r/ultralytics/yolov5) <img src="https://img.shields.io/docker/pulls/ultralytics/yolov5?logo=docker" alt="Docker Pulls"> for an encapsulated environment, following our [Docker Quickstart Guide](../../guides/docker-quickstart.md).

## Step 1: Create and Configure Your Deep Learning VM

Let's begin by creating a virtual machine optimized for [deep learning](https://www.ultralytics.com/glossary/deep-learning-dl):

1.  Navigate to the [GCP marketplace](https://cloud.google.com/marketplace) and select the **Deep Learning VM**.
2.  Choose an **n1-standard-8** instance; it offers a balance of 8 vCPUs and 30 GB of memory, suitable for many ML tasks.
3.  Select a [GPU](https://www.ultralytics.com/glossary/gpu-graphics-processing-unit). The choice depends on your workload; even a basic T4 GPU will significantly accelerate model training.
4.  Check the box for 'Install NVIDIA GPU driver automatically on first startup?' for a seamless setup.
5.  Allocate a 300 GB SSD Persistent Disk to prevent I/O bottlenecks.
6.  Click 'Deploy' and allow GCP to provision your custom Deep Learning VM.

This VM comes pre-loaded with essential tools and frameworks, including the [Anaconda](https://www.anaconda.com/) Python distribution, which conveniently bundles many necessary dependencies for YOLOv5.

![GCP Marketplace illustration of setting up a Deep Learning VM](https://github.com/ultralytics/docs/releases/download/0/gcp-deep-learning-vm-setup.avif)

## Step 2: Prepare the VM for YOLOv5

After setting up the environment, let's get YOLOv5 installed and ready:

```bash
# Clone the YOLOv5 repository
git clone https://github.com/ultralytics/yolov5
cd yolov5

# Install dependencies
pip install -r requirements.txt
```

This setup process ensures you have a Python environment version 3.8.0 or newer and [PyTorch](https://www.ultralytics.com/glossary/pytorch) 1.8 or later. Our scripts automatically download [models](https://github.com/ultralytics/yolov5/tree/master/models) and [datasets](https://github.com/ultralytics/yolov5/tree/master/data) from the latest YOLOv5 [release](https://github.com/ultralytics/yolov5/releases), simplifying the process of starting model training.

## Step 3: Train and Deploy Your YOLOv5 Models

With the setup complete, you are ready to [train](../../modes/train.md), [validate](../../modes/val.md), [predict](../../modes/predict.md), and [export](../../modes/export.md) with YOLOv5 on your GCP VM:

```bash
# Train a YOLOv5 model on your dataset (e.g., yolov5s)
python train.py --data coco128.yaml --weights yolov5s.pt --img 640

# Validate the trained model to check Precision, Recall, and mAP
python val.py --weights yolov5s.pt --data coco128.yaml

# Run inference using the trained model on images or videos
python detect.py --weights yolov5s.pt --source path/to/your/images_or_videos

# Export the trained model to various formats like ONNX, CoreML, TFLite for deployment
python export.py --weights yolov5s.pt --include onnx coreml tflite
```

Using just a few commands, YOLOv5 enables you to train custom [object detection](https://docs.ultralytics.com/tasks/detect/) models tailored to your specific needs or utilize pretrained weights for rapid results across various tasks. Explore different [model deployment options](../../guides/model-deployment-options.md) after exporting.

![Terminal command image illustrating model training on a GCP Deep Learning VM](https://github.com/ultralytics/docs/releases/download/0/terminal-command-model-training.avif)

## Allocate Swap Space (Optional)

If you are working with particularly large datasets that might exceed your VM's RAM, consider adding swap space to prevent memory errors:

```bash
# Allocate a 64GB swap file
sudo fallocate -l 64G /swapfile

# Set the correct permissions for the swap file
sudo chmod 600 /swapfile

# Set up the Linux swap area
sudo mkswap /swapfile

# Enable the swap file
sudo swapon /swapfile

# Verify the swap space allocation (should show increased swap memory)
free -h
```

## Training Custom Datasets

To train YOLOv5 on your custom dataset within GCP, follow these general steps:

1.  Prepare your dataset according to the YOLOv5 format (images and corresponding label files). See our [datasets overview](../../datasets/index.md) for guidance.
2.  Upload your dataset to your GCP VM using `gcloud compute scp` or the web console's SSH feature.
3.  Create a dataset configuration YAML file (`custom_dataset.yaml`) that specifies the paths to your training and validation data, the number of classes, and class names.
4.  Begin the [training process](../../modes/train.md) using your custom dataset YAML and potentially starting from pretrained weights:

    ```bash
    # Example: Train YOLOv5s on a custom dataset for 100 epochs
    python train.py --img 640 --batch 16 --epochs 100 --data custom_dataset.yaml --weights yolov5s.pt
    ```

For comprehensive instructions on preparing data and training with custom datasets, consult the [Ultralytics YOLOv5 Train documentation](../../modes/train.md).

## Leveraging Cloud Storage

For efficient data management, especially with large datasets or numerous experiments, integrate your YOLOv5 workflow with [Google Cloud Storage](https://cloud.google.com/storage):

```bash
# Ensure Google Cloud SDK is installed and initialized
# If not installed: curl https://sdk.cloud.google.com/ | bash
# Then initialize: gcloud init

# Example: Copy your dataset from a GCS bucket to your VM
gsutil cp -r gs://your-data-bucket/my_dataset ./datasets/

# Example: Copy trained model weights from your VM to a GCS bucket
gsutil cp -r ./runs/train/exp/weights gs://your-models-bucket/yolov5_custom_weights/
```

This approach allows you to store large datasets and trained models securely and cost-effectively in the cloud, minimizing the storage requirements on your VM instance.

## Concluding Thoughts

Congratulations! You are now equipped to harness the capabilities of Ultralytics YOLOv5 combined with the computational power of Google Cloud Platform. This setup provides scalability, efficiency, and versatility for your object detection projects. Whether for personal exploration, academic research, or building industrial [solutions](../../solutions/index.md), you have taken a significant step into the world of AI and ML on the cloud.

Consider using [Ultralytics HUB](../../hub/index.md) for a streamlined, no-code experience to train and manage your models.

Remember to document your progress, share insights with the vibrant Ultralytics community, and utilize resources like [GitHub discussions](https://github.com/ultralytics/yolov5/discussions) for collaboration and support. Now, go forth and innovate with YOLOv5 and GCP!

Want to continue enhancing your ML skills? Dive into our [documentation](../../quickstart.md) and explore the [Ultralytics Blog](https://www.ultralytics.com/blog) for more tutorials and insights. Let your AI adventure continue!
