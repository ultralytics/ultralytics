---
comments: true
description: Learn to freeze YOLOv5 layers for efficient transfer learning, reducing resources and speeding up training while maintaining accuracy.
keywords: YOLOv5, transfer learning, freeze layers, machine learning, deep learning, model training, PyTorch, Ultralytics
---

# Transfer Learning with Frozen Layers in YOLOv5

📚 This guide explains how to **freeze** YOLOv5 🚀 layers when implementing [transfer learning](https://www.ultralytics.com/glossary/transfer-learning). Transfer learning is a powerful technique that allows you to quickly retrain a model on new data without having to retrain the entire network. By freezing part of the initial weights and only updating the rest, you can significantly reduce computational resources and training time, though this approach may slightly impact final model accuracy.

## Before You Start

Clone the repo and install [requirements.txt](https://github.com/ultralytics/yolov5/blob/master/requirements.txt) in a [**Python>=3.8.0**](https://www.python.org/) environment, including [**PyTorch>=1.8**](https://pytorch.org/get-started/locally/). [Models](https://github.com/ultralytics/yolov5/tree/master/models) and [datasets](https://github.com/ultralytics/yolov5/tree/master/data) download automatically from the latest YOLOv5 [release](https://github.com/ultralytics/yolov5/releases).

```bash
git clone https://github.com/ultralytics/yolov5  # clone
cd yolov5
pip install -r requirements.txt  # install
```

## How Layer Freezing Works

When you freeze layers in a neural network, you're essentially setting their parameters to be non-trainable. The gradients for these layers are set to zero, preventing any weight updates during backpropagation. This is implemented in YOLOv5's training process as follows:

```python
# Freeze
freeze = [f"model.{x}." for x in range(freeze)]  # layers to freeze
for k, v in model.named_parameters():
    v.requires_grad = True  # train all layers
    if any(x in k for x in freeze):
        print(f"freezing {k}")
        v.requires_grad = False
```

## Exploring Model Architecture

To effectively freeze specific parts of the model, it's helpful to understand the layer structure. You can view all module names with:

```python
for k, v in model.named_parameters():
    print(k)

"""Output:
model.0.conv.conv.weight
model.0.conv.bn.weight
model.0.conv.bn.bias
model.1.conv.weight
model.1.bn.weight
model.1.bn.bias
model.2.cv1.conv.weight
model.2.cv1.bn.weight
...
"""
```

The YOLOv5 architecture consists of a backbone (layers 0-9) and a head (remaining layers):

```yaml
# YOLOv5 v6.0 backbone
backbone:
    # [from, number, module, args]
    - [-1, 1, Conv, [64, 6, 2, 2]] # 0-P1/2
    - [-1, 1, Conv, [128, 3, 2]] # 1-P2/4
    - [-1, 3, C3, [128]]
    - [-1, 1, Conv, [256, 3, 2]] # 3-P3/8
    - [-1, 6, C3, [256]]
    - [-1, 1, Conv, [512, 3, 2]] # 5-P4/16
    - [-1, 9, C3, [512]]
    - [-1, 1, Conv, [1024, 3, 2]] # 7-P5/32
    - [-1, 3, C3, [1024]]
    - [-1, 1, SPPF, [1024, 5]] # 9

# YOLOv5 v6.0 head
head:
    - [-1, 1, Conv, [512, 1, 1]]
    - [-1, 1, nn.Upsample, [None, 2, "nearest"]]
    - [[-1, 6], 1, Concat, [1]] # cat backbone P4
    - [-1, 3, C3, [512, False]] # 13
    # ... remaining head layers
```

## Freezing Options

### Freeze Backbone Only

To freeze only the backbone (layers 0-9), which is useful for adapting the model to new classes while retaining learned feature extraction capabilities:

```bash
python train.py --freeze 10
```

This approach is particularly effective when your new dataset shares similar low-level features with the original training data but has different classes or objects.

### Freeze All Except Detection Layers

To freeze the entire model except for the final output convolution layers in the Detect module:

```bash
python train.py --freeze 24
```

This approach is ideal when you want to maintain most of the model's learned features but need to adapt it to detect a different number of classes.

## Performance Comparison

We trained YOLOv5m on the VOC dataset using different freezing strategies, starting from the official COCO pretrained weights:

```bash
python train.py --batch 48 --weights yolov5m.pt --data voc.yaml --epochs 50 --cache --img 512 --hyp hyp.finetune.yaml
```

### Accuracy Results

The results demonstrate that freezing layers accelerates training but slightly reduces final [accuracy](https://www.ultralytics.com/glossary/accuracy):

![Freezing training mAP50 results](https://github.com/ultralytics/docs/releases/download/0/freezing-training-map50-results.avif)

![Freezing training mAP50-95 results](https://github.com/ultralytics/docs/releases/download/0/freezing-training-map50-95-results.avif)

<img width="922" alt="Table results" src="https://github.com/ultralytics/docs/releases/download/0/table-results.avif">

### Resource Utilization

Freezing more layers reduces GPU memory requirements and utilization, making this technique valuable for training larger models or using higher resolution images:

![Training GPU memory allocated percent](https://github.com/ultralytics/docs/releases/download/0/training-gpu-memory-allocated-percent.avif)

![Training GPU memory utilization percent](https://github.com/ultralytics/docs/releases/download/0/training-gpu-memory-utilization-percent.avif)

## When to Use Layer Freezing

Layer freezing in transfer learning is particularly beneficial in scenarios such as:

1. **Limited computational resources**: When GPU memory or processing power is constrained
2. **Small datasets**: When your new dataset is too small to train a full model without overfitting
3. **Quick adaptation**: When you need to rapidly adapt a model to a new domain
4. **Fine-tuning for specific tasks**: When adapting a general model to a specialized application

For more information on transfer learning techniques and their applications, see the [transfer learning glossary entry](https://www.ultralytics.com/glossary/transfer-learning).

## Supported Environments

Ultralytics provides a range of ready-to-use environments, each pre-installed with essential dependencies such as [CUDA](https://developer.nvidia.com/cuda-zone), [CUDNN](https://developer.nvidia.com/cudnn), [Python](https://www.python.org/), and [PyTorch](https://pytorch.org/), to kickstart your projects.

- **Free GPU Notebooks**: <a href="https://bit.ly/yolov5-paperspace-notebook"><img src="https://assets.paperspace.io/img/gradient-badge.svg" alt="Run on Gradient"></a> <a href="https://colab.research.google.com/github/ultralytics/yolov5/blob/master/tutorial.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"></a> <a href="https://www.kaggle.com/models/ultralytics/yolov5"><img src="https://kaggle.com/static/images/open-in-kaggle.svg" alt="Open In Kaggle"></a>
- **Google Cloud**: [GCP Quickstart Guide](../environments/google_cloud_quickstart_tutorial.md)
- **Amazon**: [AWS Quickstart Guide](../environments/aws_quickstart_tutorial.md)
- **Azure**: [AzureML Quickstart Guide](../environments/azureml_quickstart_tutorial.md)
- **Docker**: [Docker Quickstart Guide](../environments/docker_image_quickstart_tutorial.md) <a href="https://hub.docker.com/r/ultralytics/yolov5"><img src="https://img.shields.io/docker/pulls/ultralytics/yolov5?logo=docker" alt="Docker Pulls"></a>

## Project Status

<a href="https://github.com/ultralytics/yolov5/actions/workflows/ci-testing.yml"><img src="https://github.com/ultralytics/yolov5/actions/workflows/ci-testing.yml/badge.svg" alt="YOLOv5 CI"></a>

This badge indicates that all [YOLOv5 GitHub Actions](https://github.com/ultralytics/yolov5/actions) Continuous Integration (CI) tests are successfully passing. These CI tests rigorously check the functionality and performance of YOLOv5 across various key aspects: [training](https://github.com/ultralytics/yolov5/blob/master/train.py), [validation](https://github.com/ultralytics/yolov5/blob/master/val.py), [inference](https://github.com/ultralytics/yolov5/blob/master/detect.py), [export](https://github.com/ultralytics/yolov5/blob/master/export.py), and [benchmarks](https://github.com/ultralytics/yolov5/blob/master/benchmarks.py). They ensure consistent and reliable operation on macOS, Windows, and Ubuntu, with tests conducted every 24 hours and upon each new commit.
