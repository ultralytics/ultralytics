---
comments: true
description: Kickstart your real-time object detection journey with Ultralytics YOLOv5! This guide covers installation, inference, and training to help you master YOLOv5 quickly.
keywords: YOLOv5, Quickstart, real-time object detection, AI, ML, PyTorch, inference, training, Ultralytics, machine learning, deep learning, PyTorch Hub, COCO dataset
---

# YOLOv5 Quickstart 🚀

Embark on your journey into the dynamic realm of real-time [object detection](https://www.ultralytics.com/glossary/object-detection) with Ultralytics YOLOv5! This guide is crafted to serve as a comprehensive starting point for AI enthusiasts and professionals aiming to master YOLOv5. From initial setup to advanced [training techniques](tutorials/train-custom-data.md), we've got you covered. By the end of this guide, you'll have the knowledge to implement YOLOv5 into your projects confidently using state-of-the-art [deep learning](https://www.ultralytics.com/glossary/deep-learning-dl) methods. Let's ignite the engines and soar into YOLOv5!

## Install

Prepare for launch by cloning the [YOLOv5 repository](https://github.com/ultralytics/yolov5) and establishing the environment. This ensures that all the necessary [requirements](https://github.com/ultralytics/yolov5/blob/master/requirements.txt) are installed. Check that you have [**Python>=3.8.0**](https://www.python.org/) and [**PyTorch>=1.8**](https://pytorch.org/get-started/locally/) ready for takeoff. These foundational tools are crucial for running YOLOv5 effectively.

```bash
git clone https://github.com/ultralytics/yolov5 # clone repository
cd yolov5
pip install -r requirements.txt # install dependencies
```

## Inference with PyTorch Hub

Experience the simplicity of YOLOv5 [PyTorch Hub](./tutorials/pytorch-hub-model-loading.md) inference, where [models](https://github.com/ultralytics/yolov5/tree/master/models) are seamlessly downloaded from the latest YOLOv5 [release](https://github.com/ultralytics/yolov5/releases). This method leverages the power of [PyTorch](https://www.ultralytics.com/glossary/pytorch) for easy model loading and execution, making it straightforward to get predictions.

```python
import torch

# Model loading
model = torch.hub.load("ultralytics/yolov5", "yolov5s")  # Can be 'yolov5n' - 'yolov5x6', or 'custom'

# Inference on images
img = "https://ultralytics.com/images/zidane.jpg"  # Can be a file, Path, PIL, OpenCV, numpy, or list of images

# Run inference
results = model(img)

# Display results
results.print()  # or .show(), .save(), .crop(), .pandas(), etc.
```

## Inference with detect.py

Harness `detect.py` for versatile inference on various sources. It automatically fetches [models](https://github.com/ultralytics/yolov5/tree/master/models) from the latest YOLOv5 [release](https://github.com/ultralytics/yolov5/releases) and saves results with ease. This script is ideal for command-line usage and integrating YOLOv5 into larger systems, supporting inputs like images, videos, directories, webcams, and even [live streams](https://en.wikipedia.org/wiki/Streaming_media).

```bash
python detect.py --weights yolov5s.pt --source 0                              # webcam
python detect.py --weights yolov5s.pt --source image.jpg                      # image
python detect.py --weights yolov5s.pt --source video.mp4                      # video
python detect.py --weights yolov5s.pt --source screen                         # screenshot
python detect.py --weights yolov5s.pt --source path/                          # directory
python detect.py --weights yolov5s.pt --source list.txt                       # list of images
python detect.py --weights yolov5s.pt --source list.streams                   # list of streams
python detect.py --weights yolov5s.pt --source 'path/*.jpg'                   # glob pattern
python detect.py --weights yolov5s.pt --source 'https://youtu.be/LNwODJXcvt4' # YouTube video
python detect.py --weights yolov5s.pt --source 'rtsp://example.com/media.mp4' # RTSP, RTMP, HTTP stream
```

## Training

Replicate the YOLOv5 [COCO dataset](https://cocodataset.org/#home) benchmarks by following the [training instructions](tutorials/train-custom-data.md) below. The necessary [models](https://github.com/ultralytics/yolov5/tree/master/models) and [datasets](../datasets/detect/coco.md) (like `coco128.yaml` or the full `coco.yaml`) are pulled directly from the latest YOLOv5 [release](https://github.com/ultralytics/yolov5/releases). Training YOLOv5n/s/m/l/x on a V100 [GPU](https://www.ultralytics.com/glossary/gpu-graphics-processing-unit) should typically take 1/2/4/6/8 days respectively (note that [Multi-GPU training](./tutorials/multi-gpu-training.md) setups work faster). Maximize performance by using the highest possible `--batch-size` or use `--batch-size -1` for the YOLOv5 [AutoBatch](https://github.com/ultralytics/yolov5/pull/5092) feature, which automatically finds the optimal [batch size](https://www.ultralytics.com/glossary/batch-size). The following batch sizes are ideal for V100-16GB GPUs. See the [Architecture Summary](tutorials/architecture-description.md) for details on model configuration files (`*.yaml`).

```bash
# Train YOLOv5n on COCO128 for 3 epochs
python train.py --data coco128.yaml --epochs 3 --weights yolov5n.pt --batch-size 128

# Train YOLOv5s on COCO for 300 epochs
python train.py --data coco.yaml --epochs 300 --weights '' --cfg yolov5s.yaml --batch-size 64

# Train YOLOv5m on COCO for 300 epochs
python train.py --data coco.yaml --epochs 300 --weights '' --cfg yolov5m.yaml --batch-size 40

# Train YOLOv5l on COCO for 300 epochs
python train.py --data coco.yaml --epochs 300 --weights '' --cfg yolov5l.yaml --batch-size 24

# Train YOLOv5x on COCO for 300 epochs
python train.py --data coco.yaml --epochs 300 --weights '' --cfg yolov5x.yaml --batch-size 16
```

<img width="800" src="https://cdn.ul.run/i/8477aa5a6a0d08a87baddd689c34aef4.avif" alt="YOLOv5 training curves for COCO dataset">

To conclude, YOLOv5 is not only a state-of-the-art tool for object detection but also a testament to the power of [machine learning](https://www.ultralytics.com/glossary/machine-learning-ml) in transforming the way we interact with the world through visual understanding. As you progress through this guide and begin applying YOLOv5 to your projects, remember that you are at the forefront of a technological revolution, capable of achieving remarkable feats in [computer vision](https://www.ultralytics.com/glossary/computer-vision-cv). Should you need further insights or support from fellow visionaries, you're invited to our [GitHub repository](https://github.com/ultralytics/yolov5), home to a thriving community of developers and researchers. Explore further resources like [Ultralytics Platform](https://platform.ultralytics.com) for dataset management and model training without code, or check out our [Solutions](https://www.ultralytics.com/solutions) page for real-world applications and inspiration. Keep exploring, keep innovating, and enjoy the marvels of YOLOv5. Happy detecting! 🌠🔍

## FAQ

### Do I need to clone the repository to use YOLOv5?

Not for inference. PyTorch Hub downloads the code and weights on demand with `torch.hub.load("ultralytics/yolov5", "yolov5s")`. Cloning is required to run the `train.py`, `val.py`, `detect.py`, and `export.py` scripts.

### Which YOLOv5 model should I start with?

`yolov5s.pt` is the best balance of speed and accuracy for most projects. Use `yolov5n.pt` for edge devices, `yolov5m.pt` to `yolov5x.pt` when accuracy matters more than speed, and the P6 variants such as `yolov5s6.pt` for training at 1280 pixels. See the [pretrained checkpoints table](https://github.com/ultralytics/yolov5#pretrained-checkpoints) for the full list.

### How do I train on my own dataset?

Put images and YOLO-format labels in parallel `images/` and `labels/` folders, describe them in a `dataset.yaml`, and run `python train.py --data dataset.yaml --weights yolov5s.pt`. The [Train Custom Data](tutorials/train-custom-data.md) tutorial walks through each step.

### Is YOLOv5 still the right choice for a new project?

YOLOv5 is stable and remains fully supported in the standalone repository, but it is no longer the latest model. For new work, [YOLO26](../models/yolo26.md) in the `ultralytics` package is faster, more accurate, and exports end-to-end without NMS, using the same `pip install ultralytics` workflow described in the [Quickstart](../quickstart.md).
