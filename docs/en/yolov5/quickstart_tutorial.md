---
comments: true
description: Kickstart your real-time object detection journey with YOLOv5! This guide covers installation, inference, and training to help you master YOLOv5 quickly.
keywords: YOLOv5, Quickstart, real-time object detection, AI, ML, PyTorch, inference, training, Ultralytics, machine learning, deep learning
---

# YOLOv5 Quickstart 🚀

Embark on your journey into the dynamic realm of real-time object detection with YOLOv5! This guide is crafted to serve as a comprehensive starting point for AI enthusiasts and professionals aiming to master YOLOv5. From initial setup to advanced training techniques, we've got you covered. By the end of this guide, you'll have the knowledge to implement YOLOv5 into your projects confidently. Let's ignite the engines and soar into YOLOv5!

## Install

Prepare for launch by cloning the repository and establishing the environment. This ensures that all the necessary [requirements](https://github.com/ultralytics/yolov5/blob/master/requirements.txt) are installed. Check that you have [**Python>=3.8.0**](https://www.python.org/) and [**PyTorch>=1.8**](https://pytorch.org/get-started/locally/) ready for takeoff.

```bash
git clone https://github.com/ultralytics/yolov5  # clone repository
cd yolov5
pip install -r requirements.txt  # install dependencies
```

## Inference with PyTorch Hub

Experience the simplicity of YOLOv5 [PyTorch Hub](./tutorials/pytorch_hub_model_loading.md) inference, where [models](https://github.com/ultralytics/yolov5/tree/master/models) are seamlessly downloaded from the latest YOLOv5 [release](https://github.com/ultralytics/yolov5/releases).

```python
import torch

# Model loading
model = torch.hub.load("ultralytics/yolov5", "yolov5s")  # Can be 'yolov5n' - 'yolov5x6', or 'custom'

# Inference on images
img = "https://ultralytics.com/images/zidane.jpg"  # Can be a file, Path, PIL, OpenCV, numpy, or list of images

# Run inference
results = model(img)

# Display results
results.print()  # Other options: .show(), .save(), .crop(), .pandas(), etc.
```

## Inference with detect.py

Harness `detect.py` for versatile inference on various sources. It automatically fetches [models](https://github.com/ultralytics/yolov5/tree/master/models) from the latest YOLOv5 [release](https://github.com/ultralytics/yolov5/releases) and saves results with ease.

```bash
python detect.py --weights yolov5s.pt --source 0                               # webcam
                                               img.jpg                         # image
                                               vid.mp4                         # video
                                               screen                          # screenshot
                                               path/                           # directory
                                               list.txt                        # list of images
                                               list.streams                    # list of streams
                                               'path/*.jpg'                    # glob
                                               'https://youtu.be/LNwODJXcvt4'  # YouTube
                                               'rtsp://example.com/media.mp4'  # RTSP, RTMP, HTTP stream
```

## Training

Replicate the YOLOv5 [COCO](https://github.com/ultralytics/yolov5/blob/master/data/scripts/get_coco.sh) benchmarks with the instructions below. The necessary [models](https://github.com/ultralytics/yolov5/tree/master/models) and [datasets](https://github.com/ultralytics/yolov5/tree/master/data) are pulled directly from the latest YOLOv5 [release](https://github.com/ultralytics/yolov5/releases). Training YOLOv5n/s/m/l/x on a V100 GPU should typically take 1/2/4/6/8 days respectively (note that [Multi-GPU](./tutorials/multi_gpu_training.md) setups work faster). Maximize performance by using the highest possible `--batch-size` or use `--batch-size -1` for the YOLOv5 [AutoBatch](https://github.com/ultralytics/yolov5/pull/5092) feature. The following batch sizes are ideal for V100-16GB GPUs.

```bash
python train.py --data coco.yaml --epochs 300 --weights '' --cfg yolov5n.yaml  --batch-size 128
                                                                 yolov5s                    64
                                                                 yolov5m                    40
                                                                 yolov5l                    24
                                                                 yolov5x                    16
```

<img width="800" src="https://user-images.githubusercontent.com/26833433/90222759-949d8800-ddc1-11ea-9fa1-1c97eed2b963.png" alt="YOLO training curves">

To conclude, YOLOv5 is not only a state-of-the-art tool for object detection but also a testament to the power of machine learning in transforming the way we interact with the world through visual understanding. As you progress through this guide and begin applying YOLOv5 to your projects, remember that you are at the forefront of a technological revolution, capable of achieving remarkable feats. Should you need further insights or support from fellow visionaries, you're invited to our [GitHub repository](https://github.com/ultralytics/yolov5) home to a thriving community of developers and researchers. Keep exploring, keep innovating, and enjoy the marvels of YOLOv5. Happy detecting! 🌠🔍
