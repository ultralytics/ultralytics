---
comments: true
description: Master YOLOv5 with ease! Our guide covers installation, inference, training, and more to power your object detection tasks using PyTorch. Ideal for beginners and experts alike.
keywords: YOLOv5, Object Detection, PyTorch Tutorial, Model Training, Image Inference, Ultralytics, AI, Machine Learning
---

# YOLOv5 Quickstart 🚀

Welcome to the world of YOLOv5! Whether you're a beginner or an expert in object detection, this guide is your key to unlocking the full potential of YOLOv5. Follow along for a comprehensive walkthrough on installation, running inference, and training your own models. Let's get started!

## Install

Begin your YOLOv5 journey by cloning the repo and setting up the environment, installing all [requirements](https://github.com/ultralytics/yolov5/blob/master/requirements.txt) correctly. Ensure you have [**Python>=3.8.0**](https://www.python.org/) and [**PyTorch>=1.8**](https://pytorch.org/get-started/locally/) installed.

```bash
git clone https://github.com/ultralytics/yolov5  # clone
cd yolov5
pip install -r requirements.txt  # install
```

## Inference with PyTorch Hub

YOLOv5 [PyTorch Hub](https://docs.ultralytics.com/yolov5/tutorials/pytorch_hub_model_loading) inference. [Models](https://github.com/ultralytics/yolov5/tree/master/models) download automatically from the latest YOLOv5 [release](https://github.com/ultralytics/yolov5/releases).

```python
import torch

# Model
model = torch.hub.load("ultralytics/yolov5", "yolov5s")  # or yolov5n - yolov5x6, custom

# Images
img = "https://ultralytics.com/images/zidane.jpg"  # or file, Path, PIL, OpenCV, numpy, list

# Inference
results = model(img)

# Results
results.print()  # or .show(), .save(), .crop(), .pandas(), etc.
```

## Inference with detect.py

`detect.py` runs inference on a variety of sources, downloading [models](https://github.com/ultralytics/yolov5/tree/master/models) automatically from the latest YOLOv5 [release](https://github.com/ultralytics/yolov5/releases) and saving results to `runs/detect`.

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

The commands below reproduce YOLOv5 [COCO](https://github.com/ultralytics/yolov5/blob/master/data/scripts/get_coco.sh)
results. [Models](https://github.com/ultralytics/yolov5/tree/master/models)
and [datasets](https://github.com/ultralytics/yolov5/tree/master/data) download automatically from the latest YOLOv5 [release](https://github.com/ultralytics/yolov5/releases). Training times for YOLOv5n/s/m/l/x are 1/2/4/6/8 days on a V100 GPU ([Multi-GPU](https://docs.ultralytics.com/yolov5/tutorials/multi_gpu_training) times faster). Use the largest `--batch-size` possible, or pass `--batch-size -1` for YOLOv5 [AutoBatch](https://github.com/ultralytics/yolov5/pull/5092). Batch sizes shown for V100-16GB.

```bash
python train.py --data coco.yaml --epochs 300 --weights '' --cfg yolov5n.yaml  --batch-size 128
                                                                 yolov5s                    64
                                                                 yolov5m                    40
                                                                 yolov5l                    24
                                                                 yolov5x                    16
```

<img width="800" src="https://user-images.githubusercontent.com/26833433/90222759-949d8800-ddc1-11ea-9fa1-1c97eed2b963.png" alt="YOLO training curves">

In conclusion, YOLOv5 is a versatile and powerful tool for object detection, offering flexibility and ease of use for both newcomers and seasoned practitioners in the field of computer vision. By following the steps outlined above, you can quickly get started with YOLOv5, harnessing its capabilities for a wide range of applications. Remember, the journey into AI and machine learning is ongoing, and YOLOv5 is an excellent companion on this exciting path. For further information, tips, and community support, visit our [GitHub repository](https://github.com/ultralytics/yolov5) and join the vibrant Ultralytics community. Happy detecting! 🌟
