---
comments: true
description: Learn to export YOLOv5 models to various formats like TFLite, ONNX, CoreML and TensorRT. Increase model efficiency and deployment flexibility with our step-by-step guide.
keywords: YOLOv5 export, TFLite, ONNX, CoreML, TensorRT, model conversion, YOLOv5 tutorial, PyTorch export
---

# TFLite, ONNX, CoreML, TensorRT Export

📚 This guide explains how to export a trained YOLOv5 🚀 model from [PyTorch](https://www.ultralytics.com/glossary/pytorch) to various deployment formats including ONNX, TensorRT, CoreML and more.

## Before You Start

Clone repo and install [requirements.txt](https://github.com/ultralytics/yolov5/blob/master/requirements.txt) in a [**Python>=3.8.0**](https://www.python.org/) environment, including [**PyTorch>=1.8**](https://pytorch.org/get-started/locally/). [Models](https://github.com/ultralytics/yolov5/tree/master/models) and [datasets](https://github.com/ultralytics/yolov5/tree/master/data) download automatically from the latest YOLOv5 [release](https://github.com/ultralytics/yolov5/releases).

```bash
git clone https://github.com/ultralytics/yolov5 # clone
cd yolov5
pip install -r requirements.txt # install
```

For [TensorRT](https://developer.nvidia.com/tensorrt) export example (requires GPU) see our Colab [notebook](https://colab.research.google.com/github/ultralytics/yolov5/blob/master/tutorial.ipynb#scrollTo=VTRwsvA9u7ln&line=2&uniqifier=1) appendix section. <a href="https://colab.research.google.com/github/ultralytics/yolov5/blob/master/tutorial.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"></a>

## Supported Export Formats

YOLOv5 inference is officially supported in 12 formats:

!!! tip "Performance Tips"

    - Export to ONNX or OpenVINO for up to 3x CPU speedup. See [CPU Benchmarks](https://github.com/ultralytics/yolov5/pull/6613).
    - Export to TensorRT for up to 5x GPU speedup. See [GPU Benchmarks](https://github.com/ultralytics/yolov5/pull/6963).

| Format                                                       | `export.py --include` | Model                     |
| :----------------------------------------------------------- | :-------------------- | :------------------------ |
| [PyTorch](https://pytorch.org/)                              | -                     | `yolov5s.pt`              |
| [TorchScript](../../integrations/torchscript.md)             | `torchscript`         | `yolov5s.torchscript`     |
| [ONNX](../../integrations/onnx.md)                           | `onnx`                | `yolov5s.onnx`            |
| [OpenVINO](../../integrations/openvino.md)                   | `openvino`            | `yolov5s_openvino_model/` |
| [TensorRT](../../integrations/tensorrt.md)                   | `engine`              | `yolov5s.engine`          |
| [CoreML](../../integrations/coreml.md)                       | `coreml`              | `yolov5s.mlmodel`         |
| [TensorFlow SavedModel](../../integrations/tf-savedmodel.md) | `saved_model`         | `yolov5s_saved_model/`    |
| [TensorFlow GraphDef](../../integrations/tf-graphdef.md)     | `pb`                  | `yolov5s.pb`              |
| [TensorFlow Lite](../../integrations/tflite.md)              | `tflite`              | `yolov5s.tflite`          |
| [TensorFlow Edge TPU](../../integrations/edge-tpu.md)        | `edgetpu`             | `yolov5s_edgetpu.tflite`  |
| [TensorFlow.js](../../integrations/tfjs.md)                  | `tfjs`                | `yolov5s_web_model/`      |
| [PaddlePaddle](../../integrations/paddlepaddle.md)           | `paddle`              | `yolov5s_paddle_model/`   |

## Benchmarks

Benchmarks below run on a Colab Pro with the YOLOv5 tutorial notebook <a href="https://colab.research.google.com/github/ultralytics/yolov5/blob/master/tutorial.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"></a>. To reproduce:

```bash
python benchmarks.py --weights yolov5s.pt --imgsz 640 --device 0
```

### Colab Pro V100 GPU

```
benchmarks: weights=/content/yolov5/yolov5s.pt, imgsz=640, batch_size=1, data=/content/yolov5/data/coco128.yaml, device=0, half=False, test=False
Checking setup...
YOLOv5 🚀 v6.1-135-g7926afc torch 1.10.0+cu111 CUDA:0 (Tesla V100-SXM2-16GB, 16160MiB)
Setup complete ✅ (8 CPUs, 51.0 GB RAM, 46.7/166.8 GB disk)

Benchmarks complete (458.07s)
                   Format  mAP@0.5:0.95  Inference time (ms)
0                 PyTorch        0.4623                10.19
1             TorchScript        0.4623                 6.85
2                    ONNX        0.4623                14.63
3                OpenVINO           NaN                  NaN
4                TensorRT        0.4617                 1.89
5                  CoreML           NaN                  NaN
6   TensorFlow SavedModel        0.4623                21.28
7     TensorFlow GraphDef        0.4623                21.22
8         TensorFlow Lite           NaN                  NaN
9     TensorFlow Edge TPU           NaN                  NaN
10          TensorFlow.js           NaN                  NaN
```

### Colab Pro CPU

```
benchmarks: weights=/content/yolov5/yolov5s.pt, imgsz=640, batch_size=1, data=/content/yolov5/data/coco128.yaml, device=cpu, half=False, test=False
Checking setup...
YOLOv5 🚀 v6.1-135-g7926afc torch 1.10.0+cu111 CPU
Setup complete ✅ (8 CPUs, 51.0 GB RAM, 41.5/166.8 GB disk)

Benchmarks complete (241.20s)
                   Format  mAP@0.5:0.95  Inference time (ms)
0                 PyTorch        0.4623               127.61
1             TorchScript        0.4623               131.23
2                    ONNX        0.4623                69.34
3                OpenVINO        0.4623                66.52
4                TensorRT           NaN                  NaN
5                  CoreML           NaN                  NaN
6   TensorFlow SavedModel        0.4623               123.79
7     TensorFlow GraphDef        0.4623               121.57
8         TensorFlow Lite        0.4623               316.61
9     TensorFlow Edge TPU           NaN                  NaN
10          TensorFlow.js           NaN                  NaN
```

## Export a Trained YOLOv5 Model

This command exports a pretrained YOLOv5s model to TorchScript and ONNX formats. `yolov5s.pt` is the 'small' model, the second-smallest model available. Other options are `yolov5n.pt`, `yolov5m.pt`, `yolov5l.pt` and `yolov5x.pt`, along with their P6 counterparts i.e. `yolov5s6.pt` or you own custom training checkpoint i.e. `runs/exp/weights/best.pt`. For details on all available models please see our README [table](https://github.com/ultralytics/yolov5#pretrained-checkpoints).

```bash
python export.py --weights yolov5s.pt --include torchscript onnx
```

!!! tip

    Add `--half` to export models at FP16 half [precision](https://www.ultralytics.com/glossary/precision) for smaller file sizes

Output:

```text
export: data=data/coco128.yaml, weights=['yolov5s.pt'], imgsz=[640, 640], batch_size=1, device=cpu, half=False, inplace=False, train=False, keras=False, optimize=False, int8=False, dynamic=False, simplify=False, opset=12, verbose=False, workspace=4, nms=False, agnostic_nms=False, topk_per_class=100, topk_all=100, iou_thres=0.45, conf_thres=0.25, include=['torchscript', 'onnx']
YOLOv5 🚀 v6.2-104-ge3e5122 Python-3.8.0 torch-1.12.1+cu113 CPU

Downloading https://github.com/ultralytics/yolov5/releases/download/v6.2/yolov5s.pt to yolov5s.pt...
100% 14.1M/14.1M [00:00<00:00, 274MB/s]

Fusing layers...
YOLOv5s summary: 213 layers, 7225885 parameters, 0 gradients

PyTorch: starting from yolov5s.pt with output shape (1, 25200, 85) (14.1 MB)

TorchScript: starting export with torch 1.12.1+cu113...
TorchScript: export success ✅ 1.7s, saved as yolov5s.torchscript (28.1 MB)

ONNX: starting export with onnx 1.12.0...
ONNX: export success ✅ 2.3s, saved as yolov5s.onnx (28.0 MB)

Export complete (5.5s)
Results saved to /content/yolov5
Detect:          python detect.py --weights yolov5s.onnx
Validate:        python val.py --weights yolov5s.onnx
PyTorch Hub:     model = torch.hub.load('ultralytics/yolov5', 'custom', 'yolov5s.onnx')
Visualize:       https://netron.app/
```

The 3 exported models will be saved alongside the original PyTorch model:

<p align="center"><img width="700" src="https://github.com/ultralytics/docs/releases/download/0/yolo-export-locations.avif" alt="YOLO export locations"></p>

[Netron Viewer](https://github.com/lutzroeder/netron) is recommended for visualizing exported models:

<p align="center"><img width="850" src="https://github.com/ultralytics/docs/releases/download/0/yolo-model-visualization.avif" alt="YOLO model visualization"></p>

## Exported Model Usage Examples

`detect.py` runs inference on exported models:

```bash
python detect.py --weights yolov5s.pt             # PyTorch
python detect.py --weights yolov5s.torchscript    # TorchScript
python detect.py --weights yolov5s.onnx           # ONNX Runtime or OpenCV DNN with dnn=True
python detect.py --weights yolov5s_openvino_model # OpenVINO
python detect.py --weights yolov5s.engine         # TensorRT
python detect.py --weights yolov5s.mlmodel        # CoreML (macOS only)
python detect.py --weights yolov5s_saved_model    # TensorFlow SavedModel
python detect.py --weights yolov5s.pb             # TensorFlow GraphDef
python detect.py --weights yolov5s.tflite         # TensorFlow Lite
python detect.py --weights yolov5s_edgetpu.tflite # TensorFlow Edge TPU
python detect.py --weights yolov5s_paddle_model   # PaddlePaddle
```

`val.py` runs validation on exported models:

```bash
python val.py --weights yolov5s.pt             # PyTorch
python val.py --weights yolov5s.torchscript    # TorchScript
python val.py --weights yolov5s.onnx           # ONNX Runtime or OpenCV DNN with dnn=True
python val.py --weights yolov5s_openvino_model # OpenVINO
python val.py --weights yolov5s.engine         # TensorRT
python val.py --weights yolov5s.mlmodel        # CoreML (macOS Only)
python val.py --weights yolov5s_saved_model    # TensorFlow SavedModel
python val.py --weights yolov5s.pb             # TensorFlow GraphDef
python val.py --weights yolov5s.tflite         # TensorFlow Lite
python val.py --weights yolov5s_edgetpu.tflite # TensorFlow Edge TPU
python val.py --weights yolov5s_paddle_model   # PaddlePaddle
```

Use PyTorch Hub with exported YOLOv5 models:

```python
import torch

# Model
model = torch.hub.load("ultralytics/yolov5", "custom", "yolov5s.pt")
model = torch.hub.load("ultralytics/yolov5", "custom", "yolov5s.torchscript")  # TorchScript
model = torch.hub.load("ultralytics/yolov5", "custom", "yolov5s.onnx")  # ONNX Runtime
model = torch.hub.load("ultralytics/yolov5", "custom", "yolov5s_openvino_model")  # OpenVINO
model = torch.hub.load("ultralytics/yolov5", "custom", "yolov5s.engine")  # TensorRT
model = torch.hub.load("ultralytics/yolov5", "custom", "yolov5s.mlmodel")  # CoreML (macOS Only)
model = torch.hub.load("ultralytics/yolov5", "custom", "yolov5s_saved_model")  # TensorFlow SavedModel
model = torch.hub.load("ultralytics/yolov5", "custom", "yolov5s.pb")  # TensorFlow GraphDef
model = torch.hub.load("ultralytics/yolov5", "custom", "yolov5s.tflite")  # TensorFlow Lite
model = torch.hub.load("ultralytics/yolov5", "custom", "yolov5s_edgetpu.tflite")  # TensorFlow Edge TPU
model = torch.hub.load("ultralytics/yolov5", "custom", "yolov5s_paddle_model")  # PaddlePaddle

# Images
img = "https://ultralytics.com/images/zidane.jpg"  # or file, Path, PIL, OpenCV, numpy, list

# Inference
results = model(img)

# Results
results.print()  # or .show(), .save(), .crop(), .pandas(), etc.
```

## OpenCV DNN inference

[OpenCV](https://www.ultralytics.com/glossary/opencv) inference with ONNX models:

```bash
python export.py --weights yolov5s.pt --include onnx

python detect.py --weights yolov5s.onnx --dnn # detect
python val.py --weights yolov5s.onnx --dnn    # validate
```

## C++ Inference

YOLOv5 OpenCV DNN C++ inference on exported ONNX model examples:

- [https://github.com/Hexmagic/ONNX-yolov5/blob/master/src/test.cpp](https://github.com/Hexmagic/ONNX-yolov5/blob/master/src/test.cpp)
- [https://github.com/doleron/yolov5-opencv-cpp-python](https://github.com/doleron/yolov5-opencv-cpp-python)

YOLOv5 OpenVINO C++ inference examples:

- [https://github.com/dacquaviva/yolov5-openvino-cpp-python](https://github.com/dacquaviva/yolov5-openvino-cpp-python)
- [https://github.com/UNeedCryDear/yolov5-seg-opencv-dnn-cpp](https://github.com/UNeedCryDear/yolov5-seg-opencv-onnxruntime-cpp)

## TensorFlow.js Web Browser Inference

- [https://aukerul-shuvo.github.io/YOLOv5_TensorFlow-JS/](https://aukerul-shuvo.github.io/YOLOv5_TensorFlow-JS/)

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
