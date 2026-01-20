---
comments: true
description: Learn how to deploy YOLOv5 using Neural Magic's DeepSparse for GPU-class performance on CPUs. Discover easy integration, flexible deployments, and more.
keywords: YOLOv5, DeepSparse, Neural Magic, YOLO deployment, Sparse inference, Deep learning, Model sparsity, CPU optimization, No hardware accelerators, AI deployment
---

<!--
Copyright (c) 2021 - present / Neuralmagic, Inc. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

   http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing,
software distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
-->

# Deploy YOLOv5 with Neural Magic's DeepSparse

Welcome to software-delivered AI.

This guide explains how to deploy YOLOv5 with Neural Magic's DeepSparse.

DeepSparse is an inference runtime with exceptional performance on CPUs. For instance, compared to the ONNX Runtime baseline, DeepSparse offers a 5.8x speed-up for YOLOv5s, running on the same machine!

<p align="center">
  <img width="60%" src="https://cdn.jsdelivr.net/gh/ultralytics/assets@main/docs/yolov5-speed-improvement.avif" alt="YOLOv5 DeepSparse vs ONNX Runtime speed comparison chart">
</p>

For the first time, your [deep learning](https://www.ultralytics.com/glossary/deep-learning-dl) workloads can meet the performance demands of production without the complexity and costs of hardware accelerators. Put simply, DeepSparse gives you the performance of GPUs and the simplicity of software:

- **Flexible Deployments**: Run consistently across cloud, data center, and edge with any hardware provider from Intel to AMD to ARM
- **Infinite Scalability**: Scale vertically to 100s of cores, out with standard Kubernetes, or fully-abstracted with Serverless
- **Easy Integration**: Clean APIs for integrating your model into an application and monitoring it in production

## How Does DeepSparse Achieve GPU-Class Performance?

DeepSparse takes advantage of model sparsity to gain its performance speedup.

Sparsification through pruning and quantization is a broadly studied technique, allowing order-of-magnitude reductions in the size and compute needed to execute a network, while maintaining high [accuracy](https://www.ultralytics.com/glossary/accuracy). DeepSparse is sparsity-aware, meaning it skips the zeroed out parameters, shrinking amount of compute in a forward pass. Since the sparse computation is now memory bound, DeepSparse executes the network depth-wise, breaking the problem into Tensor Columns, vertical stripes of computation that fit in cache.

<p align="center">
  <img width="60%" src="https://cdn.jsdelivr.net/gh/ultralytics/assets@main/docs/tensor-columns.avif" alt="DeepSparse tensor columns for sparse neural network inference">
</p>

Sparse networks with compressed computation, executed depth-wise in cache, allows DeepSparse to deliver GPU-class performance on CPUs!

## How Do I Create A Sparse Version of YOLOv5 Trained on My Data?

Neural Magic's open-source model repository, [SparseZoo](https://github.com/neuralmagic/sparsezoo/blob/main/README.md), contains pre-sparsified checkpoints of each YOLOv5 model. Using [SparseML](https://github.com/neuralmagic/sparseml), which is integrated with Ultralytics, you can fine-tune a sparse checkpoint onto your data with a single CLI command.

[Checkout Neural Magic's YOLOv5 documentation for more details](https://www.redhat.com/en/about/press-releases/red-hat-completes-acquisition-neural-magic-fuel-optimized-generative-ai-innovation-across-hybrid-cloud).

## DeepSparse Usage

We will walk through an example benchmarking and deploying a sparse version of YOLOv5s with DeepSparse.

### Install DeepSparse

Run the following to install DeepSparse. We recommend you use a virtual environment with Python.

```bash
pip install "deepsparse[server,yolo,onnxruntime]"
```

### Collect an ONNX File

DeepSparse accepts a model in the ONNX format, passed either as:

- A SparseZoo stub which identifies an ONNX file in the SparseZoo
- A local path to an ONNX model in a filesystem

The examples below use the standard dense and pruned-quantized YOLOv5s checkpoints, identified by the following SparseZoo stubs:

```bash
zoo:cv/detection/yolov5-s/pytorch/ultralytics/coco/base-none
zoo:cv/detection/yolov5-s/pytorch/ultralytics/coco/pruned65_quant-none
```

### Deploy a Model

DeepSparse offers convenient APIs for integrating your model into an application.

To try the deployment examples below, pull down a sample image and save it as `basilica.jpg` with the following:

```bash
wget -O basilica.jpg https://raw.githubusercontent.com/neuralmagic/deepsparse/main/src/deepsparse/yolo/sample_images/basilica.jpg
```

#### Python API

`Pipelines` wrap pre-processing and output post-processing around the runtime, providing a clean interface for adding DeepSparse to an application. The DeepSparse-Ultralytics integration includes an out-of-the-box `Pipeline` that accepts raw images and outputs the bounding boxes.

Create a `Pipeline` and run inference:

```python
from deepsparse import Pipeline

# list of images in local filesystem
images = ["basilica.jpg"]

# create Pipeline
model_stub = "zoo:cv/detection/yolov5-s/pytorch/ultralytics/coco/pruned65_quant-none"
yolo_pipeline = Pipeline.create(
    task="yolo",
    model_path=model_stub,
)

# run inference on images, receive bounding boxes + classes
pipeline_outputs = yolo_pipeline(images=images, iou_thres=0.6, conf_thres=0.001)
print(pipeline_outputs)
```

If you are running in the cloud, you may get an error that OpenCV cannot find `libGL.so.1`. You can either install the missing library:

```bash
apt-get install libgl1
```

Or use the headless Ultralytics package that avoids GUI dependencies entirely:

```bash
pip install ultralytics-opencv-headless
```

#### HTTP Server

DeepSparse Server runs on top of the popular [FastAPI](https://fastapi.tiangolo.com/) web framework and [Uvicorn](https://www.uvicorn.org/) web server. With just a single CLI command, you can easily setup a model service endpoint with DeepSparse. The Server supports any Pipeline from DeepSparse, including [object detection](https://www.ultralytics.com/glossary/object-detection) with YOLOv5, enabling you to send raw images to the endpoint and receive the bounding boxes.

Spin up the Server with the pruned-quantized YOLOv5s:

```bash
deepsparse.server \
  --task yolo \
  --model_path zoo:cv/detection/yolov5-s/pytorch/ultralytics/coco/pruned65_quant-none
```

An example request, using Python's `requests` package:

```python
import json

import requests

# list of images for inference (local files on client side)
path = ["basilica.jpg"]
files = [("request", open(img, "rb")) for img in path]

# send request over HTTP to /predict/from_files endpoint
url = "http://0.0.0.0:5543/predict/from_files"
resp = requests.post(url=url, files=files)

# response is returned in JSON
annotations = json.loads(resp.text)  # dictionary of annotation results
bounding_boxes = annotations["boxes"]
labels = annotations["labels"]
```

#### Annotate CLI

You can also use the annotate command to have the engine save an annotated photo on disk. Try `--source 0` to annotate your live webcam feed!

```bash
deepsparse.object_detection.annotate --model_filepath zoo:cv/detection/yolov5-s/pytorch/ultralytics/coco/pruned65_quant-none --source basilica.jpg
```

Running the above command will create an `annotation-results` folder and save the annotated image inside.

<p align="center">
<img src="https://cdn.jsdelivr.net/gh/ultralytics/assets@main/docs/basilica-annotated.avif" alt="YOLOv5 detection results with bounding boxes" width="60%">
</p>

## Benchmarking Performance

We will compare DeepSparse's throughput to ONNX Runtime's throughput on YOLOv5s, using DeepSparse's benchmarking script.

The benchmarks were run on an AWS `c6i.8xlarge` instance (16 cores).

### Batch 32 Performance Comparison

#### ONNX Runtime Baseline

At batch 32, ONNX Runtime achieves 42 images/sec with the standard dense YOLOv5s:

```bash
deepsparse.benchmark zoo:cv/detection/yolov5-s/pytorch/ultralytics/coco/base-none -s sync -b 32 -nstreams 1 -e onnxruntime

# Original Model Path: zoo:cv/detection/yolov5-s/pytorch/ultralytics/coco/base-none
# Batch Size: 32
# Scenario: sync
# Throughput (items/sec): 41.9025
```

#### DeepSparse Dense Performance

While DeepSparse offers its best performance with optimized sparse models, it also performs well with the standard dense YOLOv5s.

At batch 32, DeepSparse achieves 70 images/sec with the standard dense YOLOv5s, a **1.7x performance improvement over ORT**!

```bash
deepsparse.benchmark zoo:cv/detection/yolov5-s/pytorch/ultralytics/coco/base-none -s sync -b 32 -nstreams 1

# Original Model Path: zoo:cv/detection/yolov5-s/pytorch/ultralytics/coco/base-none
# Batch Size: 32
# Scenario: sync
# Throughput (items/sec): 69.5546
```

#### DeepSparse Sparse Performance

When sparsity is applied to the model, DeepSparse's performance gains over ONNX Runtime is even stronger.

At batch 32, DeepSparse achieves 241 images/sec with the pruned-quantized YOLOv5s, a **5.8x performance improvement over ORT**!

```bash
deepsparse.benchmark zoo:cv/detection/yolov5-s/pytorch/ultralytics/coco/pruned65_quant-none -s sync -b 32 -nstreams 1

# Original Model Path: zoo:cv/detection/yolov5-s/pytorch/ultralytics/coco/pruned65_quant-none
# Batch Size: 32
# Scenario: sync
# Throughput (items/sec): 241.2452
```

### Batch 1 Performance Comparison

DeepSparse is also able to gain a speed-up over ONNX Runtime for the latency-sensitive, batch 1 scenario.

#### ONNX Runtime Baseline

At batch 1, ONNX Runtime achieves 48 images/sec with the standard, dense YOLOv5s.

```bash
deepsparse.benchmark zoo:cv/detection/yolov5-s/pytorch/ultralytics/coco/base-none -s sync -b 1 -nstreams 1 -e onnxruntime

# Original Model Path: zoo:cv/detection/yolov5-s/pytorch/ultralytics/coco/base-none
# Batch Size: 1
# Scenario: sync
# Throughput (items/sec): 48.0921
```

#### DeepSparse Sparse Performance

At batch 1, DeepSparse achieves 135 items/sec with a pruned-quantized YOLOv5s, **a 2.8x performance gain over ONNX Runtime!**

```bash
deepsparse.benchmark zoo:cv/detection/yolov5-s/pytorch/ultralytics/coco/pruned65_quant-none -s sync -b 1 -nstreams 1

# Original Model Path: zoo:cv/detection/yolov5-s/pytorch/ultralytics/coco/pruned65_quant-none
# Batch Size: 1
# Scenario: sync
# Throughput (items/sec): 134.9468
```

Since `c6i.8xlarge` instances have VNNI instructions, DeepSparse's throughput can be pushed further if weights are pruned in blocks of 4.

At batch 1, DeepSparse achieves 180 items/sec with a 4-block pruned-quantized YOLOv5s, a **3.7x performance gain over ONNX Runtime!**

```bash
deepsparse.benchmark zoo:cv/detection/yolov5-s/pytorch/ultralytics/coco/pruned35_quant-none-vnni -s sync -b 1 -nstreams 1

# Original Model Path: zoo:cv/detection/yolov5-s/pytorch/ultralytics/coco/pruned35_quant-none-vnni
# Batch Size: 1
# Scenario: sync
# Throughput (items/sec): 179.7375
```

## Get Started With DeepSparse

**Research or Testing?** DeepSparse Community is free for research and testing. Get started with their [Documentation](https://www.redhat.com/en/about/press-releases/red-hat-completes-acquisition-neural-magic-fuel-optimized-generative-ai-innovation-across-hybrid-cloud).

For more information on deploying YOLOv5 with DeepSparse, check out the [Neural Magic's DeepSparse documentation](https://www.redhat.com/en/about/press-releases/red-hat-completes-acquisition-neural-magic-fuel-optimized-generative-ai-innovation-across-hybrid-cloud) and the [Ultralytics blog post on DeepSparse integration](https://www.ultralytics.com/blog/deploy-yolov5-with-neural-magics-deepsparse-for-gpu-class-performance-on-cpus).
