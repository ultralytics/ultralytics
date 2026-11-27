---
title: Export YOLO26 to TorchScript
comments: true
description: Export Ultralytics YOLO26 models to PyTorch TorchScript with format=torchscript. Covers the .torchscript file and its metadata, FP16 on CUDA, embedded NMS, output shapes, LibTorch C++ inference, devices, and benchmarks.
keywords: YOLO26, TorchScript, PyTorch TorchScript, TorchScript export, torchscript file, LibTorch, C++ inference, torch.jit.load, torch::jit::load, FP16, CUDA, NVIDIA Triton, Ultralytics
---

# PyTorch TorchScript Export

!!! warning "PyTorch is retiring TorchScript"

    PyTorch has [deprecated TorchScript](https://docs.pytorch.org/docs/stable/jit.html) and is gradually removing its features. For new mobile and edge deployments, use the supported [ExecuTorch integration](executorch.md). Ultralytics retains regular TorchScript export for legacy C++ compatibility.

[TorchScript](https://docs.pytorch.org/docs/stable/jit.html) is the serialized program format of [PyTorch](https://www.ultralytics.com/glossary/pytorch) that runs a model without its Python code, in the LibTorch C++ library or in PyTorch itself. Exporting an [Ultralytics YOLO26](../models/yolo26.md) model with `format="torchscript"` traces it with PyTorch into a single `yolo26n.torchscript` file, in FP32 or FP16, that a C++ application loads with LibTorch and that `YOLO()` runs on CPUs and NVIDIA GPUs.

## At a Glance

| Property          | Details                                                                                                                                                                                                                                                                               |
| ----------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `format` argument | `torchscript`                                                                                                                                                                                                                                                                         |
| Output            | `yolo26n.torchscript`, one file with the [metadata](#output-format) inside                                                                                                                                                                                                            |
| Runs on           | CPUs and NVIDIA GPUs through PyTorch and through LibTorch in the [C++ example](#run-with-libtorch), and NVIDIA GPUs through Triton. Ultralytics runs it on the CPU and on CUDA GPUs; Apple GPUs (`device="mps"`) fail for YOLO26, see [Limitations](#limitations-and-troubleshooting) |
| Runtime           | PyTorch (`torch.jit.load`) in Python, LibTorch (`torch::jit::load`) in C++, and the PyTorch backend of NVIDIA Triton Inference Server                                                                                                                                                 |
| Precision         | FP32 (default) and FP16. YOLO26n is 10.2 to 10.3 MB in FP32 and 5.4 MB in FP16                                                                                                                                                                                                        |
| Export host       | Linux and macOS, with PyTorch 1.8 or newer and no other package. FP32 exports run on the CPU; FP16 needs a CUDA GPU or Apple silicon with PyTorch 2.0 or newer. Windows was not run. See [Quickstart](#quickstart)                                                                    |

!!! tip "Export in the cloud"

    [Ultralytics Platform](../platform/train/models.md#export-model) can export a trained model to TorchScript, so you do not need a local export environment.

## Quickstart

Install the Ultralytics package in a virtual environment, here on Linux or macOS:

```bash
python3 -m venv torchscript-env && source torchscript-env/bin/activate
pip install ultralytics
```

The export needs only PyTorch, which `pip install ultralytics` installs: in the environments used for this page, the first export installed and replaced no package, and the export, prediction, and validation ran the same with `YOLO_AUTOINSTALL=false`. PyTorch 2.14 and newer log deprecation `FutureWarning` messages during the export and when the model loads; the export still completes, see [Limitations and Troubleshooting](#limitations-and-troubleshooting). The export has these version requirements:

| Requirement      | Version                                                                                                                                                                                                                     |
| ---------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Python           | 3.8 or newer, as for the Ultralytics package. Exports, prediction, and validation ran with every version from 3.8 to 3.14                                                                                                   |
| PyTorch          | 1.8 or newer, as for the Ultralytics package; no other package is needed. Exports for this page ran with PyTorch 1.8.0, 2.4.1, 2.8.0, 2.14.1, and 2.15.0, with 2.14.0 on macOS, and with the pre-release 2.16.0.dev20261126 |
| LibTorch (C++)   | Only for C++ applications. The [LibTorch example](#run-with-libtorch) ran with LibTorch 2.15.0 for the CPU and for CUDA 13.2, and OpenCV 4.14.0, built with CMake 3.28 and GCC 13.3                                         |
| Operating system | Exports ran on Ubuntu 24.04 x86-64 and on macOS 27 with an Apple M4; Windows and ARM64 Linux were not run                                                                                                                   |

For other installation methods, see the [installation guide](../quickstart.md). Then export a [YOLO26 model](https://platform.ultralytics.com/ultralytics/yolo26) and load the exported file to [run inference](../modes/predict.md) or [validate](../modes/val.md) its accuracy:

!!! example "Export"

    === "Python"

        ```python
        from ultralytics import YOLO

        # Load a YOLO26 model
        model = YOLO("yolo26n.pt")

        # Export the model to TorchScript format
        model.export(format="torchscript")  # creates 'yolo26n.torchscript'
        ```

    === "CLI"

        ```bash
        # Export a YOLO26n PyTorch model to TorchScript format
        yolo export model=yolo26n.pt format=torchscript # creates 'yolo26n.torchscript'
        ```

On a shared AMD EPYC 9655 server, every export of YOLO26n took 0.2 to 1.1 s, as the export log reports, and of YOLO26x 0.5 to 1.1 s. The exported Ultralytics YOLO26 model runs on the export host, so you can test it there:

!!! example "Predict"

    === "Python"

        ```python
        from ultralytics import YOLO

        # Load the exported TorchScript model
        model = YOLO("yolo26n.torchscript")

        # Run inference
        results = model("https://ultralytics.com/images/bus.jpg")
        ```

    === "CLI"

        ```bash
        # Run inference with the exported TorchScript model
        yolo predict model=yolo26n.torchscript source='https://ultralytics.com/images/bus.jpg'
        ```

!!! example "Validate"

    === "Python"

        ```python
        from ultralytics import YOLO

        # Load the exported TorchScript model
        model = YOLO("yolo26n.torchscript")

        # Validate accuracy on the COCO8 dataset
        metrics = model.val(data="coco8.yaml")
        ```

    === "CLI"

        ```bash
        # Validate the exported TorchScript model
        yolo val model=yolo26n.torchscript data=coco8.yaml
        ```

## Supported Tasks

TorchScript export supports all seven Ultralytics tasks: [detection](../tasks/detect.md), [segmentation](../tasks/segment.md), [semantic segmentation](../tasks/semantic.md), [depth estimation](../tasks/depth.md), [classification](../tasks/classify.md), [pose estimation](../tasks/pose.md), and [OBB](../tasks/obb.md). Semantic segmentation and depth estimation are available only with YOLO26, the only family that ships those heads. For this page, the YOLO26n checkpoint of every task was exported with each value of `nms`, except `nms=True` for classification, which fails (see [Output Format](#output-format)), and run on the export host; the `nms=None` export was validated on its small dataset (`coco8.yaml`, `coco8-seg.yaml`, `cityscapes8.yaml`, `depth8.yaml`, `imagenet10`, `coco8-pose.yaml`, and `dota8.yaml`), where the FP32 TorchScript model reached the same metrics as the PyTorch model run with `rect=False`; the FP16 export of every task ran on a CUDA GPU. [YOLO11](../models/yolo11.md) detection and segmentation, [YOLOv10](../models/yolov10.md), [YOLOv8](../models/yolov8.md), [YOLOv5u](../models/yolov5.md), and [RT-DETR](../models/rtdetr.md) detection models were exported, run, and validated as well.

## Export Arguments

| Argument   | Type             | Default         | Description                                                                                                                                                                                                                                                                                                                                                                                 |
| ---------- | ---------------- | --------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `format`   | `str`            | `'torchscript'` | Target format for the exported model, defining compatibility with various deployment environments.                                                                                                                                                                                                                                                                                          |
| `imgsz`    | `int` or `tuple` | `640`           | Image size of the model input: an integer for square images or a tuple `(height, width)`. Fixed in the `.torchscript` file unless `dynamic=True`. When not passed, the export uses the size the checkpoint was trained at: `640` for detect and segment, `1024` for the Cityscapes semantic checkpoint, `768` for depth, `224` for classify, `640` for pose, and `1024` for OBB.            |
| `quantize` | `int` or `str`   | `None`          | Precision of the weights and the computation: `16` (FP16, also spelled `'fp16'`) or `32`/unset (FP32). FP16 needs a CUDA GPU or Apple silicon at export (`device=0` or `device="mps"`); INT8 is not supported, see [Precision](#precision). Replaces the deprecated `half` flag.                                                                                                            |
| `batch`    | `int`            | `1`             | Batch size of the example input that the export traces with, stored in the metadata.                                                                                                                                                                                                                                                                                                        |
| `dynamic`  | `bool`           | `False`         | Lets the model take other image sizes at inference.                                                                                                                                                                                                                                                                                                                                         |
| `nms`      | `bool`, optional | `None`          | `None` writes the raw output, which needs NMS for YOLO detection, segmentation, pose, and OBB models; `True` embeds NMS in these models; `False` keeps the NMS-free head of models that have one, such as YOLO26; a model without one, such as YOLO11, logs `This model has no one-to-one head; using one-to-many outputs.` and writes the raw output. See [Output Format](#output-format). |
| `device`   | `str`            | `None`          | Device that runs the PyTorch model while the export traces it, such as `cpu`, `0` for the first CUDA GPU, or `mps`.                                                                                                                                                                                                                                                                         |

For the arguments shared by all formats, see the [Export mode documentation](../modes/export.md). Other format-specific arguments fail, for example `opset=12` with `argument 'opset' is not supported for format='torchscript'`, and so do `simplify=False`, `optimize=True`, `workspace`, `data`, and a `fraction` other than 1.0. With `nms=True` or `nms=False`, the export also takes `max_det` and `agnostic_nms`, and with `nms=True` the `conf` and `iou` of the embedded NMS.

### Precision

`quantize` selects the precision of the weights stored in the `.torchscript` file and of the traced computation:

| `quantize`       | Result                                                                                                                                                                      |
| ---------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| unset or `32`    | FP32 weights. YOLO26n is 10.2 to 10.3 MB                                                                                                                                    |
| `16` or `'fp16'` | FP16 weights with `device=0` or `device="mps"`; YOLO26n is 5.4 MB. On the CPU, the export fails with `FP16 TorchScript export is only supported on GPU, i.e. use device=0.` |
| `8` or `'int8'`  | Fails with `quantize=8 (INT8) is not supported; format='torchscript' supports quantize=16 (FP16), 32 or None (FP32).`                                                       |
| `'w8a16'`        | Fails with `quantize='w8a16' is not supported; format='torchscript' supports quantize=16 (FP16), 32 or None (FP32).`                                                        |

- **FP16 export**: on a CUDA GPU or Apple silicon, Ultralytics converts the model and the example input to FP16 before it traces them, so the file stores FP16 weights and an FP16 computation.
- **FP16 at inference**: `YOLO()` computes in FP16 only on a GPU and only when `quantize=16` is passed to `predict` or `val`, with an FP16 or an FP32 file; without it, Ultralytics converts an FP16 file to FP32 when it loads it. On the RTX PRO 6000, YOLO26n took 2.1 ms with the FP16 file and `quantize=16`, 1.9 ms with the FP16 file without it, and 2.1 ms with the FP32 file and `quantize=16`, see [Benchmarks](#benchmarks). On the CPU, Ultralytics always computes in FP32, and an FP16 file gave the same detections there.

To export Ultralytics YOLO26 in FP16 on a CUDA GPU and run it in FP16:

!!! example "FP16 on a CUDA GPU"

    === "Python"

        ```python
        from ultralytics import YOLO

        model = YOLO("yolo26n.pt")

        # Export in FP16 on the first CUDA GPU
        model.export(format="torchscript", quantize=16, device=0)  # creates 'yolo26n.torchscript'

        # Load the exported model and run inference in FP16 on the GPU
        model = YOLO("yolo26n.torchscript")
        results = model("https://ultralytics.com/images/bus.jpg", device=0, quantize=16)
        ```

    === "CLI"

        ```bash
        # Export in FP16 on the first CUDA GPU
        yolo export model=yolo26n.pt format=torchscript quantize=16 device=0 # creates 'yolo26n.torchscript'

        # Run inference in FP16 on the GPU
        yolo predict model=yolo26n.torchscript source='https://ultralytics.com/images/bus.jpg' device=0 quantize=16
        ```

The sizes of the `.torchscript` file for the official YOLO26n checkpoints at their default export size, in MB (10^6 bytes). The export log prints the size in MiB, as `(9.8 MB)` and `(5.1 MB)` for the detection model:

| Task     | Default `imgsz` | FP32 | FP16 |
| -------- | --------------- | ---- | ---- |
| Detect   | `640`           | 10.2 | 5.4  |
| Segment  | `640`           | 11.6 | 6.1  |
| Semantic | `1024`          | 6.8  | 3.6  |
| Depth    | `768`           | 25.9 | 13.2 |
| Classify | `224`           | 11.5 | 5.9  |
| Pose     | `640`           | 12.4 | 6.5  |
| OBB      | `1024`          | 10.6 | 5.6  |

The file also stores the source paths of the installed Ultralytics package, more than 1,600 times for YOLO26n, so its size changes by a few kB with the install location: the FP32 YOLO26n files of this page were 10,246,105 to 10,254,232 bytes.

## Output Format

The export writes `yolo26n.torchscript` next to the `.pt` file. The file is a zip archive of the traced program and its weights; the Ultralytics metadata (class names, task, image size, batch size, stride, and export arguments) is stored as JSON in its extra file `config.txt`, where Ultralytics reads it when it loads the model, and where `torch.jit.load` and `torch::jit::load` return it through their extra-files argument, see [Run with LibTorch](#run-with-libtorch).

The model takes one input of shape `(batch, 3, height, width)`: RGB, channels first (NCHW), `float32` (`float16` for an FP16 export), scaled to 0-1. Ultralytics resizes each image to that size with gray padding (letterbox; classification models are resized and center-cropped instead), and the box coordinates in the output refer to the padded image. The shapes below are for the official YOLO26n checkpoints at their default export size, with the Cityscapes checkpoint `yolo26n-sem.pt` for semantic segmentation:

| Task     | Default `imgsz` | `nms=None` (default)                                                     | `nms=False`                            | `nms=True`                                                 |
| -------- | --------------- | ------------------------------------------------------------------------ | -------------------------------------- | ---------------------------------------------------------- |
| Detect   | `640`           | `(1, 84, 8400)`                                                          | `(1, 300, 6)`                          | `(1, 300, 6)`                                              |
| Segment  | `640`           | `(1, 116, 8400)` and `(1, 32, 160, 160)`                                 | `(1, 300, 38)` and `(1, 32, 160, 160)` | `(1, 300, 38)` and `(1, 32, 160, 160)`                     |
| Semantic | `1024`          | `(1, 19, 1024, 1024)`, a score for each of the 19 classes and each pixel | Same as `nms=None`                     | Same as `nms=None`                                         |
| Depth    | `768`           | `(1, 1, 768, 768)`, depth map                                            | Same as `nms=None`                     | Same as `nms=None`                                         |
| Classify | `224`           | `(1, 1000)`, class probabilities                                         | Same as `nms=None`                     | Fails, see [Limitations](#limitations-and-troubleshooting) |
| Pose     | `640`           | `(1, 56, 8400)`: 4 box values, 1 class score, then 17 keypoints x 3      | `(1, 300, 57)`                         | `(1, 300, 57)`                                             |
| OBB      | `1024`          | `(1, 20, 21504)`: 4 box values, 15 class scores, then the angle          | `(1, 300, 7)`                          | `(1, 300, 7)`                                              |

- **Raw output (`nms=None`)**: detection, segmentation, pose, and OBB models write shape `(batch, values, anchors)`. For each anchor: the box center x, center y, width, and height in input pixels, then the class scores, then the values of the task (32 mask coefficients, 17 x 3 keypoint values, or the OBB angle in radians). Apply NMS to this output yourself. See the [ONNX output format](onnx.md#output-format) for the layout of each task.
- **NMS-free head (`nms=False`)**: YOLO26 detection, segmentation, pose, and OBB models write up to `max_det` (300 by default) detections per image, each with the box corners x1, y1, x2, y2 in input pixels, the score, the class index, and then the values of the task, so no NMS is needed. OBB rows hold the rotated box center x, center y, width, and height, the score, the class index, and the angle in radians. Keep the rows whose score is above your confidence threshold.
- **Embedded NMS (`nms=True`)**: YOLO detection, segmentation, pose, and OBB models write the same layout after NMS inside the model, keeping the boxes above the confidence threshold of the export, `conf=0.25` unless you pass another `conf`, and at most `max_det` rows (300 by default; `max_det=100` gave `(1, 100, 6)`). Semantic segmentation and depth models log a warning such as `'nms=True' is not valid for semantic models. Forcing 'nms=None'.` and export as with `nms=None`, and classification models fail with `'nms=True' is not valid for classification models.`
- **Semantic segmentation and depth**: the semantic output is a `float32` score for each of the 19 Cityscapes classes and each pixel, and Ultralytics takes the class with the highest score; this differs from the [ONNX export](onnx.md#output-format), which writes a `uint8` class map. The depth output is a `float32` depth map. In an FP16 export, both are `float16`. Both are the same for every value of `nms`.
- **RT-DETR**: the output of RT-DETR-L has shape `(1, 300, 6)`, which needs no NMS. `nms=True` logs `'nms=True' is not available for end2end models or non-YOLO heads. Forcing 'nms=None'.` and exports the same output.

The table below compares the raw outputs (`nms=None`) on the x86-64 export host with those of the [ONNX](onnx.md) export of the same checkpoint on one image, `bus.jpg`, and for OBB on the aerial image `boats.jpg`. The FP32 column is the largest absolute difference of any output value, where box values are in pixels; the FP16 column is the correlation of the output of the FP16 export, run on an RTX PRO 6000 GPU, with the ONNX output. For semantic segmentation, the class with the highest TorchScript score is compared with the class map of ONNX:

| Task              | FP32 against ONNX               | FP16 on a GPU against ONNX        |
| ----------------- | ------------------------------- | --------------------------------- |
| Detect            | 0.0042                          | 1.0000                            |
| Segment           | 0.0038, prototypes below 0.0001 | 1.0000, prototypes 1.0000         |
| Semantic          | Same class on every pixel       | Same class on 99.9% of the pixels |
| Depth             | below 0.0001                    | 1.0000                            |
| Classify          | below 0.0001                    | 1.0000, same top-1 class          |
| Pose              | 0.0015                          | 1.0000                            |
| OBB (`boats.jpg`) | 0.0048                          | 1.0000                            |

On the same images, the detections that the FP32 `nms=False` exports of the detection, segmentation, pose, and OBB models and the FP32 `nms=True` export of the detection model kept above a score of 0.25 matched those of the ONNX export with the same setting, with the same classes and values within 0.001.

`YOLO("yolo26n.torchscript")` reads the layout from the metadata, so you only need these details when you run the `.torchscript` file in your own code.

### Run with LibTorch

The examples below run Ultralytics YOLO26 detection with LibTorch in C++, and with `torch.jit` in Python, without the Ultralytics package. They read an export with `nms=False`, whose `(1, 300, 6)` output needs no NMS, and assume a 640x640 input; the first line of each example is the export command. Both load the metadata through the extra-files argument; the Python example reads the class names from it, and the C++ example prints class indices. Save the Python example as `main.py` and run it next to `yolo26n.torchscript` and `bus.jpg`. For this page, both examples ran on the Linux export host, the Python one with PyTorch 2.15.0 and also with 2.14.0 on macOS, and printed the same 5 objects and the same boxes, to the pixel, as `YOLO("yolo26n.torchscript")`.

!!! example "LibTorch"

    === "C++"

        ```cpp
        // Export first: yolo export model=yolo26n.pt format=torchscript nms=False
        #include <torch/script.h>
        #include <opencv2/opencv.hpp>

        #include <algorithm>
        #include <cmath>
        #include <iostream>

        int main(int argc, char* argv[]) {
            if (argc < 3) {
                std::cerr << "Usage: yolo26 <model.torchscript> <image>" << std::endl;
                return 1;
            }
            // Load the model; config.txt holds the Ultralytics metadata (class names, imgsz, task) as JSON
            torch::jit::ExtraFilesMap extra;
            extra["config.txt"] = "";
            torch::jit::Module model = torch::jit::load(argv[1], torch::kCPU, extra);
            model.eval();

            // Letterbox the image to 640x640 with gray padding, as Ultralytics does
            cv::Mat img = cv::imread(argv[2]);
            const int size = 640;
            float r = std::min(size / static_cast<float>(img.rows), size / static_cast<float>(img.cols));
            int nw = std::round(img.cols * r), nh = std::round(img.rows * r);
            int left = (size - nw) / 2, top = (size - nh) / 2;
            cv::Mat resized, canvas(size, size, CV_8UC3, cv::Scalar(114, 114, 114));
            cv::resize(img, resized, cv::Size(nw, nh), 0, 0, cv::INTER_LINEAR);
            resized.copyTo(canvas(cv::Rect(left, top, nw, nh)));
            cv::cvtColor(canvas, canvas, cv::COLOR_BGR2RGB);

            // HWC uint8 -> NCHW float32 in 0-1
            torch::Tensor x = torch::from_blob(canvas.data, {1, size, size, 3}, torch::kUInt8);
            x = x.permute({0, 3, 1, 2}).to(torch::kFloat32).div(255);

            // An export with nms=False outputs (1, 300, 6): x1, y1, x2, y2, score, class index
            torch::NoGradGuard no_grad;
            torch::Tensor y = model.forward({x}).toTensor()[0];
            for (int i = 0; i < y.size(0); i++) {
                float score = y[i][4].item<float>();
                if (score < 0.25) continue;
                // Undo the letterbox and clip to the image
                float x1 = std::clamp((y[i][0].item<float>() - left) / r, 0.f, static_cast<float>(img.cols));
                float y1 = std::clamp((y[i][1].item<float>() - top) / r, 0.f, static_cast<float>(img.rows));
                float x2 = std::clamp((y[i][2].item<float>() - left) / r, 0.f, static_cast<float>(img.cols));
                float y2 = std::clamp((y[i][3].item<float>() - top) / r, 0.f, static_cast<float>(img.rows));
                std::cout << "class " << y[i][5].item<int>() << " score " << score << " box " << std::round(x1) << " " << std::round(y1)
                          << " " << std::round(x2) << " " << std::round(y2) << std::endl;
            }
            return 0;
        }
        ```

    === "CMakeLists.txt"

        ```cmake
        cmake_minimum_required(VERSION 3.18)
        project(yolo26)
        find_package(Torch REQUIRED)
        find_package(OpenCV REQUIRED)
        add_executable(yolo26 main.cpp)
        target_link_libraries(yolo26 "${TORCH_LIBRARIES}" ${OpenCV_LIBS})
        set_property(TARGET yolo26 PROPERTY CXX_STANDARD 17)
        ```

    === "Python"

        ```python
        # Export first: yolo export model=yolo26n.pt format=torchscript nms=False
        import json

        import cv2
        import numpy as np
        import torch

        # Load the model; config.txt holds the Ultralytics metadata (class names, imgsz, task) as JSON
        extra = {"config.txt": ""}
        model = torch.jit.load("yolo26n.torchscript", map_location="cpu", _extra_files=extra)
        names = json.loads(extra["config.txt"])["names"]

        # Letterbox the image to 640x640 with gray padding, as Ultralytics does
        img = cv2.imread("bus.jpg")
        h, w = img.shape[:2]
        r = min(640 / h, 640 / w)
        nw, nh = round(w * r), round(h * r)
        left, top = (640 - nw) // 2, (640 - nh) // 2
        canvas = np.full((640, 640, 3), 114, np.uint8)
        canvas[top : top + nh, left : left + nw] = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
        x = torch.from_numpy(canvas[..., ::-1].copy()).permute(2, 0, 1)[None].float() / 255  # BGR to RGB, NCHW, 0-1

        # An export with nms=False outputs (1, 300, 6): x1, y1, x2, y2, score, class index
        with torch.no_grad():
            y = model(x)[0]
        for x1, y1, x2, y2, score, cls in y[y[:, 4] > 0.25].tolist():
            box = np.clip([(x1 - left) / r, (y1 - top) / r, (x2 - left) / r, (y2 - top) / r], 0, [w, h, w, h]).round()
            print(names[str(int(cls))], round(score, 2), box.astype(int).tolist())
        ```

To build the C++ example, download LibTorch from the [PyTorch website](https://pytorch.org/get-started/locally/), put `main.cpp` and `CMakeLists.txt` in one directory, and run `cmake -S . -B build -DCMAKE_PREFIX_PATH="/path/to/libtorch;/path/to/opencv"`, `cmake --build build`, and `./build/yolo26 yolo26n.torchscript bus.jpg`. For a CUDA GPU, use a CUDA build of LibTorch, load the model with `torch::kCUDA`, and move the input with `x.to(torch::kCUDA)`. With LibTorch 2.15.0 for CUDA 13.2 on the RTX PRO 6000, the C++ example printed the same 5 objects and boxes, for exports from the CPU and from the GPU, after two fixes:

- **CUDA toolkit**: CMake needs a CUDA toolkit with `nvcc` (12.6 or newer; 12.8 was used). Below 12.6 it stops with "PyTorch requires CUDA 12.6 or above.", and with no toolkit with "Your installed Caffe2 version uses CUDA but I cannot find the CUDA libraries. Please set the proper CUDA prefixes and / or install CUDA."
- **cuDNN**: the LibTorch archive ships only `libcudnn.so.9`, and the example stopped with `Invalid handle. Cannot load symbol cudnnGetVersion` until the other cuDNN 9 libraries (here from the `nvidia-cudnn-cu13` 9.26.0.51 wheel) were on `LD_LIBRARY_PATH`.

## Deploy

`YOLO("yolo26n.torchscript")` runs the model with PyTorch, on the CPU or a CUDA GPU:

- **`device`**: the model runs on the device passed to `predict` or `val`, whichever device it was exported on. An export from the CPU ran on a CUDA GPU with `device=0`, and FP32 and FP16 exports from a CUDA GPU ran with `device="cpu"`, with the same detections on `bus.jpg` and the same mAP50-95 in `val` on `coco8.yaml`, also in a process with no visible GPU. On the CPU, Ultralytics computes an FP16 export in FP32. On Apple silicon, exports from `device="mps"` ran on the CPU, but `device="mps"` fails for YOLO26, see [Limitations and Troubleshooting](#limitations-and-troubleshooting).
- **Fixed input shape**: unless the model was exported with `dynamic=True`, it runs at its export `imgsz`, which Ultralytics reads from the metadata, and an `imgsz` passed to `predict` or `val` is ignored: YOLO26n exported at 640 gave the same detections with `imgsz=320`, `1280`, and `480x640`. With `nms=None` or `nms=False`, the batch size is not fixed: models exported with `batch=2` and `batch=4` reached the same accuracy in batched `val` on `coco8.yaml` as with `batch=1`. An `nms=True` export runs at its export `batch`: Ultralytics splits or pads the input to that size, and your own code must pass exactly that many images.
- **`dynamic=True`**: the model runs at the `imgsz` passed to `predict` or `val`, and YOLO26n gave the same detections as the PyTorch model at 320, 1280, and 480x640; `val` ran with `batch=4` at `imgsz=480`.
- **File name**: Ultralytics recognizes the format by `.torchscript` in the file name and reads the metadata only from a file whose name ends in `.torchscript`, see [Limitations and Troubleshooting](#limitations-and-troubleshooting).
- **Several models**: a detection and a segmentation model ran alternately in one process.

### NVIDIA Triton Inference Server

The [PyTorch backend of Triton](https://github.com/triton-inference-server/pytorch_backend) runs TorchScript models "using the PyTorch C++ API", with the platform `pytorch_libtorch`, which its README calls the "Legacy LibTorch framework". It loads `model.pt` from the version directory of the model unless `default_model_filename` in `config.pbtxt` names another file, so Triton can read the Ultralytics export from a read-only mount of its directory, without renaming or copying `yolo26n.torchscript`. This `config.pbtxt` serves an FP32 YOLO26n detection export at 640 with any value of `nms`:

```protobuf
platform: "pytorch_libtorch"
default_model_filename: "yolo26n.torchscript"
max_batch_size: 0
input [{ name: "input__0", data_type: TYPE_FP32, dims: [1, 3, 640, 640] }]
output [{ name: "output__0", data_type: TYPE_FP32, dims: [-1, -1, -1] }]
```

Save it as `model_repository/yolo26n/config.pbtxt`, export Ultralytics YOLO26 into its own directory, and start Triton on the GPU with that directory mounted as version `1`. The empty `model_repository/yolo26n/1` directory must exist, because Docker cannot create a mount point inside the read-only repository:

```bash
mkdir -p yolo26n-export model_repository/yolo26n/1
cd yolo26n-export && yolo export model=yolo26n.pt format=torchscript && cd ..
docker run -d --rm --name yolo26n-triton --gpus all -p 8000:8000 \
  -v "$PWD/model_repository:/models:ro" -v "$PWD/yolo26n-export:/models/yolo26n/1:ro" \
  nvcr.io/nvidia/tritonserver:26.11-py3 \
  tritonserver --model-repository=/models --model-control-mode=explicit --load-model=yolo26n
```

Triton serves the model over HTTP at `localhost:8000/v2/models/yolo26n/infer`, with the input `input__0` that the [Python example](#run-with-libtorch) prepares as `x`. Triton keeps the model it has loaded: after a new export into `yolo26n-export`, load it again with `curl -X POST localhost:8000/v2/repository/models/yolo26n/load`. That request needs `--model-control-mode=explicit`; without it, Triton answers `explicit model load / unload is not allowed if polling is enabled`. For this page, Triton 26.11 (server 2.75.0) ran these commands on one RTX PRO 6000, with the instance on the GPU (`KIND_GPU`) by default. The export with `nms=None` returned `(1, 84, 8400)`. After a re-export with `nms=False` into `yolo26n-export`, Triton kept returning `(1, 84, 8400)` until the load request; after it, and after a further re-export with `nms=True` and load request, Triton returned `(1, 300, 6)` with the same 5 objects and boxes on `bus.jpg` as `YOLO()` on the new file, with scores within 0.01. To serve Ultralytics models from Triton with ONNX, see the [Triton guide](../guides/triton-inference-server.md).

### Deployment Guides

- **[Loading a TorchScript Model in C++](https://docs.pytorch.org/tutorials/advanced/cpp_export.html)**: the PyTorch tutorial for `torch::jit::load` and CMake, which PyTorch marks with "TorchScript is deprecated, please use torch.export instead."
- **[PyTorch C++ API](https://docs.pytorch.org/cppdocs/)**: the LibTorch reference.
- **[ExecuTorch](executorch.md)**: the Ultralytics export for the current PyTorch route to phones and embedded devices, with its own `torch.export` → `.pte` pipeline; see also the [ExecuTorch documentation](https://docs.pytorch.org/executorch/).
- **[Model deployment options](../guides/model-deployment-options.md)**: compare TorchScript with the other export formats.

## Limitations and Troubleshooting

- **FP16 export on the CPU**: `quantize=16` without a GPU fails with `FP16 TorchScript export is only supported on GPU, i.e. use device=0.` Export on a CUDA GPU with `device=0` or on Apple silicon with `device="mps"`; the FP16 file then also runs on a CPU through `YOLO()`, in FP32.
- **INT8**: `quantize=8` fails with `quantize=8 (INT8) is not supported; format='torchscript' supports quantize=16 (FP16), 32 or None (FP32).` For INT8 models, export to [ONNX](onnx.md), [OpenVINO](openvino.md), or [TensorRT](tensorrt.md).
- **Apple GPUs (`device="mps"`)**: `YOLO("yolo26n.torchscript")` loads, but its first prediction with `device="mps"` fails with `TypeError: Cannot convert a MPS Tensor to float64 dtype as the MPS framework doesn't support float64. Please use float32 instead.` This happened with PyTorch 2.14.0 on macOS for YOLO26 and YOLO11 models of every export variant run, exported on the CPU or with `device="mps"`, while a YOLOv8n export ran on `mps`. Run the model on the CPU of the Mac, or export to [Core ML](coreml.md) for the Apple GPU and Neural Engine.
- **PyTorch 2.16 pre-release**: PyTorch 2.16.0.dev20261126 no longer has `torch.jit.load` and `torch.jit.trace`, and the [Python example](#run-with-libtorch) fails there with `AttributeError: module 'torch.jit' has no attribute 'load'`. The Ultralytics export, `YOLO("yolo26n.torchscript")`, and its prediction ran with this version.
- **Deprecation warnings**: PyTorch 2.14 and newer log ``FutureWarning: `torch.jit.trace` is deprecated. Please switch to `torch.compile` or `torch.export`.``, ``FutureWarning: `torch.jit.trace_method` is deprecated. Please switch to `torch.compile` or `torch.export`.``, and ``FutureWarning: `torch.jit.load` is deprecated. Please switch to `torch.export`.``, and on Python 3.14 ``FutureWarning: `torch.jit.trace` is not supported in Python 3.14+ and may break. Please switch to `torch.compile` or `torch.export`.`` and ``FutureWarning: `torch.jit.load` is not supported in Python 3.14+ and may break. Please switch to `torch.export`.`` instead. The exports, predictions, and validations for this page that logged them completed with the same results as on Python 3.8 and PyTorch 1.8.0.
- **`val` with embedded NMS**: an `nms=True` export keeps only the boxes above the `conf` of the export, 0.25 by default, so `val`, which counts boxes down to a score of 0.001 (0.01 for OBB), reports a lower mAP: on the x86-64 export host, YOLO26n reached 50.3 mAP50-95 on `coco8.yaml` with `nms=True` against 67.5 with `nms=None`. For validation, export with `nms=True conf=0.001` (66.9), and deploy the export with the confidence you need.
- **Embedded NMS for classification**: `nms=True` fails with `'nms=True' is not valid for classification models.`
- **File name**: keep `.torchscript` at the end of the name, for example `yolo26n_640.torchscript`. A copy named `model.torchscript.bak` loaded with the class names `class0`, `class1`, and so on; `yolo26n.ts` and `detector.bin` failed with `is not a supported model format`; and `detector.pt` failed with `detector.pt is a TorchScript archive, not an Ultralytics PyTorch checkpoint.`
- **Output file**: every export of one checkpoint writes the same `yolo26n.torchscript`, whatever its `imgsz`, `quantize`, or `nms`, and replaces an existing file. Rename the file before you export a second variant.
- **Checkpoints trained with `quantize=8`**: a checkpoint fine-tuned with [quantization-aware training](../modes/export.md#quantization-aware-training) fails with `format='torchscript' cannot export a QAT model: its Q/DQ ranges are only read by the 'onnx' and 'engine' backends. Export a non-QAT checkpoint to this format instead.` Export it to [ONNX](onnx.md) or [TensorRT](tensorrt.md) instead.
- **Not run for this page**: exports on Windows and ARM64 Linux.

For other errors, see the [Common Issues guide](../guides/yolo-common-issues.md).

## Benchmarks

The tables below compare PyTorch, the ONNX export, and TorchScript exports of every YOLO26 detection model on the export host: on its CPU, an **AMD EPYC 9655** server shared with other jobs, and on one **NVIDIA RTX PRO 6000** GPU. Every process was limited to the 8 CPUs of one 8-core CCD of the processor with `taskset -c 40-47`, where PyTorch and TorchScript ran with the 8 threads that Ultralytics sets.

=== "CPU latency"

    Milliseconds per image on the CPU, lower is better, as the mean of two runs. The fastest format for each model is in bold.

    | Model   | `imgsz` | PyTorch  | ONNX FP32 | TorchScript FP32 |
    | ------- | ------- | -------- | --------- | ---------------- |
    | YOLO26n | 640     | **9.9**  | 16.6      | 17.3             |
    | YOLO26s | 640     | **25.7** | 28.4      | 37.0             |
    | YOLO26m | 640     | 59.4     | **58.4**  | 78.7             |
    | YOLO26l | 640     | 75.4     | **68.0**  | 97.9             |
    | YOLO26x | 640     | 144.8    | **132.3** | 191.6            |

=== "GPU latency"

    Milliseconds per image on one RTX PRO 6000, lower is better, as the mean of two runs. FP16 ran with `quantize=16` passed to `val`. The fastest format for each model is in bold.

    | Model   | `imgsz` | PyTorch FP32 | PyTorch FP16 | TorchScript FP32 | TorchScript FP16 |
    | ------- | ------- | ------------ | ------------ | ---------------- | ---------------- |
    | YOLO26n | 640     | 2.6          | 2.8          | **1.9**          | 2.1              |
    | YOLO26s | 640     | 2.7          | 2.9          | 2.2              | **2.1**          |
    | YOLO26m | 640     | 3.6          | 3.4          | 3.3              | **2.3**          |
    | YOLO26l | 640     | 4.5          | 4.5          | 4.5              | **3.2**          |
    | YOLO26x | 640     | 6.5          | 4.5          | 6.5              | **3.9**          |

=== "Accuracy"

    mAP<sup>val</sup> 50-95 on 500 images of COCO val2017, higher is better. FP32 TorchScript reached the same values on the GPU, within 0.1.

    | Model   | PyTorch | ONNX FP32 | TorchScript FP32 | PyTorch FP16 (GPU) | TorchScript FP16 (GPU) |
    | ------- | ------- | --------- | ---------------- | ------------------ | ---------------------- |
    | YOLO26n | 42.1    | 42.1      | 42.1             | 42.1               | 42.2                   |
    | YOLO26s | 49.5    | 49.5      | 49.5             | 49.4               | 49.5                   |
    | YOLO26m | 53.6    | 53.6      | 53.6             | 53.5               | 53.6                   |
    | YOLO26l | 55.5    | 55.5      | 55.5             | 55.5               | 55.5                   |
    | YOLO26x | 55.7    | 55.7      | 55.7             | 55.7               | 55.7                   |

=== "Size"

    Size of the `.pt` checkpoint, the ONNX file, and the `.torchscript` file in MB (10^6 bytes). The smallest file for each model is in bold.

    | Model   | PyTorch | ONNX FP32 | TorchScript FP32 | TorchScript FP16 |
    | ------- | ------- | --------- | ---------------- | ---------------- |
    | YOLO26n | 5.5     | 9.9       | 10.2             | **5.4**          |
    | YOLO26s | 20.4    | 38.3      | 38.6             | **19.6**         |
    | YOLO26m | 44.3    | 82.0      | 82.3             | **41.4**         |
    | YOLO26l | 53.2    | 99.6      | 100.2            | **50.5**         |
    | YOLO26x | 118.7   | 223.3     | 223.8            | **112.3**        |

- **TorchScript FP32** reproduced the accuracy of PyTorch for all five YOLO26 detection models. On the CPU, it took 1.3 to 1.7 times as long as PyTorch, which Ultralytics runs in the channels-last memory format on x86 CPUs under Linux and Windows, and 1.0 to 1.1 times as long as PyTorch with `channels_last=False`; against ONNX, it took 1.0 to 1.4 times as long.
- **FP16 on the GPU**: the FP16 TorchScript export took 0.60 to 0.95 times the time of the FP32 export for YOLO26s, YOLO26m, YOLO26l, and YOLO26x, and 1.11 times for YOLO26n; against PyTorch FP16, it took 0.7 to 0.9 times as long. Its file is half the size of FP32, and the accuracy changed by 0.0 to 0.1 mAP.
- **All CPUs**: without `taskset`, on all 384 logical CPUs, YOLO26n took 27.9 ms in TorchScript FP32, against 13.9 ms in PyTorch and 12.5 ms in ONNX, with the same 8 threads, as the mean of two runs.

Measured with `ultralytics` 8.4.219, PyTorch 2.15.0 (CUDA 13.2 build), and ONNX Runtime 1.32.0 at `batch=1`, on the NVIDIA RTX PRO 6000 Blackwell Server Edition. PyTorch ran with `rect=False`, at the same 640x640 input as the exported models. Accuracy is on the last 500 images of COCO val2017 in sorted order, so it differs from the values for the full validation set on the [YOLO26 model page](../models/yolo26.md). Latency is the inference stage that `val` reports and does not include preprocessing or postprocessing; the two runs differed by up to 20%.

## About TorchScript

![TorchScript model serialization and deployment workflow overview](https://cdn.ul.run/i/ad24d0edc6ca8ab83e4e358f91fe1f02.avif)

TorchScript turns a PyTorch model into a program that runs without Python, either by tracing, which records the operations of one run on an example input, or by scripting, which compiles a subset of Python. Ultralytics exports by tracing. PyTorch has deprecated TorchScript in favor of `torch.export`, see the [TorchScript documentation](https://docs.pytorch.org/docs/stable/jit.html); Ultralytics keeps the export for LibTorch applications that load `.torchscript` files.

![TorchScript scripting vs tracing comparison](https://cdn.ul.run/i/7b474752918355e482355c0e00d038ff.avif)

## FAQ

### Should I use TorchScript, ONNX, or ExecuTorch for a new Ultralytics YOLO26 deployment?

Ultralytics keeps TorchScript export for applications that already load `.torchscript` files with LibTorch, because PyTorch has deprecated TorchScript and its 2.16 pre-release no longer has `torch.jit.load` in Python. For a new C++ or server deployment, [ONNX](onnx.md) runs in ONNX Runtime on CPUs and GPUs, and [TensorRT](tensorrt.md) gives the fastest inference on NVIDIA GPUs; for phones and embedded boards, PyTorch recommends [ExecuTorch](executorch.md), which Ultralytics exports with `format="executorch"`.

### Does an Ultralytics TorchScript model run on a GPU?

Yes: an Ultralytics TorchScript export runs on NVIDIA GPUs with `device=0` in `YOLO()`, and a file exported on the CPU ran on a CUDA GPU without a new export; the C++ example ran on a CUDA GPU with LibTorch as well, see [Run with LibTorch](#run-with-libtorch). For FP16 on the GPU, export with `quantize=16 device=0` and predict with `quantize=16`, see [Precision](#precision). On Apple GPUs, `device="mps"` fails for YOLO26 and YOLO11 models, see [Limitations and Troubleshooting](#limitations-and-troubleshooting).

### Why does `val` report a lower mAP for an Ultralytics TorchScript export with `nms=True`?

An Ultralytics TorchScript export with `nms=True` drops every box below the `conf` of the export, 0.25 by default, before `val` sees it, so the mAP of `val` is lower than for the raw output: 50.3 against 67.5 mAP50-95 for YOLO26n on `coco8.yaml` on the x86-64 export host. Export with `conf=0.001` to validate (66.9), or validate the `nms=None` or `nms=False` export, and keep a deployment threshold for the model you ship.

### Why is an Ultralytics TorchScript model slower than the PyTorch model on a CPU?

On the x86 server CPU used for this page, an Ultralytics TorchScript export of YOLO26 took 1.3 to 1.7 times as long as the PyTorch model through `YOLO()`, see [Benchmarks](#benchmarks). Part of the gap is the memory format: Ultralytics runs PyTorch models in channels-last format on x86 CPUs under Linux and Windows, which it does not apply to TorchScript files, and with `channels_last=False`, TorchScript took 1.0 to 1.1 times as long as PyTorch. For CPU inference, [ONNX](onnx.md) and [OpenVINO](openvino.md) exports are usually the faster choice; on NVIDIA GPUs, see [TensorRT](tensorrt.md).
