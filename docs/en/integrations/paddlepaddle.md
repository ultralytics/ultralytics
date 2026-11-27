---
comments: true
description: Export Ultralytics YOLO26 models to Baidu PaddlePaddle with format=paddle for Paddle Inference on CPUs and NVIDIA GPUs. Covers installation, output files and shapes, CPU threads, limitations, and benchmarks.
keywords: YOLO26, PaddlePaddle, Paddle export, Paddle Inference, X2Paddle, pdiparams, PIR, Baidu, model deployment, CPU inference, paddlepaddle-gpu
---

# Baidu PaddlePaddle Export

[PaddlePaddle](https://www.paddlepaddle.org.cn/en) is Baidu's open-source [deep learning](https://www.ultralytics.com/glossary/deep-learning-dl) framework, and Paddle Inference is its native library for running trained models. Exporting an [Ultralytics YOLO26](../models/yolo26.md) model with `format="paddle"` converts it with Baidu's [X2Paddle](https://github.com/PaddlePaddle/X2Paddle) tool and creates a `yolo26n_paddle_model/` directory that holds an FP32 Paddle Inference model.

## At a Glance

| Property          | Details                                                                                                                                                                                                                                                   |
| ----------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `format` argument | `paddle`                                                                                                                                                                                                                                                  |
| Output            | `yolo26n_paddle_model/`, a directory with the model in `inference_model/`, the Python code that X2Paddle generated, and [`metadata.yaml`](#output-format)                                                                                                 |
| Runs on           | CPUs through Paddle Inference, and NVIDIA GPUs with the [`paddlepaddle-gpu` build](#nvidia-gpus), tested on an RTX PRO 6000 Blackwell                                                                                                                     |
| Runtime           | Paddle Inference: the `paddle.inference` module of the `paddlepaddle` package, or the Paddle Inference C++ library                                                                                                                                        |
| Precision         | FP32 only. YOLO26n is 10.0 MB                                                                                                                                                                                                                             |
| Export host       | Linux x86-64 with Python 3.8 to 3.13 and macOS on Apple silicon with Python 3.13, where the exports for this page ran; ARM64 Linux and Windows have `paddlepaddle` packages but were not run, and NVIDIA Jetson is not supported. The export needs no GPU |

!!! tip "Export in the cloud"

    [Ultralytics Platform](../platform/train/models.md#export-model) can export a trained model to PaddlePaddle, so you do not need a local export environment.

## Quickstart

Install the Ultralytics package in a virtual environment, here on Linux or macOS:

```bash
python3 -m venv paddle-env && source paddle-env/bin/activate
pip install ultralytics
```

The first export installs `paddlepaddle`, `x2paddle`, `six`, and the packages they require, such as `protobuf`: 6 to 12 packages in the environments used for this page, none of which replaced an installed package. The export logs `Restart runtime or rerun command for updates to take effect` and still completes in the same run.

To install them before the first export, run `pip install ultralytics "paddlepaddle>=3.0.0,<3.3.0" x2paddle six` instead, and on ARM64, including Apple silicon, `paddlepaddle==3.0.0`, the only version the export accepts there; the export then installs nothing. The export has these version requirements:

| Requirement      | Version                                                                                                                                                                                                                                                                                                                                                                                                                                                      |
| ---------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| Python           | 3.8 to 3.13, the versions with `paddlepaddle` packages. Exports ran with every version from 3.8 to 3.13. On Python 3.8, pip installs `paddlepaddle` 3.0.0, the last version for it                                                                                                                                                                                                                                                                           |
| `paddlepaddle`   | 3.0.0 or newer and older than 3.3.0, because Paddle Inference 3.3 [fails with oneDNN on CPUs](https://github.com/PaddlePaddle/Paddle/issues/77340). The first export installs the newest version in that range (3.0.0 on Python 3.8) and replaces a version outside it: an installed 3.3.1 was replaced by 3.2.2. On ARM64, including Apple silicon, the export requires exactly 3.0.0 and replaces any other version. Exports also ran with 3.0.0 and 3.1.0 |
| `x2paddle`       | No version requirement. All exports for this page used 1.6.0. `six` is installed with it, because X2Paddle imports it without declaring it                                                                                                                                                                                                                                                                                                                   |
| Operating system | `paddlepaddle` 3.1 to 3.2 has packages for Linux x86-64 and ARM64, macOS on Apple silicon, and Windows x64; 3.0.0 also for macOS on Intel. Exports for this page ran on Ubuntu 24.04 x86-64 and on macOS 27 with an Apple M4                                                                                                                                                                                                                                 |

For other installation methods, see the [installation guide](../quickstart.md). Then export your model and load the exported directory to run inference or validate its accuracy:

!!! example "Export"

    === "Python"

        ```python
        from ultralytics import YOLO

        # Load a YOLO26 model
        model = YOLO("yolo26n.pt")

        # Export the model to PaddlePaddle format
        model.export(format="paddle")  # creates 'yolo26n_paddle_model/'
        ```

    === "CLI"

        ```bash
        # Export a YOLO26n PyTorch model to PaddlePaddle format
        yolo export model=yolo26n.pt format=paddle # creates 'yolo26n_paddle_model/'
        ```

On a shared AMD EPYC 9655 server, exports of YOLO26n took 2 to 3 s with the packages installed, and 8 to 18 s for the first export in a new environment, which also installs them. Exports of YOLO26x took 6 to 7 s. The exported model runs on the export host, so you can test it there:

!!! example "Predict"

    === "Python"

        ```python
        from ultralytics import YOLO

        # Load the exported PaddlePaddle model
        model = YOLO("yolo26n_paddle_model")

        # Run inference
        results = model("https://ultralytics.com/images/bus.jpg")
        ```

    === "CLI"

        ```bash
        # Run inference with the exported PaddlePaddle model
        yolo predict model=yolo26n_paddle_model source='https://ultralytics.com/images/bus.jpg'
        ```

!!! example "Validate"

    === "Python"

        ```python
        from ultralytics import YOLO

        # Load the exported PaddlePaddle model
        model = YOLO("yolo26n_paddle_model")

        # Validate accuracy on the COCO8 dataset
        metrics = model.val(data="coco8.yaml")
        ```

    === "CLI"

        ```bash
        # Validate the exported PaddlePaddle model
        yolo val model=yolo26n_paddle_model data=coco8.yaml
        ```

## Supported Tasks

PaddlePaddle export supports all seven Ultralytics tasks: [detection](../tasks/detect.md), [segmentation](../tasks/segment.md), [semantic segmentation](../tasks/semantic.md), [depth estimation](../tasks/depth.md), [classification](../tasks/classify.md), [pose estimation](../tasks/pose.md), and [OBB](../tasks/obb.md). Semantic segmentation and depth estimation are available only with [YOLO26](../models/yolo26.md), the only family that ships those heads. For this page, the YOLO26n checkpoint of every task was exported and run on the export host, and so were [YOLO11](../models/yolo11.md), [YOLOv10](../models/yolov10.md), [YOLOv8](../models/yolov8.md), and [YOLOv5](../models/yolov5.md) detection models. [RT-DETR](../models/rtdetr.md) does not convert, see [Limitations and Troubleshooting](#limitations-and-troubleshooting).

## Export Arguments

| Argument   | Type             | Default    | Description                                                                                                                                                                                                                                                                                                                                                                                                                              |
| ---------- | ---------------- | ---------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `format`   | `str`            | `'paddle'` | Target format for the exported model, defining compatibility with various deployment environments.                                                                                                                                                                                                                                                                                                                                       |
| `imgsz`    | `int` or `tuple` | `640`      | Image size of the model input, fixed in the exported model: an integer for square images or a tuple `(height, width)`. When not passed, the export uses the size the checkpoint was trained at. The official YOLO26 checkpoints use `640` for detect and segment, `640` for semantic models pretrained on ADE20K and `1024` for those pretrained on Cityscapes, `768` for depth, `224` for classify, `640` for pose, and `1024` for OBB. |
| `batch`    | `int`            | `1`        | Batch size of the exported model, fixed in the input shape.                                                                                                                                                                                                                                                                                                                                                                              |
| `quantize` | `int` or `str`   | `None`     | Precision: only `32` or unset (FP32). Other values raise an error, see [Precision](#precision). Replaces the deprecated `half`/`int8` flags.                                                                                                                                                                                                                                                                                             |
| `nms`      | `bool`, optional | `None`     | Detection, segmentation, pose, and OBB models always export the raw output that needs NMS: `True` (embedded NMS) and `False` (the NMS-free head) are not available, see [Output Format](#output-format).                                                                                                                                                                                                                                 |
| `device`   | `str`            | `None`     | Device that traces the model during the export: `cpu` (default), `0` for the first CUDA GPU, or `mps` on Apple silicon. With `mps`, the exported files had the same sizes and gave the same detections on one image as with `cpu`; tracing on a CUDA GPU was not run.                                                                                                                                                                    |

For the arguments shared by all formats, see the [Export mode documentation](../modes/export.md).

### Precision

The export writes FP32 weights and computes in FP32; it has no FP16 or INT8 option:

| `quantize`    | Result                                                                                                 |
| ------------- | ------------------------------------------------------------------------------------------------------ |
| unset or `32` | FP32 model, 10.0 MB for YOLO26n                                                                        |
| `16`          | Fails with `quantize=16 (FP16) is not supported; format='paddle' supports quantize=32 or None (FP32).` |
| `8`           | Fails with `quantize=8 (INT8) is not supported; format='paddle' supports quantize=32 or None (FP32).`  |
| `'w8a16'`     | Fails with `quantize='w8a16' is not supported; format='paddle' supports quantize=32 or None (FP32).`   |
| `'w8a32'`     | Fails with `quantize='w8a32' is not supported; format='paddle' supports quantize=32 or None (FP32).`   |

The deprecated flags `half=True` and `int8=True` fail with the same messages as `quantize=16` and `quantize=8`.

## Output Format

The export writes one directory:

| File                              | Contents                                                                                                                 | YOLO26n |
| --------------------------------- | ------------------------------------------------------------------------------------------------------------------------ | ------- |
| `inference_model/model.json`      | The network as a Paddle program (PIR) in JSON, which Paddle Inference loads                                              | 0.3 MB  |
| `inference_model/model.pdiparams` | The FP32 weights of that program                                                                                         | 9.7 MB  |
| `x2paddle_code.py`                | The network as PaddlePaddle Python code, generated by X2Paddle                                                           | 29 kB   |
| `model.pdparams`                  | The weights for `x2paddle_code.py`                                                                                       | 9.7 MB  |
| `__pycache__/`                    | Python bytecode of `x2paddle_code.py`, written while the export runs it                                                  | 44 kB   |
| `metadata.yaml`                   | Class names, task, image size, batch size, stride, and export arguments, which Ultralytics reads when it loads the model | 1.5 kB  |

Inference needs only `inference_model/` and `metadata.yaml`, 10.0 MB for YOLO26n: a copy of the directory with only these files ran the same detections. The other files hold the same weights a second time, so the whole directory is about twice as large, and the export log prints its size in MiB, as `(18.9 MB)` for YOLO26n. The sizes of the model files, `model.json` and `model.pdiparams`, for the official YOLO26n checkpoints at their default export size, with the Cityscapes checkpoint `yolo26n-sem.pt` for semantic segmentation, in MB (10^6 bytes):

| Task     | Default `imgsz` | Model files, MB |
| -------- | --------------- | --------------- |
| Detect   | `640`           | 10.0            |
| Segment  | `640`           | 11.3            |
| Semantic | `1024`          | 6.4             |
| Depth    | `768`           | 20.9            |
| Classify | `224`           | 11.3            |
| Pose     | `640`           | 12.2            |
| OBB      | `1024`          | 10.4            |

The model takes one input with shape `(batch, 3, height, width)`: RGB, channels first (NCHW), `float32`, scaled to 0-1. Ultralytics resizes each image to that size with gray padding (letterbox; classification models are resized and center-cropped instead), and the box coordinates in the output refer to the padded image. The task sets the outputs, which are `float32`. The shapes below are for the official YOLO26n checkpoints at their default export size, with the Cityscapes checkpoint `yolo26n-sem.pt` for semantic segmentation:

| Task     | Default `imgsz` | Outputs                                                             |
| -------- | --------------- | ------------------------------------------------------------------- |
| Detect   | `640`           | `(1, 84, 8400)`                                                     |
| Segment  | `640`           | `(1, 116, 8400)` and `(1, 32, 160, 160)`                            |
| Semantic | `1024`          | `(1, 19, 1024, 1024)`, a score for each class and pixel             |
| Depth    | `768`           | `(1, 1, 768, 768)`, depth map                                       |
| Classify | `224`           | `(1, 1000)`, class probabilities                                    |
| Pose     | `640`           | `(1, 56, 8400)`: 4 box values, 1 class score, then 17 keypoints x 3 |
| OBB      | `1024`          | `(1, 20, 21504)`: 4 box values, 15 class scores, then the angle     |

- **Raw output**: detection, segmentation, pose, and OBB models write shape `(batch, values, anchors)`. For each anchor: the box center x, center y, width, and height in input pixels, then the class scores, then the values of the task (32 mask coefficients, 17 x 3 keypoint values, or the OBB angle in radians). Apply NMS to this output yourself. See the [ONNX output format](onnx.md#output-format) for the layout of each task.
- **`nms`**: the output is the same for every value of `nms`. `nms=False` logs `This export format does not support end2end models, disabling the end2end branch.` for models with an NMS-free head, such as YOLO26, and `nms=True` logs `format=paddle does not support embedded NMS; using nms=None.`
- **Segmentation**: the 32 mask coefficients are the last 32 values of each anchor, and the second output holds the mask prototypes.
- **Semantic segmentation**: the output holds a score for each class and pixel, not the class map of the [ONNX export](onnx.md#output-format), so take the argmax over the class axis in your own code.

The table below compares the raw outputs with those of the [ONNX](onnx.md) export of the same checkpoint on one image, `bus.jpg`, as the largest absolute difference of any output value, where box values are in pixels:

| Task     | Largest difference from ONNX |
| -------- | ---------------------------- |
| Detect   | 0.0031                       |
| Segment  | 0.0023                       |
| Semantic | Same class on every pixel    |
| Depth    | 0.0001                       |
| Classify | below 0.0001                 |
| Pose     | 0.0024                       |
| OBB      | 0.0042                       |

These are the outputs as `YOLO()` runs the model. With the default settings of Paddle Inference on x86-64 CPUs, the depth and OBB outputs differ by up to 6.2 and 138.9, see [Limitations and Troubleshooting](#limitations-and-troubleshooting).

`YOLO("yolo26n_paddle_model")` reads the layout from `metadata.yaml`, so you only need these details when you run the model in your own code. The keys of `metadata.yaml` include `names`, `task`, `imgsz`, `stride`, `batch`, `end2end`, `args`, and, for pose models, `kpt_shape`.

### Run with Paddle Inference

To run the model outside the Ultralytics package, load it with the `paddle.inference` module of the `paddlepaddle` package:

```python
# Export first: yolo export model=yolo26n.pt format=paddle
import numpy as np
import paddle.inference as pdi

config = pdi.Config(
    "yolo26n_paddle_model/inference_model/model.json", "yolo26n_paddle_model/inference_model/model.pdiparams"
)
config.set_cpu_math_library_num_threads(8)  # Paddle Inference uses one CPU thread by default
predictor = pdi.create_predictor(config)

image = np.zeros((1, 3, 640, 640), dtype=np.float32)  # your image, letterboxed to 640x640: RGB, NCHW, float32, 0-1
predictor.get_input_handle(predictor.get_input_names()[0]).copy_from_cpu(image)
predictor.run()
output = predictor.get_output_handle(predictor.get_output_names()[0]).copy_to_cpu()  # raw predictions
assert output.shape == (1, 84, 8400)
```

The class names are in `yolo26n_paddle_model/metadata.yaml`. On x86-64 CPUs, also call `config.delete_pass("inplace_pass")` for OBB models and `config.delete_pass("onednn_placement_pass")` for depth models before `create_predictor`, as `YOLO()` does, see [Limitations and Troubleshooting](#limitations-and-troubleshooting). Paddle Inference also has a C++ API with the same `Config` and predictor calls, see the [Paddle Inference documentation](https://www.paddlepaddle.org.cn/inference/master/guides/introduction/index_intro.html).

## Deploy

`YOLO("yolo26n_paddle_model")` loads `inference_model/model.json` and `model.pdiparams` with Paddle Inference, which Ultralytics installs on first use if it is missing:

- **CPU threads**: On x86-64 CPUs, `YOLO()` runs the model on up to 8 threads. Paddle Inference computes on one thread unless the application sets another count, and Ultralytics sets none; but with oneDNN, which the default settings of Paddle Inference turn on on x86-64 CPUs, it followed the PyTorch thread count, which Ultralytics sets to 8, or to one less than the number of CPUs on smaller machines, and on Linux to no more than the CPUs the process may use, before it loads the model. To choose the count yourself, run the model with [Paddle Inference](#run-with-paddle-inference) and call `config.set_cpu_math_library_num_threads()`. On the AMD EPYC 9655 server, limited to 8 CPUs with `taskset`, one raw inference of YOLO26n took 52 to 54 ms on one thread and 29 ms on 8, whether set in the `Config` or through `torch.set_num_threads(8)`; YOLO26x took 1012 to 1043 ms and 276 to 297 ms. With oneDNN turned off, only the `Config` setting counts: YOLO26n took 158 to 159 ms on one thread and 66 to 67 ms on 8. For depth models, `YOLO()` runs only the convolutions with oneDNN, so the other layers run on one thread, see [Limitations and Troubleshooting](#limitations-and-troubleshooting).
- **Fixed input shape**: the model always runs at its export `imgsz` and `batch`, which Ultralytics reads from `metadata.yaml`. An `imgsz` passed to `predict` or `val` is ignored; export again to change it. Smaller batches are padded to the export batch size, and by the Ultralytics code larger ones are split into chunks of it, so any number of images works.
- **`device`**: without a CUDA GPU the model runs on the CPU. On Apple silicon, Paddle Inference runs on the CPU too; `device="mps"` is accepted and gave the same detections on one image as `device="cpu"`. For GPUs, see [NVIDIA GPUs](#nvidia-gpus).
- **Directory name**: pass the directory and keep `_paddle_model` in its name, because Ultralytics recognizes the format by it, see [Limitations and Troubleshooting](#limitations-and-troubleshooting).
- **Files to copy**: `inference_model/` and `metadata.yaml` are enough, see [Output Format](#output-format). Without `metadata.yaml`, a YOLO26n detection model exported at 640 still loaded, with the class names `class0`, `class1`, and so on; other tasks and sizes were not tried.
- **Several models in one process**: a detection and a segmentation model loaded in one process ran alternately without errors.

### NVIDIA GPUs

Paddle Inference runs on NVIDIA GPUs with the `paddlepaddle-gpu` build, which you install instead of `paddlepaddle`, as both provide the same `paddle` module. Its 3.x versions are published on the [PaddlePaddle package index](https://www.paddlepaddle.org.cn/en/install/quick), not on PyPI, which has only 2.x `paddlepaddle-gpu` builds, below the 3.0.0 that Ultralytics requires; install it with the command from that page for your CUDA version. By the Ultralytics code:

- **Installation**: when `paddlepaddle-gpu` is installed, Ultralytics checks that it is 3.0.0 or newer, stops with an error if it is older, and never replaces it with the CPU build.
- **Inference**: When PyTorch sees CUDA, `device=0` requests GPU 0 from Paddle Inference with a 2,048 MB initial memory pool. The CPU wheel logs `Please use PaddlePaddle with GPU version.` and leaves `config.use_gpu()` false. Ultralytics then applies the same x86-64 oneDNN workarounds for OBB and depth as for `device="cpu"`. See the [CPU fallback benchmarks](#cpu-fallback-on-a-cuda-host).
- **Export**: `device=0` traces the model on the GPU; the conversion itself needs no GPU and no GPU build.

On an NVIDIA RTX PRO 6000 Blackwell with the CUDA 13.0 build of `paddlepaddle-gpu` 3.2.2, `device=0` enabled GPU inference. With 640-pixel FP32 exports, YOLO26n-OBB reached 0.8944 mAP50-95 on `dota8.yaml` and YOLO26n-depth reached 0.2088 RMSE on `depth8.yaml`. On a host without a visible CUDA GPU, use `device="cpu"`.

### Deployment Guides

- **[Paddle Inference](https://www.paddlepaddle.org.cn/inference/master/guides/introduction/index_intro.html)**: the documentation of Paddle Inference, with the Python and C++ APIs.
- **[Paddle Inference `Config` for CPUs](https://www.paddlepaddle.org.cn/inference/master/api_reference/python_api_doc/Config/CPUConfig.html)** and **[for GPUs](https://www.paddlepaddle.org.cn/inference/master/api_reference/python_api_doc/Config/GPUConfig.html)**: the thread count, oneDNN, and GPU settings used above.
- **[PaddlePaddle installation](https://www.paddlepaddle.org.cn/en/install/quick)**: the commands for the CPU and GPU builds.
- **[Ultralytics and PaddlePaddle](https://www.ultralytics.com/blog/exporting-ultralytics-yolo11-using-the-paddlepaddle-integration)**: an overview of the integration on the Ultralytics blog.
- **[Model deployment options](../guides/model-deployment-options.md)**: compare PaddlePaddle with the other export formats.

## Limitations and Troubleshooting

- **Depth and OBB models on x86-64 CPUs**: Paddle Inference turns on oneDNN by default on x86-64 CPUs, and two of its bugs change the outputs of these models: in-place operations after oneDNN layers overwrite tensors that other layers still read, which gives OBB models wrong angles and box centers, and its bilinear resize ignores `align_corners=True`, which the depth head uses. `YOLO()` avoids both: for OBB models it turns off in-place operations with `config.delete_pass("inplace_pass")`, and for depth models it runs only the convolutions with oneDNN through `config.delete_pass("onednn_placement_pass")`. On the 4 validation images of `dota8.yaml`, YOLO26n-OBB then reached 0.894 mAP50-95 and YOLO11n-OBB 0.817, and on those of `depth8.yaml`, YOLO26n-depth reached a delta1 of 0.854 and an RMSE of 0.205, the same as ONNX. With the default settings, which your own Paddle Inference code uses unless it calls the same `delete_pass`, they reached 0.383, 0.439, and 0.851 and 0.260. On the AMD EPYC 9655 server limited to 8 CPUs, in the inference stage that `val` reports, YOLO26n-OBB took 94 ms per image instead of 82 with the default settings, and YOLO26n-depth 148 ms instead of 83, as the mean of several runs. Turning oneDNN off with `config.disable_mkldnn()` also gives the ONNX values but is slower: on 8 threads set with `config.set_cpu_math_library_num_threads(8)`, YOLO26n-OBB took 229 to 233 ms and YOLO26n-depth 326 to 334 ms.
- **RT-DETR**: the export of `rtdetr-l.pt` fails with `Model is not supported yet.`
- **Python 3.14 and newer**: `paddlepaddle` has no packages for them, so pip finds no version to install, for example `Could not find a version that satisfies the requirement paddlepaddle<3.3.0,>=3.0.0 (from versions: none)` on Linux x86-64, and the export fails with `No module named 'paddle'`. Use Python 3.13 or older.
- **`paddlepaddle` 3.3 or newer**: the first export and the first prediction replace it with a version older than 3.3, see the version requirements in [Quickstart](#quickstart). In an environment that needs a newer `paddlepaddle` for other code, export in a separate environment.
- **Automatic installation turned off**: with `YOLO_AUTOINSTALL=false` and a plain `pip install ultralytics`, the export logs `Ultralytics requirement ['paddlepaddle>=3.0.0,<3.3.0'] not found, AutoUpdate disabled by YOLO_AUTOINSTALL=False` and fails with `No module named 'x2paddle'`. Install the packages first with the command in [Quickstart](#quickstart); the export then needs no installation.
- **NVIDIA Jetson**: by the Ultralytics code, the export stops with `Jetson Paddle exports not supported yet`. Export on another machine.
- **Passing a file or the `inference_model/` directory**: `YOLO("yolo26n_paddle_model/inference_model/model.pdiparams")`, the `model.json` file, and the `inference_model/` directory all fail with `is not a supported model format`. Pass the exported directory, and keep `_paddle_model` in its name: a copy named `yolo26n_pd` failed the same way, while `my_paddle_model_v2` and `yolo26n_paddle_model_old` loaded.
- **Other files in the directory**: Ultralytics loads the first `.json` file and the first `.pdiparams` file it finds in the directory and its subdirectories. With a `labels.json` file added next to `metadata.yaml`, loading failed with `(InvalidArgument) Invalid model file.` An export also keeps such files: every export of one checkpoint writes the same `yolo26n_paddle_model/` directory, whatever its `imgsz` or `batch`, overwrites `model.json` and `model.pdiparams`, and keeps any other files there. With a `.pdiparams` file added to `inference_model/`, loading failed with `Deserialize to tensor failed, maybe the loaded file is not a paddle model` when that file was found first. Keep other files out of the directory.
- **Fixed input shape**: `dynamic=True` fails with `argument 'dynamic' is not supported for format='paddle'`. Export again to change the image size or the batch size.
- **Checkpoints trained with `quantize=8`**: a checkpoint fine-tuned with [quantization-aware training](../modes/export.md#quantization-aware-training) fails with `format='paddle' cannot export a QAT model: its Q/DQ ranges are only read by the 'onnx' and 'engine' backends`. Export the checkpoint you started the fine-tuning from instead, or export the fine-tuned checkpoint to [ONNX](onnx.md) or [TensorRT](tensorrt.md).

For other errors, see the [Common Issues guide](../guides/yolo-common-issues.md).

## Benchmarks

There are no earlier PaddlePaddle benchmarks from Ultralytics. The tables below compare PyTorch, ONNX, and PaddlePaddle models of every YOLO26 detection model on the CPU of the export host, an **AMD EPYC 9655** server shared with other jobs. Every process was limited to the 8 CPUs of one 8-core CCD of the processor with `taskset -c 40-47`. PyTorch and ONNX Runtime ran on the 8 threads that Ultralytics sets, and Paddle Inference, with oneDNN, on the same 8 threads, see [Deploy](#deploy). The results apply to this setup.

=== "Latency"

    Milliseconds per image, lower is better, as the mean of two runs. The fastest format for each model is in bold.

    | Model   | `imgsz` | PyTorch  | ONNX FP32 | PaddlePaddle FP32 |
    | ------- | ------- | -------- | --------- | ----------------- |
    | YOLO26n | 640     | **10.0** | 16.6      | 33.4              |
    | YOLO26s | 640     | **25.5** | 31.3      | 63.0              |
    | YOLO26m | 640     | 65.1     | **57.9**  | 125.9             |
    | YOLO26l | 640     | 74.1     | **70.7**  | 166.1             |
    | YOLO26x | 640     | 144.4    | **130.4** | 280.1             |

=== "Accuracy"

    mAP<sup>val</sup> 50-95 on 500 images of COCO val2017, higher is better.

    | Model   | PyTorch | ONNX FP32 | PaddlePaddle FP32 |
    | ------- | ------- | --------- | ----------------- |
    | YOLO26n | 42.0    | 42.0      | 42.0              |
    | YOLO26s | 49.4    | 49.4      | 49.4              |
    | YOLO26m | 53.4    | 53.4      | 53.4              |
    | YOLO26l | 55.4    | 55.4      | 55.4              |
    | YOLO26x | 55.6    | 55.6      | 55.6              |

=== "Size"

    Size of the `.pt` checkpoint, the `.onnx` file, and the PaddlePaddle model files (`model.json` and `model.pdiparams`) in MB (10^6 bytes).

    | Model   | PyTorch | ONNX FP32 | PaddlePaddle FP32 |
    | ------- | ------- | --------- | ----------------- |
    | YOLO26n | 5.5     | 9.9       | 10.0              |
    | YOLO26s | 20.4    | 38.3      | 38.4              |
    | YOLO26m | 44.3    | 82.0      | 82.1              |
    | YOLO26l | 53.2    | 99.6      | 99.8              |
    | YOLO26x | 118.7   | 223.3     | 223.5             |

- **Latency**: PyTorch was the fastest for YOLO26n and YOLO26s, and ONNX for the larger models. PaddlePaddle took 2.0 to 2.3 times as long as ONNX and 1.9 to 3.3 times as long as PyTorch.
- **Accuracy**: PaddlePaddle and PyTorch gave the same mAP as ONNX at every size.
- **Size**: the PaddlePaddle model files were 0.1 to 0.2 MB larger than the ONNX file.
- **All CPUs**: without `taskset`, on all 384 logical CPUs, YOLO26n took 33.8 ms as PaddlePaddle, 14.7 ms as PyTorch, and 14.6 ms as ONNX, all on 8 threads, as the mean of two runs.

Measured with `ultralytics` 8.4.218, `paddlepaddle` 3.2.2, X2Paddle 1.6.0, PyTorch 2.15.0, and ONNX Runtime 1.32.0 at `batch=1`. PyTorch ran with `rect=False`, at the same 640x640 input as the exported models. Accuracy is on the last 500 images of COCO val2017 in sorted order, so it differs from the values for the full validation set on the [YOLO26 model page](../models/yolo26.md). Latency is the inference stage that `val` reports and does not include preprocessing or postprocessing; the two runs differed by up to 26%, and runs that overlapped a burst of other jobs on the server were repeated.

### CPU Fallback on a CUDA Host

On a server with an AMD EPYC 9655 and an NVIDIA RTX PRO 6000 Blackwell, `device=0` with the `paddlepaddle` 3.2.2 CPU wheel ran Paddle Inference on the CPU. Each pair compares the same FP32 export at `imgsz=640` and `batch=1` with the default oneDNN passes and with the [task-specific workarounds](#limitations-and-troubleshooting) Ultralytics applies on fallback. OBB mAP50-95 is on all 458 [DOTAv1](../datasets/obb/dota-v2.md#dota-v10) validation images (28,853 objects), and depth RMSE is on all 654 [NYU Depth V2](../datasets/depth/nyu-depth-v2.md) validation images.

| Model   | OBB mAP50-95, default | OBB mAP50-95, corrected ↑ | Depth RMSE, default | Depth RMSE, corrected ↓ |
| ------- | --------------------: | ------------------------: | ------------------: | ----------------------: |
| YOLO26n |                0.0721 |                    0.3577 |              0.5518 |                  0.5465 |
| YOLO26s |                0.0742 |                    0.4024 |              0.5720 |                  0.5392 |
| YOLO26m |                0.0803 |                    0.4414 |              0.5669 |                  0.5282 |
| YOLO26l |                0.0864 |                    0.4461 |              0.5641 |                  0.5277 |
| YOLO26x |                0.0864 |                    0.4609 |              0.5575 |                  0.5198 |

## About PaddlePaddle

Developed by [Baidu](https://www.baidu.com/), PaddlePaddle (**PA**rallel **D**istributed **D**eep **LE**arning) is an open-source deep learning framework with tools for training and deployment, including Paddle Inference. For the full reference, see the [PaddlePaddle documentation](https://www.paddlepaddle.org.cn/documentation/docs/en/guides/index_en.html).

<p align="center">
  <img width="75%" src="https://github.com/PaddlePaddle/Paddle/blob/develop/doc/imgs/logo.png?raw=true" alt="PaddlePaddle deep learning framework logo">
</p>

<p align="center">
  <br>
  <iframe loading="lazy" width="720" height="405" src="https://www.youtube.com/embed/c5eFrt2KuzY"
    title="YouTube video player" frameborder="0"
    allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share"
    allowfullscreen>
  </iframe>
  <br>
  <strong>Watch:</strong> How to Export Ultralytics YOLO Models to PaddlePaddle Format | Key Features of PaddlePaddle Format
</p>

## FAQ

### Can I deploy the exported model with Paddle Lite, Paddle Serving, or Paddle.js?

Not as exported. The export writes the PIR format of PaddlePaddle 3, `model.json` and `model.pdiparams`, which Paddle Inference reads. The `opt` tool of Paddle Lite 2.12 from PyPI failed on it with `Check failed: (idx < static_cast<int32_t>(BlocksSize())): 0!<0 idx >= blocks.size()`. [Paddle Serving](https://github.com/PaddlePaddle/Serving) was archived in October 2026, and its README recommends NVIDIA Triton Inference Server; for Triton, export to [ONNX](onnx.md) or [TensorRT](tensorrt.md). [Paddle.js](https://github.com/PaddlePaddle/Paddle.js), last released in 2021, converts the earlier PaddlePaddle model format, and was not tried for this page.

### Should I use PaddlePaddle or ONNX for CPU deployment?

On the CPU of the benchmark server, ONNX: both gave the same mAP, and the PaddlePaddle models took 2.0 to 2.3 times as long per image, see [Benchmarks](#benchmarks). OBB and depth models take longer still as PaddlePaddle, because of the settings `YOLO()` uses to keep their outputs correct, see [Limitations and Troubleshooting](#limitations-and-troubleshooting). PaddlePaddle fits an application that already runs Paddle Inference.
