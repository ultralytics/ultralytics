# Ultralytics YOLO ONNX Runtime Inference in Rust

<img alt="Rust" src="https://img.shields.io/badge/Rust-000000.svg?logo=rust&logoColor=white"> <img alt="ONNX Runtime" src="https://img.shields.io/badge/ONNX%20Runtime-717272.svg?logo=onnx&logoColor=white"> <a href="https://docs.ultralytics.com" target="_blank"><img alt="Ultralytics Docs" src="https://img.shields.io/badge/Ultralytics-Docs-042AFF.svg?logo=ultralytics&logoColor=white"></a>

This repository provides a Rust demonstration for performing Ultralytics YOLO tasks like [Detection](https://docs.ultralytics.com/tasks/detect), [Segmentation](https://docs.ultralytics.com/tasks/segment), [Classification](https://docs.ultralytics.com/tasks/classify), and [Pose Estimation](https://docs.ultralytics.com/tasks/pose) with [YOLOv8](https://docs.ultralytics.com/models/yolov8) and [YOLO26](https://docs.ultralytics.com/models/yolo26) ONNX models using the [ONNXRuntime](https://onnxruntime.ai/).

## ✨ Recently Updated

- Runs Ultralytics YOLO26 ONNX exports for detection, segmentation, classification, and pose.

For a maintained Rust library and CLI covering every Ultralytics YOLO task, see [Ultralytics Inference for Rust](https://docs.ultralytics.com/inference).

## 🚀 Features

- Supports `Detection`, `Segmentation`, `Classification`, and `Pose(Keypoints)-Detection` tasks.
- Supports `FP16` & `FP32` [ONNX](https://onnx.ai/) models.
- Supports `CPU`, `CUDA`, and `TensorRT` execution providers to accelerate computation.
- Supports dynamic input shapes (`batch`, `width`, `height`).

## 🛠️ Installation

### 1. Install Rust

Please follow the official Rust installation guide: [https://rust-lang.org/tools/install](https://rust-lang.org/tools/install).

### 2. ONNXRuntime Linking

- The `download-binaries` and `copy-dylibs` features of the `ort` crate (enabled in `Cargo.toml`) download prebuilt ONNX Runtime binaries at build time, so no manual setup is needed on supported platforms.
- To link your own ONNX Runtime build instead, refer to the [ORT documentation](https://ort.pyke.io/setup/linking).

### 3. [Optional] Install CUDA & CuDNN & TensorRT

- The CUDA execution provider requires [CUDA](https://developer.nvidia.com/cuda/toolkit) `12.x` and [cuDNN](https://developer.nvidia.com/cuda/cuda-x-libraries/cudnn) `9.x`.
- The TensorRT execution provider additionally requires [TensorRT](https://developer.nvidia.com/tensorrt) `10.x`.

## ▶️ Get Started

### 1. Export Ultralytics YOLO ONNX Models

First, install the Ultralytics package:

```bash
pip install -U ultralytics
```

Then, export the desired [Ultralytics YOLO26](https://docs.ultralytics.com/models/yolo26) (or [YOLOv8](https://docs.ultralytics.com/models/yolov8)) models to the ONNX format. See the [Export documentation](https://docs.ultralytics.com/modes/export) for more details.

```bash
# Export ONNX models with fixed input shapes
yolo export model=yolo26n.pt format=onnx
yolo export model=yolo26n-seg.pt format=onnx
yolo export model=yolo26n-cls.pt format=onnx
yolo export model=yolo26n-pose.pt format=onnx

# Export an ONNX model with dynamic shapes (batch, height, width)
yolo export model=yolo26n.pt format=onnx dynamic=True
```

### 2. Run Inference

This command will perform inference using the specified ONNX model on the source image using the CPU.

```bash
cargo run --release -- --model MODEL_PATH.onnx --source SOURCE_IMAGE.jpg
```

#### Using GPU Acceleration

Set `--cuda` to use the CUDA execution provider for faster inference on NVIDIA GPUs.

```bash
cargo run --release -- --cuda --model MODEL_PATH.onnx --source SOURCE_IMAGE.jpg
```

Set `--trt` to use the TensorRT execution provider. You can also set `--fp16` simultaneously to leverage the TensorRT FP16 engine for potentially even greater speed, especially on compatible hardware.

```bash
cargo run --release -- --trt --fp16 --model MODEL_PATH.onnx --source SOURCE_IMAGE.jpg
```

#### Specifying Device and Batch Size

Set `--device-id` to select a specific GPU device. If the CUDA execution provider is unavailable, `ort` falls back to the `CPU` execution provider; an invalid device ID on a CUDA system fails instead.

```bash
cargo run --release -- --cuda --device-id 0 --model MODEL_PATH.onnx --source SOURCE_IMAGE.jpg
```

The CLI runs inference on the single `--source` image. `--batch` sets the batch size the model is configured for and must match the batch of a fixed-shape export. With `--trt` and a model exported with dynamic batch dimensions, `--batch-min`, `--batch`, and `--batch-max` set the minimum, optimal, and maximum batch sizes of the TensorRT optimization profile. Refer to the [TensorRT Execution Provider documentation](https://onnxruntime.ai/docs/execution-providers/TensorRT-ExecutionProvider.html#explicit-shape-range-for-dynamic-shape-input) for details.

#### Dynamic Image Size

Set `--height` and `--width` to perform inference with dynamic image sizes; both are required for models exported with dynamic input shapes (`dynamic=True`).

```bash
cargo run --release -- --cuda --height 640 --width 480 --model MODEL_PATH_dynamic.onnx --source SOURCE_IMAGE.jpg
```

#### Profiling Performance

Set `--profile` to measure the time consumed in each stage of the inference pipeline (preprocessing, H2D transfer, inference, D2H transfer, postprocessing). **Note:** Models often require a few "warm-up" runs (1-3 iterations) before reaching optimal performance. Ensure you run the command enough times to get a stable performance evaluation.

```bash
cargo run --release -- --trt --fp16 --profile --model MODEL_PATH.onnx --source SOURCE_IMAGE.jpg
```

Example Profile Output (yolov8m.onnx, batch=1, 3 runs, trt, fp16, RTX 3060Ti):

```text
==> 0 # Warm-up run
[Model Preprocess]: 12.75788ms
[ORT H2D]: 237.118µs
[ORT Inference]: 507.895469ms
[ORT D2H]: 191.655µs
[Model Inference]: 508.34589ms
[Model Postprocess]: 1.061122ms
==> 1 # Stable run
[Model Preprocess]: 13.658655ms
[ORT H2D]: 209.975µs
[ORT Inference]: 5.12372ms
[ORT D2H]: 182.389µs
[Model Inference]: 5.530022ms
[Model Postprocess]: 1.04851ms
==> 2 # Stable run
[Model Preprocess]: 12.475332ms
[ORT H2D]: 246.127µs
[ORT Inference]: 5.048432ms
[ORT D2H]: 187.117µs
[Model Inference]: 5.493119ms
[Model Postprocess]: 1.040906ms
```

#### Other Options

- `--conf`: Confidence threshold for detections \[default: 0.3].
- `--iou`: IoU (Intersection over Union) threshold for Non-Maximum Suppression (NMS) \[default: 0.5].
- `--kconf`: Confidence threshold for keypoints (in Pose Estimation) \[default: 0.55].
- `--plot`: Plot the inference results with random RGB colors and save the output image to the `runs` directory.
- `--task`, `--nc`, `--nk`, `--nm`: Task, number of classes, keypoints, and mask coefficients. These are read from the ONNX metadata of Ultralytics exports; set them only for models without it.

You can view all available command-line arguments by running:

```bash
# Clone the repository if you haven't already
# git clone https://github.com/ultralytics/ultralytics
# cd ultralytics/examples/YOLOv8-ONNXRuntime-Rust

cargo run --release -- --help
```

## 🖼️ Examples

![Ultralytics YOLO Tasks](https://cdn.ul.run/i/c99d914c3958d0755b5a3d7204b6f24a.avif)

### Classification

Running an ONNX classification model on the `CPU`. The plotted result image will be saved in the `runs` directory.

```bash
cargo run --release -- --model yolo26n-cls.onnx --source ../../ultralytics/assets/bus.jpg --plot --profile
```

Example output:

```text
Summary:
> Task: Classify (Ultralytics 8.4.218) # Version might differ
> EP: CPU
> Dtype: Float32
> Batch: 1 (Const), Height: 224 (Const), Width: 224 (Const)
> nc: 1000 nk: 0, nm: 0, conf: 0.3, kconf: 0.55, iou: 0.5

[Model Preprocess]: 3.438084ms
[ORT H2D]: 42ns
[ORT Prepare Value]: 6.75µs
[ORT Inference]: 2.464833ms
[ORT D2H]: 1.75µs
[Model Inference]: 2.508334ms
[Model Postprocess]: 2.458µs
[YOLOResult { Probs(top5): Some([(654, 0.5228457), (734, 0.42368317), (874, 0.0143324835), (627, 0.012642897), (656, 0.0077146036)]), Bboxes: None, Keypoints: None, Masks: None }] # Class IDs and confidences
```

### Object Detection

Using the `CUDA` execution provider.

```bash
cargo run --release -- --cuda --model yolo26n.onnx --source ../../ultralytics/assets/bus.jpg --plot
```

### Pose Detection

Using the `TensorRT` execution provider.

```bash
cargo run --release -- --trt --model yolo26n-pose.onnx --source ../../ultralytics/assets/bus.jpg --plot
```

### Instance Segmentation

Using the `TensorRT` execution provider with an FP16 engine (`--fp16`).

```bash
cargo run --release -- --trt --fp16 --model yolo26n-seg.onnx --source ../../ultralytics/assets/bus.jpg --plot
```

## 🤝 Contributing

Contributions are welcome! If you find an issue or have a suggestion, please open an issue or pull request on the main [Ultralytics repository](https://github.com/ultralytics/ultralytics).
