# YOLOv8-ONNXRuntime-Rust for All Key YOLO Tasks

This repository provides a Rust demonstration for performing Ultralytics YOLOv8 tasks like [Classification](https://docs.ultralytics.com/tasks/classify/), [Segmentation](https://docs.ultralytics.com/tasks/segment/), [Detection](https://docs.ultralytics.com/tasks/detect/), [Pose Estimation](https://docs.ultralytics.com/tasks/pose/), and [Oriented Bounding Box (OBB)](https://docs.ultralytics.com/tasks/obb/) detection using the [ONNXRuntime](https://onnxruntime.ai/).

## ✨ Recently Updated

- Added YOLOv8-OBB demo.
- Updated ONNXRuntime dependency to 1.19.x.

Newly updated YOLOv8 example code is located in [this repository](https://github.com/jamjamjon/usls/tree/main/examples/yolo).

## 🚀 Features

- Supports `Classification`, `Segmentation`, `Detection`, `Pose(Keypoints)-Detection`, and `OBB` tasks.
- Supports `FP16` & `FP32` [ONNX](https://onnx.ai/) models.
- Supports `CPU`, `CUDA`, and `TensorRT` execution providers to accelerate computation.
- Supports dynamic input shapes (`batch`, `width`, `height`).

## 🛠️ Installation

### 1. Install Rust

Please follow the official Rust installation guide: [https://www.rust-lang.org/tools/install](https://rust-lang.org/tools/install/).

### 2. ONNXRuntime Linking

- #### For detailed setup instructions, refer to the [ORT documentation](https://ort.pyke.io/setup/linking).

- #### For Linux or macOS Users:
  - Download the ONNX Runtime package from the [Releases page](https://github.com/microsoft/onnxruntime/releases).
  - Set up the library path by exporting the `ORT_DYLIB_PATH` environment variable:
    ```bash
    export ORT_DYLIB_PATH=/path/to/onnxruntime/lib/libonnxruntime.so.1.19.0 # Adjust version/path as needed
    ```

### 3. [Optional] Install CUDA & CuDNN & TensorRT

- The CUDA execution provider requires [CUDA](https://developer.nvidia.com/cuda-toolkit) v11.6+.
- The TensorRT execution provider requires CUDA v11.4+ and [TensorRT](https://developer.nvidia.com/tensorrt) v8.4+. You may also need [cuDNN](https://developer.nvidia.com/cudnn).

## ▶️ Get Started

### 1. Export the Ultralytics YOLOv8 ONNX Models

First, install the Ultralytics package:

```bash
pip install -U ultralytics
```

Then, export the desired [Ultralytics YOLOv8](https://docs.ultralytics.com/models/yolov8/) models to the ONNX format. See the [Export documentation](https://docs.ultralytics.com/modes/export/) for more details.

```bash
# Export ONNX model with dynamic shapes (recommended for flexibility)
yolo export model=yolov8m.pt format=onnx simplify dynamic
yolo export model=yolov8m-cls.pt format=onnx simplify dynamic
yolo export model=yolov8m-pose.pt format=onnx simplify dynamic
yolo export model=yolov8m-seg.pt format=onnx simplify dynamic
# yolo export model=yolov8m-obb.pt format=onnx simplify dynamic # Add OBB export if needed

# Export ONNX model with constant shapes (if dynamic shapes are not required)
# yolo export model=yolov8m.pt format=onnx simplify
# yolo export model=yolov8m-cls.pt format=onnx simplify
# yolo export model=yolov8m-pose.pt format=onnx simplify
# yolo export model=yolov8m-seg.pt format=onnx simplify
# yolo export model=yolov8m-obb.pt format=onnx simplify
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

Set `--device_id` to select a specific GPU device. If the specified device ID is invalid (e.g., setting `device_id 1` when only one GPU exists), `ort` will automatically fall back to the `CPU` execution provider without causing a panic.

```bash
cargo run --release -- --cuda --device_id 0 --model MODEL_PATH.onnx --source SOURCE_IMAGE.jpg
```

Set `--batch` to perform inference with a specific batch size.

```bash
cargo run --release -- --cuda --batch 2 --model MODEL_PATH.onnx --source SOURCE_IMAGE.jpg
```

If you're using `--trt` with a model exported with dynamic batch dimensions, you can explicitly specify the minimum, optimal, and maximum batch sizes for TensorRT optimization using `--batch-min`, `--batch`, and `--batch-max`. Refer to the [TensorRT Execution Provider documentation](https://onnxruntime.ai/docs/execution-providers/TensorRT-ExecutionProvider.html#explicit-shape-range-for-dynamic-shape-input) for details.

#### Dynamic Image Size

Set `--height` and `--width` to perform inference with dynamic image sizes. **Note:** The ONNX model must have been exported with dynamic input shapes (`dynamic=True`).

```bash
cargo run --release -- --cuda --width 480 --height 640 --model MODEL_PATH_dynamic.onnx --source SOURCE_IMAGE.jpg
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
- `--iou`: IoU (Intersection over Union) threshold for Non-Maximum Suppression (NMS) \[default: 0.45].
- `--kconf`: Confidence threshold for keypoints (in Pose Estimation) \[default: 0.55].
- `--plot`: Plot the inference results with random RGB colors and save the output image to the `runs` directory.

You can view all available command-line arguments by running:

```bash
# Clone the repository if you haven't already
# git clone https://github.com/ultralytics/ultralytics
# cd ultralytics/examples/YOLOv8-ONNXRuntime-Rust

cargo run --release -- --help
```

## 🖼️ Examples

![Ultralytics YOLO Tasks](https://raw.githubusercontent.com/ultralytics/assets/main/im/banner-tasks.png)

### Classification

Running a dynamic shape ONNX classification model on the `CPU` with a specific image size (`--height 224 --width 224`). The plotted result image will be saved in the `runs` directory.

```bash
cargo run --release -- --model ../assets/weights/yolov8m-cls-dyn.onnx --source ../assets/images/dog.jpg --height 224 --width 224 --plot --profile
```

Example output:

```text
Summary:
> Task: Classify (Ultralytics 8.0.217) # Version might differ
> EP: Cpu
> Dtype: Float32
> Batch: 1 (Dynamic), Height: 224 (Dynamic), Width: 224 (Dynamic)
> nc: 1000 nk: 0, nm: 0, conf: 0.3, kconf: 0.55, iou: 0.45

[Model Preprocess]: 16.363477ms
[ORT H2D]: 50.722µs
[ORT Inference]: 16.295808ms
[ORT D2H]: 8.37µs
[Model Inference]: 16.367046ms
[Model Postprocess]: 3.527µs
[
    YOLOResult {
        Probs(top5): Some([(208, 0.6950566), (209, 0.13823675), (178, 0.04849795), (215, 0.019029364), (212, 0.016506357)]), # Class IDs and confidences
        Bboxes: None,
        Keypoints: None,
        Masks: None,
    },
]
```

### Object Detection

Using the `CUDA` execution provider and a dynamic image size (`--height 640 --width 480`).

```bash
cargo run --release -- --cuda --model ../assets/weights/yolov8m-dynamic.onnx --source ../assets/images/bus.jpg --plot --height 640 --width 480
```

### Pose Detection

Using the `TensorRT` execution provider.

```bash
cargo run --release -- --trt --model ../assets/weights/yolov8m-pose.onnx --source ../assets/images/bus.jpg --plot
```

### Instance Segmentation

Using the `TensorRT` execution provider with an FP16 model (`--fp16`).

```bash
cargo run --release -- --trt --fp16 --model ../assets/weights/yolov8m-seg.onnx --source ../assets/images/0172.jpg --plot
```

## 🤝 Contributing

Contributions are welcome! If you find any issues or have suggestions for improvement, please feel free to open an issue or submit a pull request to the main [Ultralytics repository](https://github.com/ultralytics/ultralytics).
