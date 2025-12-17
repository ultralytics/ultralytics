# YOLO-Series ONNXRuntime Rust Demo for Core YOLO Tasks

This repository provides a [Rust](https://rust-lang.org/) demo showcasing key [Ultralytics YOLO](https://docs.ultralytics.com/) series tasks such as [Classification](https://docs.ultralytics.com/tasks/classify/), [Segmentation](https://docs.ultralytics.com/tasks/segment/), [Detection](https://docs.ultralytics.com/tasks/detect/), [Pose Estimation](https://docs.ultralytics.com/tasks/pose/), and Oriented Bounding Box ([OBB](https://docs.ultralytics.com/tasks/obb/)) detection using the [ONNXRuntime](https://github.com/microsoft/onnxruntime). It supports various YOLO models (v5 through 11) across multiple computer vision tasks.

## ✨ Introduction

- This example leverages the latest versions of both the [ONNX Runtime](https://onnxruntime.ai/) and popular YOLO models.
- We utilize the [usls crate](https://github.com/jamjamjon/usls/tree/main) to streamline YOLO model inference in Rust, providing efficient data loading, visualization, and optimized inference performance. This allows developers to easily integrate state-of-the-art object detection into their Rust applications.

## 🚀 Features

- **Extensive Model Compatibility**: Supports a wide range of YOLO versions including [YOLOv5](https://docs.ultralytics.com/models/yolov5/), [YOLOv6](https://docs.ultralytics.com/models/yolov6/), [YOLOv7](https://docs.ultralytics.com/models/yolov7/), [YOLOv8](https://docs.ultralytics.com/models/yolov8/), [YOLOv9](https://docs.ultralytics.com/models/yolov9/), [YOLOv10](https://docs.ultralytics.com/models/yolov10/), [YOLO11](https://docs.ultralytics.com/models/yolo11/), [YOLO-World](https://docs.ultralytics.com/models/yolo-world/), [RT-DETR](https://docs.ultralytics.com/models/rtdetr/), and others.
- **Versatile Task Coverage**: Includes examples for `Classification`, `Segmentation`, `Detection`, `Pose`, and `OBB`.
- **Precision Flexibility**: Works seamlessly with `FP16` and `FP32` precision [ONNX models](https://docs.ultralytics.com/integrations/onnx/).
- **Execution Providers**: Accelerated support for `CPU`, [CUDA](https://developer.nvidia.com/cuda-toolkit), [CoreML](https://developer.apple.com/documentation/coreml), and [TensorRT](https://docs.ultralytics.com/integrations/tensorrt/).
- **Dynamic Input Shapes**: Dynamically adjusts to variable `batch`, `width`, and `height` dimensions for flexible model input.
- **Flexible Data Loading**: The `DataLoader` component handles images, folders, videos, and real-time video streams.
- **Real-Time Display and Video Export**: The `Viewer` provides real-time frame visualization and video export functions, similar to OpenCV’s `imshow()` and `imwrite()`.
- **Enhanced Annotation and Visualization**: The `Annotator` facilitates comprehensive result rendering, supporting bounding boxes (HBB), oriented bounding boxes (OBB), polygons, masks, keypoints, and text labels.

## 🛠️ Setup Instructions

### 1. ONNXRuntime Linking

<details>
<summary>You have two options to link the ONNXRuntime library:</summary>

- **Option 1: Manual Linking**
  - For detailed setup instructions, consult the [ONNX Runtime linking documentation](https://ort.pyke.io/setup/linking).
  - **Linux or macOS**:
    1. Download the appropriate ONNX Runtime package from the official [Releases page](https://github.com/microsoft/onnxruntime/releases).
    2. Set up the library path by exporting the `ORT_DYLIB_PATH` environment variable, pointing to the downloaded library file:
       ```bash
       # Example path, replace with your actual path
       export ORT_DYLIB_PATH=/path/to/onnxruntime/lib/libonnxruntime.so.1.19.0
       ```

- **Option 2: Automatic Download**
  - Use the `--features auto` flag with Cargo to let the build script handle downloading the library automatically:
    ```bash
    cargo run -r --example yolo --features auto
    ```

</details>

### 2. [Optional] Install CUDA, CuDNN, and TensorRT

- The CUDA execution provider requires [NVIDIA CUDA Toolkit](https://developer.nvidia.com/cuda-toolkit) version `12.x`.
- The TensorRT execution provider requires both CUDA `12.x` and [NVIDIA TensorRT](https://developer.nvidia.com/tensorrt) `10.x`. Ensure [cuDNN](https://developer.nvidia.com/cudnn) is also correctly installed.

### 3. [Optional] Install ffmpeg

To enable viewing video frames and saving video inferences, install the `rust-ffmpeg` crate's dependencies. Follow the instructions provided here:
[https://github.com/zmwangx/rust-ffmpeg/wiki/Notes-on-building#dependencies](https://github.com/zmwangx/rust-ffmpeg/wiki/Notes-on-building#dependencies)

## ▶️ Get Started

Run the examples using Cargo. The `--` separates Cargo arguments from the example's arguments.

```bash
# Run a custom model (e.g., YOLOv8 detection)
cargo run -r -- --task detect --ver v8 --nc 6 --model path/to/your/model.onnx

# Classify examples
cargo run -r -- --task classify --ver v5 --scale s --width 224 --height 224 --nc 1000  # YOLOv5 Classification
cargo run -r -- --task classify --ver v8 --scale n --width 224 --height 224 --nc 1000  # YOLOv8 Classification
cargo run -r -- --task classify --ver v11 --scale n --width 224 --height 224 --nc 1000 # YOLO11 Classification

# Detect examples
cargo run -r -- --task detect --ver v5 --scale n     # YOLOv5 Detection
cargo run -r -- --task detect --ver v6 --scale n     # YOLOv6 Detection
cargo run -r -- --task detect --ver v7 --scale t     # YOLOv7 Detection
cargo run -r -- --task detect --ver v8 --scale n     # YOLOv8 Detection
cargo run -r -- --task detect --ver v9 --scale t     # YOLOv9 Detection
cargo run -r -- --task detect --ver v10 --scale n    # YOLOv10 Detection
cargo run -r -- --task detect --ver v11 --scale n    # YOLO11 Detection
cargo run -r -- --task detect --ver rtdetr --scale l # RT-DETR Detection

# Pose examples
cargo run -r -- --task pose --ver v8 --scale n  # YOLOv8-Pose Estimation
cargo run -r -- --task pose --ver v11 --scale n # YOLO11-Pose Estimation

# Segment examples
cargo run -r -- --task segment --ver v5 --scale n                              # YOLOv5-Segment
cargo run -r -- --task segment --ver v8 --scale n                              # YOLOv8-Segment
cargo run -r -- --task segment --ver v11 --scale n                             # YOLO11-Segment
cargo run -r -- --task segment --ver v8 --model path/to/FastSAM-s-dyn-f16.onnx # FastSAM Segmentation

# OBB (Oriented Bounding Box) examples
cargo run -r -- --ver v8 --task obb --scale n --width 1024 --height 1024 --source images/dota.png  # YOLOv8-OBB
cargo run -r -- --ver v11 --task obb --scale n --width 1024 --height 1024 --source images/dota.png # YOLO11-OBB
```

**Use `cargo run -- --help` to see all available options.**

For more detailed information and advanced usage, please refer to the [usls-yolo example documentation](https://github.com/jamjamjon/usls/tree/main/examples/yolo).

## 🤝 Contributing

Contributions are welcome! If you'd like to improve this demo or add new features, please feel free to submit issues or pull requests on the repository. Your input helps make the Ultralytics ecosystem better for everyone. Check out the [Ultralytics Contribution Guide](https://docs.ultralytics.com/help/contributing/) for more details.
