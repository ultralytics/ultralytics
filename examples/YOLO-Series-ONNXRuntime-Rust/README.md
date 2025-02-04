# YOLO-Series ONNXRuntime Rust Demo for Core YOLO Tasks

This repository provides a Rust demo for key YOLO-Series tasks such as `Classification`, `Segmentation`, `Detection`, `Pose Detection`, and `OBB` using ONNXRuntime. It supports various YOLO models (v5 - 11) across multiple vision tasks.

## Introduction

- This example leverages the latest versions of both ONNXRuntime and YOLO models.
- We utilize the [usls](https://github.com/jamjamjon/usls/tree/main) crate to streamline YOLO model inference, providing efficient data loading, visualization, and optimized inference performance.

## Features

- **Extensive Model Compatibility**: Supports `YOLOv5`, `YOLOv6`, `YOLOv7`, `YOLOv8`, `YOLOv9`, `YOLOv10`, `YOLO11`, `YOLO-world`, `RTDETR`, and others, covering a wide range of YOLO versions.
- **Versatile Task Coverage**: Includes `Classification`, `Segmentation`, `Detection`, `Pose`, and `OBB`.
- **Precision Flexibility**: Works with `FP16` and `FP32` ONNX models.
- **Execution Providers**: Accelerated support for `CPU`, `CUDA`, `CoreML`, and `TensorRT`.
- **Dynamic Input Shapes**: Dynamically adjusts to variable `batch`, `width`, and `height` dimensions for flexible model input.
- **Flexible Data Loading**: The `DataLoader` handles images, folders, videos, and video streams.
- **Real-Time Display and Video Export**: `Viewer` provides real-time frame visualization and video export functions, similar to OpenCV’s `imshow()` and `imwrite()`.
- **Enhanced Annotation and Visualization**: The `Annotator` facilitates comprehensive result rendering, with support for bounding boxes (HBB), oriented bounding boxes (OBB), polygons, masks, keypoints, and text labels.

## Setup Instructions

### 1. ONNXRuntime Linking

<details>
<summary>You have two options to link the ONNXRuntime library:</summary>

- **Option 1: Manual Linking**

  - For detailed setup, consult the [ONNX Runtime linking documentation](https://ort.pyke.io/setup/linking).
  - **Linux or macOS**:
    1. Download the ONNX Runtime package from the [Releases page](https://github.com/microsoft/onnxruntime/releases).
    2. Set up the library path by exporting the `ORT_DYLIB_PATH` environment variable:
       ```shell
       export ORT_DYLIB_PATH=/path/to/onnxruntime/lib/libonnxruntime.so.1.19.0
       ```

- **Option 2: Automatic Download**
  - Use the `--features auto` flag to handle downloading automatically:
    ```shell
    cargo run -r --example yolo --features auto
    ```

</details>

### 2. \[Optional\] Install CUDA, CuDNN, and TensorRT

- The CUDA execution provider requires CUDA version `12.x`.
- The TensorRT execution provider requires both CUDA `12.x` and TensorRT `10.x`.

### 3. \[Optional\] Install ffmpeg

To view video frames and save video inferences, install `rust-ffmpeg`. For instructions, see:  
[https://github.com/zmwangx/rust-ffmpeg/wiki/Notes-on-building#dependencies](https://github.com/zmwangx/rust-ffmpeg/wiki/Notes-on-building#dependencies)

## Get Started

```Shell
# customized
cargo run -r -- --task detect --ver v8 --nc 6 --model xxx.onnx  # YOLOv8

# Classify
cargo run -r -- --task classify --ver v5 --scale s --width 224 --height 224 --nc 1000  # YOLOv5
cargo run -r -- --task classify --ver v8 --scale n --width 224 --height 224 --nc 1000  # YOLOv8
cargo run -r -- --task classify --ver v11 --scale n --width 224 --height 224 --nc 1000  # YOLO11

# Detect
cargo run -r -- --task detect --ver v5 --scale n  # YOLOv5
cargo run -r -- --task detect --ver v6 --scale n  # YOLOv6
cargo run -r -- --task detect --ver v7 --scale t  # YOLOv7
cargo run -r -- --task detect --ver v8 --scale n  # YOLOv8
cargo run -r -- --task detect --ver v9 --scale t  # YOLOv9
cargo run -r -- --task detect --ver v10 --scale n  # YOLOv10
cargo run -r -- --task detect --ver v11 --scale n  # YOLO11
cargo run -r -- --task detect --ver rtdetr --scale l  # RTDETR

# Pose
cargo run -r -- --task pose --ver v8 --scale n   # YOLOv8-Pose
cargo run -r -- --task pose --ver v11 --scale n  # YOLO11-Pose

# Segment
cargo run -r -- --task segment --ver v5 --scale n  # YOLOv5-Segment
cargo run -r -- --task segment --ver v8 --scale n  # YOLOv8-Segment
cargo run -r -- --task segment --ver v11 --scale n  # YOLOv8-Segment
cargo run -r -- --task segment --ver v8 --model yolo/FastSAM-s-dyn-f16.onnx  # FastSAM

# OBB
cargo run -r -- --ver v8 --task obb --scale n --width 1024 --height 1024 --source images/dota.png  # YOLOv8-Obb
cargo run -r -- --ver v11 --task obb --scale n --width 1024 --height 1024 --source images/dota.png  # YOLO11-Obb
```

**`cargo run -- --help` for more options**

For more details, please refer to [usls-yolo](https://github.com/jamjamjon/usls/tree/main/examples/yolo).
