---
comments: true
description: Ultralytics Inference for Rust is a high-performance YOLO inference library and CLI built on ONNX Runtime, with GPU acceleration and zero Python runtime.
keywords: Ultralytics Inference, Rust, YOLO, ONNX Runtime, object detection, segmentation, pose, OBB, classification, semantic segmentation, CUDA, TensorRT, CoreML, edge AI, real-time inference
---

# Ultralytics Inference for Rust

[![GitHub](https://img.shields.io/badge/GitHub-ultralytics%2Finference-181717?logo=github&logoColor=white)](https://github.com/ultralytics/inference)
[![Crates.io](https://img.shields.io/crates/v/ultralytics-inference?logo=rust&logoColor=white&label=crates.io&color=CE422B)](https://crates.io/crates/ultralytics-inference)
[![docs.rs](https://img.shields.io/docsrs/ultralytics-inference?logo=docs.rs&logoColor=white&label=docs.rs)](https://docs.rs/ultralytics-inference)
[![Downloads](https://img.shields.io/crates/d/ultralytics-inference?logo=rust&logoColor=white&label=downloads&color=CE422B)](https://crates.io/crates/ultralytics-inference)
[![MSRV](https://img.shields.io/crates/msrv/ultralytics-inference?logo=rust&logoColor=white&color=CE422B)](https://crates.io/crates/ultralytics-inference)

[Ultralytics Inference](https://github.com/ultralytics/inference) is a high-performance [YOLO](https://www.ultralytics.com/yolo) inference library and command-line tool written in [Rust](https://rust-lang.org/). It runs exported [ONNX](../integrations/onnx.md) models through [ONNX Runtime](https://onnxruntime.ai/) to deliver fast, memory-safe predictions on images, videos, webcams, and streams, with no Python runtime required at inference time.

The project ships as a single crate, `ultralytics-inference`, that you can use two ways: as a **CLI** for quick predictions and batch jobs, or as a **library** embedded directly in your Rust application. It supports every Ultralytics [task](../tasks/index.md) and a broad set of hardware backends through a uniform device interface.

## Why Rust inference?

- **Native speed and a small footprint.** Compiles to a native binary with no interpreter, ideal for servers, containers, and [edge devices](https://www.ultralytics.com/glossary/edge-ai).
- **Memory safety.** Rust's ownership model removes whole classes of runtime errors without a garbage collector.
- **All YOLO tasks.** Detect, segment, pose, OBB, classify, and semantic segmentation from one API.
- **Broad hardware support.** CPU plus CUDA, TensorRT, CoreML, OpenVINO, DirectML, ROCm, and XNNPACK execution providers selected at build time.
- **GPU-side preprocessing.** An optional fused CUDA kernel keeps letterbox, normalize, and layout conversion on the device for a zero-copy input path.
- **Auto-download.** Known YOLO model names and sample assets download automatically on first use.

!!! tip "Looking for the Python package?"

    This page covers the standalone Rust crate. For the Python workflow (training, validation, export, and prediction) see the main [Quickstart](../quickstart.md) and [Predict mode](../modes/predict.md). Export any Ultralytics model to ONNX with the [ONNX integration](../integrations/onnx.md), then run it here.

## Installation

Rust 1.89 or newer is required. The [video](#cargo-features) feature additionally needs FFmpeg 7+ installed on the system.

=== "CLI"

    ```bash
    # Install the command-line tool from crates.io
    cargo install ultralytics-inference

    # Or with GPU support compiled in
    cargo install ultralytics-inference --features cuda,tensorrt
    ```

    The binary is placed at `~/.cargo/bin/ultralytics-inference` (Linux and macOS) or `%USERPROFILE%\.cargo\bin\` on Windows.

=== "Library"

    ```bash
    # Add the crate to your project
    cargo add ultralytics-inference
    ```

    ```toml
    # Or add it manually to Cargo.toml
    [dependencies]
    ultralytics-inference = "0.0.27"
    ```

## CLI quickstart

The CLI exposes a `predict` subcommand. With no arguments it downloads a nano detection model and sample images, runs inference, and saves the annotated results to `runs/detect/predict`.

```bash
# Detect on the built-in samples (downloads model and images)
ultralytics-inference predict

# Detect on your own image
ultralytics-inference predict --model yolo26n.onnx --source image.jpg

# Segmentation (auto-downloads yolo26n-seg.onnx)
ultralytics-inference predict --task segment --source image.jpg

# Pose on a video, shown live in a window
ultralytics-inference predict --task pose --source video.mp4 --show

# Tune thresholds and filter to specific classes
ultralytics-inference predict --source image.jpg --conf 0.5 --iou 0.45 --classes "0,1,2"

# Run a whole folder on the GPU in half precision
ultralytics-inference predict --source images/ --device cuda:0 --half
```

Common flags:

| Flag             | Default        | Description                                                           |
| ---------------- | -------------- | --------------------------------------------------------------------- |
| `--model`, `-m`  | `yolo26n.onnx` | Path to an ONNX model; a known YOLO name is downloaded automatically. |
| `--task`         | `detect`       | One of `detect`, `segment`, `pose`, `obb`, `classify`, `semantic`.    |
| `--source`, `-s` | sample         | Image, directory, glob, video, webcam index, or URL.                  |
| `--conf`         | `0.25`         | Confidence threshold.                                                 |
| `--iou`          | `0.7`          | IoU threshold for non-maximum suppression.                            |
| `--imgsz`        | model metadata | Inference image size.                                                 |
| `--device`       | `cpu`          | Execution device, for example `cuda:0`, `coreml`, `tensorrt:0`.       |
| `--half`         | `false`        | FP16 half-precision inference.                                        |
| `--save`         | `true`         | Save annotated results to `runs/<task>/predict`.                      |
| `--show`         | `false`        | Display results in a window.                                          |
| `--classes`      | all            | Filter detections by class IDs, for example `"0,1,2"`.                |

## Library quickstart

Load a model and run a prediction. Model metadata such as class names, task type, and image size is read automatically from the ONNX file.

```rust
use ultralytics_inference::YOLOModel;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Metadata (classes, task, imgsz) is parsed from the model.
    let mut model = YOLOModel::load("yolo26n.onnx")?;

    let results = model.predict("image.jpg")?;

    for result in &results {
        if let Some(boxes) = &result.boxes {
            for i in 0..boxes.len() {
                let class_id = boxes.cls()[i] as usize;
                let conf = boxes.conf()[i];
                let name = result.names.get(&class_id).map_or("unknown", |s| s.as_str());
                println!("{name} {conf:.2}");
            }
        }
    }

    Ok(())
}
```

Use `InferenceConfig` to control thresholds, image size, precision, and device with a builder API:

```rust
use ultralytics_inference::{Device, InferenceConfig, YOLOModel};

let config = InferenceConfig::new()
    .with_confidence(0.5)
    .with_iou(0.45)
    .with_imgsz(640, 640)
    .with_device(Device::Cuda(0))
    .with_half(true);

let mut model = YOLOModel::load_with_config("yolo26n.onnx", config)?;
let results = model.predict("image.jpg")?;
```

Each task populates a different field on `Results`. Each tab below is a complete, runnable program; the model and sample inputs download automatically on first run. Swap `predict_default()` for `predict("image.jpg")` to run on your own files.

=== "Detect"

    ```rust
    use ultralytics_inference::YOLOModel;

    fn main() -> Result<(), Box<dyn std::error::Error>> {
        let mut model = YOLOModel::load("yolo26n.onnx")?;
        let results = model.predict_default()?;

        for result in &results {
            if let Some(boxes) = &result.boxes {
                println!("{} detections", boxes.len());
                let xyxy = boxes.xyxy(); // rows of [x1, y1, x2, y2]
                for i in 0..boxes.len() {
                    let class_id = boxes.cls()[i] as usize;
                    let name = result.names.get(&class_id).map_or("unknown", |s| s.as_str());
                    println!("  {name} {:.2} {:?}", boxes.conf()[i], xyxy.row(i).to_vec());
                }
            }
        }

        Ok(())
    }
    ```

=== "Segment"

    ```rust
    use ultralytics_inference::YOLOModel;

    fn main() -> Result<(), Box<dyn std::error::Error>> {
        let mut model = YOLOModel::load("yolo26n-seg.onnx")?;
        let results = model.predict_default()?;

        for result in &results {
            if let Some(masks) = &result.masks {
                let (n, h, w) = masks.data.dim(); // mask data shape (N, H, W)
                println!("{n} instance masks ({h}x{w})");
            }
            if let Some(boxes) = &result.boxes {
                for i in 0..boxes.len() {
                    let class_id = boxes.cls()[i] as usize;
                    let name = result.names.get(&class_id).map_or("unknown", |s| s.as_str());
                    println!("  {name} {:.2}", boxes.conf()[i]);
                }
            }
        }

        Ok(())
    }
    ```

=== "Pose"

    ```rust
    use ultralytics_inference::YOLOModel;

    fn main() -> Result<(), Box<dyn std::error::Error>> {
        let mut model = YOLOModel::load("yolo26n-pose.onnx")?;
        let results = model.predict_default()?;

        for result in &results {
            if let Some(kpts) = &result.keypoints {
                let (n, k, _) = kpts.xy().dim(); // keypoint coords shape (N, K, 2)
                println!("{n} pose(s), {k} keypoints each");

                // Optional per-keypoint confidence, shape (N, K)
                if let Some(conf) = kpts.conf() {
                    println!("  keypoint confidence values: {}", conf.len());
                }
            }
        }

        Ok(())
    }
    ```

=== "Classify"

    ```rust
    use ultralytics_inference::YOLOModel;

    fn main() -> Result<(), Box<dyn std::error::Error>> {
        let mut model = YOLOModel::load("yolo26n-cls.onnx")?;
        let results = model.predict_default()?;

        for result in &results {
            if let Some(probs) = &result.probs {
                let top1 = probs.top1();
                let name = result.names.get(&top1).map_or("unknown", |s| s.as_str());
                println!("top-1: {name} ({:.2})", probs.top1conf());

                for (id, conf) in probs.top5().into_iter().zip(probs.top5conf()) {
                    let name = result.names.get(&id).map_or("unknown", |s| s.as_str());
                    println!("  {name} {conf:.2}");
                }
            }
        }

        Ok(())
    }
    ```

=== "OBB"

    ```rust
    use ultralytics_inference::YOLOModel;

    fn main() -> Result<(), Box<dyn std::error::Error>> {
        let mut model = YOLOModel::load("yolo26n-obb.onnx")?;
        let results = model.predict_default()?;

        for result in &results {
            if let Some(obb) = &result.obb {
                println!("{} oriented boxes", obb.len());
                let xywhr = obb.xywhr(); // rows of [cx, cy, w, h, angle]
                for i in 0..obb.len() {
                    let class_id = obb.cls()[i] as usize;
                    let name = result.names.get(&class_id).map_or("unknown", |s| s.as_str());
                    println!("  {name} {:.2} {:?}", obb.conf()[i], xywhr.row(i).to_vec());
                }
            }
        }

        Ok(())
    }
    ```

=== "Semantic"

    ```rust
    use ultralytics_inference::YOLOModel;

    fn main() -> Result<(), Box<dyn std::error::Error>> {
        let mut model = YOLOModel::load("yolo26n-sem.onnx")?;
        let results = model.predict_default()?;

        for result in &results {
            if let Some(sem) = &result.semantic_mask {
                let (h, w) = sem.data.dim(); // per-pixel class map shape (H, W)
                println!("class map {h}x{w}");

                for class_id in sem.class_ids() {
                    let name = result.names.get(&class_id).map_or("unknown", |s| s.as_str());
                    println!("  present: {name}");
                }
            }
        }

        Ok(())
    }
    ```

## Supported tasks

All Ultralytics [tasks](../tasks/index.md) are supported. When `--model` is omitted, the matching nano model for the selected task is downloaded automatically.

| Task                  | `--task`   | Output                        | Default model       |
| --------------------- | ---------- | ----------------------------- | ------------------- |
| Detection             | `detect`   | Bounding boxes and classes    | `yolo26n.onnx`      |
| Instance segmentation | `segment`  | Boxes plus per-instance masks | `yolo26n-seg.onnx`  |
| Pose                  | `pose`     | Boxes plus keypoints          | `yolo26n-pose.onnx` |
| Oriented boxes        | `obb`      | Rotated bounding boxes        | `yolo26n-obb.onnx`  |
| Classification        | `classify` | Class probabilities           | `yolo26n-cls.onnx`  |
| Semantic segmentation | `semantic` | Per-pixel class map           | `yolo26n-sem.onnx`  |

## Model compatibility

Any Ultralytics model exported to ONNX can be loaded from a local file. Auto-download is available for standard YOLO26, YOLO11, and YOLOv8 model names in sizes `n`, `s`, `m`, `l`, and `x`:

| Model family | Auto-downloadable variants                                            |
| ------------ | --------------------------------------------------------------------- |
| YOLO26       | `yolo26{n,s,m,l,x}.onnx`, `-seg`, `-pose`, `-obb`, `-cls`, and `-sem` |
| YOLO11       | `yolo11{n,s,m,l,x}.onnx`, `-seg`, `-pose`, `-obb`, and `-cls`         |
| YOLOv8       | `yolov8{n,s,m,l,x}.onnx`, `-seg`, `-pose`, `-obb`, and `-cls`         |

Semantic segmentation (`-sem`) is YOLO26-only.

## Input sources

The `--source` argument (and the `Source` type in the library) accepts many input kinds, auto-detected from the string:

| Source    | Example                         | Notes                         |
| --------- | ------------------------------- | ----------------------------- |
| Image     | `image.jpg`                     | Single file.                  |
| Directory | `images/`                       | All images in the folder.     |
| Glob      | `images/*.jpg`                  | Shell-style pattern.          |
| Video     | `video.mp4`                     | Requires the `video` feature. |
| Webcam    | `0`                             | Requires the `video` feature. |
| Stream    | `rtsp://...`                    | Requires the `video` feature. |
| URL       | `https://example.com/image.jpg` | Remote image download.        |

## Devices and execution providers

Inference runs on CPU by default. GPU and accelerator backends are compiled in as [Cargo features](#cargo-features) and selected at runtime with `--device` (CLI) or `Device` (library).

| Device string | `Device` variant      | Build feature | Hardware              |
| ------------- | --------------------- | ------------- | --------------------- |
| `cpu`         | `Device::Cpu`         | built in      | Any CPU               |
| `cuda:0`      | `Device::Cuda(0)`     | `cuda`        | NVIDIA GPU            |
| `tensorrt:0`  | `Device::TensorRt(0)` | `tensorrt`    | NVIDIA GPU, optimized |
| `coreml`      | `Device::CoreMl`      | `coreml`      | Apple Silicon / macOS |
| `openvino`    | `Device::OpenVino`    | `openvino`    | Intel CPU / iGPU      |
| `directml:0`  | `Device::DirectMl(0)` | `directml`    | Windows GPU           |
| `rocm:0`      | `Device::Rocm(0)`     | `rocm`        | AMD GPU               |
| `xnnpack`     | `Device::Xnnpack`     | `xnnpack`     | Optimized CPU         |

```bash
# Build the CLI with the providers you need
cargo install ultralytics-inference --features cuda,tensorrt
```

## GPU acceleration and CUDA preprocessing

On NVIDIA hardware, the `cuda` feature enables the CUDA execution provider, and `tensorrt` adds the TensorRT provider for further optimization. For the lowest possible latency, the `cuda-preprocess` feature moves preprocessing onto the GPU.

`cuda-preprocess` runs letterbox resizing, normalization, and the HWC-to-CHW layout conversion as a single fused CUDA kernel, then feeds the result to the model as a zero-copy device tensor. This removes the per-image CPU preprocessing cost and the host-to-device copy, which matters most for high-throughput batches and real-time streams.

```bash
# Build with fused GPU preprocessing (implies cuda + tensorrt)
cargo build --release --features cuda-preprocess
```

The fast path is used automatically, with no API change, when all of the following hold: the feature is compiled in, the device is CUDA or TensorRT, the task is detect, segment, pose, OBB, or semantic segmentation, and the model uses FP32 input. It is enabled by default and can be turned off per model:

```rust
use ultralytics_inference::{Device, InferenceConfig};

let config = InferenceConfig::new()
    .with_device(Device::TensorRt(0))
    .with_cuda_preprocess(false); // force CPU preprocessing
```

!!! note "Match your CUDA toolkit"

    `cuda-preprocess` requires a matching CUDA toolkit at build time and uses NVRTC at runtime for the fused preprocessing kernel. See the [CUDA and TensorRT acceleration guide](https://docs.rs/ultralytics-inference/latest/ultralytics_inference/cuda_guide/index.html) for version requirements and troubleshooting.

## Cargo features

Features are enabled at build time. The defaults cover annotation and live display.

| Feature           | Default | Purpose                                                                |
| ----------------- | ------- | ---------------------------------------------------------------------- |
| `annotate`        | yes     | Draw boxes, masks, keypoints, and labels; required for `--save`.       |
| `visualize`       | yes     | Real-time window display for `--show`.                                 |
| `video`           | no      | Read and write video files (requires FFmpeg 7+).                       |
| `cuda`            | no      | NVIDIA CUDA execution provider.                                        |
| `tensorrt`        | no      | NVIDIA TensorRT execution provider.                                    |
| `cuda-preprocess` | no      | Fused GPU preprocessing with zero-copy input (implies cuda, tensorrt). |
| `coreml`          | no      | Apple CoreML execution provider.                                       |
| `openvino`        | no      | Intel OpenVINO execution provider.                                     |
| `rocm`            | no      | AMD ROCm execution provider.                                           |
| `directml`        | no      | Windows DirectML execution provider.                                   |

Convenience groups bundle related providers: `nvidia` (cuda, tensorrt), `amd` (rocm, migraphx), `intel` (openvino, onednn), `mobile` (nnapi, coreml, qnn), and `all` (annotate, visualize, video). Additional providers such as `nnapi`, `qnn`, `xnnpack`, `webgpu`, and others are also available.

Enable features when installing the CLI or adding the library:

```bash
cargo install ultralytics-inference --features video
cargo install ultralytics-inference --features cuda,tensorrt
```

```toml
[dependencies]
ultralytics-inference = { version = "0.0.27", features = ["video"] }
```

## Output and saving

By default, predictions are annotated and saved to an auto-incrementing run directory:

```text
runs/
└── detect/
    └── predict/          # then predict2, predict3, ...
        └── image.jpg     # annotated result
```

The subfolder matches the task (`runs/segment/`, `runs/pose/`, and so on). For video sources the annotated output is written as a video file; pass `--save-frames` to write individual frames instead. For the `semantic` task, `--save-json` writes per-pixel class-map PNGs under a `results/` subfolder. Annotated image and video saving require the `annotate` feature; semantic class-map PNG export does not. Video input and output require the `video` feature.

## FAQ

### Do I need Python installed?

No. The crate runs exported ONNX models directly through ONNX Runtime. Python is only needed if you train or [export](../modes/export.md) models with the Ultralytics package beforehand.

### Which models can I run?

Any Ultralytics YOLO model exported to ONNX, including [YOLO26](../models/yolo26.md), [YOLO11](../models/yolo11.md), and [YOLOv8](../models/yolov8.md). Known model names download automatically; you can also point `--model` at any local `.onnx` file.

### How do I get a model file?

Export from the Python package, for example with the [ONNX integration](../integrations/onnx.md), or let the CLI download a standard nano model for the chosen task on first run.

### Is video supported?

Yes, with the `video` feature enabled and FFmpeg 7+ installed on the system. This covers video files, webcams, and RTSP/RTMP/HTTP streams.

### What do the `annotate` and `visualize` features do?

Both are enabled by default. `annotate` draws boxes, masks, keypoints, and class labels onto the image and is required for `--save` to write annotated results. `visualize` opens a live window for `--show`. For a smaller, headless build that only returns results programmatically, disable them with `cargo build --no-default-features` (add back individual features as needed).

### Where is the full API reference?

This page is a high-level overview. The complete, type-by-type API reference for every public struct, method, and configuration option is published on [docs.rs](https://docs.rs/ultralytics-inference/latest/ultralytics_inference/), generated directly from the source.
