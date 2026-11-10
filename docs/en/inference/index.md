---
comments: true
description: Ultralytics Inference for Rust is a high-performance YOLO inference library and CLI built on ONNX Runtime, with GPU acceleration, tiled inference, scene text OCR and no Python runtime.
keywords: Ultralytics Inference, Rust, YOLO, ONNX Runtime, object detection, segmentation, pose, OBB, classification, semantic segmentation, depth estimation, OCR, tiled inference, CUDA, TensorRT, CoreML, OpenVINO, edge AI, real-time inference
---

# Ultralytics Inference for Rust

<div align="center">
    <a href="https://github.com/ultralytics/inference"><img src="https://img.shields.io/badge/GitHub-ultralytics%2Finference-181717?logo=github&logoColor=white" alt="Ultralytics Inference GitHub"></a>
    <a href="https://crates.io/crates/ultralytics-inference"><img src="https://img.shields.io/crates/v/ultralytics-inference?logo=rust&logoColor=white&label=crates.io&color=CE422B" alt="Ultralytics Inference Crates.io"></a>
    <a href="https://docs.rs/ultralytics-inference"><img src="https://img.shields.io/docsrs/ultralytics-inference?logo=docs.rs&logoColor=white&label=docs.rs" alt="Ultralytics Inference docs.rs"></a>
    <a href="https://crates.io/crates/ultralytics-inference"><img src="https://img.shields.io/crates/d/ultralytics-inference?logo=rust&logoColor=white&label=downloads&color=CE422B" alt="Ultralytics Inference Downloads"></a>
    <a href="https://crates.io/crates/ultralytics-inference"><img src="https://img.shields.io/crates/msrv/ultralytics-inference?logo=rust&logoColor=white&color=CE422B" alt="Ultralytics Inference MSRV"></a>
</div>

[Ultralytics Inference](https://github.com/ultralytics/inference) is a high-performance [YOLO](https://www.ultralytics.com/yolo) inference library and command-line tool written in [Rust](https://rust-lang.org/). It runs exported [ONNX](../integrations/onnx.md) models through [ONNX Runtime](https://onnxruntime.ai/) to deliver fast, memory-safe predictions on images, videos, webcams, and streams, with no Python runtime required at inference time.

The project ships as a single crate, `ultralytics-inference`, that you can use two ways: as a **CLI** for quick predictions and batch jobs, or as a **library** embedded directly in your Rust application. It supports every Ultralytics [task](../tasks/index.md) and a broad set of hardware backends through a uniform device interface. For browser inference, the same repository publishes the `@ultralytics/yolo` npm package; see its [setup guide](https://github.com/ultralytics/inference/tree/main/web).

## Why Rust inference?

- **Native speed and a small footprint.** Compiles to a native binary with no interpreter, ideal for servers, containers, and [edge devices](https://www.ultralytics.com/glossary/edge-ai).
- **Memory safety.** Rust's ownership model removes whole classes of runtime errors without a garbage collector.
- **All YOLO tasks.** Detect, segment, semantic segmentation, depth estimation, classify, pose, and OBB from one API, plus scene text OCR and tiled inference for large images.
- **Broad hardware support.** CPU plus CUDA, TensorRT, CoreML, OpenVINO, DirectML, ROCm, MIGraphX, and XNNPACK execution providers selected at build time, and native OpenVINO IR models.
- **GPU-side preprocessing.** An optional fused CUDA kernel runs letterbox, normalize, and layout conversion on the GPU and hands the result to the model without leaving the device.
- **Auto-download.** Known YOLO model names, the OCR models, and sample assets download automatically on first use.
- **Same output as Python.** Masks, mask outlines, and drawn results follow the Ultralytics Python package, so `--save` produces the same image as `plot()`.

!!! tip "Looking for the Python package?"

    This page covers the standalone Rust crate. For the Python workflow (training, validation, export, and prediction) see the main [Quickstart](../quickstart.md) and [Predict mode](../modes/predict.md). Export any Ultralytics model to ONNX with the [ONNX integration](../integrations/onnx.md), then run it here.

## Installation

Rust 1.92 or newer is required. The [video](#cargo-features) feature additionally needs FFmpeg 6 to 9 installed on the system.

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
    ultralytics-inference = "0.0.58"
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

# Depth estimation (auto-downloads yolo26n-depth.onnx)
ultralytics-inference predict --task depth --source image.jpg

# Tune thresholds and filter to specific classes
ultralytics-inference predict --source image.jpg --conf 0.5 --iou 0.45 --classes "0,1,2"

# Run a whole folder on the GPU in half precision
ultralytics-inference predict --source images/ --device cuda:0 --quantize 16

# Tile a large aerial image into overlapping 640 px windows to find small objects
ultralytics-inference predict --model yolo26n-obb.onnx --source aerial.jpg --tile 640

# Read scene text (auto-downloads yolo26n-text.onnx and the PaddleOCR models)
ultralytics-inference predict --ocr --source sign.jpg

# Thicker lines, no confidences in labels
ultralytics-inference predict --source image.jpg --line-width 4 --show-conf false
```

All `predict` flags:

| Flag                     | Default         | Description                                                                                                                                   |
| ------------------------ | --------------- | --------------------------------------------------------------------------------------------------------------------------------------------- |
| `--model`, `-m`          | `yolo26n.onnx`  | Path to an ONNX model or OpenVINO export; a known YOLO name is downloaded automatically.                                                      |
| `--task`                 | `detect`        | One of `detect`, `segment`, `semantic`, `depth`, `classify`, `pose`, `obb`; selects the nano model when `--model` is omitted.                 |
| `--source`, `-s`         | sample          | Image, directory, glob, video, webcam index, stream, or URL.                                                                                  |
| `--conf`                 | `0.25`          | Confidence threshold.                                                                                                                         |
| `--iou`                  | `0.5`           | IoU threshold for non-maximum suppression.                                                                                                    |
| `--max-det`              | `300`           | Maximum number of detections per image.                                                                                                       |
| `--imgsz`                | model metadata  | Inference image size.                                                                                                                         |
| `--rect [<BOOL>]`        | `true`          | Rectangular inference with minimal padding.                                                                                                   |
| `--tile [<TILE>]`        | off             | Tile size for overlapping-window inference; a bare `--tile` uses `--imgsz`. See [tiled inference](#tiled-inference).                          |
| `--retina-masks`         | `false`         | Build segmentation masks at the original image size instead of the inference size.                                                            |
| `--ocr`                  | `false`         | Read the text in each box, oriented box, or mask with PaddleOCR; without `--model` this runs `yolo26n-text.onnx`. See [OCR](#scene-text-ocr). |
| `--batch`                | `1`             | Batch size for inference.                                                                                                                     |
| `--quantize`             | model precision | Inference precision: `8`, `16`, `32`, `int8`, `fp16`, `fp32`, `w8a8`, `w16a16`, `w32a32`, `w8a16`, or `w8a32`.                                |
| `--save [<BOOL>]`        | `true`          | Save annotated results to `runs/<task>/predict`.                                                                                              |
| `--save-frames`          | `false`         | Save individual frames for video input instead of a video file.                                                                               |
| `--save-json`            | `false`         | Save semantic segmentation class-map PNGs.                                                                                                    |
| `--show`                 | `false`         | Display results in a window.                                                                                                                  |
| `--show-labels [<BOOL>]` | `true`          | Show class names in box labels.                                                                                                               |
| `--show-conf [<BOOL>]`   | `true`          | Show confidences in box labels.                                                                                                               |
| `--show-boxes [<BOOL>]`  | `true`          | Draw boxes and their labels.                                                                                                                  |
| `--line-width`           | from image size | Box line width in pixels, which also sizes labels.                                                                                            |
| `--device`               | auto            | Execution device, for example `cpu`, `cuda:0`, `tensorrt:0`, `coreml`, `intel:gpu`, `directml:0`, `rocm:0`, `migraphx:0`, `xnnpack`.          |
| `--verbose [<BOOL>]`     | `true`          | Print per-image results and timing.                                                                                                           |
| `--classes`              | all             | Filter detections by class IDs, for example `0`, `"0,1,2"`, or `"[0, 1, 2]"`.                                                                 |

`[<BOOL>]` marks a flag whose value is optional: passing it bare (`--rect`) sets it to `true`, and `--rect false` turns it off. The legacy `--half` flag is still accepted, maps to `--quantize 16`, and prints the same deprecation warning as the Python package.

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
use ultralytics_inference::{Device, InferenceConfig, Quantization, YOLOModel};

let config = InferenceConfig::new()
    .with_confidence(0.5)
    .with_iou(0.45)
    .with_imgsz(640, 640)
    .with_device(Device::Cuda(0))
    .with_quantize(Quantization::Fp16);

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

                // Mask outlines in pixels, like Ultralytics masks.xy (masks.xyn is normalized)
                for outline in masks.xy() {
                    println!("  outline with {} points", outline.nrows());
                }
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

=== "Depth"

    ```rust
    use ultralytics_inference::YOLOModel;

    fn main() -> Result<(), Box<dyn std::error::Error>> {
        let mut model = YOLOModel::load("yolo26n-depth.onnx")?;
        let results = model.predict_default()?;

        for result in &results {
            if let Some(depth) = &result.depth {
                let (h, w) = depth.data.dim(); // per-pixel depth map shape (H, W), meters
                println!("depth map {h}x{w}");

                // Valid pixels only (depth > 0); None when the map has none.
                if let (Some(min), Some(max)) = (depth.min_depth(), depth.max_depth()) {
                    println!("  range {min:.2}-{max:.2} m");
                }

                // Read the depth of any pixel directly, in meters.
                println!("  center {:.2} m", depth.data[[h / 2, w / 2]]);
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

## Supported tasks

All Ultralytics [tasks](../tasks/index.md) are supported. When `--model` is omitted, the matching nano model for the selected task is downloaded automatically.

| Task                  | `--task`   | Output                        | Default model        |
| --------------------- | ---------- | ----------------------------- | -------------------- |
| Detection             | `detect`   | Bounding boxes and classes    | `yolo26n.onnx`       |
| Instance segmentation | `segment`  | Boxes plus per-instance masks | `yolo26n-seg.onnx`   |
| Semantic segmentation | `semantic` | Per-pixel class map           | `yolo26n-sem.onnx`   |
| Depth estimation      | `depth`    | Per-pixel depth map in meters | `yolo26n-depth.onnx` |
| Classification        | `classify` | Class probabilities           | `yolo26n-cls.onnx`   |
| Pose                  | `pose`     | Boxes plus keypoints          | `yolo26n-pose.onnx`  |
| Oriented boxes        | `obb`      | Rotated bounding boxes        | `yolo26n-obb.onnx`   |

`--ocr` adds scene text reading on top of any detect, segment, or OBB model; without `--model` it runs the `yolo26n-text.onnx` text detector. See [scene text OCR](#scene-text-ocr).

## Model compatibility

Any Ultralytics model exported to ONNX can be loaded from a local file. Auto-download is available for the standard [YOLO26](https://platform.ultralytics.com/ultralytics/yolo26), [YOLO11](https://platform.ultralytics.com/ultralytics/yolo11), and [YOLOv8](https://platform.ultralytics.com/ultralytics/yolov8) model names in sizes `n`, `s`, `m`, `l`, and `x`:

| Model family | Auto-downloadable variants                                                               |
| ------------ | ---------------------------------------------------------------------------------------- |
| YOLO26       | `yolo26{n,s,m,l,x}.onnx`, `-seg`, `-sem`, `-depth`, `-cls`, `-pose`, `-obb`, and `-text` |
| YOLO11       | `yolo11{n,s,m,l,x}.onnx`, `-seg`, `-cls`, `-pose`, and `-obb`                            |
| YOLOv8       | `yolov8{n,s,m,l,x}.onnx`, `-seg`, `-cls`, `-pose`, and `-obb`                            |

Semantic segmentation (`-sem`), depth estimation (`-depth`), and the `-text` scene text detectors are YOLO26-only. The PaddleOCR `PP-OCRv6_medium_rec.onnx` recognizer and `PP-LCNet_x0_25_textline_ori.onnx` orientation classifier used by `--ocr` (Apache-2.0) also download automatically. [RT-DETR](../models/rtdetr.md) detection models also run, but are not auto-downloaded: export one to ONNX first. Native [OpenVINO](../integrations/openvino.md) IR exports run with the `openvino-native` feature, see [OpenVINO IR models](#openvino-ir-models).

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

## Tiled inference

Large images with small objects, such as aerial or satellite photos, can be split into overlapping windows and merged back into one result, the same as the Python `tile` argument. Windows overlap by 20%, boxes are mapped back to the source image, and duplicates are removed per class. With a dynamic batch export, `--batch` runs several windows in one inference call. Every task is supported.

```bash
# 640 px windows; a bare --tile uses --imgsz
ultralytics-inference predict --model yolo26n-obb.onnx --source aerial.jpg --tile 640
```

```rust
use ultralytics_inference::{InferenceConfig, YOLOModel};

// with_tile(0) uses the inference size
let config = InferenceConfig::new().with_tile(640);
let mut model = YOLOModel::load_with_config("yolo26n.onnx", config)?;
let results = model.predict("aerial.jpg")?;
```

## Scene text OCR

YOLO26-Text detectors find words as oriented boxes, and `TextRecognizer` reads them with PaddleOCR PP-OCRv6, the same as the Python [OCR solution](../guides/ocr.md) with `recognizer="paddleocr"`. Boxes of detection models and masks of segmentation models work too: a mask is read from the minimum-area rectangle around its outline and keeps the outline as its polygon. Each crop is turned upright, text the orientation classifier finds upside down is read again turned around, and the texts come back in reading order. The recognizer runs on the detector's device.

```bash
# Auto-downloads yolo26n-text.onnx and the PaddleOCR models
ultralytics-inference predict --ocr --source sign.jpg

# Any size of the text detector, on the GPU
ultralytics-inference predict --ocr --model yolo26x-text.onnx --source sign.jpg --device cuda:0
```

```rust
use ultralytics_inference::YOLOModel;
use ultralytics_inference::ocr::{DEFAULT_ORIENTATION, DEFAULT_RECOGNIZER, TextRecognizer};

let mut detector = YOLOModel::load("yolo26n-text.onnx")?;
let mut ocr = TextRecognizer::new(&detector, DEFAULT_RECOGNIZER, DEFAULT_ORIENTATION)?;
let image = image::open("sign.jpg")?;
let result = detector.predict_image(&image, "sign.jpg".into())?.remove(0);
for text in ocr.read(&image, &result)? {
    println!("{} {:.2} {:?}", text.text, text.confidence, text.polygon);
}
```

`predict --ocr` draws each text like the Python solution, and `--line-width`, `--show-labels`, and `--show-conf` apply to it too.

## Devices and execution providers

GPU and accelerator backends are compiled in as [Cargo features](#cargo-features) and selected at runtime with `--device` (CLI) or `Device` (library). Without a device, the providers compiled into the build are registered in a fixed preference order (TensorRT first, then CUDA, and so on), so a build with only the default features runs on CPU.

| Device string | `Device` variant      | Build feature                   | Hardware              |
| ------------- | --------------------- | ------------------------------- | --------------------- |
| `cpu`         | `Device::Cpu`         | built in                        | Any CPU               |
| `cuda:0`      | `Device::Cuda(0)`     | `cuda`                          | NVIDIA GPU            |
| `tensorrt:0`  | `Device::TensorRt(0)` | `tensorrt`                      | NVIDIA GPU, optimized |
| `coreml`      | `Device::CoreMl`      | `coreml`                        | Apple Silicon / macOS |
| `intel:cpu`   | `Device::IntelCpu`    | `openvino` or `openvino-native` | Intel CPU             |
| `intel:gpu`   | `Device::IntelGpu`    | `openvino` or `openvino-native` | Intel GPU             |
| `intel:npu`   | `Device::IntelNpu`    | `openvino` or `openvino-native` | Intel NPU             |
| `directml:0`  | `Device::DirectMl(0)` | `directml`                      | Windows GPU           |
| `rocm:0`      | `Device::Rocm(0)`     | `rocm`                          | AMD GPU               |
| `migraphx:0`  | `Device::MiGraphX(0)` | `migraphx`                      | AMD GPU (Linux x64)   |
| `xnnpack`     | `Device::Xnnpack`     | `xnnpack`                       | Optimized CPU         |

```bash
# Build the CLI with the providers you need
cargo install ultralytics-inference --features cuda,tensorrt
```

### OpenVINO IR models

With the `openvino-native` feature, `--model` also takes a native [OpenVINO](../integrations/openvino.md) export, the `.xml` file or the folder from `yolo export format=openvino`, and runs it through the OpenVINO runtime instead of ONNX Runtime. Every task runs on `intel:cpu`, `intel:gpu`, and `intel:npu`, and the library loads it with `YOLOModel::load("yolo26n_openvino_model")`. Building needs no OpenVINO install; the runtime is loaded when the first model opens. See the [OpenVINO guide](https://docs.rs/ultralytics-inference/latest/ultralytics_inference/openvino_guide/index.html) for setup.

```bash
yolo export model=yolo26n.pt format=openvino # add quantize=16, quantize=8, dynamic=True, or batch=4
ultralytics-inference predict --model yolo26n_openvino_model --source bus.jpg --device intel:gpu
```

## GPU acceleration and CUDA preprocessing

On NVIDIA hardware, the `cuda` feature enables the CUDA execution provider, and `tensorrt` adds the TensorRT provider for further optimization (it includes `cuda`, which also runs the OCR recognizer). For the lowest possible [inference latency](https://www.ultralytics.com/glossary/inference-latency), the `cuda-preprocess` feature moves preprocessing onto the GPU.

`cuda-preprocess` runs letterbox resizing, normalization, and the HWC-to-CHW layout conversion as a single fused CUDA kernel, then feeds the result to the model as a zero-copy device tensor. The raw 8-bit frame is still uploaded to the GPU, through a pinned staging buffer on discrete GPUs; what the kernel removes is CPU resizing, normalization, and layout conversion, and the separate upload of the preprocessed tensor, which matters most for high-throughput batches and real-time streams.

```bash
# Build with fused GPU preprocessing (implies cuda + tensorrt)
cargo build --release --features cuda-preprocess
```

The fast path is used automatically, with no API change, when all of the following hold: the feature is compiled in, the device is CUDA, TensorRT, or not set, the task is detect, segment, semantic segmentation (single images only), depth estimation, pose, or OBB, and the model uses FP32 input. It is enabled by default and can be turned off per model:

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

| Feature           | Default | Purpose                                                                     |
| ----------------- | ------- | --------------------------------------------------------------------------- |
| `annotate`        | yes     | Draw boxes, masks, keypoints, labels, and OCR texts; required for `--save`. |
| `visualize`       | yes     | Real-time window display for `--show`.                                      |
| `video`           | no      | Read and write video files (requires FFmpeg 6 to 9).                        |
| `cuda`            | no      | NVIDIA CUDA execution provider.                                             |
| `tensorrt`        | no      | NVIDIA TensorRT execution provider (includes `cuda`).                       |
| `cuda-preprocess` | no      | Fused GPU preprocessing (implies cuda, tensorrt; needs the CUDA toolkit).   |
| `coreml`          | no      | Apple CoreML execution provider.                                            |
| `openvino`        | no      | Intel OpenVINO execution provider (Linux and Windows x64).                  |
| `openvino-native` | no      | Native OpenVINO IR (`.xml`) models through the OpenVINO runtime.            |
| `rocm`            | no      | AMD ROCm execution provider (needs your own ONNX Runtime build).            |
| `migraphx`        | no      | AMD MIGraphX execution provider (Linux x64).                                |
| `directml`        | no      | Windows DirectML execution provider.                                        |
| `xnnpack`         | no      | XNNPACK execution provider for optimized CPU inference.                     |

Convenience groups bundle related providers: `nvidia` (cuda, tensorrt), `amd` (rocm, migraphx), `intel` (openvino, onednn), `mobile` (nnapi, coreml, qnn), and `all` (annotate, visualize, video). Additional providers `onednn`, `nnapi`, `qnn`, `acl`, `tvm`, `rknpu`, `cann`, `webgpu`, and `azure` are also available.

Enable features when installing the CLI or adding the library:

```bash
cargo install ultralytics-inference --features video
cargo install ultralytics-inference --features cuda,tensorrt
```

```toml
[dependencies]
ultralytics-inference = { version = "0.0.58", features = ["video"] }
```

## Output and saving

By default, predictions are annotated and saved to an auto-incrementing run directory:

```text
runs/
└── detect/
    └── predict/          # then predict2, predict3, ...
        └── image.jpg     # annotated result
```

The subfolder matches the task (`runs/segment/`, `runs/pose/`, and so on). For video sources the annotated output is written as a video file; pass `--save-frames` to write individual frames instead. For the `semantic` task, `--save-json` writes per-pixel class-map PNGs under a `results/` subfolder. For the `depth` task the colorized depth map is blended over the source image, matching the Ultralytics Python `plot()` output. Annotated image and video saving require the `annotate` feature; semantic class-map PNG export does not. Video input and output require the `video` feature.

Drawing follows the Python `Results.plot()`: the line width defaults to 0.3% of the mean image side (at least 2 px), labels sit above each box or inside it when there is no room, and instance masks are blended the same way, so saved images match Python. `--line-width`, `--show-labels`, `--show-conf`, and `--show-boxes` work like the Python arguments of the same name. In the library, pass them through `AnnotateOptions`:

```rust
use ultralytics_inference::annotate::{AnnotateOptions, annotate_image_with_options};

let options = AnnotateOptions {
    line_width: Some(3),
    show_conf: false,
    ..Default::default()
};
let annotated = annotate_image_with_options(&image, &results[0], &options);
```

Segmentation masks are built at the inference size like Python; pass `--retina-masks` (or `InferenceConfig::with_retina_masks(true)`) to build them at the original image size.

## FAQ

### Do I need Python installed?

No. The crate runs exported ONNX models directly through ONNX Runtime. Python is only needed if you train or [export](../modes/export.md) models with the Ultralytics package beforehand.

### Which models can I run?

Any Ultralytics YOLO model exported to ONNX, including [YOLO26](../models/yolo26.md), [YOLO11](../models/yolo11.md), and [YOLOv8](../models/yolov8.md). Known model names download automatically; you can also point `--model` at any local `.onnx` file.

### How do I get a model file?

Export from the Python package, for example with the [ONNX integration](../integrations/onnx.md), or let the CLI download a standard nano model for the chosen task on first run.

### Can it read text in images?

Yes. `--ocr` reads scene text with a YOLO26-Text detector and PaddleOCR, like the Python [OCR solution](../guides/ocr.md), and also works on the boxes or masks of your own detect and segment models. See [scene text OCR](#scene-text-ocr).

### How do I find small objects in large images?

Use [tiled inference](#tiled-inference): `--tile 640` splits the image into overlapping 640 px windows and merges the detections back into one result.

### Is video supported?

Yes, with the `video` feature enabled and FFmpeg 6 to 9 installed on the system. This covers video files, webcams, and RTSP/RTMP/HTTP streams.

### What do the `annotate` and `visualize` features do?

Both are enabled by default. `annotate` draws boxes, masks, keypoints, and class labels onto the image and is required for `--save` to write annotated results. `visualize` opens a live window for `--show`. For a smaller, headless build that only returns results programmatically, disable them with `cargo build --no-default-features` (add back individual features as needed).

### Where is the full API reference?

This page is a high-level overview. The complete, type-by-type API reference for every public struct, method, and configuration option is published on [docs.rs](https://docs.rs/ultralytics-inference/latest/ultralytics_inference/), generated directly from the source.
