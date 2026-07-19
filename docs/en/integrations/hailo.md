---
comments: true
description: Export Ultralytics YOLO detection, segmentation, pose, OBB, classification, and semantic segmentation models directly to Hailo HEF for computer vision and edge AI.
keywords: Hailo export, Hailo HEF, export YOLO to Hailo, YOLO Hailo, Hailo-8, Hailo-8L, Hailo-10, Hailo-15, Raspberry Pi AI Kit, Raspberry Pi AI HAT+, Hailo Dataflow Compiler, Hailo DFC, HailoRT, Hailo AI accelerator, edge AI, embedded AI, computer vision, object detection, instance segmentation, pose estimation, oriented bounding box, image classification, model quantization, INT8 quantization, Ultralytics YOLO, YOLO26, YOLO11, YOLOv8
---

# Hailo Export for Ultralytics YOLO Models

Hailo AI accelerators run compiled Hailo Executable Format (HEF) models on edge devices such as the [Raspberry Pi AI Kit](https://www.raspberrypi.com/products/ai-kit/) and [AI HAT+](https://www.raspberrypi.com/documentation/accessories/ai-hat-plus.html). Ultralytics exports YOLO detection, segmentation, pose, OBB, classification, and semantic segmentation models directly to HEF with the Hailo Dataflow Compiler (DFC).

Hailo deployment is designed for computer vision at the edge: cameras, robots, industrial systems, gateways, and other devices that need local object detection without sending every frame to the cloud. A compiled HEF contains the quantized network, hardware allocation, scheduling, and optional HailoRT post-processing needed by the selected accelerator.

!!! note "Compare newer edge accelerators"

    For new hardware deployments, also evaluate [Axelera](axelera.md) and [DeepX](deepx.md), which target newer edge accelerator platforms and may offer higher performance. Hailo recommends at least 1,024 representative calibration images for best accuracy; the built-in task-specific datasets are suitable only for quick testing.

## Why Deploy Ultralytics YOLO on Hailo?

Combining Ultralytics YOLO with a Hailo neural processing unit (NPU) provides a practical path from model training to low-power edge AI inference. Common use cases include:

- **Smart cameras and video analytics**: Run real-time object detection near the camera for security, retail, traffic, and occupancy applications.
- **Robotics and autonomous systems**: Detect people, vehicles, packages, tools, or obstacles without relying on a continuous cloud connection.
- **Industrial computer vision**: Deploy custom YOLO models for inspection, counting, safety monitoring, and quality control.
- **Raspberry Pi AI projects**: Add accelerated vision inference to Raspberry Pi systems using the AI Kit or AI HAT+.
- **Edge gateways and AI PCs**: Process multiple video or sensor streams locally while reducing bandwidth and cloud-compute requirements.

Local inference can improve privacy and response time because images remain on the deployment device. Actual throughput, latency, and power use depend on the YOLO model size, input resolution, Hailo architecture, host system, and application pipeline.

## How Hailo Export Works

Ultralytics owns the complete export workflow behind `format="hailo"`:

```text
YOLO (.pt) -> ONNX -> Hailo parse -> INT8 optimization -> HEF compile
```

The exporter performs these stages automatically:

1. Exports a static ONNX graph with compiler-compatible settings.
2. Selects the head outputs for the model architecture.
3. Generates normalization, activation, and post-processing directives.
4. Builds a representative calibration stream and quantizes the model to INT8.
5. Compiles the optimized graph for the selected Hailo accelerator.
6. Saves the HEF with Ultralytics metadata and removes the intermediate ONNX file.

YOLOv8 and YOLO11 detection models use HailoRT YOLO NMS in the compiled pipeline. YOLO26 detection models use their NMS-free one-to-one outputs, so the exporter selects a different output and quantization path automatically. YOLOv8/YOLO11 segmentation, pose, and OBB compile the raw head tensors, which Ultralytics decodes at inference, and YOLOv8/YOLO11/YOLO26 classification runs softmax on chip so the HEF returns class probabilities directly. For YOLO26 semantic segmentation the exporter follows the accelerator: Hailo-8/8L (DFC v3.x) return classifier logits for host upsampling and reduction, while Hailo-10/15 (DFC v5.x) compile multi-class ArgMax heads on chip and return a compact class map. Single-class heads use the host-logit path on every target because they require a threshold instead of ArgMax. Users do not need to find ONNX end nodes, write a Hailo model script (`.alls`), or create an NMS JSON manually.

## Installation

Install Ultralytics and download the DFC wheel for your target hardware from the Hailo Developer Zone (free registration required):

```bash
pip install ultralytics
pip install /path/to/hailo_dataflow_compiler-*.whl
```

!!! note

    Hailo compilation requires Linux x86_64. Compile the model on a supported workstation, then copy the output directory to the target device. The DFC is not required for inference.

Hailo-8 and Hailo-8L use DFC v3.x. Hailo-10 and Hailo-15 use DFC v5.x. Install the compiler generation that matches the target accelerator.

!!! tip "Export in Ultralytics Platform"

    [Ultralytics Platform](https://platform.ultralytics.com/) provides managed Hailo export, so no local Hailo account or DFC installation is required.

## Export a Hailo HEF Model

Use `format="hailo"` and select the target accelerator with `name`:

```python
from ultralytics import YOLO

model = YOLO("yolo11n.pt")
output = model.export(format="hailo", name="hailo8l")
print(output)  # yolo11n_hailo_model/
```

The equivalent CLI command is:

```bash
yolo export model=yolo11n.pt format=hailo name=hailo8l
```

Hailo export is INT8-only. Ultralytics automatically downloads a task-specific calibration dataset when `data` is not provided. For custom models, use representative training or validation images:

!!! danger "Use at least 1,024 calibration images for best accuracy"

    Ultralytics forces DFC optimization level 2 and configures fine-tuning to use the actual calibration dataset size. Hailo recommends at least 1,024 diverse images; the built-in lightweight datasets compile at level 2 but may not represent the production domain. For production HEF exports, pass a representative dataset using `data="path/to/dataset.yaml"`.

```python
model.export(format="hailo", name="hailo8l", data="path/to/dataset.yaml")
```

Compilation uses a fixed input shape. Set `imgsz` to the resolution used on the device:

```python
model.export(format="hailo", name="hailo8l", imgsz=640)
```

## Supported Models and Hardware

The Hailo ecosystem covers a broad range of computer vision workloads, but the Ultralytics `format="hailo"` exporter currently validates standard YOLO detection, segmentation, pose, OBB, classification, and semantic segmentation heads. The task table describes the available exporter paths; hardware validation is listed separately below.

| Ultralytics task          | Direct Hailo export | Supported model families | Notes                                                                                        |
| :------------------------ | :-----------------: | :----------------------- | :------------------------------------------------------------------------------------------- |
| Object detection          |         ✅          | YOLOv8, YOLO11, YOLO26   | Standard Ultralytics `Detect` heads, including custom models                                 |
| Instance segmentation     |         ✅          | YOLOv8, YOLO11           | Raw head tensors decoded by Ultralytics at inference; YOLO26-seg is not currently supported  |
| Image classification      |         ✅          | YOLOv8, YOLO11, YOLO26   | Softmax runs on chip; the HEF returns class probabilities directly                           |
| Pose estimation           |         ✅          | YOLOv8, YOLO11           | Raw head tensors decoded by Ultralytics at inference; YOLO26-pose is not currently supported |
| Oriented object detection |         ✅          | YOLOv8, YOLO11           | Raw head tensors decoded by Ultralytics at inference; YOLO26-OBB is not currently supported  |
| Semantic segmentation     |         ✅          | YOLO26                   | Hailo-8/8L and single-class heads return logits; Hailo-10/15 bakes multi-class maps          |

Specialized detection families such as YOLOv10, YOLO-World, YOLOE, and RT-DETR are also ❌ unsupported. Ultralytics rejects these tasks and model families before compilation instead of producing an unvalidated HEF.

| Model family                         | Hailo-8 / Hailo-8L | Hailo-10 / Hailo-15 | Output                                                        |
| :----------------------------------- | :----------------: | :-----------------: | :------------------------------------------------------------ |
| YOLOv8 / YOLO11 detection            |         ✅         |         ✅          | HEF with HailoRT YOLO NMS                                     |
| YOLO26 detection                     |         ✅         |         ✅          | NMS-free detection-head outputs for supported runtimes        |
| YOLOv8-seg / YOLO11-seg              |         ✅         |         ✅          | Raw segmentation tensors, decoded by Ultralytics at inference |
| YOLOv8-pose / YOLO11-pose            | Hailo-8L validated |    Not validated    | Raw pose tensors, decoded by Ultralytics at inference         |
| YOLOv8-obb / YOLO11-obb              | Hailo-8L validated |    Not validated    | Raw OBB tensors, decoded by Ultralytics at inference          |
| YOLOv8-cls / YOLO11-cls / YOLO26-cls | Hailo-8L validated |    Not validated    | On-chip softmax; HEF returns class probabilities              |
| YOLO26-sem                           | Hailo-8L validated |    Not validated    | Logits, or a baked multi-class map on Hailo-10/15             |

Pose, OBB, classification, and YOLO26 semantic segmentation (Hailo-8/8L path) were validated on Hailo-8L with HailoRT 4.23 and DFC 3.33. The exporter accepts the other listed targets, but those new task paths require validation with the matching compiler and device before production use.

Select one of these `name` values:

| `name`     | Target accelerator |
| :--------- | :----------------- |
| `hailo8`   | Hailo-8            |
| `hailo8l`  | Hailo-8L           |
| `hailo10h` | Hailo-10H          |
| `hailo15h` | Hailo-15H          |
| `hailo15l` | Hailo-15L          |

`hailo8l` is the default. Install the DFC generation that matches the selected target.

### Hailo Hardware and SDK Generations

Hailo accelerator families use different compiler generations. The generated HEF must match the target hardware, so choose `name` for the device that will run inference rather than the machine performing the export.

| Hardware family       | DFC generation | Typical deployment examples                   |
| :-------------------- | :------------- | :-------------------------------------------- |
| Hailo-8 / Hailo-8L    | DFC v3.x       | Accelerator modules, Raspberry Pi AI Kit/HAT+ |
| Hailo-10H             | DFC v5.x       | Newer edge AI and Raspberry Pi deployments    |
| Hailo-15H / Hailo-15L | DFC v5.x       | Smart-camera and embedded vision applications |

The compiler runs on Linux x86_64, while the resulting HEF runs on the Hailo device through HailoRT. This separation lets you compile on a workstation or in Ultralytics Platform and deploy the small runtime artifact to an ARM or x86 edge host.

### Compatibility Notes

Hailo compilation is hardware-specific and uses a fixed input shape. Keep these constraints in mind:

- The selected `name` must match the deployment accelerator.
- Calibration images should represent the lighting, viewpoints, objects, and backgrounds expected in production.
- A HEF compiled with one `imgsz` does not become dynamically resizable at runtime.
- Custom class counts are supported because Ultralytics generates post-processing configuration from the model metadata.
- Detection models with standard Ultralytics `Detect` heads, YOLOv8/YOLO11 segmentation, pose, and OBB models, and YOLOv8/YOLO11/YOLO26 classification models, and YOLO26 semantic segmentation models are supported; YOLO26 instance segmentation, pose, and oriented bounding box, along with YOLO-World, YOLOE, YOLOv10, and RT-DETR exports, are not currently supported.
- Hailo-8/8L and Hailo-10/15 artifacts are compiled by different DFC generations and are not interchangeable.

## Calibration and INT8 Quantization

Hailo HEF export uses INT8 quantization to map the YOLO network efficiently onto the accelerator. The calibration dataset estimates activation ranges; it does not retrain the model or require labels during compilation.

When `data` is omitted, Ultralytics uses a task-specific lightweight calibration dataset, such as COCO128 for detection or cityscapes8 for semantic segmentation. For a custom computer vision model, point `data` to its dataset YAML so the compiler observes representative images from the actual deployment domain:

```python
model.export(format="hailo", name="hailo8l", data="my_dataset.yaml")
```

`fraction` selects the portion of the dataset used for calibration. More images help only when they represent the deployment domain; out-of-domain images can reduce quantized accuracy and increase optimization time. If the INT8 HEF loses accuracy relative to the original PyTorch model, first improve the calibration data before changing model or runtime settings.

### Accuracy Expectations by Model Family

Measured on a Hailo-8L with in-domain calibration (COCO128, 128 images), INT8 HEF exports retain the following share of their PyTorch mAP50 under the same evaluation protocol:

| Model   | mAP50 retention | Notes                                                    |
| :------ | :-------------- | :------------------------------------------------------- |
| YOLOv8n | ~100%           | DFL head with on-chip NMS                                |
| YOLO11n | ~96%            | Attention blocks in the backbone are more INT8-sensitive |
| YOLO26n | ~93%            | End-to-end head plus attention; see the confidence note  |

Retention compares both models at the same confidence threshold. YOLOv8 and YOLO11 HEFs bake the export-time `conf` (default 0.25) into the on-chip NMS, so validating against a PyTorch baseline at its default low threshold integrates a larger part of the precision-recall curve and overstates the quantization gap.

Beyond detection, the segmentation, pose, OBB, and classification exporter paths were validated on the same Hailo-8L (DFC 3.33, HailoRT 4.23). Each INT8 HEF was compared with its PyTorch checkpoint on the same validation split, using in-domain calibration:

| Task                  | Metric (validation split)          | YOLOv8n | YOLO11n |
| :-------------------- | :--------------------------------- | :------ | :------ |
| Instance segmentation | mask mAP50 retention (COCO128-seg) | 98.0%   | 93.6%   |
| Pose                  | box mAP50 retention (COCO8-pose)   | 98.1%   | 90.8%   |
| Oriented bounding box | mAP50 retention (DOTA128)          | ~100%   | 96.9%   |
| Classification        | top-1 retention (ImageNet val)     | 92.6%   | 95.4%   |

Segmentation, pose, and OBB were calibrated with each task's default in-domain set (COCO128-seg, COCO8-pose, DOTA128); classification was calibrated with ImageNet100. Two caveats follow from those defaults: COCO8-pose is only 8 images, so treat pose as indicative and pass a larger `data=` for production, and DOTA8 saturates mAP50 near 100% for both models, which is why OBB is read on DOTA128. Classification is also the one task where YOLO11 retains more than YOLOv8; for the others the YOLO11 attention backbone is more INT8-sensitive.

Three practical rules follow from device measurements:

1. **Calibrate in-domain, always.** Fine-tuning with out-of-domain images is equivalent to disabling fine-tuning entirely: a YOLO26n calibrated with 1,238 out-of-domain images retains the same accuracy (85.7%) as one compiled without fine-tuning. A small in-domain set beats a large out-of-domain one.
2. **Lower `conf` by about 0.05 for YOLO26 deployments.** Quantization shifts YOLO26 scores down by roughly 0.05 on average, so a threshold tuned in PyTorch drops valid detections on the HEF. Using `conf=0.20` on device matches the detection count of PyTorch at `conf=0.25`, and lowering slightly further (around `conf=0.15`) recovers essentially all of the remaining mAP50 gap at the cost of more low-confidence detections. Quantization also re-ranks roughly 20% of detections — a permanent ordering effect that no threshold undoes — but that reshuffling does not block mAP50 recovery at the lower threshold.
3. **The attention penalty is structural on Hailo-8/8L (DFC 3.33).** The attention blocks compile to `matmul` operations that keep INT8 activation inputs in every mode the compiler offers for them; the 16-bit-output mode fails allocation for this graph, and raising the precision of the surrounding layers does not help because the matmul requantizes its inputs to INT8 anyway (protecting the depthwise and output convolutions at 16-bit left mAP unchanged in our tests). When accuracy is the priority and the model is interchangeable, YOLO11 currently quantizes better than YOLO26 here; newer Hailo generations (DFC 5.x) expose more mixed-precision options and may differ.

## Exported Artifacts

Export creates a directory containing the deployable HEF and Ultralytics metadata:

```text
yolo11n_hailo_model/
├── yolo11n.hef
├── metadata.yaml
└── nms_config.json
```

- `*.hef` is the compiled model loaded by HailoRT.
- `metadata.yaml` preserves model names, task, input size, stride, and Hailo target information.
- `nms_config.json` records the generated HailoRT NMS configuration for YOLOv8 and YOLO11 detection models. YOLO26 detection and all non-detection tasks (segmentation, pose, OBB, classification, semantic) do not use this file.

The intermediate ONNX graph is removed after compilation.

## Run Inference on Hailo Hardware

Install HailoRT on the target device. Raspberry Pi AI Kit and AI HAT+ users can follow the [Raspberry Pi AI software guide](https://www.raspberrypi.com/documentation/computers/ai.html):

```bash
sudo apt install hailo-all
hailortcli fw-control identify
```

Copy the complete export directory to the device so `metadata.yaml` remains next to the HEF. Ultralytics uses HailoRT to run `predict` and `val` directly on the exported directory:

```python
from ultralytics import YOLO

model = YOLO("yolo11n_hailo_model")
results = model.predict("path/to/image.jpg")
```

For detection models, the backend converts YOLOv8 and YOLO11 HailoRT NMS output and decodes YOLO26 one-to-one outputs automatically. It decodes raw segmentation, pose, and OBB tensors, returns on-chip classification probabilities, and produces semantic class maps through host reduction on Hailo-8/8L and all single-class heads or an on-chip ArgMax for multi-class Hailo-10/15 heads. TAPPAS, GStreamer, and the Raspberry Pi `picamera2.devices.Hailo` helper remain available for application-specific pipelines.

For a GStreamer deployment, pass the HEF to `hailonet`:

```bash
gst-launch-1.0 filesrc location=video.mp4 ! decodebin ! videoconvert ! \
  hailonet hef-path=yolo11n_hailo_model/yolo11n.hef ! \
  hailofilter function-name=yolov8 ! hailooverlay ! autovideosink
```

### Hailo Deployment Options

The HEF is the same deployable model artifact across several Hailo runtime interfaces. Choose the interface that fits the application:

| Runtime option                         | Best suited for                                     |
| :------------------------------------- | :-------------------------------------------------- |
| HailoRT Python or C/C++ API            | Custom applications and direct control of inference |
| Raspberry Pi `picamera2.devices.Hailo` | Camera Module projects on Raspberry Pi              |
| GStreamer and Hailo applications       | Real-time video streams and multi-stage pipelines   |
| `hailortcli`                           | Device checks, HEF inspection, and benchmarking     |

Keep `metadata.yaml` with the HEF when the application needs Ultralytics class names, input size, stride, or other model information. The HEF itself does not replace the application-level logic for camera capture, visualization, tracking, alerts, or storage.

### Verify the Hailo Device and HEF

Before integrating a camera or video pipeline, verify the runtime and accelerator independently:

```bash
hailortcli fw-control identify
hailortcli parse-hef yolo11n_hailo_model/yolo11n.hef
```

Device-only performance measurements isolate Hailo inference from video decoding, image resizing, drawing, and application I/O. Measure the complete application separately when estimating end-to-end latency or frames per second.

## Hailo Compared with Other YOLO Export Formats

Choose an export format based on the hardware that will execute the model:

| Deployment target          | Ultralytics export format     |
| :------------------------- | :---------------------------- |
| Hailo NPU                  | Hailo HEF (`format="hailo"`)  |
| NVIDIA GPU                 | [TensorRT](tensorrt.md)       |
| Intel CPU, GPU, or NPU     | [OpenVINO](openvino.md)       |
| Apple hardware             | [CoreML](coreml.md)           |
| Qualcomm Snapdragon NPU    | [QNN](qnn.md)                 |
| Rockchip NPU               | [RKNN](rockchip-rknn.md)      |
| Raspberry Pi AI Camera     | [Sony IMX500](sony-imx500.md) |
| Portable cross-runtime use | [ONNX](onnx.md)               |

HEF is the correct choice when the final device contains a Hailo accelerator. ONNX remains useful as a portable interchange format, but HailoRT executes the hardware-specific HEF produced by the DFC rather than the original ONNX model.

## Optimize Hailo Computer Vision Performance

Model and pipeline choices often matter more than compiler flags:

- Start with a small YOLO model and increase model size only when accuracy requires it.
- Choose the lowest fixed `imgsz` that still preserves the objects important to the application.
- Use calibration images from the real camera and environment when possible.
- Keep the Hailo network active across frames instead of reopening the HEF for every inference.
- Separate device inference time from preprocessing, video decoding, post-processing, visualization, and network I/O.
- Use a streaming pipeline such as GStreamer for sustained video workloads.
- Validate the exported HEF on the exact accelerator and HailoRT version used in production.

## Export Arguments

| Argument   | Type          | Default       | Description                                        |
| :--------- | :------------ | :------------ | :------------------------------------------------- |
| `name`     | `str`         | `hailo8l`     | Target Hailo accelerator architecture              |
| `imgsz`    | `int`, `list` | `640`         | Fixed model input size                             |
| `data`     | `str`         | task-specific | Calibration dataset YAML                           |
| `fraction` | `float`       | `1.0`         | Fraction of calibration images to use              |
| `quantize` | `int`         | `8`           | Hailo export uses INT8 quantization                |
| `opset`    | `int`         | `11`          | Fixed ONNX opset required by the Hailo translation |
| `simplify` | `bool`        | `True`        | Simplify the intermediate ONNX graph               |
| `conf`     | `float`       | `0.25`        | YOLOv8/YOLO11 HailoRT NMS confidence threshold     |
| `iou`      | `float`       | `0.7`         | YOLOv8/YOLO11 HailoRT NMS IoU threshold            |

For detection export, YOLOv8 and YOLO11 receive HailoRT NMS, while YOLO26 keeps its NMS-free one-to-one outputs. Segmentation, pose, and OBB use raw head tensors, classification returns on-chip probabilities, and semantic segmentation returns raw logits on Hailo-8/8L and all single-class heads or baked class maps for multi-class Hailo-10/15 heads. Do not pass `end2end`; explicit overrides are rejected. Dynamic shapes, batches larger than one, embedded Ultralytics NMS, FP16, and FP32 are also unsupported.

## Troubleshooting Hailo Export

### Hailo Dataflow Compiler Import Error

If export reports that `hailo_sdk_client` is missing, install the DFC wheel for the target hardware generation in the same Python environment as Ultralytics. Hailo-8/8L and Hailo-10/15 require different compiler generations.

### Unsupported Operating System or Architecture

HEF compilation is supported on Linux x86_64. Export through [Ultralytics Platform](https://platform.ultralytics.com/) or use a compatible workstation if the local computer is macOS, Windows, Raspberry Pi, or another ARM system.

### Export Takes a Long Time

DFC optimization is the most expensive stage. Compilation time increases with model size, input resolution, and calibration data. A supported GPU can accelerate optimization, while CPU-only compilation can be substantially slower.

### Quantized Model Accuracy Drops

Use calibration images that resemble production inputs and include the important objects, scales, lighting conditions, and backgrounds. Compare the original PyTorch model and exported HEF on the same validation set before deployment. A moderate family-dependent gap remains even with good calibration; see [Accuracy Expectations by Model Family](#accuracy-expectations-by-model-family) for the measured baselines.

### HEF Does Not Load on the Device

Confirm that `name` matched the physical Hailo architecture and that the device driver, firmware, and HailoRT packages are mutually compatible. Inspect the artifact with `hailortcli parse-hef` and verify the accelerator with `hailortcli fw-control identify`.

### Output Parsing Looks Incorrect

Keep `metadata.yaml` beside the HEF so Ultralytics can select the matching YOLOv8, YOLO11, or YOLO26 post-processing path. Custom HailoRT applications must likewise match post-processing to the exported model family.

## FAQ

### Can I compile a HEF on a Raspberry Pi?

No. Run the DFC on a supported Linux x86_64 system and deploy the resulting HEF to the Raspberry Pi.

### Do I need an NVIDIA GPU?

A supported GPU greatly reduces DFC optimization time. CPU compilation is possible but can take substantially longer.

### Which YOLO models support Hailo export?

Direct export supports detection models with the standard YOLOv8, YOLO11, or YOLO26 detection head, YOLOv8/YOLO11 segmentation, pose, and OBB models, and YOLOv8/YOLO11/YOLO26 classification models. This includes custom-trained models built from those standard architectures. YOLO26 semantic segmentation models are also supported. YOLO26 instance segmentation, pose, and OBB, along with YOLOv10, YOLO-World, YOLOE, and RT-DETR, are rejected rather than producing an unvalidated HEF.

### Can I export a custom-trained YOLO model?

Yes. Use the same `format="hailo"` command with the custom `.pt` weights and pass the training dataset YAML through `data` for representative INT8 calibration. Class names and class count are read from the model metadata.

### Does Hailo export support dynamic image sizes?

No. The DFC compiles a fixed input shape into the HEF. Choose `imgsz` during export to match the resolution used by the deployment pipeline.

### Why does YOLO26 produce different Hailo outputs?

YOLO26 uses an NMS-free one-to-one detection head. Ultralytics compiles those output tensors directly instead of attaching the HailoRT YOLOv8-style NMS used for YOLOv8 and YOLO11.

### What is the difference between the DFC and HailoRT?

The Hailo Dataflow Compiler converts and quantizes the model into a hardware-specific HEF on a Linux x86_64 build machine. HailoRT loads and runs that HEF on the target device.

### Should I deploy the ONNX or HEF file?

Deploy the compiled HEF to the Hailo runtime. ONNX is an intermediate representation used during export and is removed after successful compilation.

### Where can I get the Hailo DFC?

Download the compiler wheel for your hardware generation from the Hailo Developer Zone. The compiler is required only to create the HEF; HailoRT runs it on the target accelerator.

## Summary

Ultralytics Hailo export provides a direct path from a trained YOLO model to a deployable HEF:

1. Load a YOLOv8, YOLO11, or YOLO26 detection or classification model, a YOLOv8/YOLO11 segmentation, pose, or OBB model, or a YOLO26 semantic segmentation model.
2. Export with `format="hailo"` and select the target architecture.
3. Calibrate and compile locally with the matching DFC, or use managed export in Ultralytics Platform.
4. Copy the HEF and `metadata.yaml` to the Hailo-powered edge device.
5. Run inference with HailoRT, Raspberry Pi Picamera2, or a GStreamer video pipeline.

For other computer vision deployment targets, see [Export mode](../modes/export.md), [Benchmark mode](../modes/benchmark.md), and the [integrations guide](index.md). Related hardware guides include [ONNX](onnx.md), [OpenVINO](openvino.md), [TensorRT](tensorrt.md), [NCNN](ncnn.md), [RKNN](rockchip-rknn.md), [Sony IMX500](sony-imx500.md), and [Qualcomm QNN](qnn.md).
