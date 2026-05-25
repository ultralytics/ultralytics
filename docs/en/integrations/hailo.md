---
comments: true
description: Learn how to export any Ultralytics YOLO model to Hailo HEF format for high-performance inference on Hailo AI accelerators at the edge.
keywords: YOLO11, YOLOv8, Hailo, HEF, model export, Ultralytics, edge AI, NPU, embedded devices, deep learning, quantization, Hailo-8, Hailo DFC, Data Flow Compiler
---

# Hailo Export for Ultralytics YOLO Models

Deploying computer vision models on edge devices requires a format optimized for the target hardware. The [Hailo](https://hailo.ai/) AI processor delivers high-performance inference on edge platforms including the [Raspberry Pi AI Kit](https://www.raspberrypi.com/products/ai-kit/), without relying on cloud connectivity.

This guide walks you through exporting any Ultralytics YOLO model to Hailo's **HEF (Hailo Executable Format)** using the **Hailo Data Flow Compiler (DFC)** SDK. The result is a compiled model ready to run on Hailo-8, Hailo-8L, and Hailo-15 accelerators.

<p align="center">
  <img width="100%" src="https://cdn.jsdelivr.net/gh/ultralytics/assets@main/docs/hailo-overview.avif" alt="Hailo AI accelerator overview">
</p>

## Why Export to Hailo HEF?

[Hailo](https://hailo.ai/) designs dedicated AI accelerators built specifically for edge inference. Their chips deliver industry-leading performance-per-watt, making them ideal for real-time computer vision on embedded and IoT devices.

**Key reasons to use Hailo:**

- **High throughput**: Hailo-8 delivers up to 26 TOPS, enabling real-time detection at high frame rates.
- **Low power consumption**: Designed for always-on edge deployment with a small power budget.
- **Raspberry Pi AI Kit support**: Hailo-8L (13 TOPS) powers the official Raspberry Pi AI Kit, adding hardware-accelerated inference to Raspberry Pi 5.
- **NMS on-chip**: Hailo hardware can run Non-Maximum Suppression post-processing natively, reducing host CPU load.
- **INT8 quantization**: The DFC automatically quantizes your model from FP32 to INT8 using a calibration dataset, with minimal accuracy loss.

## Export Workflow Overview

Unlike single-step exports, converting to HEF involves a multi-stage pipeline using the Hailo DFC SDK:

```text
YOLO (.pt) → ONNX → HAR (parse) → HAR (optimize/quantize) → HEF (compile)
```

1. **Export to ONNX** using Ultralytics
2. **Parse** the ONNX model into Hailo's intermediate HAR format
3. **Load model script** (`.alls`) with normalization and post-processing directives
4. **Calibrate and quantize** using representative images (auto-downloaded via Ultralytics)
5. **Compile** to a deployable HEF file

## Installation

### Step 1: Install Ultralytics

```bash
pip install ultralytics
```

### Step 2: Install Hailo DFC SDK

The Hailo DFC is required for parsing, optimization, and compilation. Download the Python wheel from the [Hailo Developer Zone](https://hailo.ai/developer-zone/) (free registration required) and install it:

```bash
pip install hailo_sdk_client
```

!!! note

    The Hailo DFC SDK requires a **Linux x86_64** machine. Export and compilation cannot be performed on ARM devices (including Raspberry Pi). Copy the resulting `.hef` file to your Hailo-powered device for deployment.

## Complete Export Script

The script below runs the full pipeline in one go — from a Ultralytics `.pt` file to a compiled `.hef` file. It exports to ONNX using Ultralytics, then compiles with the Hailo DFC using COCO128 (auto-downloaded) as the calibration dataset.

!!! example "Full Pipeline"

    ```python
    import random
    from pathlib import Path

    import numpy as np
    from hailo_sdk_client import ClientRunner
    from PIL import Image
    from ultralytics import YOLO
    from ultralytics.data.utils import check_det_dataset
    from ultralytics.utils import DATASETS_DIR

    # ── Configuration ─────────────────────────────────────────────────────────
    MODEL = "yolo11n"   # any Ultralytics detection model: yolo11n, yolov8s, yolov9c …
    HW_ARCH = "hailo8"  # hailo8 | hailo8l | hailo15h
    IMGSZ = 640
    CALIB_IMAGES = 1024

    # End nodes for YOLO11 / YOLOv8 detection head — see table below for other models
    END_NODES = [
        "/model.23/cv2.0/cv2.0.2/Conv",
        "/model.23/cv3.0/cv3.0.2/Conv",
        "/model.23/cv2.1/cv2.1.2/Conv",
        "/model.23/cv3.1/cv3.1.2/Conv",
        "/model.23/cv2.2/cv2.2.2/Conv",
        "/model.23/cv3.2/cv3.2.2/Conv",
    ]

    # ── Step 1: Export to ONNX ────────────────────────────────────────────────
    model = YOLO(f"{MODEL}.pt")
    model.export(format="onnx", imgsz=IMGSZ, opset=11)  # creates {MODEL}.onnx

    # ── Step 2: Parse ONNX with Hailo DFC ────────────────────────────────────
    # The DFC prints the detected end nodes after parsing — use them if unsure.
    runner = ClientRunner(hw_arch=HW_ARCH)
    runner.translate_onnx_model(f"{MODEL}.onnx", end_node_names=END_NODES)

    # ── Step 3: Load model script (normalization + NMS) ───────────────────────
    alls = (
        "normalization1 = normalization([0.0, 0.0, 0.0], [255.0, 255.0, 255.0])\n"
        "change_output_activation(conv54, sigmoid)\n"
        "change_output_activation(conv65, sigmoid)\n"
        "change_output_activation(conv80, sigmoid)\n"
        'nms_postprocess("yolo11n_nms_config.json", meta_arch=yolov8, engine=cpu)\n'
        "allocator_param(width_splitter_defuse=disabled)"
    )
    runner.load_model_script(alls)

    # ── Step 4: Build calibration dataset (auto-downloads COCO128) ───────────
    check_det_dataset("coco128.yaml")
    calib_dir = DATASETS_DIR / "coco128" / "images" / "train2017"
    image_files = list(calib_dir.glob("*.jpg")) + list(calib_dir.glob("*.png"))

    calibset = np.zeros((CALIB_IMAGES, IMGSZ, IMGSZ, 3), dtype=np.float32)
    for i in range(CALIB_IMAGES):
        img = Image.open(random.choice(image_files)).convert("RGB").resize((IMGSZ, IMGSZ))
        calibset[i] = np.array(img, dtype=np.float32)

    # ── Step 5: Optimize and quantize ────────────────────────────────────────
    runner.optimize(calibset)
    runner.save_har(f"{MODEL}.o.har")  # optional — saves intermediate HAR

    # ── Step 6: Compile to HEF ───────────────────────────────────────────────
    hef = runner.compile()
    with open(f"{MODEL}.hef", "wb") as f:
        f.write(hef)

    print(f"Compiled HEF saved to: {MODEL}.hef")
    ```

The resulting `{MODEL}.hef` file is ready to deploy on any compatible Hailo device.

## Step-by-Step Breakdown

### Step 1: Export to ONNX

Ultralytics exports your trained model to ONNX format, which the Hailo DFC ingests as input. Set `opset=11` for best DFC compatibility.

```python
from ultralytics import YOLO

model = YOLO("yolo11n.pt")
model.export(format="onnx", imgsz=640, opset=11)
```

### Step 2: Parse the ONNX Model

The `translate_onnx_model` call converts the ONNX graph into Hailo's intermediate HAR representation. The `end_node_names` list tells the DFC where to cut the graph — just before NMS — so Hailo can attach its own hardware post-processing.

```python
from hailo_sdk_client import ClientRunner

runner = ClientRunner(hw_arch="hailo8")
runner.translate_onnx_model("yolo11n.onnx", end_node_names=END_NODES)
```

!!! tip "Finding end nodes"

    The DFC prints a suggestion after parsing:

    ```
    [info] In order to use HailoRT post-processing capabilities, these end node names should be used: ...
    ```

    Copy those node names if you are unsure which ones to use, or if you are working with a custom or less common architecture.

### Step 3: Load the Model Script

The model script (`.alls`) configures input normalization and NMS post-processing. The `meta_arch=yolov8` setting applies to both YOLOv8 and YOLO11 since they share the same detection head.

```python
alls = (
    "normalization1 = normalization([0.0, 0.0, 0.0], [255.0, 255.0, 255.0])\n"
    "change_output_activation(conv54, sigmoid)\n"
    "change_output_activation(conv65, sigmoid)\n"
    "change_output_activation(conv80, sigmoid)\n"
    'nms_postprocess("yolo11n_nms_config.json", meta_arch=yolov8, engine=cpu)\n'
    "allocator_param(width_splitter_defuse=disabled)"
)
runner.load_model_script(alls)
```

!!! note

    The `change_output_activation` layer names (`conv54`, `conv65`, `conv80`) are assigned by the DFC during parsing and are **model-specific**. If you are compiling a different model size or architecture, check the DFC output for the correct names, or use a pre-defined `.alls` file from [Hailo's model zoo](https://github.com/hailo-ai/hailo_model_zoo).

    Remove the `nms_postprocess` line if you prefer to run NMS on the host CPU instead.

### Step 4: Build the Calibration Dataset

INT8 quantization requires a representative set of images. The script below uses COCO128, which Ultralytics downloads automatically via `check_det_dataset`:

```python
from ultralytics.data.utils import check_det_dataset
from ultralytics.utils import DATASETS_DIR

check_det_dataset("coco128.yaml")  # downloads to DATASETS_DIR if not present
calib_dir = DATASETS_DIR / "coco128" / "images" / "train2017"
```

!!! tip

    Use at least 64 images for calibration. More images (512–1024) generally improve quantization accuracy. For best results, use images from your deployment domain rather than COCO128.

### Step 5: Optimize and Quantize

```python
runner.optimize(calibset)
runner.save_har(f"{MODEL}.o.har")  # optional intermediate checkpoint
```

This step applies quantization-aware fine-tuning and layer noise analysis. A GPU is strongly recommended — without one, this step can take several hours.

### Step 6: Compile to HEF

```python
hef = runner.compile()
with open(f"{MODEL}.hef", "wb") as f:
    f.write(hef)
```

## Supported Models and End Nodes

All Ultralytics detection models can be exported to HEF. The `end_node_names` parameter must match the detection head output nodes in the ONNX graph, which vary by architecture.

### YOLO11 and YOLOv8

YOLO11 and YOLOv8 share the same decoupled detection head. The layer index differs by one between the two families:

| Model Family | Detection Head Layer | End Node Pattern                              |
| ------------ | -------------------- | --------------------------------------------- |
| YOLO11 (all) | `model.23`           | `/model.23/cv2.0/cv2.0.2/Conv` (6 nodes)      |
| YOLOv8 (all) | `model.22`           | `/model.22/cv2.0/cv2.0.2/Conv` (6 nodes)      |

**YOLO11 end nodes** (all sizes: n, s, m, l, x):

```python
END_NODES = [
    "/model.23/cv2.0/cv2.0.2/Conv",
    "/model.23/cv3.0/cv3.0.2/Conv",
    "/model.23/cv2.1/cv2.1.2/Conv",
    "/model.23/cv3.1/cv3.1.2/Conv",
    "/model.23/cv2.2/cv2.2.2/Conv",
    "/model.23/cv3.2/cv3.2.2/Conv",
]
```

**YOLOv8 end nodes** (all sizes: n, s, m, l, x):

```python
END_NODES = [
    "/model.22/cv2.0/cv2.0.2/Conv",
    "/model.22/cv3.0/cv3.0.2/Conv",
    "/model.22/cv2.1/cv2.1.2/Conv",
    "/model.22/cv3.1/cv3.1.2/Conv",
    "/model.22/cv2.2/cv2.2.2/Conv",
    "/model.22/cv3.2/cv3.2.2/Conv",
]
```

### Other Architectures

For other Ultralytics models (YOLOv9, YOLOv10, YOLO-World, RT-DETR, etc.), run the parse step without `end_node_names` first, read the suggested nodes from the DFC log output, then re-run with those nodes:

```python
# First pass: let the DFC suggest end nodes
runner = ClientRunner(hw_arch=HW_ARCH)
runner.translate_onnx_model(f"{MODEL}.onnx")
# Check the printed log for: "[info] In order to use HailoRT post-processing..."
```

Pre-compiled `.alls` scripts and NMS config files for many YOLO variants are available in the [Hailo Model Zoo](https://github.com/hailo-ai/hailo_model_zoo).

## Supported Hardware Architectures

| Architecture | Device    | Performance | Common Use Case             |
| ------------ | --------- | ----------- | --------------------------- |
| `hailo8`     | Hailo-8   | 26 TOPS     | Standard edge AI module     |
| `hailo8l`    | Hailo-8L  | 13 TOPS     | Raspberry Pi AI Kit         |
| `hailo15h`   | Hailo-15H | 40 TOPS     | Higher-performance variant  |

Set `HW_ARCH` in the script to match your target device before compiling.


## Running Inference on Hailo Hardware

Once you have the `.hef` file, copy it to your Hailo-powered device and run inference using the **HailoRT** Python API (`hailo_platform` package). Unlike the DFC export steps, inference runs directly on the edge device.

!!! note

    The inference code below runs **on the Hailo-powered device** (e.g. Raspberry Pi + AI Kit), not on the x86 machine used for compilation.

### Step 1: Install HailoRT on the Device

On the target device, install the `hailo_platform` Python package from the [Hailo Developer Zone](https://hailo.ai/developer-zone/). For Raspberry Pi AI Kit users, follow the [official setup guide](https://www.raspberrypi.com/documentation/accessories/ai-kit.html) which installs HailoRT automatically.

```bash
pip install hailort
```

### Step 2: Quick Sanity Check

Before running Python inference, confirm the Hailo device is recognized:

```bash
hailortcli fw-control identify
```

You should see the device type, firmware version, and serial number printed.

### Step 3: Run Inference

The script below runs object detection on a single image using the compiled HEF file and the `hailo_platform` Python API. It handles preprocessing, inference, and drawing bounding boxes from the on-chip NMS output.

!!! example "Inference Script"

    ```python
    import numpy as np
    from hailo_platform import (
        HEF,
        VDevice,
        ConfigureParams,
        HailoStreamInterface,
        InputVStreamParams,
        OutputVStreamParams,
        FormatType,
        InferVStreams,
    )
    from PIL import Image, ImageDraw, ImageFont

    # ── Configuration ──────────────────────────────────────────────────────────
    HEF_PATH = "yolo11n.hef"   # path to your compiled HEF file
    SOURCE = "bus.jpg"          # image path, 0 for webcam, or a video path
    IMGSZ = 640
    CONF = 0.25

    COCO_NAMES = [
        "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train",
        "truck", "boat", "traffic light", "fire hydrant", "stop sign",
        "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
        "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
        "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard",
        "sports ball", "kite", "baseball bat", "baseball glove", "skateboard",
        "surfboard", "tennis racket", "bottle", "wine glass", "cup", "fork",
        "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
        "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair",
        "couch", "potted plant", "bed", "dining table", "toilet", "tv",
        "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave",
        "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase",
        "scissors", "teddy bear", "hair drier", "toothbrush",
    ]

    # ── Load HEF and connect to device ─────────────────────────────────────────
    hef = HEF(HEF_PATH)
    params = VDevice.create_params()
    target = VDevice(params)

    configure_params = ConfigureParams.create_from_hef(hef, interface=HailoStreamInterface.PCIe)
    network_groups = target.configure(hef, configure_params)
    network_group = network_groups[0]
    network_group_params = network_group.create_params()

    # ── Setup I/O virtual streams ───────────────────────────────────────────────
    input_vstreams_params = InputVStreamParams.make(
        network_group, quantized=False, format_type=FormatType.FLOAT32
    )
    output_vstreams_params = OutputVStreamParams.make(
        network_group, quantized=False, format_type=FormatType.FLOAT32
    )

    # ── Preprocess ─────────────────────────────────────────────────────────────
    orig = Image.open(SOURCE).convert("RGB")
    ow, oh = orig.size
    resized = orig.resize((IMGSZ, IMGSZ))
    input_data = np.expand_dims(np.array(resized, dtype=np.float32), axis=0)  # (1,640,640,3)
    input_name = hef.get_input_vstream_infos()[0].name

    # ── Inference ──────────────────────────────────────────────────────────────
    with InferVStreams(network_group, input_vstreams_params, output_vstreams_params) as pipeline:
        with network_group.activate(network_group_params):
            pipeline.send({input_name: input_data})
            raw = pipeline.recv()

    # ── Parse on-chip NMS output and draw results ──────────────────────────────
    # When compiled with nms_postprocess the HEF outputs detections grouped by
    # class: shape (batch, num_classes, max_dets, 5) where 5 = [y1,x1,y2,x2,score]
    draw = ImageDraw.Draw(orig)
    output_key = list(raw.keys())[0]
    batch_dets = raw[output_key][0]  # shape: (num_classes, max_dets, 5)

    for cls_idx, cls_dets in enumerate(batch_dets):
        for det in cls_dets:
            score = float(det[4])
            if score < CONF:
                continue
            y1, x1, y2, x2 = det[:4]
            # Scale from model coords (0-640) back to original image size
            x1 = int(x1 * ow / IMGSZ)
            y1 = int(y1 * oh / IMGSZ)
            x2 = int(x2 * ow / IMGSZ)
            y2 = int(y2 * oh / IMGSZ)
            label = f"{COCO_NAMES[cls_idx]} {score:.2f}"
            draw.rectangle([x1, y1, x2, y2], outline="red", width=2)
            draw.text((x1 + 2, y1 + 2), label, fill="red")

    orig.save("output.jpg")
    print("Saved output.jpg")
    ```

!!! tip

    The detection output format assumes the HEF was compiled with on-chip NMS (`nms_postprocess` in the `.alls` script). If you compiled **without** NMS, the raw outputs are the 6 detection head tensors and you must run NMS on the host CPU separately.

### Raspberry Pi AI Kit

The Raspberry Pi AI Kit uses Hailo-8L (13 TOPS). To use it:

1. Set `HW_ARCH = "hailo8l"` before compiling your HEF on the x86 machine.
2. Copy the `.hef` to your Raspberry Pi.
3. Install HailoRT by following the [official Raspberry Pi AI Kit setup guide](https://www.raspberrypi.com/documentation/accessories/ai-kit.html).
4. Run the inference script above.

For camera-based inference on Raspberry Pi, the [picamera2 Hailo examples](https://github.com/raspberrypi/picamera2/tree/main/examples/hailo) provide ready-to-use scripts for live detection with the Camera Module.

### Video Inference with TAPPAS

For high-throughput video pipelines, [TAPPAS](https://github.com/hailo-ai/tappas) provides GStreamer elements that stream video through the Hailo chip in real time:

```bash
gst-launch-1.0 filesrc location=video.mp4 ! decodebin ! \
    hailonet hef-path=yolo11n.hef ! \
    hailofilter function-name=yolov8 ! \
    hailooverlay ! autovideosink
```

See the [TAPPAS documentation](https://github.com/hailo-ai/tappas) for full pipeline configuration options.

## Summary

This guide covered the complete workflow to export any Ultralytics YOLO model to Hailo HEF format:

1. Export to ONNX with Ultralytics (`model.export(format="onnx")`).
2. Parse the ONNX model with the Hailo DFC and specify detection head end nodes.
3. Configure normalization and NMS via a model script.
4. Quantize with a calibration dataset (COCO128 via Ultralytics).
5. Compile to a `.hef` file ready for Hailo-8, Hailo-8L, or Hailo-15.

For further details, see the [Hailo Developer Zone](https://hailo.ai/developer-zone/) and the [Hailo Model Zoo](https://github.com/hailo-ai/hailo_model_zoo). For other Ultralytics export options, visit the [integration guide page](index.md).

## FAQ

### What Hailo devices are supported?

The Hailo DFC supports Hailo-8 (`hailo8`), Hailo-8L (`hailo8l`), and Hailo-15H (`hailo15h`). Set `HW_ARCH` to match your target. The Raspberry Pi AI Kit uses Hailo-8L.

### Which Ultralytics models can be exported?

Any Ultralytics detection model can be exported to HEF, including YOLO11, YOLOv8, YOLOv9, YOLOv10, and others. The `end_node_names` must be set correctly for each architecture — see the [Supported Models](#supported-models-and-end-nodes) section above.

### Why does the model script use `meta_arch=yolov8` for YOLO11?

YOLO11 uses the same decoupled detection head architecture as YOLOv8. The Hailo DFC uses `meta_arch=yolov8` for NMS configuration for both model families.

### Do I need a GPU for the optimization step?

A GPU is strongly recommended for the quantization-aware fine-tuning in `runner.optimize()`. Without one, the process still works but is significantly slower (several hours vs. ~10–20 minutes with a GPU).

### How do I find the correct end nodes for my model?

Run `runner.translate_onnx_model(...)` without specifying `end_node_names`. The DFC will print a suggestion in the log:

```text
[info] In order to use HailoRT post-processing capabilities, these end node names should be used: ...
```

Use those node names in your `END_NODES` list.

### Where can I get the Hailo DFC SDK and NMS config files?

The Hailo DFC SDK (Python wheel) and pre-defined `.alls` / NMS config files are available from the [Hailo Developer Zone](https://hailo.ai/developer-zone/) and [Hailo Model Zoo](https://github.com/hailo-ai/hailo_model_zoo) after registering for a free account.
