---
comments: true
description: Export Ultralytics YOLO detection models directly to Hailo HEF for Hailo-8, Hailo-8L, Hailo-10, and Hailo-15 accelerators.
keywords: Hailo export, Hailo HEF, export YOLO to Hailo, YOLO Hailo, Hailo-8, Hailo-8L, Hailo-10, Hailo-15, Raspberry Pi AI Kit, Raspberry Pi AI HAT+, Hailo Dataflow Compiler, Hailo DFC, HailoRT, INT8 quantization, Ultralytics YOLO, YOLO26, YOLO11, YOLOv8
---

# Hailo Export for Ultralytics YOLO Models

[Hailo](https://hailo.ai/) AI accelerators run compiled Hailo Executable Format (HEF) models on edge devices such as the [Raspberry Pi AI Kit](https://www.raspberrypi.com/products/ai-kit/) and [AI HAT+](https://www.raspberrypi.com/documentation/accessories/ai-hat-plus.html). Ultralytics exports YOLO detection models directly to HEF with the Hailo Dataflow Compiler (DFC).

## Installation

Install Ultralytics and download the DFC wheel for your target hardware from the [Hailo Developer Zone](https://hailo.ai/developer-zone/) (free registration required):

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

Hailo export is INT8-only. Ultralytics automatically downloads the default COCO128 calibration dataset when `data` is not provided. For custom models, use representative training or validation images:

```python
model.export(format="hailo", name="hailo8l", data="path/to/dataset.yaml", fraction=0.25)
```

Compilation uses a fixed input shape. Set `imgsz` to the resolution used on the device:

```python
model.export(format="hailo", name="hailo8l", imgsz=640)
```

## Supported Models and Hardware

Direct Hailo export currently supports YOLO object detection models. Other tasks fail with a clear error instead of producing an unvalidated HEF.

| Model family    | Hailo-8 / Hailo-8L | Hailo-10 / Hailo-15 | Output                                                 |
| :-------------- | :----------------: | :-----------------: | :----------------------------------------------------- |
| YOLOv8 / YOLO11 |         ✅         |         ✅          | HEF with HailoRT YOLO NMS                              |
| YOLO26          |         ✅         |         ✅          | NMS-free detection-head outputs for supported runtimes |

Select one of these `name` values:

| `name`     | Target accelerator |
| :--------- | :----------------- |
| `hailo8`   | Hailo-8            |
| `hailo8l`  | Hailo-8L           |
| `hailo10h` | Hailo-10H          |
| `hailo15h` | Hailo-15H          |
| `hailo15l` | Hailo-15L          |

`hailo8l` is the default. Install the DFC generation that matches the selected target.

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
- `nms_config.json` records the generated HailoRT NMS configuration for YOLOv8 and YOLO11 detection models. YOLO26 does not use this file.

The intermediate ONNX graph is removed after compilation.

## Run Inference on Hailo Hardware

Install HailoRT on the target device. Raspberry Pi AI Kit and AI HAT+ users can follow the [Raspberry Pi AI software guide](https://www.raspberrypi.com/documentation/computers/ai.html):

```bash
sudo apt install hailo-all
hailortcli fw-control identify
```

Copy the complete export directory to the device so `metadata.yaml` remains next to the HEF. Load the `*.hef` with HailoRT, TAPPAS, or the Raspberry Pi `picamera2.devices.Hailo` helper.

YOLOv8 and YOLO11 exports include HailoRT NMS and return detections grouped by class. YOLO26 exports expose the six NMS-free detection-head tensors; use a runtime pipeline that supports YOLO26 post-processing.

For a GStreamer deployment, pass the HEF to `hailonet`:

```bash
gst-launch-1.0 filesrc location=video.mp4 ! decodebin ! videoconvert ! \
  hailonet hef-path=yolo11n_hailo_model/yolo11n.hef ! \
  hailofilter function-name=yolov8 ! hailooverlay ! autovideosink
```

See the [Hailo applications repository](https://github.com/hailo-ai/hailo-apps) for current HailoRT and GStreamer examples.

## Export Arguments

| Argument   | Type          | Default   | Description                           |
| :--------- | :------------ | :-------- | :------------------------------------ |
| `name`     | `str`         | `hailo8l` | Target Hailo accelerator architecture |
| `imgsz`    | `int`, `list` | `640`     | Fixed model input size                |
| `data`     | `str`         | `coco128` | Calibration dataset YAML              |
| `fraction` | `float`       | `1.0`     | Fraction of calibration images to use |
| `quantize` | `int`         | `8`       | Hailo export uses INT8 quantization   |
| `end2end`  | `bool`        | model     | Select the NMS-free YOLO26 head       |

## FAQ

### Can I compile a HEF on a Raspberry Pi?

No. Run the DFC on a supported Linux x86_64 system and deploy the resulting HEF to the Raspberry Pi.

### Do I need an NVIDIA GPU?

A supported GPU greatly reduces DFC optimization time. CPU compilation is possible but can take substantially longer.

### Where can I get the Hailo DFC?

Download the compiler wheel for your hardware generation from the [Hailo Developer Zone](https://hailo.ai/developer-zone/). The compiler is required only to create the HEF; HailoRT runs it on the target accelerator.

For other export targets, see [Export mode](../modes/export.md) and the [integrations guide](index.md).
