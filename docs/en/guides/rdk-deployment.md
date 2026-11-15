---
title: YOLO Deployment on D-Robotics RDK X5
comments: true
description: Export Ultralytics YOLO models to the D-Robotics RDK X5 BPU with the hb_mapper toolchain and run detection, segmentation, classification, pose and OBB inference on the board or in x86 simulation.
keywords: RDK X5, D-Robotics, RDK, Ultralytics YOLO, model export, BPU, hb_mapper, hbm_runtime, INT8 quantization, edge AI, embedded deployment, computer vision, Ultralytics
---

# D-Robotics RDK X5 Deployment for Ultralytics YOLO Models

Ultralytics exports the YOLO model family to the [D-Robotics RDK X5](https://developer.d-robotics.cc/rdk_doc_center) BPU with `format="rdk"`: the export compiles an INT8 BPU body with the D-Robotics `hb_mapper` toolchain and generates an ONNX `tail.onnx` decode graph that runs on the host.

## Supported Models and Tasks

RDK X5 supports the Ultralytics YOLO model families, including YOLOv5u, YOLOv8, YOLOv9, YOLOv10, YOLO11, YOLO12, YOLOv13 and YOLO26. See the [D-Robotics Model Zoo](https://github.com/D-Robotics/rdk_model_zoo/tree/main/samples/vision/ultralytics_yolo) for the model sizes and tasks available for each family. Use checkpoints compatible with your installed Ultralytics version.

RDK export supports detection, instance segmentation, classification, pose estimation and oriented bounding boxes:

| Task                             | Example model     |
| -------------------------------- | ----------------- |
| [Detect](../tasks/detect.md)     | `yolo26n.pt`      |
| [Segment](../tasks/segment.md)   | `yolo26n-seg.pt`  |
| [Classify](../tasks/classify.md) | `yolo26n-cls.pt`  |
| [Pose](../tasks/pose.md)         | `yolo26n-pose.pt` |
| [OBB](../tasks/obb.md)           | `yolo26n-obb.pt`  |

Models with an NMS-free one-to-one head, such as YOLO26, also support export with `nms=False`.

## Export Workflow

The exporter uses `hb_mapper` to compile the model into an INT8 `.bin` for the X5 BPU. ONNX Runtime executes the companion `tail.onnx` decode graph, and Ultralytics handles task postprocessing and validation.

## Requirements

Set up two Python environments on an x86_64 Linux machine:

1. **Ultralytics environment** — CPython ≥ 3.8 with `pip install ultralytics`. A CPU-only PyTorch build is sufficient.

2. **D-Robotics toolchain environment** — CPython 3.10 with `hb_mapper` from the OpenExplore X5 toolchain or `rdkx5-yolo-mapper` for compilation. Mapper 1.0.1 provides `rdkx5-sim` for x86 simulation. Put the required CLI tools on `PATH` of the shell that runs the export:

```bash
export PATH=/opt/rdk-x5-toolchain/bin:$PATH # your toolchain install location
hb_mapper --help                            # required for export
rdkx5-sim --help                            # required for x86 simulation
```

!!! note "Mapper simulation support"

    The public [rdkx5-yolo-mapper 1.0.1](https://pypi.org/project/rdkx5-yolo-mapper/1.0.1/) package provides `hb_mapper`, `rdkx5-sim` and `rdkx5-sim-doctor`. Install it in the Python 3.10 toolchain environment with `pip install rdkx5-yolo-mapper==1.0.1`. Its Linux x86_64 wheel requires glibc 2.35 or newer (for example, Ubuntu 22.04).

## Export a Model

=== "Python"

    ```python
    from ultralytics import YOLO

    model = YOLO("yolo11n.pt")
    model.export(format="rdk", imgsz=640, data="coco8.yaml", fraction=1.0)

    # YOLO26 NMS-free one-to-one head (detect/segment/pose/obb):
    YOLO("yolo26n.pt").export(format="rdk", imgsz=640, data="coco8.yaml", nms=False)
    ```

=== "CLI"

    ```bash
    yolo export model=yolo11n.pt format=rdk imgsz=640 data=coco8.yaml
    yolo export model=yolo26n.pt format=rdk imgsz=640 data=coco8.yaml nms=False # one-to-one head
    ```

Export arguments:

- `data` and `fraction` select the INT8 calibration images. Use images representative of your deployment domain. RDK exports use INT8 quantization (`quantize=8`).
- `imgsz` is the static deployment size (even dimensions; the NV12 input needs even height and width). Detection-family models are typically exported at 640, classifiers at 224.
- `batch=1` and `dynamic=False` define the compiled model. Ultralytics processes larger inference batches as successive single images.
- `nms` — `None` (default) exports the one-to-many branch for host-side NMS; `nms=False` selects the model's NMS-free one-to-one head when available with its built-in top-k using your export-time `max_det`/`agnostic_nms`; `nms=True` is not supported for RDK and falls back to `nms=None` with a warning.
- Use model, data and workspace paths without literal apostrophes; spaces are supported.

The export leaves the deployment in `{model}_rdk_model/`. Copy that whole directory (the `.bin`, `tail.onnx` and `metadata.yaml` are all required) to the board or another machine.

## Run Inference on the RDK X5 Board

Board inference requires an X5-compatible `hbm_runtime` SDK and `onnxruntime` in the Python environment running Ultralytics. Use the SDK supplied for your X5 board software.

```python
from ultralytics import YOLO

model = YOLO("yolo11n_rdk_model")  # the deployment directory, or the .bin inside it

results = model("https://ultralytics.com/images/bus.jpg", imgsz=640)
for box in results[0].boxes:
    print(box.xyxy, box.conf, box.cls)
```

The backend supports [predict](../modes/predict.md), [val](../modes/val.md), batched inputs, plotting and the standard `Results` API. Use `imgsz` matching the compiled size; images are letterboxed (resized and center-cropped for classifiers) to the deployment geometry.

## Run Inference in x86 Simulation

On an x86_64 Linux host with the toolchain on `PATH`, the same code path runs the compiled `.bin` through `rdkx5-sim` with no board present:

```python
from ultralytics import YOLO

model = YOLO("yolo11n_rdk_model")  # loads and runs through rdkx5-sim on x86_64 Linux
print(model("bus.jpg", imgsz=640)[0].boxes)
```

The simulator can also be called directly to inspect a compiled model:

```bash
rdkx5-sim yolo11n.bin --describe # model I/O: names, dtypes, shapes, layouts, strides
rdkx5-sim yolo11n.bin --inputs in.npz --outputs out.npz
```

Use x86 simulation for model development and accuracy evaluation. Measure deployment latency on the RDK X5.

## Validate the Exported Model

```python
from ultralytics import YOLO

metrics = YOLO("yolo11n_rdk_model").val(data="coco8.yaml", imgsz=640)
```

Compare accuracy with `YOLO("yolo11n.pt").val(...)` using the same dataset and settings. Use representative calibration images for INT8 quantization.

## Compiler Performance Estimates

The toolchain ships `hb_perf`, which produces static HTML/PNG/JSON reports of estimated per-node cost from a compiled `.bin`:

```bash
hb_perf yolo11n.bin # requires Graphviz: sudo apt-get install graphviz
```

Use these estimates to identify expensive nodes, and measure deployment latency on the RDK X5.
