---
title: YOLO Deployment on D-Robotics RDK X5
comments: true
description: Export Ultralytics YOLO11 and YOLO26 models to the D-Robotics RDK X5 BPU with the hb_mapper toolchain and run detection, segmentation, classification, pose and OBB inference on the board or in x86 simulation.
keywords: RDK X5, D-Robotics, RDK, YOLO11, YOLO26, model export, BPU, hb_mapper, hbm_runtime, INT8 quantization, edge AI, embedded deployment, computer vision, Ultralytics
---

# D-Robotics RDK X5 Deployment for Ultralytics YOLO Models

The [D-Robotics RDK X5](https://developer.d-robotics.cc/rdk_x5) is an embedded developer kit whose BPU (Brain Processing Unit) runs quantized neural networks directly in hardware. Ultralytics exports the standard YOLO11 and YOLO26 model families to the RDK X5 BPU format with `format="rdk"`: the model's standard ONNX graph is split into a BPU body (compiled to an INT8 `.bin` by the D-Robotics `hb_mapper` toolchain) and a `tail.onnx` decode graph that runs on the host through ONNX Runtime — on the board through the `hbm_runtime` SDK, or on an x86_64 Linux host through Mapper's external `rdkx5-sim` CLI.

## Supported Tasks and Heads

RDK export supports the five standard YOLO tasks for both model families. YOLO26 additionally has an NMS-free one-to-one head that can be exported instead of the default one-to-many branch.

| Task                             | YOLO11              | YOLO26              | One-to-one (`nms=False`) |
| -------------------------------- | ------------------- | ------------------- | ------------------------ |
| [Detect](../tasks/detect.md)     | `yolo11n.pt` …      | `yolo26n.pt` …      | YOLO26 only              |
| [Segment](../tasks/segment.md)   | `yolo11n-seg.pt` …  | `yolo26n-seg.pt` …  | YOLO26 only              |
| [Classify](../tasks/classify.md) | `yolo11n-cls.pt` …  | `yolo26n-cls.pt` …  | not applicable           |
| [Pose](../tasks/pose.md)         | `yolo11n-pose.pt` … | `yolo26n-pose.pt` … | YOLO26 only              |
| [OBB](../tasks/obb.md)           | `yolo11n-obb.pt` …  | `yolo26n-obb.pt` …  | YOLO26 only              |

The validated matrix is the standard YOLO11/YOLO26 heads above; other head protocols (YOLOE, YOLOWorld, RT-DETR) and the semantic-segmentation and depth tasks are not supported for RDK export.

## How the Pipeline Works

1. **Standard ONNX export** — the model exports exactly like `format="onnx"` (opset 11, static single image), with the usual `output0`/`output1` outputs.
2. **Body/tail split** — the export cuts the graph at the head's terminal convolutions (and the classifier's final fully-connected layer, which stays on the BPU). Everything after that cut — DFL, anchors, sigmoid, keypoint/angle math, one-to-one top-k, the classifier's softmax — becomes the generated `tail.onnx` that ONNX Runtime executes on the host, so no decode is re-implemented outside the exported graph itself.
3. **INT8 compilation** — `hb_mapper makertbin` quantizes the body (Bayes-E architecture, `O3`, NV12 input) with calibration images taken from the `data` you pass to `export`. When `rdkx5-sim` is on `PATH`, the export also verifies the compiled model's actual I/O against the tail graph before publishing.
4. **Deployment directory** — the export writes `{model}_rdk_model/` containing the compiled `{model}.bin`, its sibling `tail.onnx`, `metadata.yaml` (the standard Ultralytics export metadata plus a `format: rdk` marker), the quantization info and the compiler log. A failed re-export never destroys the previous deployment.
5. **Runtime decode** — the backend validates the actual model I/O (single NV12 input, dequantized float32 outputs matching the tail inputs by name and shape), runs the body and feeds its outputs to the tail. NMS, mask assembly and all thresholds stay with the standard Ultralytics predictor.

## Requirements

Export needs **two separate environments on an x86_64 Linux machine**. Ultralytics never installs or manages the toolchain for you.

1. **Ultralytics environment** — any CPython ≥ 3.8 (the validated setup uses 3.10) with `pip install ultralytics`. A CPU-only PyTorch build is sufficient.

2. **D-Robotics toolchain environment** — CPython 3.10 with `hb_mapper` from the OpenExplore X5 toolchain or `rdkx5-yolo-mapper` for compilation. x86 simulation additionally requires a Mapper build that provides `rdkx5-sim`. Put the required CLI tools on `PATH` of the shell that runs the export:

```bash
export PATH=/opt/rdk-x5-toolchain/bin:$PATH  # your toolchain install location
hb_mapper --help                          # required for export
rdkx5-sim --help                          # required for x86 simulation; optional for export I/O verification
```

!!! note "Mapper simulation support"

    The public [rdkx5-yolo-mapper 1.0.1](https://pypi.org/project/rdkx5-yolo-mapper/1.0.1/) package provides `hb_mapper`, `rdkx5-sim` and `rdkx5-sim-doctor`. Install it in the Python 3.10 toolchain environment with `pip install rdkx5-yolo-mapper==1.0.1`. Its Linux x86_64 wheel requires glibc 2.35 or newer (for example, Ubuntu 22.04). Version 1.0.0 does not provide simulation support.

## Export a Model

=== "Python"

    ```python
    from ultralytics import YOLO

    model = YOLO("yolo11n.pt")  # detection family at 640 px, classification at 224 px
    model.export(format="rdk", imgsz=640, data="coco8.yaml", fraction=1.0)

    # YOLO26 NMS-free one-to-one head (detect/segment/pose/obb):
    YOLO("yolo26n.pt").export(format="rdk", imgsz=640, data="coco8.yaml", nms=False)
    ```

=== "CLI"

    ```bash
    yolo export model=yolo11n.pt format=rdk imgsz=640 data=coco8.yaml
    yolo export model=yolo26n.pt format=rdk imgsz=640 data=coco8.yaml nms=False  # one-to-one head
    ```

Export arguments:

- `data` and `fraction` select the INT8 calibration images. Calibration quality drives quantization accuracy — use images representative of your deployment domain, not just any small set. RDK export is always INT8 (`quantize=8`); FP32/FP16 are not supported for this format.
- `imgsz` is the static deployment size (even dimensions; the NV12 input needs even height and width). Detection-family models are typically exported at 640, classifiers at 224.
- `batch` is fixed to 1 and `dynamic` is not supported — the compiled model is a static single-image deployment, and Ultralytics serializes larger batches automatically.
- `nms` — `None` (default) exports the one-to-many branch for host-side NMS; `nms=False` selects the YOLO26 NMS-free one-to-one head with its built-in top-k using your export-time `max_det`/`agnostic_nms`; `nms=True` is not supported for RDK and falls back to `nms=None` with a warning.
- Paths containing spaces are supported. Paths containing literal apostrophes are not — relocate the model, data or workspace instead of quoting them.

The export leaves the deployment in `{model}_rdk_model/`. Copy that whole directory (the `.bin`, `tail.onnx` and `metadata.yaml` are all required) to the board or another machine.

## Run Inference on the RDK X5 Board

Board inference needs an X5-compatible `hbm_runtime` SDK importable in the **same Python environment Ultralytics is installed in** — verify which SDK build your environment provides against your compiled model, as Ultralytics never installs or manages it. Do **not** install the public `hbm-runtime` PyPI wheel (an unrelated package), and do not set any SDK check bypass variables. The tail graph runs through ONNX Runtime — if that environment has no `onnxruntime`, install it into a separate virtual environment together with Ultralytics instead of the system packages.

```python
from ultralytics import YOLO

model = YOLO("yolo11n_rdk_model")  # the deployment directory, or the .bin inside it

results = model("https://ultralytics.com/images/bus.jpg", imgsz=640)
for box in results[0].boxes:
    print(box.xyxy, box.conf, box.cls)
```

Predict and validation work exactly as with any other exported format — [predict](../modes/predict.md), [val](../modes/val.md), batched inputs (serialized to single images automatically), plotting and the full `Results` API included. Use `imgsz` matching the compiled size; images are letterboxed (resized and center-cropped for classifiers) to the deployment geometry.

## Run Inference in x86 Simulation

On an x86_64 Linux host with the toolchain on `PATH`, the same code path runs the compiled `.bin` through `rdkx5-sim` with no board present:

```python
from ultralytics import YOLO

model = YOLO("yolo11n_rdk_model")  # loads and runs through rdkx5-sim on x86_64 Linux
print(model("bus.jpg", imgsz=640)[0].boxes)
```

The simulator can also be called directly to inspect a compiled model:

```bash
rdkx5-sim yolo11n.bin --describe          # model I/O: names, dtypes, shapes, layouts, strides
rdkx5-sim yolo11n.bin --inputs in.npz --outputs out.npz
```

!!! note "Simulation is host CPU execution"

    Running the model through `rdkx5-sim` executes the compiled graph on the host CPU. It verifies numerical and contract behavior, not speed: x86 wall time is **not** RDK X5 latency. Measure on the board for real timings.

## Validate the Exported Model

```python
from ultralytics import YOLO

metrics = YOLO("yolo11n_rdk_model").val(data="coco8.yaml", imgsz=640)
```

INT8 quantization shifts confidences and geometry slightly compared to the FP32 PyTorch model — compare against `YOLO("yolo11n.pt").val(...)` under identical data and settings, and expect differences consistent with quantization rather than bit-exact equality. A representative calibration set is the main lever if the gap is too large.

## Compiler Performance Estimates

The toolchain ships `hb_perf`, which produces static HTML/PNG/JSON reports of estimated per-node cost from a compiled `.bin`:

```bash
hb_perf yolo11n.bin  # requires Graphviz: sudo apt-get install graphviz
```

These are compiler-side estimates for locating expensive nodes. Neither `hb_perf` reports nor x86 simulation wall time are measured on-board BPU latency, and neither evaluates model accuracy.

## FAQ

**Can I load the `.bin` alone?** A bare `.bin` is not routable — the runtime needs the sibling `metadata.yaml` (with the `format: rdk` marker) and `tail.onnx`, so load the export directory or the `.bin` inside one. Generic third-party `.bin` files are rejected rather than guessed.

**What about deployments from older Ultralytics versions?** Exports from before the body/tail split carry no `tail.onnx` and fail at load with a re-export instruction. Re-export the model with `YOLO(model).export(format="rdk")` — no legacy decode path is retained.

**What about quantized outputs?** Only deployments whose outputs are dequantized to float32 with no output quantization feed the tail; anything else fails at load with a re-export instruction instead of guessing scales.
