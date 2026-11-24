---
comments: true
description: Export Ultralytics YOLO26 models to AXERA NPUs with format=axera for AX650 INT8 or AX8860 16-bit float inference. Covers the axnn/Pulsar2 toolchain, calibration, the FP32 tail split, output files, and measured accuracy and latency.
keywords: YOLO26, AXERA, AXERA export, AX650, AX8860, axnn, Pulsar2, NPU, INT8 quantization, BF16, W8A16, model deployment, edge AI, embedded inference, tail.onnx
---

# AXERA Export

[AXERA](https://axera-tech.com/) NPUs, such as the AX650 and AX8860, run models compiled into the `.axmodel` format by Pulsar2, the AXERA compiler shipped inside the `axnn` Python package. Exporting an [Ultralytics YOLO26](../models/yolo26.md) model with `format="axera"` creates a `yolo26n_axera_model/` directory that holds the compiled graph, its calibration set and metadata. The export runs on Linux and needs no AXERA hardware.

## At a Glance

| Property          | Details                                                                                                             |
| ----------------- | ------------------------------------------------------------------------------------------------------------------- |
| `format` argument | `axera`                                                                                                             |
| Output            | `yolo26n_axera_model/`, a directory with `compiled.axmodel`, `metadata.yaml`, `config.json` and the calibration set |
| Runs on           | AX650 and AX8860 NPUs, through the `axengine` runtime on the device                                                 |
| Precision         | AX650: INT8 or `w8a16`; AX8860: 16-bit float (BF16), INT8 or `w8a16`                                                |
| Export host       | Linux, x86-64 or aarch64; the export itself needs no AXERA device                                                   |
| Toolchain         | [`axnn`](https://huggingface.co/AXERA-TECH/Pulsar2/tree/main) 9.0 or newer, which bundles Pulsar2                   |

## Quickstart

Install the `axnn` wheel that matches your Python version and CPU architecture on a Linux machine, then install Ultralytics:

```bash
pip install ultralytics
pip install /path/to/axnn-*.whl
```

!!! warning "Toolchain versions"

    `axnn` is a vendor wheel, not a PyPI package, so Ultralytics cannot install it for you. Download the latest release from [AXERA-TECH/Pulsar2](https://huggingface.co/AXERA-TECH/Pulsar2/tree/main) and install the file that matches your interpreter and architecture. The export requires `axnn` 9.0 or newer and reports a clear error when the package is missing or too old.

Export a model:

!!! example "Export"

    === "Python"

        ```python
        from ultralytics import YOLO

        model = YOLO("yolo26n.pt")
        model.export(format="axera", name="AX650", data="coco8.yaml")
        ```

    === "CLI"

        ```bash
        yolo export model=yolo26n.pt format=axera name=AX650 data="coco8.yaml"
        ```

`name` selects the chip and with it the default precision: `AX650` builds INT8, `AX8860` builds a 16-bit float graph. Pass `quantize` to override the default, for example `quantize="w8a16"` for INT8 weights with 16-bit activations.

## Supported Tasks

AXERA export covers every Ultralytics task: detection, instance segmentation, semantic segmentation, depth estimation, classification, pose estimation and OBB; the export validates the task name and rejects anything outside that list. Validation in this integration uses YOLO26: detection at all five sizes (n/s/m/l/x), instance segmentation with `yolo26s-seg`, and pose, OBB, semantic segmentation, depth and classification with `yolo26n` checkpoints. YOLO11 detection (n and s) has also been exported and run; other model families follow the same path but are not part of the validated matrix. RT-DETR and YOLO-World are not validated.

## Export Arguments

| Argument   | Type                      | Default      | Description                                                                                   |
| ---------- | ------------------------- | ------------ | --------------------------------------------------------------------------------------------- |
| `format`   | `str`                     | —            | `axera`                                                                                       |
| `name`     | `str`                     | `AX650`      | Chip name: `AX650` or `AX8860`                                                                |
| `quantize` | `int` or `str`            | per chip     | `8` (INT8), `16` (16-bit float) or `"w8a16"`. `AX650` + `16` raises an error                  |
| `imgsz`    | `int` or tuple            | `640`        | Square static input size                                                                      |
| `batch`    | `int`                     | `1`          | Static batch, only `1` is supported                                                           |
| `data`     | `str`                     | task default | Dataset YAML used for calibration images                                                      |
| `fraction` | `float`, `int`, or `list` | `64`         | Calibration subset: a ratio, an image count, or `[train, val, test]`. Unset exports 64 images |
| `opset`    | `int`                     | `13`         | ONNX opset used for the intermediate graph                                                    |
| `simplify` | `bool`                    | `True`       | Run the ONNX simplifier before compiling                                                      |

Dynamic shapes and embedded NMS are not supported: `dynamic` must stay `False` and `nms` must stay `None`.

### Target Chips and Quantization

| Chip     | Precision              | Weights | Activations |
| -------- | ---------------------- | ------- | ----------- |
| `AX650`  | INT8 (default)         | S8      | U8          |
| `AX650`  | `w8a16`                | S8      | U16         |
| `AX8860` | 16-bit float (default) | BF16    | BF16        |
| `AX8860` | INT8                   | S8      | U8          |
| `AX8860` | `w8a16`                | S8      | U16         |

Every build writes the same `compiled.axmodel`; the Weights and Activations columns list the quantization types, and parentheses mark each chip's default precision.

Quantization and calibration details:

- **INT8**: `quantize=8` uses Pulsar2's MinMax calibration and the default S8/U8 quantization configuration.
- **Mixed precision**: AXERA's `"w8a16"` means S8 weights with U16 activations, not the "16-bit float activations" used by other backends.
- **16-bit float**: `quantize=16` exports as BF16 because the quantizer does not accept the `FP16` dtype mapping; requesting `16` on the AX650 fails before export.
- **Calibration data**: the exporter prepares images from the `val` split of `data` (change it with `split`), packs them as `calib.zip` in Pulsar2 `Image` mode, and uses the first 64 images by default. `fraction` is adjustable with no hard cap. The four images in `coco8.yaml` only validate the pipeline; use data that covers your deployment scenes.
- **Detection head split**: fixed-point activations cannot hold both pixel coordinates and class scores in one scale, so INT8 and `w8a16` detect, segment, pose and OBB models are cut at the per-level head convolutions: the body compiles to the NPU, while box decode, mask prototypes, keypoints and angle values run in FP32 on the host. 16-bit float builds, plus semantic segmentation, depth and classification, keep the whole graph.

## Output Format

The export writes a directory next to the source checkpoint:

```text
yolo26n_axera_model/
├── compiled.axmodel    # deployable NPU graph, loaded by axengine or ax_run_model on the board
├── metadata.yaml       # task, names, imgsz, stride, batch and axera.* (target/quantize/npu_mode/outputs)
├── config.json         # Pulsar2 configuration that produced the model (quantization, input processors, output dtype)
├── build_context.json  # Pulsar2 build status and context (Success on success)
├── model.onnx          # simplified full intermediate ONNX graph
├── body.onnx           # quantized detection: the subgraph that was compiled
├── tail.onnx           # quantized detection: host-side FP32 box decode
├── calib/              # staged calibration images (64 by default)
├── calib.zip           # calibration archive read by Pulsar2 Image mode
├── quant/              # quantization results and debug information
├── frontend/           # front-end graph optimization results
└── compiler/           # compiler back-end intermediate results
```

`quant/`, `frontend/`, `compiler/` and `build_context.json` are Pulsar2 intermediates, useful only for recompiling and debugging, and can be deleted. Keep at least `compiled.axmodel` and `metadata.yaml` for deployment, plus `tail.onnx` for a quantized detection build.

**Size comparison** (640 input, YOLO26 detection):

| Model   | ONNX (MB) | AX650 INT8 (MB) | AX8860 BF16 (MB) |
| ------- | --------- | --------------- | ---------------- |
| yolo26n | 9.9       | 3.3             | 7.8              |
| yolo26s | 38.3      | 10.7            | 27.3             |
| yolo26m | 82.0      | 22.4            | 56.2             |
| yolo26l | 99.6      | 27.3            | 66.7             |
| yolo26x | 223.3     | 59.1            | 151.1            |

`ONNX (MB)` is the size of the standard ONNX export (`format=onnx`) at 640 input. `AX650 INT8 (MB)` and `AX8860 BF16 (MB)` are the sizes of the Pulsar2 compiled artifact. All sizes use 10^6 bytes per MB.

**Input**: `compiled.axmodel` takes a `(1, H, W, 3)` NHWC uint8 tensor, RGB, ranging from 0 to 255; the graph already includes the division by 255, so no external preprocessing is needed. This differs from the ONNX export, which takes NCHW float32 scaled to 0–1. `AxeraBackend` converts the predictor's `[0, 1]` tensor back to 0–255 uint8 before feeding it in.

**Output**: depends on the task and precision. Measured outputs per task:

| Task                  | Default imgsz | 16-bit float output                    | INT8 / `w8a16` output                                                         |
| --------------------- | ------------- | -------------------------------------- | ----------------------------------------------------------------------------- |
| Detection             | 640           | `(1, 84, 8400)`                        | 6 head tensors (3 scales × box/cls), restored by `tail.onnx`                  |
| Segmentation          | 640           | `(1, 116, 8400)` + `(1, 32, 160, 160)` | 10 tensors (9 head + protos), restored to the same two outputs by `tail.onnx` |
| Semantic segmentation | 1024          | `(1, 19, 1024, 1024)`                  | Same shape as 16-bit, whole-graph quantized (no `tail.onnx`)                  |
| Depth                 | 768           | `(1, 1, 768, 768)`                     | Same shape as 16-bit, whole-graph quantized (no `tail.onnx`)                  |
| Classification        | 224           | `(1, 1000)`                            | Same shape as 16-bit, whole-graph quantized (no `tail.onnx`)                  |
| Pose                  | 640           | `(1, 56, 8400)`                        | 9 head tensors (3 scales × box/cls/kpt), restored by `tail.onnx`              |
| OBB                   | 640           | `(1, 20, 8400)`                        | 9 head tensors (3 scales × box/cls/angle), restored by `tail.onnx`            |

**Raw predictions**: the 16-bit float models and `tail.onnx` return the same layout as the default ONNX export: for each anchor, box center x, center y, width and height in input-pixel coordinates, then class scores, and finally task-specific values. Apply NMS yourself; `YOLO(...)` does it automatically.

**INT8 task-head tensors**: detection returns two tensors per scale (box values and class scores); segmentation returns three per scale (box, scores and mask coefficients) plus a protos tensor `output1`; pose returns three per scale (box, scores and 17×3 keypoints); OBB returns three per scale (box, scores and angle). Feed them to `tail.onnx` in the order recorded in `metadata.yaml` under `axera.outputs`.

**Metadata**: `YOLO("yolo26n_axera_model")` reads `metadata.yaml`, runs `tail.onnx` and applies NMS automatically, so you only need the details above when writing your own inference code. Keep the `_axera_model` suffix when loading; a renamed directory is treated as a model name and triggers a download. Metadata keys include `names`, `task`, `imgsz`, `stride`, `batch`, plus `axera.target`, `axera.quantize` (`8`, `16` or `"w8a16"`), `axera.npu_mode` (`NPU3` on the AX650, `NPU4` on the AX8860), `axera.calibration.size` and `axera.outputs`.

## Deploy

On an AXERA device, install the `axengine` runtime that matches the toolchain, then copy `compiled.axmodel` and `metadata.yaml`, plus `tail.onnx` for a quantized detection build. Ultralytics picks `axengine` automatically:

```python
from ultralytics import YOLO

model = YOLO("yolo26n_axera_model")
model.predict("bus.jpg")
model.val(data="coco8.yaml")
```

Without `axengine`, `AxeraBackend` falls back to the `axnn.simulator`, which runs the compiled graph on the host for reference numbers. That path is useful for validating an export before deployment, but its latency is not NPU latency.

Board-side evaluation can also use the preinstalled `/opt/bin/ax_run_model` tool, which feeds raw tensors and writes one file per output tensor. Score those outputs with the same NMS and metric code Ultralytics uses, running `tail.onnx` first when the artifact has one.

`benchmark(format="axera")` is not supported: it skips before exporting, because a benchmark row would report simulator latency as if it were measured on the NPU. Measure accuracy with `val()` and latency with `ax_run_model -w 10 -r 100` on the device instead.

## Limitations and Troubleshooting

- **`AXERA export requires Linux`**: the `axnn` wheels are Linux-only.
- **`axnn ... is required` / version error**: install a matching `axnn` wheel from [AXERA-TECH/Pulsar2](https://huggingface.co/AXERA-TECH/Pulsar2/tree/main); 9.0 or newer is required.
- **`AX650 exports INT8 or 'w8a16'`**: 16-bit float is only available on the AX8860; use `name="AX8860"` or `quantize="w8a16"`.

## Benchmarks

Measured on the full 128-image `coco128` validation split at 640 pixels, with the same letterbox and metric code as `val()`.

!!! tip "Performance"

    | Model   | Chip     | Precision        | mAP50 | mAP50-95 | Inference time (ms/im) |
    | ------- | -------- | ---------------- | ----- | -------- | ---------------------- |
    | yolo26n | `AX650`  | INT8 + FP32 tail | 0.635 | 0.455    | 1.45                   |
    | yolo26s | `AX650`  | INT8 + FP32 tail | 0.719 | 0.535    | 3.26                   |
    | yolo26m | `AX650`  | INT8 + FP32 tail | 0.767 | 0.588    | 8.78                   |
    | yolo26l | `AX650`  | INT8 + FP32 tail | 0.790 | 0.604    | 11.28                  |
    | yolo26x | `AX650`  | INT8 + FP32 tail | 0.804 | 0.635    | 24.36                  |
    | yolo26n | `AX8860` | BF16             | 0.641 | 0.471    | 0.92                   |
    | yolo26s | `AX8860` | BF16             | 0.738 | 0.553    | 1.63                   |
    | yolo26m | `AX8860` | BF16             | 0.776 | 0.600    | 3.47                   |
    | yolo26l | `AX8860` | BF16             | 0.786 | 0.602    | 4.36                   |
    | yolo26x | `AX8860` | BF16             | 0.813 | 0.642    | 11.07                  |

    Benchmarked with `ultralytics 8.4.216`

    !!! note

        Validation for the above benchmarks was done using the [COCO128](https://platform.ultralytics.com/ultralytics/datasets/coco128) dataset. Inference time is the pure NPU graph time from `ax_run_model -w 10 -r 100` on the device, excluding pre- and post-processing, and excluding the host-side decode in `tail.onnx`.

## About AXERA

[AXERA](https://www.axera-tech.com/) (Axera Semiconductor Co., Ltd., `0600.HK`) was founded in May 2019 and builds AI inference SoCs for on-device computing, edge AI inference and smart vehicles. It was listed on the main board of the Hong Kong Stock Exchange on February 10, 2026. Its in-house technologies include the AXNeutron mixed-precision NPU, the AXProton AI-ISP, and the Pulsar2 compiler and SDK.

<p align="center">
  <img width="50%" src="https://pulsar2-docs.readthedocs.io/zh-cn/latest/_images/deploy-pipeline.png" alt="Axera Pulsar2 export for NPU deployment">
</p>

The two chips on this page target different segments:

| Chip                           | Position                                                       | Key specifications                                                                                   | NPU architecture | Available `npu_mode`     |
| ------------------------------ | -------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------- | ---------------- | ------------------------ |
| `AX650` (AX650A/AX650N series) | Cost-sensitive INT8 vision and edge inference                  | Octa-core Cortex-A55; 72 TOPS@INT4 / 18 TOPS@INT8; 8K@30 ISP; H.264/H.265 8K@30 encode, 8K@60 decode | Neutron v3       | `NPU1` / `NPU2` / `NPU3` |
| `AX8860`                       | High-concurrency video analytics, local LLMs and multimodality | 256 TOPS (equivalent, TPP<1600); 16-core Cortex-A55 @1.65 GHz; 136 GB/s memory bandwidth; 8K codec   | Neutron v7       | `NPU1` / `NPU2` / `NPU4` |

The AX650 targets cost- and power-sensitive INT8 workloads; this page's AX650 export uses INT8 by default and compiles for all three NPU cores (`NPU3`). The AX8860 adds 16-bit float compute for models that lose too much accuracy in INT8; this page's AX8860 export uses BF16 by default and compiles for all four NPU cores (`NPU4`), and you can also select INT8 or `w8a16` explicitly.

Note that `axera` (AXERA, this page) and `axelera` (Axelera AI), which Ultralytics also supports, are different export formats; their chips, toolchains and artifacts are not interchangeable.

Further reading:

- [AXERA website](https://www.axera-tech.com/)
- [AXNeutron mixed-precision NPU](https://www.axera-tech.com/en/tech/npu)
- [AX650A product page](https://www.axera-tech.com/en/product/2929.html)
- [AX8860 product page](https://www.axera-tech.com/en/product/3294.html)
- [Pulsar2 toolchain documentation](https://pulsar2-docs.readthedocs.io/zh-cn/latest/)
- [AXERA-TECH/Pulsar2 releases](https://huggingface.co/AXERA-TECH/Pulsar2/tree/main)

## FAQ

### Do I need an AXERA device to export a model?

No. The export is a pure compile step on Linux, and the `axnn.simulator` can run the compiled graph on the host for reference accuracy.

### Can I run an AXERA model on my PC before copying it to the device?

Yes. Without `axengine`, `AxeraBackend` loads `compiled.axmodel` in the `axnn.simulator`, so `predict()` and `val()` work on x86 or aarch64 Linux. Report those numbers as simulator results, not NPU measurements.

### Why is the 16-bit build BF16 instead of FP16?

The export writes BF16 weights and activations because the quantizer does not accept `data_type: FP16`. The runtime still reports the precision as `16`.

### How many calibration images does an AXERA export use?

64 by default, taken from the start of the sorted `val` split. Pass `fraction` to use a ratio or an explicit count; there is no hard cap.

### Why does the artifact contain `tail.onnx`?

Fixed-point detect, segment, pose and OBB heads cannot keep box coordinates, class scores and task values in one activation scale. The quantized graph stops at the head convolutions and `tail.onnx` performs the decode in floating point on the host (mask prototypes for segmentation, keypoints for pose and angles for OBB). It is written for `quantize=8` and `quantize="w8a16"` detect, segment, pose and OBB builds; 16-bit float builds keep the whole model.
