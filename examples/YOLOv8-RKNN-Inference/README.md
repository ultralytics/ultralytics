# YOLOv8 - RKNN Inference

This repository provides an example implementation for running [Ultralytics YOLOv8](https://docs.ultralytics.com/models/yolov8) models on Rockchip RKNN NPUs (e.g. RK3566, RK3568, RK3576, RK3588) using [rknn-toolkit2](https://github.com/airockchip/rknn-toolkit2). It includes:

- `export_onnx.py` — export a YOLOv8 `.pt` to a **separate-head** ONNX graph for RKNN INT8 quantization,
- `rknn_inference.py` — image and video inference with a `.rknn` model (runs on the PC simulator or on the NPU device),
- `requirements.txt` — Python dependencies.

> **Tested on**: Orange Pi 5 Max (RK3588). Use `--target rk3588` when running on the device.

> The scripts are adapted from the author's `detect_rknn.py` / `detect_video_rknn.py` /
> `export_6out_onnx.py` (image / video inference and separate-head export).

## Why a separate-head ONNX?

The default Ultralytics ONNX export decodes the boxes and concatenates all
predictions into a single `[1, 4+nc, 8400]` tensor. The separate-head export keeps
per-scale tensors instead:

```
6-output (recommended):  [box(1, 4*reg_max, h, w), Sigmoid(cls)(1, nc, h, w)] x 3
9-output:                [box, cls_logit, score_sum] x 3
```

Keeping `Sigmoid(cls)` fused in the graph (without `score_sum`) lets rknn-toolkit2
fuse the sigmoid into the preceding convolution during INT8 quantization. This
preserves the confidence scale — otherwise the INT8 confidence is compressed at
the top and stair-stepped (see the discussion in
[airockchip/rknn_model_zoo yolov8](https://github.com/airockchip/rknn_model_zoo/tree/main/examples/yolov8)).

## ⚙️ Installation

1. **rknn-toolkit2** — not pip-installable; download it from Rockchip:

   ```bash
   git clone https://github.com/airockchip/rknn-toolkit2.git
   cd rknn-toolkit2/rknn-toolkit2
   pip install -r packages/requirements_cp310-*.txt
   pip install packages/rknn_toolkit2-*-cp310-*.whl
   ```

   `rknn-toolkit2` runs on the PC (simulator) or drives an RKNPU device over the
   network (via `--target`). This example uses the `rknn.api.RKNN` interface; to
   run on-device with the lite runtime (`rknnlite.api.RKNNLite`), adapt the
   `load_rknn()` helper accordingly.

2. **Example dependencies**:

   ```bash
   git clone https://github.com/ultralytics/ultralytics.git
   cd ultralytics/examples/YOLOv8-RKNN-Inference
   pip install -r requirements.txt
   pip install -e ../..  # local ultralytics package, imported by export_onnx.py
   ```

## 🚀 Usage

### 1. Export the separate-head ONNX

```bash
python export_onnx.py yolov8n.pt yolov8n_6out.onnx           # 6-output (recommended)
python export_onnx.py yolov8n.pt yolov8n_9out.onnx --heads 9 # 9-output
```

`--imgsz` defaults to 640 and must match the size used for conversion and
inference (pass the same value to `rknn_inference.py --imgsz`).

### 2. Convert ONNX to RKNN (with rknn-toolkit2)

Use the following minimal script (or the `convert.py` from
[airockchip/rknn_model_zoo](https://github.com/airockchip/rknn_model_zoo)):

```python
from rknn.api import RKNN

target_platform = "rk3588"  # RK3566 / RK3568 / RK3576 / RK3588 - your board
rknn = RKNN(verbose=False)
rknn.config(mean_values=[[0, 0, 0]], std_values=[[255, 255, 255]], target_platform=target_platform)
rknn.load_onnx(model="yolov8n_6out.onnx")
rknn.build(do_quantization=True, dataset="./dataset.txt")  # list of calibration images
rknn.export_rknn("yolov8n_6out.rknn")
rknn.release()
```

Set `target_platform` to your board and pass the **same** value to `--target` at
inference time. Use a calibration dataset that matches your deployment
preprocessing (letterbox gray-114) so the INT8 confidence is not flattened — see
the rknn_model_zoo yolov8 README, section "Quantization notes".

### 3. Run inference

On the PC (simulator):

```bash
python rknn_inference.py --model yolov8n_6out.rknn --image bus.jpg [img2.jpg ...]
python rknn_inference.py --model yolov8n_6out.rknn --video demo.mp4 --out result.avi
```

On an NPU device (e.g. RK3588):

```bash
python rknn_inference.py --model yolov8n_6out.rknn --target rk3588 --image bus.jpg
```

The 6-output and 9-output layouts (3 or 4 scales) are detected automatically from
the model's output count. Annotated results are saved next to the input
(`*_rknn.jpg` / `*_rknn.avi`).

## 🖼️ Expected Results

Running the exported 6-output `yolov8n` on a standard test image such as
[`ultralytics/assets/bus.jpg`](../../assets/bus.jpg) produces output similar to:

```
person @ (211 241 283 507) 0.88
person @ (109 235 224 536) 0.88
person @ (477 223 560 521) 0.87
bus   @ (99 135 550 456) 0.85
```

## 🤝 Contributing

This example is part of the Ultralytics community examples. Contributions are
welcome — see the repository [contributing guidelines](../../CONTRIBUTING.md).
