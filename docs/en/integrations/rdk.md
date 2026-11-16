---
comments: true
description: Export Ultralytics YOLO26 models to the D-Robotics RDK X5 with format=rdk. Covers the hb_mapper toolchain, INT8 calibration, output files, tail.onnx, the x86 simulator, and deployment on the RDK X5 BPU.
keywords: YOLO26, D-Robotics, RDK X5, RDK export, BPU, hb_mapper, rdkx5-yolo-mapper, rdkx5-sim, NV12, INT8 quantization, edge AI, robotics, embedded devices, model deployment
---

# D-Robotics RDK Export

The [D-Robotics RDK X5](https://developer.d-robotics.cc/en/rdkx5) is a robotics development board whose BPU (Brain Processing Unit) accelerates quantized neural networks. Exporting an [Ultralytics YOLO26](../models/yolo26.md) model with `format="rdk"` creates a `yolo26n_rdk_model/` directory that holds an INT8 `.bin` model compiled for the RDK X5 BPU with the D-Robotics `hb_mapper` compiler.

## At a Glance

| Property          | Details                                                                                                                                               |
| ----------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------- |
| `format` argument | `rdk`                                                                                                                                                 |
| Output            | `yolo26n_rdk_model/`, a directory with `yolo26n.bin` and `metadata.yaml`; detect, segment, pose, and OBB models add [`tail.onnx`](#output-format)     |
| Runs on           | The BPU of the D-Robotics RDK X5                                                                                                                      |
| Runtime           | The D-Robotics runtime on the board, or the `rdkx5-sim` simulator on an x86-64 Linux PC. Ultralytics exports RDK models and does not load them        |
| Precision         | INT8 only, calibrated on `data`. YOLO26n is 3.4 MB                                                                                                    |
| Export host       | Linux x86-64 with the `rdkx5-yolo-mapper` package in a separate Python 3.10 environment. Export on a PC and copy the exported directory to the RDK X5 |

## Quickstart

The export calls the `hb_mapper` compiler from the [`rdkx5-yolo-mapper`](https://pypi.org/project/rdkx5-yolo-mapper/) package. That package runs only on Python 3.10 and pins old versions of `numpy`, `onnx`, `protobuf`, and `opencv-python`, so install it in its own virtual environment on an x86-64 Linux machine and add that environment to the end of `PATH`:

```bash
python3.10 -m venv rdk-mapper
rdk-mapper/bin/pip install rdkx5-yolo-mapper==1.0.1
export PATH="$PATH:$PWD/rdk-mapper/bin"
hb_mapper --version # hb_mapper, version 1.24.3
```

Install the Ultralytics package in your usual environment with `pip install ultralytics onnx onnxslim onnxruntime`; see the [installation guide](../quickstart.md) for other methods. Then export your model on the PC and copy the exported directory to the board.

!!! example "Export"

    === "Python"

        ```python
        from ultralytics import YOLO

        # Load a YOLO26 model
        model = YOLO("yolo26n.pt")

        # Export the model to RDK format, calibrated on your dataset
        model.export(format="rdk", data="coco8.yaml")  # creates 'yolo26n_rdk_model/'
        ```

    === "CLI"

        ```bash
        # Export a YOLO26n PyTorch model to RDK format, calibrated on your dataset
        yolo export model=yolo26n.pt format=rdk data=coco8.yaml # creates 'yolo26n_rdk_model/'
        ```

The export of YOLO26n at `imgsz=640` took about 2.5 minutes with the 4 validation images of `coco8.yaml` and about 6.5 minutes with 40 images, on a server CPU. Most of that time is calibration and compilation inside `hb_mapper`.

## Supported Tasks

RDK export works for all seven Ultralytics tasks with [YOLO26](../models/yolo26.md) models: [detection](../tasks/detect.md), [segmentation](../tasks/segment.md), [semantic segmentation](../tasks/semantic.md), [depth estimation](../tasks/depth.md), [classification](../tasks/classify.md), [pose estimation](../tasks/pose.md), and [OBB](../tasks/obb.md). A [YOLO11](../models/yolo11.md) detection model was also exported. Accuracy was measured for detection and classification, see [Accuracy](#accuracy); the other tasks were compiled and their outputs checked for shape only.

## Export Arguments

| Argument   | Type                      | Default | Description                                                                                                                                                                                                                                                                                                     |
| ---------- | ------------------------- | ------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `format`   | `str`                     | `'rdk'` | Target format for the exported model, defining compatibility with various deployment environments.                                                                                                                                                                                                              |
| `imgsz`    | `int` or `tuple`          | `640`   | Image size of the model input: an integer for square images or a tuple `(height, width)`. When not passed, the export uses the size the checkpoint was trained at.                                                                                                                                              |
| `data`     | `str`                     | `None`  | Dataset YAML used for INT8 calibration; classification instead takes a dataset directory or a built-in dataset name. Calibration uses the `val` split unless `split` is set. If omitted, Ultralytics uses the small default dataset for the model task, such as `coco8.yaml` for detection.                     |
| `fraction` | `float`, `int`, or `list` | `1.0`   | Calibration subset: a ratio, an image count (integers above `1`), or a `[train, val, test]` list, from which the entry of the calibration split is used.                                                                                                                                                        |
| `nms`      | `bool`, optional          | `None`  | `None` (default) returns raw predictions for external NMS. `False` exports the NMS-free head of models that have one, such as YOLO26, and `tail.onnx` then returns up to `max_det` detections. Embedded NMS (`True`) is not available, see [Limitations and Troubleshooting](#limitations-and-troubleshooting). |

RDK models are always INT8 with a batch size of 1 and a fixed input shape, so `batch`, `dynamic`, `opset`, and `simplify` are not accepted, and `quantize` accepts only `8`. For the arguments shared by all formats, see the [Export mode documentation](../modes/export.md).

### Calibration

`hb_mapper` reads the calibration images to choose the INT8 ranges of the model, so they should look like the images the model will see in deployment. The export logs `>300 images recommended for INT8 calibration` for a smaller set. `coco8.yaml` has only 4 validation images, which is enough to test the export but not to calibrate a model for deployment. Calibration is the slowest part of the export, and its time grows with the number of images.

## Output Format

The export writes one directory:

| File            | Contents                                                                                                                                         |
| --------------- | ------------------------------------------------------------------------------------------------------------------------------------------------ |
| `yolo26n.bin`   | The INT8 model for the RDK X5 BPU                                                                                                                |
| `tail.onnx`     | Only for detect, segment, pose, and OBB models: the operations that decode the head outputs into raw predictions, to run on the CPU in `float32` |
| `metadata.yaml` | Class names, task, image size, stride, and export arguments                                                                                      |

The intermediate `yolo26n.onnx` file stays next to the directory. The `.bin` model takes one image in NV12 format, the layout that the camera pipeline of the board delivers: a `uint8` array of shape `(height * 3 / 2, width)` that holds the Y plane followed by the interleaved U and V planes. The conversion to RGB and the division by 255 are part of the model. All outputs are `float32`.

The shapes below are for the official YOLO26n checkpoints at their default export size, with the Cityscapes checkpoint `yolo26n-sem.pt` for semantic segmentation:

| Task     | Default `imgsz` | `.bin` output                                           | `tail.onnx` output                       |
| -------- | --------------- | ------------------------------------------------------- | ---------------------------------------- |
| Detect   | `640`           | 6 head tensors                                          | `(1, 84, 8400)`                          |
| Segment  | `640`           | 9 head tensors and `(1, 32, 160, 160)`                  | `(1, 116, 8400)` and `(1, 32, 160, 160)` |
| Semantic | `1024`          | `(1, 19, 1024, 1024)`, a score for each class and pixel | No `tail.onnx`                           |
| Depth    | `768`           | `(1, 1, 768, 768)`, depth map                           | No `tail.onnx`                           |
| Classify | `224`           | `(1, 1000, 1, 1)`, class probabilities                  | No `tail.onnx`                           |
| Pose     | `640`           | 9 head tensors                                          | `(1, 56, 8400)`                          |
| OBB      | `1024`          | 9 head tensors                                          | `(1, 20, 21504)`                         |

- **Head tensors**: a detection model returns two tensors for each of the three detection scales, one with box values and one with class scores, named `out0` to `out5`. Segmentation, pose, and OBB models return one more tensor per scale with their extra values. Feed them to the inputs of `tail.onnx` that have the same names; a segmentation model also passes its mask prototypes through `tail.onnx` unchanged.
- **Raw predictions**: `tail.onnx` returns the same raw predictions as a default ONNX export: for each anchor the box center x, center y, width, and height in input pixels, then the class scores, then the values of the task. Apply NMS yourself. See the [ONNX output format](onnx.md#output-format) for the layout of each task.
- **Why `tail.onnx`**: the box decoding works with pixel-range values that a single INT8 scale cannot hold. A YOLO26n model compiled in one piece scored 0.262 mAP<sup>val</sup> 50-95 on the images of the [Accuracy](#accuracy) section, against 0.394 with the decoding in `tail.onnx`.
- **NMS-free output**: with `nms=False`, `tail.onnx` of a YOLO26 detection model returns `(1, 300, 6)`: `max_det` rows of `x1, y1, x2, y2, score, class`.

### Run with the x86 Simulator

The `rdkx5-yolo-mapper` package includes `rdkx5-sim`, which runs a compiled `.bin` model on an x86-64 Linux PC, so you can check a model before you copy it to the board. It takes the NV12 image from an NPZ file and writes the outputs to another. `rdkx5-sim yolo26n_rdk_model/yolo26n.bin --describe` prints the names, shapes, and data types of the inputs and outputs.

```python
# Export first: yolo export model=yolo26n.pt format=rdk data=coco8.yaml
import subprocess

import cv2
import numpy as np
import onnxruntime as ort

from ultralytics.data.augment import LetterBox

image = LetterBox((640, 640))(image=cv2.imread("bus.jpg"))  # BGR, 640x640
i420 = cv2.cvtColor(image, cv2.COLOR_BGR2YUV_I420).reshape(-1)  # Y plane, then U, then V
nv12 = np.concatenate((i420[:409600], i420[409600:].reshape(2, -1).T.reshape(-1))).reshape(960, 640)

np.savez("input.npz", images=nv12)
cmd = ["rdkx5-sim", "yolo26n_rdk_model/yolo26n.bin", "--inputs", "input.npz", "--outputs", "output.npz"]
subprocess.run(cmd, check=True)
outputs = np.load("output.npz")  # six head tensors named out0 to out5

tail = ort.InferenceSession("yolo26n_rdk_model/tail.onnx", providers=["CPUExecutionProvider"])
feed = {t.name: outputs[t.name].reshape(t.shape) for t in tail.get_inputs()}
predictions = tail.run(None, feed)[0]  # shape (1, 84, 8400), raw predictions that need NMS
assert predictions.shape == (1, 84, 8400), "export a detection model at imgsz=640"
```

The simulator is for checking accuracy, not speed: it took several seconds per image at `imgsz=640`. Measure latency on the board.

## Accuracy

INT8 quantization changes the outputs, so validate an exported model on your own data. These results are from the x86 simulator, with the decoding in `tail.onnx` and the standard Ultralytics NMS and metrics:

| Model       | Validation images               | Calibration images | Metric                  | PyTorch FP32  | RDK INT8      |
| ----------- | ------------------------------- | ------------------ | ----------------------- | ------------- | ------------- |
| YOLO26n     | 500 from COCO val2017           | 40                 | mAP<sup>val</sup> 50-95 | 0.442         | 0.394         |
| YOLO26n     | 500 from COCO val2017           | 300                | mAP<sup>val</sup> 50-95 | 0.442         | 0.385         |
| YOLO26n-cls | 1000 from ImageNet val, 1/class | 200                | top-1 / top-5 accuracy  | 0.702 / 0.885 | 0.680 / 0.878 |

The calibration images are other images of the same validation sets; calibrating YOLO26n on 300 images in place of 40 did not improve the result. The models have not been run on RDK X5 hardware.

## Deploy

Copy the `yolo26n_rdk_model/` directory to the RDK X5 and run `yolo26n.bin` with the D-Robotics runtime of the board; see the [RDK documentation](https://developer.d-robotics.cc/rdk_doc/en/RDK) and the [RDK Model Zoo](https://github.com/D-Robotics/rdk_model_zoo) for the runtime APIs and examples. For a detect, segment, pose, or OBB model, pass the outputs to `tail.onnx` with [ONNX Runtime](https://onnxruntime.ai/) as in the simulator example above, then apply NMS.

- **Ultralytics does not load RDK models**: `YOLO("yolo26n_rdk_model")` fails, see [Limitations and Troubleshooting](#limitations-and-troubleshooting). Run the model in your own code, and read the class names and image size from `metadata.yaml`.
- **Fixed input shape**: the model always runs at its export `imgsz` with a batch size of 1. Export again to change the image size.
- **CPU operations**: `hb_mapper` leaves a few operations on the CPU of the board, such as the Softmax of the attention blocks in YOLO26 and YOLO11 models. Its log lists them.

## Limitations and Troubleshooting

- **Export host**: export on an x86-64 Linux machine. `rdkx5-yolo-mapper` has builds only for x86-64 Linux with Python 3.10 and glibc 2.35 or newer.
- **`hb_mapper` not found**: without the compiler on `PATH`, the export fails with `RDK export requires the D-Robotics 'hb_mapper' compiler, which was not found on PATH.` Add the `bin` directory of the toolchain environment to the end of `PATH`, as in the [Quickstart](#quickstart).
- **Executable stack error**: on a system with glibc 2.41, such as Debian 13, `hb_mapper` fails with `cannot enable executable stack as shared object requires: Invalid argument`. The export was tested on Debian 12, which has glibc 2.36.
- **RDK models do not load in Ultralytics**: predicting or validating with `YOLO("yolo26n_rdk_model")` fails with `model='yolo26n_rdk_model' is not a supported model format.` Use the [simulator](#run-with-the-x86-simulator) on a PC or the D-Robotics runtime on the board.
- **Precision**: an export without `quantize` logs `rdk export requires INT8 quantization, enabling it.` `quantize=16` and `quantize=32` fail with `format='rdk' supports quantize=8 (INT8).`
- **Unsupported arguments**: `batch`, `dynamic`, `opset`, and `simplify` fail with, for example, `argument 'batch' is not supported for format='rdk'`. The intermediate ONNX graph always uses opset 11, the newest that `hb_mapper` accepts.
- **Embedded NMS**: `nms=True` logs `format=rdk does not support embedded NMS; exporting native outputs for external NMS.` and writes the raw output.
- **Calibration without `data`**: the export of a detection model logs `INT8 export requires a missing 'data' arg for calibration. Using default 'data=coco8.yaml'.` and calibrates on that small dataset; the other tasks name their own default dataset. Pass `data` with images that represent your deployment data.
- **Output directory**: every export of one checkpoint writes the same `yolo26n_rdk_model/` directory and deletes the previous contents first. Rename the directory to keep an earlier export.
- **Other D-Robotics boards**: the export compiles for the BPU of the RDK X5 only.

For other errors, see the [Common Issues guide](../guides/yolo-common-issues.md).

## About D-Robotics RDK

[D-Robotics](https://en.d-robotics.cc/) builds processors and development kits for robots. The RDK X5 is its development board with a BPU for neural networks, and the `hb_mapper` compiler converts ONNX models to the `.bin` format that the BPU runs.

## FAQ

### How do I export a YOLO26 model for the D-Robotics RDK X5?

Install `rdkx5-yolo-mapper` in a separate Python 3.10 environment on an x86-64 Linux PC, add it to the end of `PATH`, and run `yolo export model=yolo26n.pt format=rdk data=coco8.yaml` with your own dataset in `data`. The export writes `yolo26n_rdk_model/`, which you copy to the board. See the [Quickstart](#quickstart).

### Why does the export need calibration data?

The BPU runs INT8 models, and `hb_mapper` sets the INT8 range of each layer from the values it sees on real images. Pass a dataset with images like the ones the model will see in deployment; see [Calibration](#calibration).

### Can I test an RDK model on my PC before I copy it to the board?

Yes, with `rdkx5-sim` from the `rdkx5-yolo-mapper` package, which runs the compiled `.bin` on an x86-64 Linux PC. See [Run with the x86 Simulator](#run-with-the-x86-simulator). `YOLO()` does not load RDK models.

### Can I run a YOLO26 model on other edge accelerators?

Yes, with the export format of each accelerator: an RDK model runs only on the RDK X5 BPU. See [Rockchip RKNN](rockchip-rknn.md), [Hailo](hailo.md), [Axelera](axelera.md), [DEEPX](deepx.md), and [TensorRT](tensorrt.md) for NVIDIA Jetson, or the [model deployment options guide](../guides/model-deployment-options.md) for an overview.
