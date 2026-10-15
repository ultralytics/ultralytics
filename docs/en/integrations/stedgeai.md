---
comments: true
description: Export Ultralytics YOLO to quantization-friendly float and QDQ INT8 ONNX with format="stedgeai" for STMicroelectronics STM32 devices through ST Edge AI Core and the STM32 AI Model Zoo.
keywords: ST Edge AI, STEdgeAI, format="stedgeai", STMicroelectronics, STM32, STM32N6, Neural-ART, STM32 AI Model Zoo, ST Edge AI Core, ONNX QDQ, INT8 quantization, per-channel quantization, microcontroller, MCU, edge AI, Ultralytics, YOLO26, YOLO11, YOLOv8, ONNX export
---

# ST Edge AI Export for Ultralytics YOLO Models

Exporting [Ultralytics YOLO](../models/index.md) models with `format="stedgeai"` produces [ONNX](onnx.md) models ready for [ST Edge AI Core](https://www.st.com/en/development-tools/stedgeai-core.html), STMicroelectronics' toolchain for deploying neural networks on STM32 microcontrollers and microprocessors such as the STM32N6 with its Neural-ART accelerator. The export writes a float ONNX model whose outputs are designed for INT8 quantization and, with `quantize=8`, an ONNX Runtime QDQ INT8 model calibrated on your data.

## Why a Dedicated ST Edge AI Format?

A YOLO detection head concatenates decoded box coordinates and class scores into one output tensor. In a standard ONNX export, box coordinates are in pixels (about `0` to `640`) while class scores are probabilities in `[0, 1]`. Post-training INT8 quantization assigns one scale to that tensor, so the small class-score values, and pose keypoint coordinates sharing the tensor, lose most of their resolution.

The ST Edge AI export normalizes box and keypoint coordinates by the input image size inside the head decode, before the concatenation, using the same normalization Ultralytics applies to its [Edge TPU](edge-tpu.md) exports. Every output channel then lies in a comparable `[0, 1]` range, so the whole graph, including the head, can be quantized to INT8 without a mixed-scale tensor.

Dense prediction heads stop before their final upsampling. Semantic segmentation emits raw class logits at 1/8 of the input resolution and depth estimation emits raw log-depth at 1/4 of the input resolution. This keeps the on-device output buffer 64x (semantic) and 16x (depth) smaller, avoids a full-resolution resize on the device, and keeps the INT8 depth output in the evenly spread log domain instead of after an exponential.

## Supported Tasks

All Ultralytics tasks are supported. Each task's output, at `imgsz=640`, is:

| Task                                          | Output                                                                                                      |
| :-------------------------------------------- | :---------------------------------------------------------------------------------------------------------- |
| [Detection](../tasks/detect.md)               | `[1, 4 + nc, anchors]`: boxes normalized to `[0, 1]`, class scores                                          |
| [Instance Segmentation](../tasks/segment.md)  | `[1, 4 + nc + 32, anchors]` with boxes normalized to `[0, 1]`, plus mask prototypes `[1, 32, 160, 160]`     |
| [Semantic Segmentation](../tasks/semantic.md) | Raw class logits `[1, nc, 80, 80]` (1/8 resolution); upsample before the per-pixel argmax                   |
| [Depth Estimation](../tasks/depth.md)         | Raw log-depth `[1, 1, 160, 160]` (1/4 resolution); decode with `exp(clip(x, -4, 5)) ** a * exp(b)` and upsample |
| [Classification](../tasks/classify.md)        | Class probabilities `[1, nc]`                                                                               |
| [Pose Estimation](../tasks/pose.md)           | `[1, 4 + nc + nk, anchors]`: boxes and keypoints normalized to `[0, 1]`                                    |
| [Oriented Detection](../tasks/obb.md)         | `[1, 4 + nc + 1, anchors]`: boxes normalized to `[0, 1]`, angle in radians; requires a square `imgsz`       |

The depth calibration `[a, b]` is stored in the model metadata as `depth_calibration`. Ultralytics predict and validate apply the depth decode and upsample semantic logits and depth maps automatically.

## Usage

The ST Edge AI format supports the [Export](../modes/export.md), [Predict](../modes/predict.md), and [Validate](../modes/val.md) modes. Predict and validate run the exported model with ONNX Runtime on the CPU and scale the normalized coordinates back to pixels, so you can measure float and INT8 accuracy on any host before deploying.

!!! example "Export"

    === "Python"

        ```python
        from ultralytics import YOLO

        # Load a YOLO26 model
        model = YOLO("yolo26n.pt")

        # Export a quantization-friendly float ONNX model
        model.export(format="stedgeai")  # creates 'yolo26n_stedgeai.onnx'

        # Also export a QDQ INT8 model calibrated on COCO8
        model.export(format="stedgeai", quantize=8, data="coco8.yaml")  # creates 'yolo26n_int8_stedgeai.onnx'
        ```

    === "CLI"

        ```bash
        # Export a quantization-friendly float ONNX model
        yolo export model=yolo26n.pt format=stedgeai # creates 'yolo26n_stedgeai.onnx'

        # Also export a QDQ INT8 model calibrated on COCO8
        yolo export model=yolo26n.pt format=stedgeai quantize=8 data=coco8.yaml # creates 'yolo26n_int8_stedgeai.onnx'
        ```

!!! example "Predict"

    === "Python"

        ```python
        from ultralytics import YOLO

        # Load the exported ST Edge AI model
        model = YOLO("yolo26n_int8_stedgeai.onnx")

        # Run inference
        results = model("https://ultralytics.com/images/bus.jpg")
        ```

    === "CLI"

        ```bash
        # Run inference with the exported ST Edge AI model
        yolo predict model=yolo26n_int8_stedgeai.onnx source='https://ultralytics.com/images/bus.jpg'
        ```

!!! example "Validate"

    === "Python"

        ```python
        from ultralytics import YOLO

        # Load the exported ST Edge AI model
        model = YOLO("yolo26n_int8_stedgeai.onnx")

        # Validate INT8 accuracy on the COCO8 dataset
        metrics = model.val(data="coco8.yaml")
        ```

    === "CLI"

        ```bash
        # Validate the exported ST Edge AI model
        yolo val model=yolo26n_int8_stedgeai.onnx data=coco8.yaml
        ```

### Export Arguments

| Argument   | Type             | Default      | Description                                                                                                                                                                                                                  |
| :--------- | :--------------- | :----------- | :--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `format`   | `str`            | `'stedgeai'` | Target format for the exported model, defining compatibility with ST Edge AI Core.                                                                                                                                           |
| `imgsz`    | `int` or `tuple` | `640`        | Desired image size for the model input, as an integer for square images or a `(height, width)` tuple. The exported model has a fixed batch size of 1 and static shapes. OBB models require a square `imgsz`.                  |
| `quantize` | `int` or `str`   | `None`       | Quantization precision. Unset or `32` exports the float model only; `8` also writes a QDQ INT8 model.                                                                                                                       |
| `data`     | `str`            | `None`       | Dataset YAML used for INT8 calibration; classification instead takes a dataset directory or a built-in dataset name. If omitted with `quantize=8`, Ultralytics uses the task's default dataset, such as `coco8.yaml` for detection. |
| `fraction` | `float`          | `1.0`        | Fraction of the calibration dataset to use.                                                                                                                                                                                  |
| `opset`    | `int`            | `None`       | ONNX opset. Defaults to `18` and is capped at `20`, the highest opset ST Edge AI Core supports; INT8 export requires `13` or higher.                                                                                          |
| `simplify` | `bool`           | `True`       | Simplifies the ONNX graph with `onnxslim`.                                                                                                                                                                                   |
| `device`   | `str`            | `None`       | Specifies the device for exporting: GPU (`device=0`) or CPU (`device=cpu`).                                                                                                                                                  |

The exported models never contain NMS: `nms=True` and the YOLO26 NMS-free head (`nms=False`) add TopK and gather operators that ST Edge AI Core does not compile, so both fall back to raw outputs for external NMS. `dynamic=True` and `batch` values other than `1` are rejected.

!!! tip "Calibrate on your own data"

    INT8 accuracy depends on how well the calibration images represent the deployment scene. Pass `data=` a dataset YAML with at least 300 representative images from your own cameras; the default calibration datasets are small public samples.

### Output Structure

```text
yolo26n_stedgeai.onnx        # float model with input-normalized coordinates and Ultralytics metadata
yolo26n_int8_stedgeai.onnx   # QDQ INT8 model, written only with quantize=8
```

Both file names end in `_stedgeai.onnx`, which is how Ultralytics recognizes them and applies the coordinate scaling at predict and validate time. Renaming them to a plain `.onnx` name would load them as standard ONNX models and return boxes in `[0, 1]` instead of pixels.

### How the INT8 Quantization Works

With `quantize=8`, the float model is quantized with [ONNX Runtime](https://onnxruntime.ai/) static post-training quantization, matching the recipe used by the [STM32 AI Model Zoo Services](https://github.com/STMicroelectronics/stm32ai-modelzoo-services):

- QDQ (QuantizeLinear/DequantizeLinear) format, with float model inputs and outputs.
- Symmetric per-channel INT8 weights.
- Asymmetric per-tensor INT8 activations.
- MinMax calibration over the `data` images, preprocessed exactly as during Ultralytics validation.

The float model is kept beside the INT8 model, so you can also quantize it with the STM32 AI Model Zoo Services or ST Edge AI Core instead.

## Deploying to STM32

1. **Export** the model with `format="stedgeai"`, using the same `imgsz` you trained with and intend to run on the device.
2. **Validate** the float and INT8 models on your dataset with `yolo val` to confirm accuracy before deployment.
3. **Compile and deploy** the ONNX model with ST Edge AI Core or the STM32 AI Model Zoo Services for your STM32 target.

!!! warning "Outputs need post-processing"

    Any consumer of the exported ONNX model outside Ultralytics, such as a custom ONNX Runtime script or on-device C post-processing, must treat box and keypoint coordinates as normalized to `[0, 1]` and multiply them by the input width and height, upsample semantic logits, and decode and upsample depth as described in [Supported Tasks](#supported-tasks). Keep `imgsz` identical across training, export, quantization and deployment, since a mismatch silently rescales every box.

## Summary

The `format="stedgeai"` export turns Ultralytics YOLO models into quantization-friendly float ONNX and QDQ INT8 ONNX models for STMicroelectronics STM32 devices. Box and keypoint coordinates are normalized by the input size so the whole graph quantizes to INT8 with a comparable range on every output channel, semantic and depth heads stop before their final upsampling, and Ultralytics predict and validate restore pixel coordinates, full-resolution maps and metric depth so you can check accuracy before deploying with ST Edge AI Core.

## FAQ

### How do I export a YOLO model for STM32 devices?

Run `yolo export model=yolo26n.pt format=stedgeai quantize=8 data=coco8.yaml`. This writes a float `yolo26n_stedgeai.onnx` model and a QDQ INT8 `yolo26n_int8_stedgeai.onnx` model for ST Edge AI Core. Omit `quantize=8` to export only the float model.

### Why are the box coordinates in my exported model between 0 and 1?

The ST Edge AI export normalizes box and keypoint coordinates by the input image size so they share the `[0, 1]` range of the class scores, which keeps INT8 quantization accurate. Multiply the `x` and `w` values by the input width and the `y` and `h` values by the input height to recover pixel coordinates. Ultralytics does this automatically when it loads a `_stedgeai.onnx` model.

### Does the ST Edge AI export include NMS?

No. Embedded NMS and the YOLO26 NMS-free head add operators that ST Edge AI Core does not compile, so the export always produces raw outputs and NMS runs in your application or in the STM32 AI Model Zoo post-processing.

### How does the ST Edge AI export differ from the standard ONNX export?

The standard [ONNX export](onnx.md) emits box coordinates in pixels and, with `quantize=8`, quantizes only convolution and matrix multiplication layers while keeping the head in float. The ST Edge AI export emits normalized coordinates, raw low-resolution semantic logits and log-depth, quantizes the whole graph in the QDQ format with per-channel weights, fixes the batch size to 1 and never embeds NMS.
