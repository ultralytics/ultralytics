---
title: TFLite Export for Edge (Deprecated)
comments: true
description: YOLO TFLite (TensorFlow Lite) export is now LiteRT. See what format="tflite" exports since Ultralytics 8.4.83, how the .tflite file changed, and how to migrate.
keywords: YOLO26, TFLite, TensorFlow Lite, TFLite export, format=tflite, LiteRT, model export, edge devices, deployment, Ultralytics, machine learning, on-device inference, model optimization
---

# TFLite Model Export for Deployment (Deprecated)

TensorFlow Lite (TFLite) is now called [LiteRT](litert.md), and both names refer to the same `.tflite` model file format. Since Ultralytics 8.4.83, `format="tflite"` is a deprecated name for `format="litert"`: the export logs a warning and writes a LiteRT model. To migrate, export with `format="litert"` and update code that reads the file: for YOLO26n it is now `yolo26n.tflite`, takes a channels-first `(1, 3, 640, 640)` input, and returns raw predictions unless you pass `nms=False`. The sections below give the details, and the [LiteRT page](litert.md) documents the current export in full.

## What Changed

The table compares the export of the official YOLO26n checkpoint in Ultralytics 8.4.82, the last release with the earlier TFLite export, and in Ultralytics 8.4.215. Both exports ran on Linux x86-64 with Python 3.12.

| Property                           | Ultralytics 8.4.82: `format="tflite"`                                                                                                                                                       | Ultralytics 8.4.215: `format="tflite"` or `"litert"`                                                      |
| ---------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------- |
| Converter                          | `onnx2tf`, from an ONNX file through a TensorFlow SavedModel                                                                                                                                | `litert-torch`, directly from the PyTorch model                                                           |
| Packages the first export installs | TensorFlow, `onnx2tf`, `tf_keras`, ONNX, ONNX Runtime, and more                                                                                                                             | `litert-torch`, `ai-edge-litert`, `ai-edge-quantizer`, and their dependencies, such as JAX; no TensorFlow |
| Returned file                      | `yolo26n_saved_model/yolo26n_float32.tflite`                                                                                                                                                | `yolo26n.tflite`, next to the checkpoint                                                                  |
| Other files written                | The SavedModel and `yolo26n_float16.tflite` in the same directory, and `yolo26n.onnx` next to the checkpoint                                                                                | None                                                                                                      |
| `quantize=16`                      | Returns `yolo26n_saved_model/yolo26n_float16.tflite`                                                                                                                                        | Stops with an error                                                                                       |
| `quantize=8`                       | Returns `yolo26n_saved_model/yolo26n_int8.tflite`, with INT8 weights and floating-point activations, and also writes `yolo26n_integer_quant.tflite` and `yolo26n_full_integer_quant.tflite` | Returns `yolo26n_int8.tflite`, with INT8 weights and activations                                          |
| Input                              | `(1, 640, 640, 3)`, channels last (NHWC)                                                                                                                                                    | `(1, 3, 640, 640)`, channels first (NCHW)                                                                 |
| Default output                     | `(1, 300, 6)`: NMS-free rows with box coordinates normalized to 0-1                                                                                                                         | `(1, 84, 8400)`: raw predictions that need NMS                                                            |
| Output with `nms=False`            | Same as the default                                                                                                                                                                         | `(1, 300, 6)`: NMS-free rows with box coordinates in input pixels                                         |
| Output with `nms=True`             | Same as the default, with the warning `'nms=True' is not available for end2end models. Forcing 'nms=False'.` A YOLO11n export returns `(1, 300, 6)` from NMS embedded in the model          | The raw output, with a warning that `format=litert` does not support embedded NMS                         |

In both releases the class names, task, and image size are stored inside the `.tflite` file as a `metadata.json` entry.

## Behavior Since 8.4.83

A Python or CLI export with `format="tflite"`, such as `yolo export model=yolo26n.pt format=tflite`, first logs this warning:

```text
WARNING ⚠️ format='tflite' is deprecated as of 8.4.83 and has been replaced by the unified Google LiteRT format. Exporting format='litert' instead. See https://docs.ultralytics.com/integrations/litert
```

It then runs the LiteRT export and writes `yolo26n.tflite` next to the checkpoint.

- **Same model**: for FP32 and `quantize="w8a32"`, the model data in the file is byte-identical to the one `format="litert"` writes, and the two files differ only in the export time stored in the metadata. A calibrated `quantize=8` or `"w8a16"` export has the same size with both values, and its bytes can differ from one run to the next: `quantize=8` runs with the same value differed too.
- **Arguments and errors**: the arguments, checks, and messages are those of [LiteRT](litert.md#export-arguments), and the messages name `litert`: `quantize=16`, for example, stops with `quantize=16 (FP16) is not supported; format='litert' supports quantize=8 (INT8), 'w8a16' (INT8 weights + INT16 activations), 'w8a32' (dynamic INT8), 32 or None (FP32).`
- **Earlier precision flags**: `int8=True` still selects `quantize=8` and writes `yolo26n_int8.tflite`; `half=True` selects `quantize=16` and stops with the same message.
- **Export host**: the export runs on Linux x86-64 and on macOS with Apple silicon, with Python 3.10 or newer. On Windows and on Linux ARM64, such as a Raspberry Pi, it stops with `LiteRT export only supported on Linux x86 and macOS`.
- **Packages**: the first export installs `litert-torch`, `ai-edge-litert`, and `ai-edge-quantizer`. After a plain `pip install ultralytics`, that install can replace PyTorch and [stop the first export](litert.md#limitations-and-troubleshooting); installing `ultralytics[export-litert]` beforehand avoids it.
- **TF.js**: `format="tfjs"` is deprecated in the same way and writes the same `yolo26n.tflite` instead of a `yolo26n_web_model/` directory. See the [TF.js page](tfjs.md).

## Migrate to LiteRT

Install the LiteRT export packages with `pip install "ultralytics[export-litert]"`, then replace the `format` value:

!!! example "Export"

    === "Python"

        ```python
        from ultralytics import YOLO

        # Load a YOLO26 model
        model = YOLO("yolo26n.pt")

        # Export the model to LiteRT format
        model.export(format="litert")  # creates 'yolo26n.tflite'
        ```

    === "CLI"

        ```bash
        # Export a YOLO26n PyTorch model to LiteRT format
        yolo export model=yolo26n.pt format=litert # creates 'yolo26n.tflite'
        ```

For prediction, validation, and the version requirements, see the [LiteRT quickstart](litert.md#quickstart). Code that reads the `.tflite` file itself needs these changes:

- **File path**: load `yolo26n.tflite` instead of `yolo26n_saved_model/yolo26n_float32.tflite`.
- **Input**: pass the images as `(batch, 3, height, width)`, channels first, instead of `(batch, height, width, 3)`. The values stay RGB `float32` scaled to 0-1.
- **Output of YOLO26 detection models**: pass `nms=False` to keep the `(1, 300, 6)` rows of `x1, y1, x2, y2, score, class` in FP32 and `quantize="w8a32"` exports. Their box coordinates are now in input pixels, not normalized to 0-1. Exports without `nms`, and `quantize=8` and `"w8a16"` exports also with `nms=False`, write the raw `(1, 84, 8400)` output, which needs NMS and has box values normalized to 0-1. The other tasks have other shapes, see [Output Format](litert.md#output-format).
- **Embedded NMS**: `nms=True`, which embedded NMS in the earlier YOLO11 export and returned `(1, 300, 6)`, now writes the raw output. Run NMS in your own code, or load the file with `YOLO()`, which applies it.
- **`quantize=16`**: export FP32 instead. LiteRT has [no FP16 export](litert.md#quantization).
- **`quantize=8`**: the `yolo26n_int8.tflite` that the earlier export returned kept INT8 weights with floating-point activations. `quantize="w8a32"` is the scheme of that kind now, and `quantize=8` also quantizes the activations, calibrated on `data`, and writes the raw output. See [Quantization](litert.md#quantization).
- **Full-integer files**: the earlier `quantize=8` export also wrote `yolo26n_full_integer_quant.tflite` with `int8` input and output. All four current precisions keep `float32` input and output. For a Coral accelerator, use the [Edge TPU export](edge-tpu.md).
- **Tensor names**: the input and output of the float files were named `images` and `Identity`. Read tensors by index, not by name.

## Load Earlier TFLite Files

`YOLO()` in the current release loads `.tflite` files that the earlier export wrote. It detects the channels-last layout from the input tensor and reads the class names, task, and image size from the `metadata.json` inside the file, so a file also loads after you move it out of `yolo26n_saved_model/`.

```python
from ultralytics import YOLO

# Load a TFLite model exported with Ultralytics 8.4.82
model = YOLO("yolo26n_saved_model/yolo26n_float32.tflite")

# Run inference
results = model("https://ultralytics.com/images/bus.jpg")
```

The files of the Ultralytics 8.4.82 YOLO26n exports gave these results with `predict` on `bus.jpg` and `val` on [COCO8](../datasets/detect/coco8.md) in Ultralytics 8.4.215:

| File                                    | Written by    | Detections on `bus.jpg`           | mAP<sub>50-95</sub> on COCO8 |
| --------------------------------------- | ------------- | --------------------------------- | ---------------------------- |
| `yolo26n_float32.tflite`                | Every export  | The bus and four persons          | 0.687                        |
| `yolo26n_float16.tflite`                | Every export  | The bus and four persons          | 0.658                        |
| `yolo26n_int8.tflite`                   | `quantize=8`  | The bus and four persons          | 0.669                        |
| `yolo26n_integer_quant.tflite`          | `quantize=8`  | Misplaced boxes                   | 0.040                        |
| `yolo26n_full_integer_quant.tflite`     | `quantize=8`  | Misplaced boxes and wrong classes | 0.006                        |
| `yolo26n.tflite`, exported with 8.4.215 | For reference | The bus and four persons          | 0.669                        |

In the two `integer_quant` files of this NMS-free YOLO26n export, one quantization scale of 0.306 covers the box coordinates, the scores, and the class ids of the output, so box coordinates normalized to 0-1 take only a few distinct values. Use `yolo26n_int8.tflite` from the same directory, or export again. A YOLO11n file exported with `nms=True` in 8.4.82 also loaded and ran `predict` and `val`.

## Measured Performance (historical)

!!! note "Frozen before/after migration record"

    These results were recorded with the Ultralytics Flutter plugin `0.6.8` and LiteRT `2.1.5` on Android 16/API 36
    on a [Xiaomi 17](https://www.mi.com/global/product/xiaomi-17/) with 12 GB LPDDR5X memory. Its 3 nm
    [Snapdragon 8 Elite Gen 5](https://www.qualcomm.com/smartphones/products/8-series/snapdragon-8-elite-gen-5)
    (SM8850) has an 8-core Qualcomm Oryon CPU (2 Prime cores up to 4.6 GHz and 6 Performance cores up to 3.62 GHz),
    Adreno GPU, and Hexagon NPU. These tables compare the legacy onnx2tf INT8 TFLite assets with candidate
    litert-torch exports evaluated during the migration. The release assets were subsequently overwritten with the
    standardized exports, so these numbers do **not** describe the current `v0.7.0` asset bytes. See the
    [LiteRT performance tables](litert.md#measured-performance) for current measurements.

Both formats ran in the same back-to-back GPU sweep at the input sizes shown. Each cell is the total time
(preprocessing + inference + postprocessing), with the per-stage split beneath it; native logs confirmed that both
formats delegated the complete graph to LiteRT OpenCL (`LITERT_CL`) on the Adreno GPU.

| Model        | Task     | size<br><sup>(pixels)</sup> | Legacy<br><sup>onnx2tf INT8 TFLite<br>(ms)</sup> | Candidate<br><sup>w8a32 LiteRT<br>(ms)</sup> |
| ------------ | -------- | --------------------------- | ------------------------------------------------ | -------------------------------------------- |
| YOLO26n      | Detect   | 640                         | 14.0<br><sup>1.8 / 8.1 / 4.2</sup>               | **13.5**<br><sup>1.9 / 8.1 / 3.5</sup>       |
| YOLO26n-seg  | Segment  | 640                         | 30.1<br><sup>1.9 / 20.3 / 8.0</sup>              | **28.6**<br><sup>1.8 / 20.1 / 6.7</sup>      |
| YOLO26n-sem  | Semantic | 640                         | **26.4**<br><sup>1.9 / 16.4 / 8.1</sup>          | 32.9<br><sup>1.8 / 23.0 / 8.2</sup>          |
| YOLO26n-cls  | Classify | 224                         | 3.5<br><sup>0.9 / 2.2 / 0.4</sup>                | **3.2**<br><sup>1.0 / 2.2 / 0.1</sup>        |
| YOLO26n-pose | Pose     | 640                         | 17.4<br><sup>2.4 / 9.9 / 5.1</sup>               | **14.0**<br><sup>1.9 / 9.3 / 2.8</sup>       |
| YOLO26n-obb  | OBB      | 640                         | 13.9<br><sup>3.0 / 8.3 / 2.7</sup>               | **13.0**<br><sup>2.9 / 7.9 / 2.3</sup>       |

The same migration run also compared YOLO26n detect quantization formats. Inference is the comparable,
format-dependent metric:

| Android format                    | GPU inference (ms) | GPU-compiles |
| --------------------------------- | ------------------ | ------------ |
| onnx2tf INT8 (legacy TFLite)      | **8.6**            | yes          |
| LiteRT w8a32 (candidate)          | 8.4                | yes          |
| LiteRT INT8 (`quantize=8`)        | 11.0               | yes          |
| LiteRT FP32                       | 8.8                | yes          |
| LiteRT w8a16 (`quantize="w8a16"`) | CPU fallback       | no           |

In this run, w8a16 fell back to CPU at about 660 ms total versus about 17 ms for the GPU formats. These results
motivated shipping `w8a32`, but delegate compatibility and performance depend on the device and runtime version.
Benchmark the target device and confirm accelerator placement in its runtime logs.

## Next Steps

- **[LiteRT](litert.md)**: the current export, with its [quantization schemes](litert.md#quantization), [output format](litert.md#output-format), [Android, iOS, and browser deployment guides](litert.md#deployment-guides), [Intel GPU and NPU inference](litert.md#intel-gpu-and-npu-inference), and [current measurements](litert.md#measured-performance).
- **[TF SavedModel](tf-savedmodel.md)**: `format="saved_model"` still converts through `onnx2tf`. Load its `yolo26n_saved_model/` directory with `YOLO()`, not the channels-last `.tflite` files that the current `format="saved_model"` export writes inside it. Unlike the files of the 8.4.82 `format="tflite"` export above, their raw `(1, 84, 8400)` output has box coordinates in input pixels, and `YOLO()` placed every box in the bottom-right image corner when it loaded such a `yolo26n_float32.tflite`.
- **[Coral Edge TPU](edge-tpu.md)**: `format="edgetpu"` compiles a full-integer `.tflite` model for Coral accelerators.
- **[MediaTek Dimensity](mediatek.md)**: Android deployment of LiteRT models on MediaTek phones.

## FAQ

### Can I run a YOLO26 TFLite model on a Raspberry Pi?

Yes. Load the `.tflite` file with `YOLO("yolo26n.tflite")` on the board; the [Raspberry Pi guide](../guides/raspberry-pi.md#raspberry-pi-5-yolo26-benchmarks) lists LiteRT results on a Raspberry Pi 5. The export itself does not run on the board, because the LiteRT export needs Linux x86-64 or macOS: export on such a machine and copy the file over. For a Coral accelerator on the board, use the separate [Edge TPU export](edge-tpu.md).

### Can I run a YOLO26 TFLite model on a microcontroller?

Ultralytics does not test LiteRT exports with [LiteRT for Microcontrollers](https://developers.google.com/edge/litert/microcontrollers/overview). The smallest YOLO26n export, `quantize="w8a32"`, is a 2.9 MB file, and every precision keeps `float32` input and output. For STM32 microcontrollers, the [ST Edge AI export](stedgeai.md) writes a fully quantized INT8 model with `quantize=8` for the STMicroelectronics toolchain.
