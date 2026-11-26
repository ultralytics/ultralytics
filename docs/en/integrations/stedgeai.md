---
comments: true
description: Export Ultralytics YOLO models with format="stedgeai" to float or INT8 ONNX for STMicroelectronics STM32 devices, check accuracy on your computer, and compile with ST Edge AI Core.
keywords: ST Edge AI, STEdgeAI, format="stedgeai", STMicroelectronics, STM32, STM32N6, Neural-ART, STM32 AI Model Zoo, ST Edge AI Core, ONNX QDQ, INT8 quantization, microcontroller, MCU, edge AI, Ultralytics, Ultralytics Platform, YOLO26, YOLO11, ONNX export
---

# ST Edge AI Export for Ultralytics YOLO Models

Export [Ultralytics YOLO](../models/index.md) models with `format="stedgeai"` to get an [ONNX](onnx.md) model shaped for [ST Edge AI Core](https://stedgeai-dc.st.com/assets/embedded-docs/index.html), the STMicroelectronics toolchain that compiles neural networks for STM32 microcontrollers, including the STM32N6 with its Neural-ART accelerator. The export writes a float model by default and a fully quantized INT8 model with `quantize=8`. Ultralytics runs either file on your computer with `yolo predict` and `yolo val`, so you can measure accuracy before you compile for the device.

!!! note "Validated on the host"

    Ultralytics validates these models with ONNX Runtime on a computer, and STMicroelectronics compiled the INT8 exports of all seven tasks for the STM32N6 with ST Edge AI Core. Measuring accuracy, latency and memory on an STM32 board are steps you run with the ST tools; see [Deploy to STM32](#deploy-to-stm32).

## Why a Dedicated ST Edge AI Format?

Microcontrollers run INT8 models, and a standard ONNX export does not quantize well end to end:

- **Mixed ranges in one output.** A YOLO detection head joins box coordinates in pixels (about `0` to `640`) and class scores (`0` to `1`) in a single tensor. INT8 quantization gives that tensor one scale, so the scores lose almost all of their resolution. The ST Edge AI export normalizes box and keypoint coordinates to `[0, 1]` by the input size, the same normalization Ultralytics uses for [Edge TPU](edge-tpu.md), so the whole graph can be quantized.
- **Large dense outputs.** Semantic segmentation and depth models normally upsample to the input resolution inside the graph. The ST Edge AI export stops before that upsampling, which makes the output buffer 64x smaller for semantic segmentation and 16x smaller for depth, and it keeps depth in the log domain, where INT8 levels are spread evenly.
- **Post-processing in the graph.** Embedded NMS and the YOLO26 NMS-free head add TopK and gather operators after the head. The export leaves them out and always returns raw predictions, so NMS runs in your application.

If you want the standard layout instead (pixel coordinates, full-resolution maps, optional embedded NMS), use the [ONNX export](onnx.md).

## Supported Tasks

All Ultralytics YOLO tasks export and run on the host. Output shapes are for `imgsz=640`:

| Task                                          | Output                                                                                               |
| :-------------------------------------------- | :--------------------------------------------------------------------------------------------------- |
| [Detection](../tasks/detect.md)               | `[1, 4 + nc, 8400]`: boxes normalized to `[0, 1]`, class scores                                      |
| [Instance Segmentation](../tasks/segment.md)  | `[1, 4 + nc + 32, 8400]` with boxes normalized to `[0, 1]`, plus mask prototypes `[1, 32, 160, 160]` |
| [Semantic Segmentation](../tasks/semantic.md) | Class logits `[1, nc, 80, 80]` at 1/8 resolution; upsample, then take the per-pixel argmax           |
| [Depth Estimation](../tasks/depth.md)         | Log-depth `[1, 1, 160, 160]` at 1/4 resolution; apply `exp(x)` for metric depth, then upsample       |
| [Classification](../tasks/classify.md)        | Class probabilities `[1, nc]`                                                                        |
| [Pose Estimation](../tasks/pose.md)           | `[1, 4 + nc + nk, 8400]`: boxes and keypoints normalized to `[0, 1]`                                 |
| [Oriented Detection](../tasks/obb.md)         | `[1, 4 + nc + 1, 8400]`: boxes normalized to `[0, 1]`, angle in radians; requires a square `imgsz`   |

For instance segmentation, the export divides mask coefficients by 16 and multiplies prototypes by 16. This keeps the coefficients close to the normalized boxes in the shared INT8 tensor. A custom decoder can multiply the coefficients and prototypes from the same export as usual. The factors cancel before quantization.

RT-DETR and YOLO-NAS models are not supported.

## Usage

!!! example "Export"

    === "Python"

        ```python
        from ultralytics import YOLO

        # Load a YOLO26 model
        model = YOLO("yolo26n.pt")

        # Export a float model
        model.export(format="stedgeai")  # creates 'yolo26n_stedgeai.onnx'

        # Export an INT8 model calibrated on COCO8
        model.export(format="stedgeai", quantize=8, data="coco8.yaml")  # creates 'yolo26n_int8_stedgeai.onnx'
        ```

    === "CLI"

        ```bash
        # Export a float model
        yolo export model=yolo26n.pt format=stedgeai # creates 'yolo26n_stedgeai.onnx'

        # Export an INT8 model calibrated on COCO8
        yolo export model=yolo26n.pt format=stedgeai quantize=8 data=coco8.yaml # creates 'yolo26n_int8_stedgeai.onnx'
        ```

!!! example "Predict and validate on the host"

    === "Python"

        ```python
        from ultralytics import YOLO

        # Load the exported ST Edge AI model
        model = YOLO("yolo26n_int8_stedgeai.onnx")

        # Run inference
        results = model("https://ultralytics.com/images/bus.jpg")

        # Measure INT8 accuracy on the COCO8 dataset
        metrics = model.val(data="coco8.yaml")
        ```

    === "CLI"

        ```bash
        # Run inference with the exported ST Edge AI model
        yolo predict model=yolo26n_int8_stedgeai.onnx source='https://ultralytics.com/images/bus.jpg'

        # Measure INT8 accuracy on the COCO8 dataset
        yolo val model=yolo26n_int8_stedgeai.onnx data=coco8.yaml
        ```

Each export writes one file. Ultralytics recognizes ST Edge AI models by the `_stedgeai.onnx` ending and then scales the coordinates back to pixels, upsamples semantic and depth maps, and converts log-depth to metric depth. Keep that ending when you rename a file; a plain `.onnx` name loads as a standard ONNX model and returns boxes in `[0, 1]`.

!!! tip "Export on Ultralytics Platform"

    [Ultralytics Platform](../platform/train/models.md#export-model) exports a trained model to ST Edge AI from the model's **Export** tab, in FP32 or INT8, and the download keeps the `_stedgeai.onnx` ending. Platform calibrates INT8 on the task's default dataset (see [Format Restrictions](../platform/train/models.md#format-restrictions)), so pass `data` in a local export to calibrate on your own images.

### Export Arguments

| Argument   | Type             | Default      | Description                                                                                                                                                                                                          |
| :--------- | :--------------- | :----------- | :------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `format`   | `str`            | `'stedgeai'` | Target format for the exported model.                                                                                                                                                                                |
| `imgsz`    | `int` or `tuple` | `640`        | Model input size, as an integer for square images or a `(height, width)` tuple. The model has a fixed batch size of 1 and static shapes. OBB models require a square `imgsz`.                                        |
| `quantize` | `int` or `str`   | `None`       | Precision. Unset or `32` writes a float model; `8` writes an INT8 model.                                                                                                                                             |
| `data`     | `str`            | `None`       | Dataset used for INT8 calibration: a dataset YAML, or a dataset directory or built-in dataset name for classification. Without it, `quantize=8` uses the task's default dataset, such as `coco8.yaml` for detection. |
| `fraction` | `float`          | `1.0`        | Fraction of the calibration dataset to use.                                                                                                                                                                          |
| `opset`    | `int`            | `None`       | ONNX opset. Defaults to the newest opset your PyTorch version exports reliably.                                                                                                                                      |
| `simplify` | `bool`           | `True`       | Simplifies the ONNX graph with `onnxslim`.                                                                                                                                                                           |
| `device`   | `str`            | `None`       | Device used for exporting: GPU (`device=0`) or CPU (`device=cpu`).                                                                                                                                                   |

`batch` and `dynamic=True` are rejected, and `nms=True` falls back to raw outputs with a warning.

## INT8 Quantization

With `quantize=8`, Ultralytics quantizes the model with [ONNX Runtime](https://onnxruntime.ai/) static quantization, using the scheme ST Edge AI Core [recommends](https://stedgeai-dc.st.com/assets/embedded-docs/quantization.html) and the [STM32 AI Model Zoo Services](https://github.com/STMicroelectronics/stm32ai-modelzoo-services) apply:

- QDQ (QuantizeLinear/DequantizeLinear) format with float model inputs and outputs.
- Signed symmetric per-channel INT8 weights and signed asymmetric INT8 activations.
- MinMax calibration over the `data` images, preprocessed exactly as during Ultralytics validation.

Unlike the standard ONNX `quantize=8` export, which leaves the head in float, every operator is quantized. To quantize with the ST tools instead, export the float model and pass it to the STM32 AI Model Zoo Services.

!!! tip "Calibrate on your own data"

    INT8 accuracy depends on how well the calibration images match the deployment scene. Pass `data=` a dataset with at least 300 representative images; the default calibration datasets hold only a few public samples.

The table below compares YOLO26n at each model's default `imgsz` (640, semantic segmentation 1024, depth 768) on the small default validation datasets, calibrated on the same images. These datasets have 4 to 8 images each, so read the numbers as an indication of which tasks are sensitive to INT8 and measure your own model with `yolo val`.

| Task     | Metric              | ONNX float    | ST Edge AI float | ST Edge AI INT8 |
| :------- | :------------------ | :------------ | :--------------- | :-------------- |
| Detect   | mAP50-95            | 0.686         | 0.686            | 0.639           |
| Segment  | box / mask mAP50-95 | 0.643 / 0.553 | 0.643 / 0.553    | 0.540 / 0.541   |
| Semantic | mIoU                | 0.610         | 0.610            | 0.609           |
| Depth    | delta1              | 0.852         | 0.852            | 0.840           |
| Pose     | box / pose mAP50-95 | 0.697 / 0.459 | 0.697 / 0.459    | 0.650 / 0.267   |
| OBB      | mAP50-95            | 0.894         | 0.894            | 0.806           |

The segmentation row uses the four COCO8-seg validation images at `imgsz=640` with ONNX Runtime 1.32. On all 5,000 COCO val2017 images, the YOLO26n-seg ST Edge AI INT8 export scored **0.325 box AP and 0.294 mask AP** with ONNX Runtime on CPU at `imgsz=512`. The previous export scored 0.148 and 0.184, respectively. Both used the same 300 COCO train2017 images for calibration and COCO evaluation at IoU thresholds 0.50–0.95.

The float export matches the standard ONNX export on every task in the table. Pose loses the most accuracy under whole-graph INT8 because keypoints share one output scale with the boxes.

## Deploy to STM32

1. **Export** the model with `format="stedgeai"` at the `imgsz` you will run on the device.
2. **Validate** the float and INT8 models on your dataset with `yolo val`.
3. **Compile** the model with ST Edge AI Core. The commands below follow the ST Edge AI Core [command-line documentation](https://stedgeai-dc.st.com/assets/embedded-docs/command_line_interface.html) and [Neural-ART getting started guide](https://stedgeai-dc.st.com/assets/embedded-docs/stneuralart_getting_started.html); see them for memory profiles and input options.

    ```bash
    # Report the model's memory and operator usage for an STM32 target
    stedgeai analyze -m yolo26n_int8_stedgeai.onnx --target stm32

    # Generate the C model for an STM32N6 with the Neural-ART accelerator
    stedgeai generate -m yolo26n_int8_stedgeai.onnx --target stm32n6 --st-neural-art
    ```

4. **Deploy** the generated model in your STM32 application. The [STM32 AI Model Zoo Services](https://github.com/STMicroelectronics/stm32ai-modelzoo-services) provide application code and tutorials for Ultralytics YOLO models on the STM32N6.

!!! warning "Outputs need post-processing"

    Code that reads the exported model outside Ultralytics, such as C post-processing on the device, must multiply box and keypoint `x` and `w` values by the input width and `y` and `h` values by the input height, run NMS, upsample semantic logits before the argmax, and apply `exp` to the depth output before upsampling it. Keep `imgsz` identical across training, export and deployment.

## FAQ

### How do I export a YOLO model for STM32 devices?

Run `yolo export model=yolo26n.pt format=stedgeai quantize=8 data=coco8.yaml` to write the INT8 model `yolo26n_int8_stedgeai.onnx`, or omit `quantize=8` to write the float model `yolo26n_stedgeai.onnx`. Then compile the file with ST Edge AI Core as shown in [Deploy to STM32](#deploy-to-stm32).

### Why are the box coordinates in my exported model between 0 and 1?

The ST Edge AI export normalizes box and keypoint coordinates by the input size so they share the `[0, 1]` range of the class scores, which lets the whole graph be quantized to INT8. Multiply `x` and `w` by the input width and `y` and `h` by the input height to recover pixels. Ultralytics does this automatically when it loads a `_stedgeai.onnx` model.

### Does the ST Edge AI export include NMS?

No. The export always returns raw predictions, so run NMS in your application or use the post-processing in the STM32 AI Model Zoo Services.

### How does the ST Edge AI export differ from the standard ONNX export?

The standard [ONNX export](onnx.md) returns pixel coordinates and full-resolution semantic and depth maps, can embed NMS, and with `quantize=8` quantizes only the convolution and matrix-multiplication layers. The ST Edge AI export returns normalized coordinates and low-resolution semantic logits and log-depth, has a fixed batch size of 1, never embeds NMS, and with `quantize=8` quantizes the whole graph.

### Can I check INT8 accuracy without an STM32 board?

Yes. `yolo val model=yolo26n_int8_stedgeai.onnx data=coco8.yaml` runs the quantized model with ONNX Runtime on your computer and reports the same metrics as any other Ultralytics model. Accuracy on the device can still differ slightly, so confirm it with the ST tools after compiling.
