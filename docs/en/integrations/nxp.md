---
title: NXP i.MX95 Export with Neutron
description: Export and run Ultralytics YOLO models on the NXP i.MX95 Neutron NPU with INT8 calibration, Python and CLI examples, and deployment guidance.
keywords: NXP YOLO, YOLO26 NXP, i.MX95 NPU, Neutron SDK, NXP model export, INT8 quantization, embedded object detection, edge AI, Ultralytics Platform
comments: true
---

# NXP i.MX95 Export with Neutron

Export [Ultralytics YOLO](../models/index.md) with `format="nxp"` to produce an INT8 `.tflite` model compiled for the **NXP i.MX95 Neutron [NPU](https://www.ultralytics.com/glossary/neural-processing-unit-npu)**. The exporter reuses the [LiteRT](litert.md) conversion and calibration pipeline, then runs NXP's Neutron SDK 3.3.0 compiler for [edge AI deployment](https://www.ultralytics.com/blog/understanding-the-real-world-applications-of-edge-ai).

This target is distinct from [Sony IMX500](sony-imx500.md) (`format="imx"`). NXP i.MX8M Plus, i.MX93, and other NXP processors use different accelerator stacks; an i.MX95 artifact is not interchangeable with those targets.

## Export

Export on **Linux x86_64**, with Python 3.10 or later. Compilation has been validated with Python 3.13 and 3.14. Ultralytics installs the conversion dependencies when needed. The Neutron SDK comes from [NXP's package repository](https://eiq.nxp.com/repository/eiq-neutron-sdk/).

```bash
pip install -U ultralytics
pip install eiq-neutron-sdk==3.3.0 --extra-index-url https://eiq.nxp.com/repository
```

!!! example "Export to NXP i.MX95"

    === "Python"

        ```python
        from ultralytics import YOLO

        model = YOLO("yolo26n.pt")
        path = model.export(format="nxp", imgsz=640, data="coco128.yaml")
        # Creates yolo26n_nxp.tflite
        ```

    === "CLI"

        ```bash
        yolo export model=yolo26n.pt format=nxp imgsz=640 data=coco128.yaml
        ```

Use calibration images representative of your deployment, including its camera, lighting, and object sizes. [INT8 quantization](https://www.ultralytics.com/glossary/model-quantization) can reduce accuracy, so compare the intermediate `yolo26n_int8.tflite` against the PyTorch model on the host. `coco128.yaml` is a convenient example, not a substitute for application-specific calibration and an independent validation dataset.

| Argument   | Description                                                                    |
| ---------- | ------------------------------------------------------------------------------ |
| `imgsz`    | Fixed input image size, for example `640` or `(480, 640)`.                     |
| `quantize` | INT8 (`8`); enabled automatically. Other precisions are unsupported.           |
| `data`     | Calibration dataset YAML, or a classification dataset.                         |
| `fraction` | Fraction or count of calibration images.                                       |
| `device`   | Device used during export and calibration. The compiled target remains i.MX95. |

Batch size is fixed to 1. Detection exports use raw predictions for external NMS; Ultralytics applies this postprocessing during prediction and validation. Embedded NMS and the YOLO26 NMS-free top-k output are not supported by this export.

## Platform

In [Ultralytics Platform](../platform/train/models.md), open a model's **Export** tab and select **NXP**. Choose the image size and download the compiled `_nxp.tflite` artifact when the export completes. Platform runs the compiler on its CPU workers; an NXP board is needed only for deployment and performance measurements. See the [Platform model export walkthrough](https://www.ultralytics.com/blog/how-to-export-ultralytics-yolo-models-using-ultralytics-platform) for the general workflow.

Platform displays INT8 as a fixed setting and uses the exporter's default calibration dataset for the model's task. It does not currently expose calibration `data` or `fraction` controls. Use the Python or CLI export examples above to calibrate with your own representative images.

## Deploy on i.MX95

Install the TensorFlow Lite runtime, `libneutron_delegate.so`, Neutron driver, and firmware supplied by NXP for a BSP compatible with **Neutron SDK 3.3.0**. Follow [NXP's deployment guide](https://community.nxp.com/t5/FRDM-Training-Hub/FRDM-i-MX-95-Pro-Hands-On-NPU-Inference-with-eIQ-Neutron/ta-p/2419604) for board setup and runtime compatibility.

The Ultralytics runtime loads the board's `tflite_runtime` and Neutron delegate. Keep the `_nxp.tflite` suffix so Ultralytics selects the NXP runtime. Run the following [prediction](../modes/predict.md) and [validation](../modes/val.md) examples on the board:

!!! example "Predict with an NXP model"

    === "Python"

        ```python
        from ultralytics import YOLO

        model = YOLO("yolo26n_nxp.tflite")
        results = model.predict("https://ultralytics.com/images/bus.jpg", imgsz=640)
        ```

    === "CLI"

        ```bash
        yolo predict model=yolo26n_nxp.tflite source=https://ultralytics.com/images/bus.jpg imgsz=640
        ```

!!! example "Validate an NXP model"

    === "Python"

        ```python
        from ultralytics import YOLO

        model = YOLO("yolo26n_nxp.tflite")
        metrics = model.val(data="coco128.yaml", imgsz=640, batch=1)
        ```

    === "CLI"

        ```bash
        yolo val model=yolo26n_nxp.tflite data=coco128.yaml imgsz=640 batch=1
        ```

For a direct TensorFlow Lite benchmark, use the board's benchmark executable:

```bash
benchmark_model --graph=yolo26n_nxp.tflite --external_delegate_path=/usr/lib/libneutron_delegate.so
```

## Operator Coverage and Performance

Compilation replaces supported subgraphs with `NeutronGraph` operations. Unsupported operators remain on CPU. A successful export does **not** imply that every operation runs on the NPU or that CPU/NPU switching overhead has been eliminated. Attention, tensor layout changes, and postprocessing can still require CPU execution.

The export log includes NXP compiler statistics: converted operators, NPU partitions, memory estimates, and estimated NPU cycles. These estimates exclude CPU operators, preprocessing, postprocessing, and runtime overhead. Measure complete frame latency, peak RAM, and deployment storage on the target board; theoretical TOPS and compiled file size cannot determine those values.

Validate the calibrated intermediate against the PyTorch model, then validate the compiled model on the board using the same dataset and image size. Host compilation and intermediate accuracy checks do not establish numerical parity or latency on the NPU. See [model deployment best practices](../guides/model-deployment-practices.md) for planning application-level validation.

## FAQ

### How do I export YOLO26 models for the NXP i.MX95 NPU?

Use `model.export(format="nxp", data="your_dataset.yaml")` in Python or `yolo export model=best.pt format=nxp data=your_dataset.yaml` in the CLI on Linux x86_64. The exporter calibrates the model to INT8 and compiles it with Neutron SDK 3.3.0, producing `best_nxp.tflite`. Follow the [export examples](#export) for a pretrained YOLO26n model.

### Can I run NXP YOLO models with Python and the Ultralytics CLI?

Yes. Load `YOLO("best_nxp.tflite")` and call `predict()` or `val()`, or use `yolo predict` and `yolo val`. Both paths use the board's TensorFlow Lite runtime and Neutron delegate. Preserve the `_nxp.tflite` filename suffix and install the compatible NXP runtime described in [Deploy on i.MX95](#deploy-on-imx95).

### Does NXP export support i.MX8M Plus or i.MX93?

This export targets **i.MX95 only**. Other NXP processors have different accelerators and deployment toolchains, so an i.MX95 Neutron artifact cannot be reused on i.MX8M Plus or i.MX93. The separate `format="imx"` option targets Sony IMX500, not NXP i.MX processors.

### Why do some YOLO operators run on the CPU after NXP export?

Neutron compiles supported subgraphs for the NPU and leaves unsupported operators in TensorFlow Lite for CPU execution. Attention, layout changes, and postprocessing can create multiple partitions and transfer overhead. Inspect the compiler's converted-operator statistics, then profile the complete application on the board; successful compilation does not guarantee full NPU execution.

### How can I measure YOLO26 accuracy, latency, and RAM on i.MX95?

Calibrate with representative images and compare PyTorch, the INT8 intermediate, and the compiled model on an independent validation set. Use [Validate mode](../modes/val.md) for accuracy, and measure frame latency, peak process RAM, and deployment storage on the i.MX95 board. Compiler memory estimates, model file size, and a TOPS rating are insufficient to determine application performance.
