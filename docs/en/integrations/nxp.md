---
title: NXP i.MX95 Export with Neutron
description: Export Ultralytics YOLO models to calibrated INT8 LiteRT models compiled for the NXP i.MX95 Neutron NPU.
comments: true
---

# NXP i.MX95 Export with Neutron

Export [Ultralytics YOLO](../models/index.md) with `format="nxp"` to produce an INT8 `.tflite` model compiled for the **NXP i.MX95 Neutron NPU**. The exporter reuses the [LiteRT](litert.md) conversion and calibration pipeline, then runs NXP's Neutron SDK 3.3.0 compiler.

This target is distinct from [Sony IMX500](sony-imx500.md) (`format="imx"`). NXP i.MX8M Plus, i.MX93, and other NXP processors use different accelerator stacks; an i.MX95 artifact is not interchangeable with those targets.

## Export

Export on **Linux x86_64**, with Python 3.10 or later. Ultralytics installs the conversion dependencies when needed. The Neutron SDK comes from [NXP's package repository](https://eiq.nxp.com/repository/eiq-neutron-sdk/).

```bash
pip install -U ultralytics
pip install eiq-neutron-sdk==3.3.0 --extra-index-url https://eiq.nxp.com/repository
yolo export model=yolo26n.pt format=nxp imgsz=640 data=coco128.yaml
```

```python
from ultralytics import YOLO

model = YOLO("yolo26n.pt")
path = model.export(format="nxp", imgsz=640, data="coco128.yaml")
# yolo26n_nxp.tflite
```

Use calibration images representative of your deployment, including its camera, lighting, and object sizes. `coco128.yaml` is a convenient example, not a substitute for application-specific calibration. The intermediate `yolo26n_int8.tflite` can be validated on the host before deploying the compiled artifact.

| Argument   | Description                                                                    |
| ---------- | ------------------------------------------------------------------------------ |
| `imgsz`    | Fixed input image size, for example `640` or `(480, 640)`.                     |
| `quantize` | INT8 (`8`); enabled automatically. Other precisions are unsupported.           |
| `data`     | Calibration dataset YAML, or a classification dataset.                         |
| `fraction` | Fraction or count of calibration images.                                       |
| `device`   | Device used during export and calibration. The compiled target remains i.MX95. |

Batch size is fixed to 1. Detection exports use raw predictions for external NMS. Embedded NMS and the YOLO26 NMS-free top-k output are not supported by this export.

## Platform

In [Ultralytics Platform](../platform/train/models.md), open a model's **Export** tab and select **NXP**. Choose the image size and download the compiled `_nxp.tflite` artifact when the export completes. Platform runs the compiler on its CPU workers; an NXP board is needed only for deployment and performance measurements.

## Deploy on i.MX95

Install the TensorFlow Lite runtime, `libneutron_delegate.so`, Neutron driver, and firmware supplied by NXP for a BSP compatible with **Neutron SDK 3.3.0**. Follow [NXP's deployment guide](https://community.nxp.com/t5/FRDM-Training-Hub/FRDM-i-MX-95-Pro-Hands-On-NPU-Inference-with-eIQ-Neutron/ta-p/2419604) for board setup and runtime compatibility.

The Ultralytics runtime loads the board's `tflite_runtime` and Neutron delegate:

```bash
yolo predict model=yolo26n_nxp.tflite source=bus.jpg
yolo val model=yolo26n_nxp.tflite data=coco128.yaml
```

For a direct TensorFlow Lite benchmark, use the board's benchmark executable:

```bash
benchmark_model --graph=yolo26n_nxp.tflite --external_delegate_path=/usr/lib/libneutron_delegate.so
```

## Operator Coverage and Performance

Compilation replaces supported subgraphs with `NeutronGraph` operations. Unsupported operators remain on CPU. A successful export does **not** imply that every operation runs on the NPU or that CPU/NPU switching overhead has been eliminated. Attention, tensor layout changes, and post-processing can still require CPU execution.

The export log includes NXP compiler statistics: converted operators, NPU partitions, memory estimates, and estimated NPU cycles. These estimates exclude CPU operators, preprocessing, post-processing, and runtime overhead. Measure complete frame latency, peak RAM, and deployment storage on the target board; theoretical TOPS and compiled file size cannot determine those values.

Validate the calibrated intermediate against the PyTorch model, then validate the compiled model on the board using the same dataset and image size. Host compilation and intermediate accuracy checks do not establish numerical parity or latency on the NPU.
