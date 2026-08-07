---
comments: true
description: Learn how to export YOLO26 detection models to RDK format for D-Robotics deployment.
keywords: YOLO26, RDK, D-Robotics, export, model deployment, hb_mapper, edge AI
---

# D-Robotics RDK Export for Ultralytics YOLO26 Models

Ultralytics provides an initial `format="rdk"` export path for D-Robotics RDK deployment.

## Current scope

This first step is limited to export only.

Current support includes:

- detection models only
- standard detection heads
- one-to-one detection heads such as YOLO26-style heads

This step does not include runtime/backend integration or board-side inference support.

## Requirements

RDK export currently requires:

- an x86_64 Linux host
- the RDK export toolchain installed with:

```bash
pip install rdkx5-yolo-mapper
```

- `hb_mapper` available in your `PATH`
- `data=...` provided during export so calibration data can be prepared from the training split

## Usage

!!! example "CLI"

    ```bash
    yolo export model=yolo26n.pt format=rdk data=coco128.yaml
    ```

!!! example "Python"

    ```python
    from ultralytics import YOLO

    model = YOLO("yolo26n.pt")
    model.export(format="rdk", data="coco128.yaml")
    ```
