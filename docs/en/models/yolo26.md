---
comments: true
description: YOLO26 from Ultralytics delivers faster, simpler, end-to-end NMS-free object detection optimized for edge and low-power devices.
keywords: YOLO26, Ultralytics YOLO, object detection, end-to-end NMS-free, simplified architecture, computer vision, AI, machine learning, edge AI, low power devices, quantization, real-time inference
---

# Ultralytics YOLO26

!!! note "Coming Soon ⚠️"

    🚧 YOLO26 models are still under development and not yet released. Performance numbers shown here are **previews only**.
    Final downloads and releases will follow soon — stay updated via [YOLO Vision 2025](https://www.ultralytics.com/events/yolovision).

## Overview

[Ultralytics](https://www.ultralytics.com/) YOLO26 is the latest evolution in the YOLO series of real-time object detectors, engineered from the ground up for **edge and low-power devices**. It introduces a streamlined design that removes unnecessary complexity while integrating targeted innovations to deliver faster, lighter, and more accessible deployment.

The architecture of YOLO26 is guided by three core principles:

- **Simplicity:** YOLO26 is a **native end-to-end model**, producing predictions directly without the need for non-maximum suppression (NMS). By eliminating this post-processing step, inference becomes faster, lighter, and easier to deploy in real-world systems. This breakthrough approach was first pioneered in [YOLOv10](../models/yolov10.md) by Ao Wang at Tsinghua University and has been further advanced in YOLO26.
- **Deployment Efficiency:** The end-to-end design cuts out an entire stage of the pipeline, dramatically simplifying integration, reducing latency, and making deployment more robust across diverse environments.
- **Training Innovation:** YOLO26 introduces the **MuSGD optimizer**, a hybrid of [SGD](https://docs.pytorch.org/docs/stable/generated/torch.optim.SGD.html) and [Muon](https://arxiv.org/abs/2502.16982) — inspired by Moonshot AI's [Kimi K2](https://www.kimi.com/) breakthroughs in LLM training. This optimizer brings enhanced stability and faster convergence, transferring optimization advances from language models into computer vision.

Together, these innovations deliver a model family that achieves higher accuracy on small objects, provides seamless deployment, and runs **up to 43% faster on CPUs** — making YOLO26 one of the most practical and deployable YOLO models to date for resource-constrained environments.

![Ultralytics YOLO26 Comparison Plots](https://github.com/ultralytics/assets/releases/download/v0.0.0/yolo-comparison-plot.png)

## Key Features

- **DFL Removal**  
  The Distribution Focal Loss (DFL) module, while effective, often complicated export and limited hardware compatibility. YOLO26 removes DFL entirely, simplifying inference and broadening support for **edge and low-power devices**.

- **End-to-End NMS-Free Inference**  
  Unlike traditional detectors that rely on NMS as a separate post-processing step, YOLO26 is **natively end-to-end**. Predictions are generated directly, reducing latency and making integration into production systems faster, lighter, and more reliable.

- **ProgLoss + STAL**  
  Improved loss functions increase detection accuracy, with notable improvements in **small-object recognition**, a critical requirement for IoT, robotics, aerial imagery, and other edge applications.

- **MuSGD Optimizer**  
  A new hybrid optimizer that combines [SGD](https://docs.pytorch.org/docs/stable/generated/torch.optim.SGD.html) with [Muon](https://arxiv.org/abs/2502.16982). Inspired by Moonshot AI's [Kimi K2](https://www.kimi.com/), MuSGD introduces advanced optimization methods from LLM training into computer vision, enabling more stable training and faster convergence.

- **Up to 43% Faster CPU Inference**  
  Specifically optimized for edge computing, YOLO26 delivers significantly faster CPU inference, ensuring real-time performance on devices without GPUs.

---

## Supported Tasks and Modes

YOLO26 is designed as a **multi-task model family**, extending YOLO's versatility across diverse computer vision challenges:

| Model       | Task                                         | Inference | Validation | Training | Export |
| ----------- | -------------------------------------------- | --------- | ---------- | -------- | ------ |
| YOLO26      | [Detection](../tasks/detect.md)              | ✅        | ✅         | ✅       | ✅     |
| YOLO26-seg  | [Instance Segmentation](../tasks/segment.md) | ✅        | ✅         | ✅       | ✅     |
| YOLO26-pose | [Pose/Keypoints](../tasks/pose.md)           | ✅        | ✅         | ✅       | ✅     |
| YOLO26-obb  | [Oriented Detection](../tasks/obb.md)        | ✅        | ✅         | ✅       | ✅     |
| YOLO26-cls  | [Classification](../tasks/classify.md)       | ✅        | ✅         | ✅       | ✅     |

This unified framework ensures YOLO26 is applicable across real-time detection, segmentation, classification, pose estimation, and oriented object detection — all with training, validation, inference, and export support.

---

## Performance Metrics

!!! tip "Performance Preview"

    The following benchmarks are **early previews**. Final numbers and downloadable weights will be released once training is complete.

    === "Detection (COCO)"

        Trained on [COCO](../datasets/detect/coco.md) with 80 pre-trained classes.
        See [Detection Docs](../tasks/detect.md) for usage once models are released.

        | Model   | size<br><sup>(pixels) | mAP<sup>val<br>50-95(e2e) | mAP<sup>val<br>50-95 | Speed<br><sup>CPU ONNX<br>(ms)  | Speed<br><sup>T4 TensorRT10<br>(ms)  | params<br><sup>(M) | FLOPs<br><sup>(B) |
        | -----   | --------------------- | --------------------      | -------------------- | ------------------------------- | ------------------------------------ | ------------------ | ----------------- |
        | YOLO26n | 640                   | 39.8                      | 40.3                 | 38.90 ± 0.7                     | 1.7 ± 0.0                            | 2.4                | 5.4               |
        | YOLO26s | 640                   | 47.2                      | 47.6                 | 87.16 ± 0.9                     | 2.7 ± 0.0                            | 9.5                | 20.7              |
        | YOLO26m | 640                   | 51.5                      | 51.7                 | 220.0 ± 1.4                     | 4.9 ± 0.1                            | 20.4               | 68.2              |
        | YOLO26l | 640                   | 53.0*                     | 53.4*                | 286.17 ± 2.0*                   | 6.5 ± 0.2*                           | 24.8               | 86.4              |
        | YOLO26x | 640                   | -                         | -                    | -                               | -                                    | -                  | -                 |

        *Metrics for YOLO26l and YOLO26x are in progress. Final benchmarks will be added here.

    === "Segmentation (COCO)"

        Performance metrics coming soon.

    === "Classification (ImageNet)"

        Performance metrics coming soon.

    === "Pose (COCO)"

        Performance metrics coming soon.

    === "OBB (DOTAv1)"

        Performance metrics coming soon.

---

## Citations and Acknowledgments

!!! tip "Ultralytics YOLO26 Publication"

    Ultralytics has not published a formal research paper for YOLO26 due to the rapidly evolving nature of the models. Instead, we focus on delivering cutting-edge models and making them easy to use. For the latest updates on YOLO features, architectures, and usage, visit our [GitHub repository](https://github.com/ultralytics/ultralytics) and [documentation](https://docs.ultralytics.com/).

If you use YOLO26 or other Ultralytics software in your work, please cite it as:

!!! quote ""

    === "BibTeX"

        ```bibtex
        @software{yolo26_ultralytics,
          author = {Glenn Jocher and Jing Qiu},
          title = {Ultralytics YOLO26},
          version = {26.0.0},
          year = {2025},
          url = {https://github.com/ultralytics/ultralytics},
          orcid = {0000-0001-5950-6979, 0000-0003-3783-7069},
          license = {AGPL-3.0}
        }
        ```

DOI pending. YOLO26 is available under [AGPL-3.0](https://github.com/ultralytics/ultralytics/blob/main/LICENSE) and [Enterprise](https://www.ultralytics.com/license) licenses.

---

## FAQ

### What are the key improvements in YOLO26 compared to YOLO11?

- **DFL Removal**: Simplifies export and expands edge compatibility
- **End-to-End NMS-Free Inference**: Eliminates NMS for faster, simpler deployment
- **ProgLoss + STAL**: Boosts accuracy, especially on small objects
- **MuSGD Optimizer**: Combines SGD and Muon (inspired by Moonshot's Kimi K2) for more stable, efficient training
- **Up to 43% Faster CPU Inference**: Major performance gains for CPU-only devices

### What tasks will YOLO26 support?

YOLO26 is designed as a **unified model family**, providing end-to-end support for multiple computer vision tasks:

- [Object Detection](../tasks/detect.md)
- [Instance Segmentation](../tasks/segment.md)
- [Image Classification](../tasks/classify.md)
- [Pose Estimation](../tasks/pose.md)
- [Oriented Object Detection (OBB)](../tasks/obb.md)

Each size variant (n, s, m, l, x) is planned to support all tasks at release.

### Why is YOLO26 optimized for edge deployment?

YOLO26 delivers **state-of-the-art edge performance** with:

- Up to 43% faster CPU inference
- Reduced model size and memory footprint
- Architecture simplified for compatibility (no DFL, no NMS)
- Flexible export formats including TensorRT, ONNX, CoreML, TFLite, and OpenVINO

### When will YOLO26 models be available?

YOLO26 models are still in training and not yet open-sourced. Performance previews are shown here, with official downloads and releases planned in the near future.
See [YOLO Vision 2025](https://www.ultralytics.com/events/yolovision) for YOLO26 talks.
