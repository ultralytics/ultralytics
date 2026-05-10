---
comments: true
description: Browse the Ultralytics Python API reference, auto-generated from source for cfg, data, engine, hub, models, nn, optim, solutions, trackers, and utils.
keywords: Ultralytics, YOLO, API reference, Python, documentation, cfg, data, engine, hub, models, nn, optim, solutions, trackers, utils
---

# Ultralytics Python API Reference

This section is the complete Python API reference for the [`ultralytics`](https://github.com/ultralytics/ultralytics) package. Every page is auto-generated directly from the source so it always stays in sync with the latest release. Use it to look up classes, functions, and method signatures while building with [Ultralytics YOLO](../models/yolo26.md).

If you're new to Ultralytics, the [Quickstart](../quickstart.md), [Modes](../modes/index.md), and [Tasks](../tasks/index.md) guides are the best place to start. Once you're writing code, come back here for the exact API details.

## Sections

- [`__init__`](__init__.md): Top-level package entry point with lazy imports for `YOLO`, `NAS`, `RTDETR`, `SAM`, `FastSAM`, `YOLOE`, and related model classes.
- [`cfg`](cfg/__init__.md): Default configuration loading, CLI argument parsing, and the global `DEFAULT_CFG` used across training, validation, prediction, and export.
- [`data`](data/dataset.md): Dataset classes, data loaders, augmentations, and format converters for detection, segmentation, classification, pose, OBB, and tracking.
- [`engine`](engine/model.md): Core training, validation, prediction, export, and tuning engine — the backbone of the `Model`, `Trainer`, `Validator`, `Predictor`, `Exporter`, and `Tuner` interfaces.
- [`hub`](hub/__init__.md): [Ultralytics HUB](https://www.ultralytics.com/hub) integration for authentication, sessions, dataset uploads, and cloud-based training.
- [`models`](models/yolo/model.md): Model implementations for YOLO, YOLOE, YOLO-World, SAM, SAM3, FastSAM, RT-DETR, and YOLO-NAS, including their predict, train, val, and export pipelines.
- [`nn`](nn/tasks.md): Neural network building blocks — backbones, necks, heads, layers, and the multi-backend `AutoBackend` runtime (PyTorch, ONNX, TensorRT, CoreML, OpenVINO, TFLite, and more).
- [`optim`](optim/muon.md): Custom optimizers, including the Muon optimizer used for advanced training experiments.
- [`solutions`](solutions/solutions.md): Ready-made [Ultralytics Solutions](../solutions/index.md) — object counting, heatmaps, AI Gym, parking management, region counting, similarity search, and more.
- [`trackers`](trackers/track.md): Multi-object trackers (`BYTETracker`, `BoTSORT`) and the unified [tracking API](../modes/track.md) that plugs them into any YOLO model.
- [`utils`](utils/__init__.md): Cross-cutting utilities — logging, metrics, plotting, ops, downloads, checks, callbacks, and integrations with [Weights & Biases](../integrations/weights-biases.md), [MLflow](../integrations/mlflow.md), [Comet](../integrations/comet.md), and other tools.

## How this reference is generated

These pages are produced automatically by [`docs/build_reference.py`](https://github.com/ultralytics/ultralytics/blob/main/docs/build_reference.py), which walks the `ultralytics` package, parses each Python module, and renders the docstrings with [mkdocstrings](https://mkdocstrings.github.io/). The best way to improve a page is to improve the docstring in the corresponding source file — open a [Pull Request](https://docs.ultralytics.com/help/contributing/) and the docs will update on the next build. 🙏

## Looking for something else?

- Conceptual guides → [Guides](../guides/index.md)
- Tutorials and how-tos → [Modes](../modes/index.md), [Tasks](../tasks/index.md), [Solutions](../solutions/index.md)
- Datasets → [Datasets](../datasets/index.md)
- Third-party tools → [Integrations](../integrations/index.md)
- Help and FAQ → [Help](../help/index.md)
