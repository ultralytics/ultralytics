---
comments: true
description: Explore the comprehensive API reference for Ultralytics YOLOv8, including detailed documentation on modules, classes, and functions across various components like data handling, models, engine, and utilities.
keywords: Ultralytics, YOLOv8, API reference, documentation, Python modules, machine learning, object detection
---

# Ultralytics YOLOv8 Reference

Welcome to the comprehensive API reference documentation for Ultralytics YOLOv8. This section provides detailed information about the modules, classes, and functions that make up the YOLOv8 ecosystem. Whether you're looking to understand the inner workings of the model, customize data handling, or integrate YOLOv8 into your own projects, you'll find the necessary details here.

## Core Components

- [cfg](cfg/__init__.md): Configuration handling and management
- [data](data/base.md): Data loading, processing, and augmentation
- [engine](engine/model.md): Core training and inference engine
- [models](models/yolo/model.md): Model architectures and building blocks
- [nn](nn/modules/block.md): Neural network modules and layers
- [trackers](trackers/track.md): Object tracking algorithms and utilities
- [utils](utils/__init__.md): Utility functions and helper modules

## Specialized Modules

- [hub](hub/__init__.md): Ultralytics HUB integration
- [solutions](solutions/object_counter.md): Ready-to-use AI solutions

## Data Handling

- [data/explorer](data/explorer/explorer.md): Dataset exploration tools
- [datasets](../datasets/index.md): Pre-configured datasets for various tasks

## Model Types

- [fastsam](models/fastsam/model.md): Fast Segment Anything Model
- [nas](models/nas/model.md): Neural Architecture Search models
- [rtdetr](models/rtdetr/model.md): Real-Time Detection Transformer
- [sam](models/sam/model.md): Segment Anything Model
- [yolo](models/yolo/model.md): You Only Look Once (YOLO) models

## Task-Specific Components

- [classify](models/yolo/classify/val.md): Image classification
- [detect](models/yolo/detect/predict.md): Object detection
- [segment](models/yolo/segment/predict.md): Instance segmentation
- [pose](models/yolo/pose/predict.md): Pose estimation
- [obb](models/yolo/obb/predict.md): Oriented Bounding Box detection

## Utilities and Tools

- [callbacks](utils/callbacks/base.md): Customizable event hooks
- [plotting](utils/plotting.md): Visualization tools
- [torch_utils](utils/torch_utils.md): PyTorch-specific utilities

This reference documentation aims to provide you with a deep understanding of the Ultralytics YOLOv8 framework. Use the navigation menu to explore specific components, or use the search function to find exactly what you're looking for.

For practical guides and tutorials on using YOLOv8, please refer to our [Guides](../guides/index.md) section. If you're new to YOLOv8, we recommend starting with our [Quickstart](../quickstart.md) guide.

Happy exploring, and feel free to contribute to our [GitHub repository](https://github.com/ultralytics/ultralytics) if you have any improvements or suggestions!
