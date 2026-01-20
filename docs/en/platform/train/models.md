---
comments: true
description: Learn how to manage, analyze, and export trained models in Ultralytics Platform with support for 17 deployment formats.
keywords: Ultralytics Platform, models, model management, export, ONNX, TensorRT, CoreML, YOLO
---

# Models

[Ultralytics Platform](https://platform.ultralytics.com) provides comprehensive model management for training, analyzing, and deploying YOLO models. Upload pretrained models or train new ones directly on the Platform.

<!-- Screenshot: platform-models-detail.avif -->

## Upload Model

Upload existing model weights to the Platform:

1. Navigate to your project
2. Click **Upload Model**
3. Select your `.pt` file
4. Add name and description
5. Click **Upload**

<!-- Screenshot: platform-models-upload.avif -->

Supported model formats:

| Format  | Extension | Description               |
| ------- | --------- | ------------------------- |
| PyTorch | `.pt`     | Native Ultralytics format |

After upload, the Platform parses model metadata:

- Task type (detect, segment, pose, OBB, classify)
- Architecture (YOLO26n, YOLO26s, etc.)
- Class names and count
- Input size and parameters

## Train Model

Train a new model directly on the Platform:

1. Navigate to your project
2. Click **Train Model**
3. Select dataset
4. Choose base model
5. Configure training parameters
6. Start training

See [Cloud Training](cloud-training.md) for detailed instructions.

## Model Overview

Each model page displays:

| Section      | Content                                 |
| ------------ | --------------------------------------- |
| **Overview** | Model metadata, task type, architecture |
| **Metrics**  | Training loss and performance charts    |
| **Plots**    | Confusion matrix, PR curves, F1 curves  |
| **Test**     | Interactive inference testing           |
| **Deploy**   | Endpoint creation and management        |
| **Export**   | Format conversion and download          |

## Training Metrics

View real-time and historical training metrics:

### Loss Curves

<!-- Screenshot: platform-models-loss.avif -->

| Loss      | Description                  |
| --------- | ---------------------------- |
| **Box**   | Bounding box regression loss |
| **Class** | Classification loss          |
| **DFL**   | Distribution Focal Loss      |

### Performance Metrics

<!-- Screenshot: platform-models-metrics.avif -->

| Metric        | Description                             |
| ------------- | --------------------------------------- |
| **mAP50**     | Mean Average Precision at IoU 0.50      |
| **mAP50-95**  | Mean Average Precision at IoU 0.50-0.95 |
| **Precision** | Ratio of correct positive predictions   |
| **Recall**    | Ratio of actual positives identified    |

## Validation Plots

After training completes, view detailed validation analysis:

### Confusion Matrix

Interactive heatmap showing prediction accuracy per class:

<!-- Screenshot: platform-models-confusion.avif -->

### PR/F1 Curves

Performance curves at different confidence thresholds:

<!-- Screenshot: platform-models-curves.avif -->

| Curve                    | Description                              |
| ------------------------ | ---------------------------------------- |
| **Precision-Recall**     | Trade-off between precision and recall   |
| **F1-Confidence**        | F1 score at different confidence levels  |
| **Precision-Confidence** | Precision at different confidence levels |
| **Recall-Confidence**    | Recall at different confidence levels    |

## Export Model

Export your model to 17 deployment formats:

1. Navigate to the **Export** tab
2. Select target format
3. Click **Export**
4. Download when complete

<!-- Screenshot: platform-models-export.avif -->

### Supported Formats (17 total)

| #   | Format            | File Extension   | Use Case                           |
| --- | ----------------- | ---------------- | ---------------------------------- |
| 1   | **ONNX**          | `.onnx`          | Cross-platform, web, most runtimes |
| 2   | **TorchScript**   | `.torchscript`   | PyTorch deployment without Python  |
| 3   | **OpenVINO**      | `.xml`, `.bin`   | Intel CPUs, GPUs, VPUs             |
| 4   | **TensorRT**      | `.engine`        | NVIDIA GPUs (fastest inference)    |
| 5   | **CoreML**        | `.mlpackage`     | Apple iOS, macOS, watchOS          |
| 6   | **TF Lite**       | `.tflite`        | Mobile (Android, iOS), edge        |
| 7   | **TF SavedModel** | `saved_model/`   | TensorFlow Serving                 |
| 8   | **TF GraphDef**   | `.pb`            | TensorFlow 1.x                     |
| 9   | **TF Edge TPU**   | `.tflite`        | Google Coral devices               |
| 10  | **TF.js**         | `.json`, `.bin`  | Browser inference                  |
| 11  | **PaddlePaddle**  | `.pdmodel`       | Baidu PaddlePaddle                 |
| 12  | **NCNN**          | `.param`, `.bin` | Mobile (Android/iOS), optimized    |
| 13  | **MNN**           | `.mnn`           | Alibaba mobile runtime             |
| 14  | **RKNN**          | `.rknn`          | Rockchip NPUs                      |
| 15  | **IMX500**        | `.imx`           | Sony IMX500 sensor                 |
| 16  | **Axelera**       | `.axelera`       | Axelera AI accelerators            |

### Format Selection Guide

**For NVIDIA GPUs:** Use **TensorRT** for maximum speed

**For Intel Hardware:** Use **OpenVINO** for Intel CPUs, GPUs, and VPUs

**For Apple Devices:** Use **CoreML** for iOS, macOS, Apple Silicon

**For Android:** Use **TF Lite** or **NCNN** for best performance

**For Web Browsers:** Use **TF.js** or **ONNX** (with ONNX Runtime Web)

**For Edge Devices:** Use **TF Edge TPU** for Coral, **RKNN** for Rockchip

**For General Compatibility:** Use **ONNX** — works with most inference runtimes

<!-- Screenshot: platform-models-export-progress.avif -->

!!! tip "Export Time"

    Export time varies by format. TensorRT exports may take several minutes due to engine optimization.

## Dataset Linking

Models can be linked to their source dataset:

- View which dataset was used for training
- Access dataset from model page
- Track data lineage

When training with Platform datasets using the `ul://` URI format, linking is automatic.

## Visibility Settings

Control who can see your model:

| Setting     | Description                     |
| ----------- | ------------------------------- |
| **Private** | Only you can access             |
| **Public**  | Anyone can view on Explore page |

To change visibility:

1. Open model actions menu
2. Click **Edit**
3. Toggle visibility
4. Click **Save**

## Delete Model

Remove a model you no longer need:

1. Open model actions menu
2. Click **Delete**
3. Confirm deletion

!!! note "Trash and Restore"

    Deleted models go to Trash for 30 days. Restore from Settings > Trash.

## FAQ

### What model architectures are supported?

Ultralytics Platform supports all YOLO architectures:

- **YOLO26**: n, s, m, l, x variants (recommended)
- **YOLO11**: n, s, m, l, x variants
- **YOLOv10**: Legacy support
- **YOLOv8**: Legacy support
- **YOLOv5**: Legacy support

### Can I download my trained model?

Yes, download your model weights from the model page:

1. Click the download icon
2. Select format (original `.pt` or exported)
3. Download starts automatically

### How do I compare models across projects?

Currently, model comparison is within projects. To compare across projects:

1. Transfer models to a single project, or
2. Export metrics and compare externally

### What's the maximum model size?

There's no strict limit, but very large models (>2GB) may have longer upload and processing times.

### Can I fine-tune pretrained models?

Yes! Upload a pretrained model, then start training from that checkpoint with your dataset. The Platform automatically uses the uploaded model as the starting point.
