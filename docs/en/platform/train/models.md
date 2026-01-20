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

### Supported Formats

| Format            | Description                  | Use Case                  |
| ----------------- | ---------------------------- | ------------------------- |
| **ONNX**          | Open Neural Network Exchange | Cross-platform deployment |
| **TorchScript**   | Serialized PyTorch           | PyTorch deployment        |
| **OpenVINO**      | Intel optimization           | Intel CPUs/GPUs           |
| **TensorRT**      | NVIDIA optimization          | NVIDIA GPUs               |
| **CoreML**        | Apple optimization           | iOS/macOS                 |
| **TFLite**        | TensorFlow Lite              | Mobile/embedded           |
| **TF SavedModel** | TensorFlow format            | TensorFlow ecosystem      |
| **TF GraphDef**   | TensorFlow frozen            | Legacy TensorFlow         |
| **PaddlePaddle**  | Baidu framework              | PaddlePaddle ecosystem    |
| **NCNN**          | Mobile inference             | Android/embedded          |
| **Edge TPU**      | Google Edge TPU              | Coral devices             |
| **TF.js**         | TensorFlow.js                | Browser deployment        |
| **MNN**           | Alibaba framework            | Mobile optimization       |
| **RKNN**          | Rockchip NPU                 | Rockchip devices          |
| **IMX**           | NXP i.MX                     | NXP platforms             |
| **Axelera**       | Metis AI                     | Edge AI accelerators      |
| **ExecuTorch**    | Meta framework               | Meta platforms            |

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
