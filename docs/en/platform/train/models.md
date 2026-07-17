---
plans: [free, pro, enterprise]
title: Trained Model Management
comments: true
description: Learn how to manage, analyze, and export trained models in Ultralytics Platform with support for 19+ deployment formats.
keywords: Ultralytics Platform, models, model management, export, ONNX, TensorRT, CoreML, YOLO
---

# Models

[Ultralytics Platform](https://platform.ultralytics.com) provides comprehensive model management for training, analyzing, and deploying YOLO models. Upload pretrained models or train new ones directly on the platform.

![Ultralytics Platform Model Page Overview Tab](https://cdn.jsdelivr.net/gh/ultralytics/assets@main/docs/platform/platform-model-page-overview-tab.avif)

## Upload Model

Upload existing model weights to the platform:

1. Navigate to your project
2. **Drag and drop** `.pt` files onto the project page or models sidebar
3. Model metadata is parsed automatically from the file

Multiple files can be uploaded simultaneously (up to 3 concurrent).

![Ultralytics Platform Model Drag Drop Upload](https://cdn.jsdelivr.net/gh/ultralytics/assets@main/docs/platform/platform-model-drag-drop-upload.avif)

Supported model formats:

| Format  | Extension | Description               |
| ------- | --------- | ------------------------- |
| PyTorch | `.pt`     | Native Ultralytics format |

After upload, the platform parses model metadata:

- Task type ([detect](../../tasks/detect.md), [segment](../../tasks/segment.md), [semantic](../../tasks/semantic.md), [pose](../../tasks/pose.md), [OBB](../../tasks/obb.md), [classify](../../tasks/classify.md))
- Architecture (YOLO26n, YOLO26s, etc.)
- Class names and count
- Input size and parameters
- Training results and metrics (if present in checkpoint)

## Train Model

Train a new model directly on the platform:

1. Navigate to your project
2. Click **New Model**
3. Select base model and dataset
4. Configure training parameters
5. Choose cloud or local training
6. Start training

See [Cloud Training](cloud-training.md) for detailed instructions.

## Model Lifecycle

```mermaid
graph LR
    A[Upload .pt]:::start --> B[Overview]:::proc
    C[Train]:::start --> B
    B --> D[Predict]:::proc
    B --> E[Export]:::proc
    B --> F[Deploy]:::proc
    E --> G[19+ Formats]:::out
    F --> H[Endpoint]:::out

    classDef start fill:#4CAF50,color:#fff
    classDef proc fill:#2196F3,color:#fff
    classDef out fill:#9C27B0,color:#fff
```

## Model Page Tabs

Each model page has the following tabs:

| Tab          | Content                                       |
| ------------ | --------------------------------------------- |
| **Overview** | Model metadata, key metrics, dataset link     |
| **Train**    | Training charts, console output, system stats |
| **Predict**  | Interactive browser inference                 |
| **Export**   | Format conversion with GPU selection          |
| **Deploy**   | Endpoint creation and management              |

### Overview Tab

Displays model metadata and key metrics:

- Model name (editable), status badge, task type
- Final metrics (mAP50, mAP50-95, precision, recall)
- Metric sparkline charts showing training progression
- Training arguments (epochs, batch size, image size, etc.)
- Dataset link (when trained with a Platform dataset)
- Download button for model weights

![Ultralytics Platform Model Overview Metrics And Args](https://cdn.jsdelivr.net/gh/ultralytics/assets@main/docs/platform/platform-model-overview-metrics-and-args.avif)

### Train Tab

The Train tab has three subtabs:

#### Charts Subtab

Interactive training metric charts showing loss curves and performance metrics over epochs:

| Chart Group         | Metrics                                        |
| ------------------- | ---------------------------------------------- |
| **Metrics**         | mAP50, mAP50-95, precision, recall             |
| **Training Loss**   | train/box_loss, train/cls_loss, train/dfl_loss |
| **Validation Loss** | val/box_loss, val/cls_loss, val/dfl_loss       |
| **Learning Rate**   | lr/pg0, lr/pg1, lr/pg2                         |

![Ultralytics Platform Model Train Charts Subtab](https://cdn.jsdelivr.net/gh/ultralytics/assets@main/docs/platform/platform-model-train-charts-subtab.avif)

#### Console Subtab

Live console output from the training process:

- Real-time log streaming during training
- Epoch progress bars and validation results
- Error detection with highlighted error banners
- ANSI color support for formatted output

![Ultralytics Platform Model Train Console Subtab](https://cdn.jsdelivr.net/gh/ultralytics/assets@main/docs/platform/platform-model-train-console-subtab.avif)

#### System Subtab

GPU and system metrics during training:

| Metric         | Description                |
| -------------- | -------------------------- |
| **GPU Util**   | GPU utilization percentage |
| **GPU Memory** | GPU memory usage           |
| **GPU Temp**   | GPU temperature            |
| **CPU Usage**  | CPU utilization            |
| **RAM**        | System memory usage        |
| **Disk**       | Disk usage                 |

![Ultralytics Platform Model Train System Subtab](https://cdn.jsdelivr.net/gh/ultralytics/assets@main/docs/platform/platform-model-train-system-subtab.avif)

### Predict Tab

Run interactive inference directly in the browser:

- Upload an image, use example images, or use webcam
- Results display with bounding boxes, masks, semantic class maps, or keypoints
- Auto-inference when an image is provided
- Supports all task types ([detect](../../tasks/detect.md), [segment](../../tasks/segment.md), [semantic](../../tasks/semantic.md), [pose](../../tasks/pose.md), [OBB](../../tasks/obb.md), [classify](../../tasks/classify.md))

!!! tip "Quick Testing"

    The Predict tab runs inference on Ultralytics Cloud, so you don't need a local GPU. Results are displayed with interactive overlays matching the model's task type.

### Export Tab

Export your model to 19+ deployment formats. See [Export Model](#export-model) below and the core [Export mode guide](../../modes/export.md) for full details.

### Deploy Tab

Create and manage dedicated inference endpoints. See [Deployments](../deploy/index.md) for details.

## Validation Plots

After training completes, view detailed validation analysis:

### Confusion Matrix

Interactive heatmap showing prediction accuracy per class:

![Ultralytics Platform Model Confusion Matrix](https://cdn.jsdelivr.net/gh/ultralytics/assets@main/docs/platform/platform-model-confusion-matrix.avif)

### PR/F1 Curves

Performance curves at different confidence thresholds:

![Ultralytics Platform Model Pr F1 Curves](https://cdn.jsdelivr.net/gh/ultralytics/assets@main/docs/platform/platform-model-pr-f1-curves.avif)

| Curve                    | Description                              |
| ------------------------ | ---------------------------------------- |
| **Precision-Recall**     | Trade-off between precision and recall   |
| **F1-Confidence**        | F1 score at different confidence levels  |
| **Precision-Confidence** | Precision at different confidence levels |
| **Recall-Confidence**    | Recall at different confidence levels    |

## Export Model

```mermaid
graph LR
    A[Select Format]:::start --> B[Configure Args]:::proc
    B --> C[Export]:::proc
    C --> D{GPU Required?}:::decide
    D -->|Yes| E[Cloud GPU Export]:::proc
    D -->|No| F[CPU Export]:::proc
    E --> G[Download]:::out
    F --> G

    classDef start fill:#4CAF50,color:#fff
    classDef proc fill:#2196F3,color:#fff
    classDef decide fill:#FF9800,color:#fff
    classDef out fill:#9C27B0,color:#fff
```

Export your model to 19+ deployment formats:

1. Navigate to the **Export** tab
2. Select target format
3. Configure export arguments (image size, half precision, dynamic, etc.)
4. For GPU-required formats (TensorRT), select a GPU type
5. Click **Export**
6. Download when complete

![Ultralytics Platform Model Export Tab Format List](https://cdn.jsdelivr.net/gh/ultralytics/assets@main/docs/platform/platform-model-export-tab-format-list.avif)

### Supported Formats

The Platform supports export to [19+ deployment formats](../../modes/export.md#export-formats): ONNX, TorchScript, OpenVINO, TensorRT, CoreML, TF SavedModel, TF GraphDef, LiteRT, TF Edge TPU, PaddlePaddle, NCNN, MNN, RKNN, Qualcomm (QNN), IMX500, Axelera, ExecuTorch, and DeepX.

### Format Selection Guide

| Target             | Recommended Format  | Notes                                                          |
| ------------------ | ------------------- | -------------------------------------------------------------- |
| **NVIDIA GPUs**    | TensorRT            | Select the same GPU family as the deployment device            |
| **NVIDIA Jetson**  | TensorRT            | Select the intended target and check its validation status     |
| **Intel Hardware** | OpenVINO            | CPUs, GPUs, and VPUs                                           |
| **Apple Devices**  | CoreML or LiteRT    | iOS, macOS, Apple Silicon                                      |
| **Android**        | LiteRT or NCNN      | LiteRT (Google's on-device runtime) or NCNN for ARM            |
| **Web Browsers**   | LiteRT.js or ONNX   | LiteRT.js or ONNX via ONNX Runtime Web                         |
| **Edge Devices**   | TF Edge TPU or RKNN | Coral and Rockchip (see [supported chips](#rknn-chip-support)) |
| **General**        | ONNX                | Works with most runtimes                                       |

![Ultralytics Platform Model Export Progress](https://cdn.jsdelivr.net/gh/ultralytics/assets@main/docs/platform/platform-model-export-progress.avif)

### NVIDIA Jetson TensorRT Targets

Ultralytics Platform offers the following Jetson target selections for TensorRT `.engine` exports. As of July 2026, the Jetson export workers use JetPack 7.2 / L4T r39.2, Python 3.12.3, NVIDIA PyTorch 2.12.0a0 (26.04 build), CUDA 13.2, and TensorRT 10.16.1.11 inside the export container.

| Target selection           | API `gpuType`          | Memory | GPU architecture   | Python | CUDA | TensorRT   | Measured YOLO26n FP16 export | Physical build/load validation                |
| -------------------------- | ---------------------- | -----: | ------------------ | ------ | ---- | ---------- | ---------------------------: | --------------------------------------------- |
| Jetson Thor T5000          | `jetson-thor-t5000`    | 128 GB | Blackwell, CC 11.0 | 3.12.3 | 13.2 | 10.16.1.11 |                      ~1m 46s | Thor in NVIDIA T4000 profile; T5000 candidate |
| Jetson Thor T4000          | `jetson-thor-t4000`    |  64 GB | Blackwell, CC 11.0 | 3.12.3 | 13.2 | 10.16.1.11 |                      ~1m 46s | Thor in NVIDIA T4000 profile                  |
| Jetson AGX Orin 64GB       | `jetson-agx-orin-64gb` |  64 GB | Ampere, CC 8.7     | 3.12.3 | 13.2 | 10.16.1.11 |                       7m 15s | Built, loaded, and inferred on AGX Orin 64GB  |
| Jetson AGX Orin 32GB       | `jetson-agx-orin-32gb` |  32 GB | Ampere, CC 8.7     | 3.12.3 | 13.2 | 10.16.1.11 |                       5m 34s | AGX Orin 64GB build/load; 32GB SKU pending    |
| Jetson Orin NX 16GB        | `jetson-orin-nx-16gb`  |  16 GB | Ampere, CC 8.7     | 3.12.3 | 13.2 | 10.16.1.11 |                       5m 09s | AGX Orin 64GB build/load; NX SKU pending      |
| Jetson Orin NX 8GB         | `jetson-orin-nx-8gb`   |   8 GB | Ampere, CC 8.7     | 3.12.3 | 13.2 | 10.16.1.11 |                       5m 01s | AGX Orin 64GB build/load; NX SKU pending      |
| Jetson Orin Nano 8GB Super | `jetson-orin-nano-8gb` |   8 GB | Ampere, CC 8.7     | 3.12.3 | 13.2 | 10.16.1.11 |                       4m 59s | AGX Orin 64GB build/load; Nano SKU pending    |
| Jetson Orin Nano 4GB       | `jetson-orin-nano-4gb` |   4 GB | Ampere, CC 8.7     | 3.12.3 | 13.2 | 10.16.1.11 |                       5m 01s | AGX Orin 64GB build/load; Nano SKU pending    |

The timings are single observed end-to-end production routing tests from July 2026, rounded to the nearest second; they are reference measurements, not an SLA or per-SKU performance benchmark. Both Thor selections are built on a T5000 Developer Kit in NVIDIA's T4000 compatibility profile. The six Orin routes are built on an AGX Orin 64GB, where every resulting engine was loaded and run. Because TensorRT engines are tied to the build GPU and software stack, the smaller-Orin artifacts are candidates until they load and infer on their listed SKUs. Test memory fit on smaller Orin SKUs and perform INT8 calibration on the target device for best results. See the [NVIDIA Jetson guide](../../guides/nvidia-jetson.md) and [TensorRT integration guide](../../integrations/tensorrt.md) for local deployment details.

!!! warning "Match the TensorRT engine build environment"

    Before loading a downloaded engine, confirm the deployment device matches the build platform and GPU family, uses the same TensorRT version, and provides a compatible CUDA runtime. For Jetson targets, the software versions are shown in the table above. If the environments do not match, export the engine locally on the deployment device instead.

### RKNN Chip Support

When exporting to RKNN format, select your target Rockchip device:

| Chip    | Description          |
| ------- | -------------------- |
| RK3588  | High-end edge SoC    |
| RK3576  | Mid-range edge SoC   |
| RK3568  | Mid-range edge SoC   |
| RK3566  | Mid-range edge SoC   |
| RK3562  | Entry-level edge SoC |
| RV1103  | Vision processor     |
| RV1106  | Vision processor     |
| RV1103B | Vision processor     |
| RV1106B | Vision processor     |
| RK2118  | AI processor         |
| RV1126B | Vision processor     |

### Export Job Lifecycle

Export jobs progress through the following statuses:

| Status        | Description                          |
| ------------- | ------------------------------------ |
| **Queued**    | Export job is waiting to start       |
| **Starting**  | Export job is initializing           |
| **Running**   | Export is in progress                |
| **Completed** | Export finished — download available |
| **Failed**    | Export failed (see error message)    |
| **Cancelled** | Export was cancelled by the user     |

!!! tip "Export Time"

    Export time varies by format and build host. TensorRT exports may take several minutes because TensorRT profiles and tunes the engine on the physical GPU shown in the [Jetson validation table](#nvidia-jetson-tensorrt-targets) or the selected cloud GPU.

### Bulk Export Actions

- **Export All**: Click `Export All` to start export jobs for all CPU-based formats with default settings.
- **Delete All Exports**: Click `Delete All` to remove all exports for the model.

### Format Restrictions

Some export formats have architecture or task restrictions:

| Format      | Restriction                                |
| ----------- | ------------------------------------------ |
| **IMX500**  | Available only for `YOLOv8n` and `YOLO11n` |
| **Axelera** | Detect models only                         |

!!! note "Additional Export Rules"

    - Classification exports do not include NMS.
    - CoreML exports with batch sizes greater than `1` use `dynamic=true`.
    - Unsupported format/model combinations are disabled in the export dialog before you launch.

## Clone Model

Clone a model to a different project:

1. Open the model page
2. Click the **Clone** button
3. Select the destination project
4. Click **Clone**

The model and its weights are copied to the target project.

## Download Model

Download your model weights:

1. Navigate to the model's **Overview** tab
2. Click the **Download** button
3. The original `.pt` file downloads automatically

Exported formats can be downloaded from the **Export** tab after export completes.

## Dataset Linking

Models can be linked to their source dataset:

- View which dataset was used for training
- Click the dataset card on the Overview tab to navigate to it
- Track data lineage

When training with Platform datasets using the [`ul://` URI format](../data/datasets.md#dataset-uri), linking is automatic.

!!! example "Dataset URI Format"

    ```bash
    # Train with a Platform dataset — linking is automatic
    yolo train model=yolo26n.pt data=ul://username/datasets/my-dataset epochs=100
    ```

    The `ul://` scheme resolves to your Platform dataset. The trained model's Overview tab will show a link back to this dataset (see [Using Platform Datasets](../api/index.md#using-platform-datasets)).

## Visibility Settings

Control who can see your model:

| Setting     | Description                     |
| ----------- | ------------------------------- |
| **Private** | Only you can access             |
| **Public**  | Anyone can view on Explore page |

To change visibility, click the visibility badge (e.g., `private` or `public`) in the page header. Visibility is set at the project level, so this controls all models in the project. Switching to private takes effect immediately. Switching to public shows a confirmation dialog before applying.

## Delete Model

Remove a model you no longer need:

1. Open model actions menu
2. Click **Delete**
3. Confirm deletion

!!! note "Trash and Restore"

    Deleted models go to Trash for 30 days. Restore from [Settings > Trash](../account/trash.md).

## See Also

- [**Inference**](../deploy/inference.md): Test models in the browser with the Predict tab
- [**Endpoints**](../deploy/endpoints.md): Deploy models to production with dedicated endpoints
- [**Cloud Training**](cloud-training.md): Configure and run training jobs on cloud GPUs
- [**Export Formats**](../../modes/export.md): Full guide to all 19+ export formats

## FAQ

### What model architectures are supported?

Ultralytics Platform fully supports all YOLO architectures with dedicated projects:

- [**YOLO26**](../../models/yolo26.md): n, s, m, l, x variants (latest, recommended) — [platform.ultralytics.com/ultralytics/yolo26](https://platform.ultralytics.com/ultralytics/yolo26)
- [**YOLO11**](../../models/yolo11.md): n, s, m, l, x variants — [platform.ultralytics.com/ultralytics/yolo11](https://platform.ultralytics.com/ultralytics/yolo11)
- [**YOLOv8**](../../models/yolov8.md): n, s, m, l, x variants — [platform.ultralytics.com/ultralytics/yolov8](https://platform.ultralytics.com/ultralytics/yolov8)
- [**YOLOv5**](../../models/yolov5.md): n, s, m, l, x variants — [platform.ultralytics.com/ultralytics/yolov5](https://platform.ultralytics.com/ultralytics/yolov5)

YOLO26 supports 6 task types: [detect](../../tasks/detect.md), [segment](../../tasks/segment.md), [semantic](../../tasks/semantic.md), [pose](../../tasks/pose.md), [OBB](../../tasks/obb.md), and [classify](../../tasks/classify.md). YOLO11 and YOLOv8 support the same set except semantic segmentation, while YOLOv5 supports detect, segment, and classify.

### Can I download my trained model?

Yes, download your model weights from the model page:

1. Click the download icon on the Overview tab
2. The original `.pt` file downloads automatically
3. Exported formats can be downloaded from the Export tab

### How do I compare models across projects?

Currently, model comparison is within projects. To compare across projects:

1. Clone models to a single project, or
2. Export metrics and compare externally

### What's the maximum model size?

Uploaded `.pt` model files are limited to 1 GB, and models near that limit may take longer to upload and process.

### Can I fine-tune pretrained models?

Yes! You can use any of the official YOLO26 models as a base, or select one of your own completed models from the model selector in the training dialog. The Platform supports fine-tuning from any uploaded checkpoint.
