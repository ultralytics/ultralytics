---
comments: true
description: Explore integration options for Ultralytics HUB. Currently featuring Roboflow for dataset integration and multiple export formats for your trained models.
keywords: Ultralytics HUB, Integrations, Roboflow, Dataset, Export, YOLOv5, YOLOv8, ONNX, CoreML, TensorRT, TensorFlow
---

# Ultralytics HUB Integrations

Welcome to the Integrations guide for [Ultralytics HUB](https://bit.ly/ultralytics_hub)!

🚧 **Under Construction** 🚧

We are in the process of expanding this section to provide you with comprehensive guidance on integrating your YOLOv5 and YOLOv8 models with various platforms and formats.

We appreciate your patience as we work to make this section comprehensive and user-friendly. Stay tuned for updates!

## Available Integrations

### Dataset

- **Roboflow**: Seamlessly import your datasets for training.

### Export

Available export formats are in the table below. You can predict or validate directly on exported models using the `ultralytics` Python package, i.e. `yolo predict model=yolov8n.onnx`.

| Format                                            | `format` Argument | Model                     | Metadata | Arguments                                                            |
| ------------------------------------------------- | ----------------- | ------------------------- | -------- | -------------------------------------------------------------------- |
| [PyTorch](https://pytorch.org/)                   | -                 | `yolov8n.pt`              | ✅       | -                                                                    |
| [TorchScript](../integrations/torchscript.md)     | `torchscript`     | `yolov8n.torchscript`     | ✅       | `imgsz`, `optimize`, `batch`                                         |
| [ONNX](../integrations/onnx.md)                   | `onnx`            | `yolov8n.onnx`            | ✅       | `imgsz`, `half`, `dynamic`, `simplify`, `opset`, `batch`             |
| [OpenVINO](../integrations/openvino.md)           | `openvino`        | `yolov8n_openvino_model/` | ✅       | `imgsz`, `half`, `int8`, `batch`                                     |
| [TensorRT](../integrations/tensorrt.md)           | `engine`          | `yolov8n.engine`          | ✅       | `imgsz`, `half`, `dynamic`, `simplify`, `workspace`, `int8`, `batch` |
| [CoreML](../integrations/coreml.md)               | `coreml`          | `yolov8n.mlpackage`       | ✅       | `imgsz`, `half`, `int8`, `nms`, `batch`                              |
| [TF SavedModel](../integrations/tf-savedmodel.md) | `saved_model`     | `yolov8n_saved_model/`    | ✅       | `imgsz`, `keras`, `int8`, `batch`                                    |
| [TF GraphDef](../integrations/tf-graphdef.md)     | `pb`              | `yolov8n.pb`              | ❌       | `imgsz`, `batch`                                                     |
| [TF Lite](../integrations/tflite.md)              | `tflite`          | `yolov8n.tflite`          | ✅       | `imgsz`, `half`, `int8`, `batch`                                     |
| [TF Edge TPU](../integrations/edge-tpu.md)        | `edgetpu`         | `yolov8n_edgetpu.tflite`  | ✅       | `imgsz`, `batch`                                                     |
| [TF.js](../integrations/tfjs.md)                  | `tfjs`            | `yolov8n_web_model/`      | ✅       | `imgsz`, `half`, `int8`, `batch`                                     |
| [PaddlePaddle](../integrations/paddlepaddle.md)   | `paddle`          | `yolov8n_paddle_model/`   | ✅       | `imgsz`, `batch`                                                     |
| [NCNN](../integrations/ncnn.md)                   | `ncnn`            | `yolov8n_ncnn_model/`     | ✅       | `imgsz`, `half`, `batch`                                             |

## Coming Soon

- Additional Dataset Integrations
- Detailed Export Integration Guides
- Step-by-Step Tutorials for Each Integration
