---
comments: true
description: Explore integration options for Ultralytics HUB. Currently featuring Roboflow for dataset integration and multiple export formats for your trained models.
keywords: Ultralytics HUB, Integrations, Roboflow, Dataset, Export, YOLOv5, YOLOv8, ONNX, CoreML, TensorRT, TensorFlow
---

# HUB Integrations

🚧 **Under Construction** 🚧

Welcome to the Integrations guide for [Ultralytics HUB](https://hub.ultralytics.com/)! We are in the process of expanding this section to provide you with comprehensive guidance on integrating your YOLOv5 and YOLOv8 models with various platforms and formats. Currently, Roboflow is our available dataset integration, with a wide range of export integrations for your trained models.

<p align="center">
  <br>
  <iframe loading="lazy" width="720" height="405" src="https://www.youtube.com/embed/lveF9iCMIzc?si=_Q4WB5kMB5qNe7q6"
    title="YouTube video player" frameborder="0"
    allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share"
    allowfullscreen>
  </iframe>
  <br>
  <strong>Watch:</strong> Train Your Custom YOLO Models In A Few Clicks with Ultralytics HUB.
</p>

## Available Integrations

### Dataset Integrations

- **Roboflow**: Seamlessly import your datasets for training.

### Export Integrations

| Format                                            | `format` Argument | Model                         | Metadata | Arguments                                                    |
|---------------------------------------------------|-------------------|-------------------------------|----------|--------------------------------------------------------------|
| [PyTorch](https://pytorch.org/)                   | -                 | `yolov8n-obb.pt`              | ✅        | -                                                            |
| [TorchScript](../integrations/torchscript.md)     | `torchscript`     | `yolov8n-obb.torchscript`     | ✅        | `imgsz`, `optimize`, `batch`                                 |
| [ONNX](../integrations/onnx.md)                   | `onnx`            | `yolov8n-obb.onnx`            | ✅        | `imgsz`, `half`, `dynamic`, `simplify`, `opset`, `batch`     |
| [OpenVINO](../integrations/openvino.md)           | `openvino`        | `yolov8n-obb_openvino_model/` | ✅        | `imgsz`, `half`, `int8`, `batch`                             |
| [TensorRT](../integrations/tensorrt.md)           | `engine`          | `yolov8n-obb.engine`          | ✅        | `imgsz`, `half`, `dynamic`, `simplify`, `workspace`, `batch` |
| [CoreML](../integrations/coreml.md)               | `coreml`          | `yolov8n-obb.mlpackage`       | ✅        | `imgsz`, `half`, `int8`, `nms`, `batch`                      |
| [TF SavedModel](../integrations/tf-savedmodel.md) | `saved_model`     | `yolov8n-obb_saved_model/`    | ✅        | `imgsz`, `keras`, `int8`, `batch`                            |
| [TF GraphDef](../integrations/tf-graphdef.md)     | `pb`              | `yolov8n-obb.pb`              | ❌        | `imgsz`, `batch`                                             |
| [TF Lite](../integrations/tflite.md)              | `tflite`          | `yolov8n-obb.tflite`          | ✅        | `imgsz`, `half`, `int8`, `batch`                             |
| [TF Edge TPU](../integrations/edge-tpu.md)        | `edgetpu`         | `yolov8n-obb_edgetpu.tflite`  | ✅        | `imgsz`, `batch`                                             |
| [TF.js](../integrations/tfjs.md)                  | `tfjs`            | `yolov8n-obb_web_model/`      | ✅        | `imgsz`, `half`, `int8`, `batch`                             |
| [PaddlePaddle](../integrations/paddlepaddle.md)   | `paddle`          | `yolov8n-obb_paddle_model/`   | ✅        | `imgsz`, `batch`                                             |
| [NCNN](../integrations/ncnn.md)                   | `ncnn`            | `yolov8n-obb_ncnn_model/`     | ✅        | `imgsz`, `half`, `batch`                                     |

## Coming Soon

- Additional Dataset Integrations
- Detailed Export Integration Guides
- Step-by-Step Tutorials for Each Integration

## Need Immediate Assistance?

While we're in the process of creating detailed guides:

- Browse through other [HUB Docs](https://docs.ultralytics.com/hub/) for detailed guides and tutorials.
- Raise an issue on our [GitHub](https://github.com/ultralytics/hub/) for technical support.
- Join our [Discord Community](https://ultralytics.com/discord/) for live discussions and community support.

We appreciate your patience as we work to make this section comprehensive and user-friendly. Stay tuned for updates!
