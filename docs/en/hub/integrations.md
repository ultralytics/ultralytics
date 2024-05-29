---
description: Explore integration options for Ultralytics HUB. Currently featuring Roboflow for dataset integration and multiple export formats for your trained models. Discover what's next for Ultralytics with our under-construction page, previewing new, groundbreaking AI and ML features coming soon.
keywords: Ultralytics HUB, Integrations, Roboflow, Dataset, Export, YOLOv5, YOLOv8, ONNX, CoreML, TensorRT, TensorFlow, coming soon, under construction, new features, AI updates, ML advancements, YOLO, technology preview
---

# Ultralytics HUB Integrations - Under Construction 🏗️🌟

We are in the process of expanding this section to provide you with comprehensive guidance on integrating your YOLOv5 and YOLOv8 models with various platforms and formats.

We appreciate your patience as we work to make this section comprehensive and user-friendly. Stay tuned for updates!

### Available Integrations

#### Dataset

- **Roboflow**: Seamlessly import your datasets for training.

#### Export

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

## Exciting New Features on the Way 🎉

- Additional Dataset Integrations
- Detailed Export Integration Guides
- Step-by-Step Tutorials for Each Integration

## Stay Updated 🚧

This placeholder page is your first stop for upcoming developments. Keep an eye out for:

- **Newsletter:** Subscribe [here](https://ultralytics.com/#newsletter) for the latest news.
- **Social Media:** Follow us [here](https://www.linkedin.com/company/ultralytics) for updates and teasers.
- **Blog:** Visit our [blog](https://ultralytics.com/blog) for detailed insights.

## We Value Your Input 🗣️

Your feedback shapes our future releases. Share your thoughts and suggestions [here](https://ultralytics.com/survey).

## Thank You, Community! 🌍

Your [contributions](https://docs.ultralytics.com/help/contributing) inspire our continuous [innovation](https://github.com/ultralytics/ultralytics). Stay tuned for the big reveal of what's next in AI and ML at Ultralytics!

---

Excited for what's coming? Bookmark this page and get ready for a transformative AI and ML journey with Ultralytics! 🛠️🤖
