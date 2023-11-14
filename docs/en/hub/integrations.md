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
  <iframe width="720" height="405" src="https://www.youtube.com/embed/lveF9iCMIzc?si=_Q4WB5kMB5qNe7q6"
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

| Format                      | `format=`Argument | Model                     | Metadata           | Available Modifiers                                 |
|-----------------------------|-------------------|:--------------------------|:------------------:|-----------------------------------------------------|
| [PyTorch][pytorch]          | -                 | `yolov8n.pt`              | :white_check_mark: | -                                                   |
| [TorchScript][torchscript]  | `torchscript`     | `yolov8n.torchscript`     | :white_check_mark: | `imgsz`, `optimize`                                 |
| [ONNX][onnx]                | `onnx`            | `yolov8n.onnx`            | :white_check_mark: | `imgsz`, `half`, `dynamic`, `simplify`, `opset`     |
| [OpenVINO][openvino]        | `openvino`        | `yolov8n_openvino_model/` | :white_check_mark: | `imgsz`, `half`                                     |
| [TensorRT][tensorrt]        | `engine`          | `yolov8n.engine`          | :white_check_mark: | `imgsz`, `half`, `dynamic`, `simplify`, `workspace` |
| [CoreML][coreml]            | `coreml`          | `yolov8n.mlpackage`       | :white_check_mark: | `imgsz`, `half`, `int8`, `nms`                      |
| [TF SavedModel][tf_saved]   | `saved_model`     | `yolov8n_saved_model/`    | :white_check_mark: | `imgsz`, `keras`                                    |
| [TF GraphDef][tf_graph]     | `pb`              | `yolov8n.pb`              | :x:                | `imgsz`                                             |
| [TF Lite][tf_lite]          | `tflite`          | `yolov8n.tflite`          | :white_check_mark: | `imgsz`, `half`, `int8`                             |
| [TF Edge TPU][tf_edge_tpu]  | `edgetpu`         | `yolov8n_edgetpu.tflite`  | :white_check_mark: | `imgsz`                                             |
| [TF.js][tf_js]              | `tfjs`            | `yolov8n_web_model/`      | :white_check_mark: | `imgsz`                                             |
| [PaddlePaddle][paddle]      | `paddle`          | `yolov8n_paddle_model/`   | :white_check_mark: | `imgsz`                                             |
| [ncnn][ncnn]                | `ncnn`            | `yolov8n_ncnn_model/`     | :white_check_mark: | `imgsz`, `half`                                     |

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

[pytorch]:     https://pytorch.org/
[torchscript]: https://pytorch.org/docs/stable/jit.html
[onnx]:        https://onnx.ai/
[openvino]:    https://docs.openvino.ai/latest/index.html
[tensorrt]:    https://developer.nvidia.com/tensorrt
[coreml]:      https://github.com/apple/coremltools
[tf_saved]:    https://www.tensorflow.org/guide/saved_model
[tf_graph]:    https://www.tensorflow.org/api_docs/python/tf/Graph
[tf_lite]:     https://www.tensorflow.org/lite
[tf_edge_tpu]: https://coral.ai/docs/edgetpu/models-intro/
[tf_js]:       https://www.tensorflow.org/js
[paddle]:      https://github.com/PaddlePaddle
[ncnn]:        https://github.com/Tencent/ncnn