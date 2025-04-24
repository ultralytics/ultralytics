{%set tip1 = ':material-information-outline:{ title="conf, iou, agnostic_nms are also available when nms=True" }' %}
{%set tip2 = ':material-information-outline:{ title="conf, iou are also available when nms=True" }' %}
{%set tip3 = ':material-information-outline:{ title="imx format only supported for YOLOv8n model currently" }' %}

| Format | `format` Argument | Model | Metadata | Arguments |
| ------ | ----------------- | ----- | -------- | --------- |

| [PyTorch](https://pytorch.org/) | - | `{{ model_name or "yolo11n" }}.pt` | ✅ | - |

| [TorchScript](../integrations/torchscript.md) | `torchscript` | `{{ model_name or "yolo11n" }}.torchscript` | ✅ | `imgsz`, `batch`, `optimize`, `half`, `nms`{{ tip1 }}, `device` |

| [ONNX](../integrations/onnx.md) | `onnx` | `{{ model_name or "yolo11n" }}.onnx` | ✅ | `imgsz`, `batch`, `dynamic`, `half`, `opset`, `simplify`, `nms`{{ tip1 }}, `device` |

| [OpenVINO](../integrations/openvino.md) | `openvino` | `{{ model_name or "yolo11n" }}_openvino_model/` | ✅ | `imgsz`, `batch`, `dynamic`, `half`, `int8`, `nms`{{ tip1 }}, `fraction`, `device`, `data` |

| [TensorRT](../integrations/tensorrt.md) | `engine` | `{{ model_name or "yolo11n" }}.engine` | ✅ | `imgsz`, `batch`, `dynamic`, `half`, `int8`, `simplify`, `nms`{{ tip1 }}, `fraction`, `device`, `data`, `workspace` |

| [CoreML](../integrations/coreml.md) | `coreml` | `{{ model_name or "yolo11n" }}.mlpackage` | ✅ | `imgsz`, `batch`, `half`, `int8`, `nms`{{ tip2 }}, `device` |
| [TF SavedModel](../integrations/tf-savedmodel.md) | `saved_model` | `{{ model_name or "yolo11n" }}_saved_model/` | ✅ | `imgsz`, `batch`, `int8`, `keras`, `nms`{{ tip1 }}, `device` |
| [TF GraphDef](../integrations/tf-graphdef.md) | `pb` | `{{ model_name or "yolo11n" }}.pb` | ❌ | `imgsz`, `batch`, `device` |
| [TF Lite](../integrations/tflite.md) | `tflite` | `{{ model_name or "yolo11n" }}.tflite` | ✅ | `imgsz`, `batch`, `half`, `int8`, `nms`{{ tip1 }}, `fraction`, `device`, `data` |
| [TF Edge TPU](../integrations/edge-tpu.md) | `edgetpu` | `{{ model_name or "yolo11n" }}_edgetpu.tflite` | ✅ | `imgsz`, `device` |
| [TF.js](../integrations/tfjs.md) | `tfjs` | `{{ model_name or "yolo11n" }}_web_model/` | ✅ | `imgsz`, `batch`, `half`, `int8`, `nms`{{ tip1 }}, `device` |
| [PaddlePaddle](../integrations/paddlepaddle.md) | `paddle` | `{{ model_name or "yolo11n" }}_paddle_model/` | ✅ | `imgsz`, `batch`, `device` |
| [MNN](../integrations/mnn.md) | `mnn` | `{{ model_name or "yolo11n" }}.mnn` | ✅ | `imgsz`, `batch`, `half`, `int8`, `device` |
| [NCNN](../integrations/ncnn.md) | `ncnn` | `{{ model_name or "yolo11n" }}_ncnn_model/` | ✅ | `imgsz`, `batch`, `half`, `device` |
| [IMX500](../integrations/sony-imx500.md){{ tip3 }} | `imx` | `{{ model_name or "yolov8n" }}_imx_model/` | ✅ | `imgsz`, `int8`, `fraction`, `device`, `data` |
| [RKNN](../integrations/rockchip-rknn.md) | `rknn` | `{{ model_name or "yolo11n" }}_rknn_model/` | ✅ | `imgsz`, `batch`, `name`, `device` |
