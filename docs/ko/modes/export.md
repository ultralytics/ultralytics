---
comments: true
description: YOLOv8 모델을 ONNX, TensorRT, CoreML 등의 다양한 형식으로 내보내는 단계별 가이드를 확인해보세요. 이제 배포를 진행해보세요!.
keywords: YOLO, YOLOv8, Ultralytics, 모델 내보내기, ONNX, TensorRT, CoreML, TensorFlow SavedModel, OpenVINO, PyTorch, 모델 내보내기
---

# Ultralytics YOLO를 사용한 모델 내보내기

<img width="1024" src="https://github.com/ultralytics/assets/raw/main/yolov8/banner-integrations.png" alt="Ultralytics YOLO 생태계 및 통합">

## 소개

모델을 훈련하는 최종 목적은 실제 환경에서 배포하기 위함입니다. Ultralytics YOLOv8의 내보내기 모드는 훈련된 모델을 다양한 형식으로 내보내어 여러 플랫폼과 디바이스에서 배포할 수 있는 범용적인 옵션을 제공합니다. 이 포괄적인 가이드는 모델 내보내기의 미묘한 점들을 설명하고 최대의 호환성과 성능을 달성하는 방법을 안내하는 것을 목표로 합니다.

<p align="center">
  <br>
  <iframe width="720" height="405" src="https://www.youtube.com/embed/WbomGeoOT_k?si=aGmuyooWftA0ue9X"
    title="YouTube 비디오 플레이어" frameborder="0"
    allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share"
    allowfullscreen>
  </iframe>
  <br>
  <strong>시청하기:</strong> 사용자 지정 훈련된 Ultralytics YOLOv8 모델을 내보내고 웹캠에서 실시간 추론을 실행하는 방법.
</p>

## YOLOv8의 내보내기 모드를 선택해야 하는 이유는 무엇인가요?

- **범용성:** ONNX, TensorRT, CoreML 등 다양한 형식으로 내보낼 수 있습니다.
- **성능:** TensorRT를 사용할 경우 최대 5배 빠른 GPU 속도 향상 및 ONNX 또는 OpenVINO를 사용하여 최대 3배 빠른 CPU 속도 향상을 얻을 수 있습니다.
- **호환성:** 모델을 다양한 하드웨어 및 소프트웨어 환경에서 배포할 수 있도록 만들어줍니다.
- **사용의 용이성:** 모델 내보내기를 위한 간단한 CLI 및 Python API 제공.

### 내보내기 모드의 주요 기능

다음은 몇 가지 주요 기능들입니다:

- **원클릭 내보내기:** 다양한 형식으로 내보내기 위한 간단한 명령어.
- **배치 내보내기:** 배치 추론이 가능한 모델들을 내보낼 수 있습니다.
- **최적화된 추론:** 내보낸 모델들은 더 빠른 추론 시간을 위해 최적화되어 있습니다.
- **튜토리얼 비디오:** 원활한 내보내기 경험을 위한 심도 있는 가이드 및 튜토리얼.

!!! Tip "팁"

    * CPU 속도 향상을 위해 ONNX 또는 OpenVINO로 내보내세요.
    * GPU 속도 향상을 위해 TensorRT로 내보내세요.

## 사용 예시

YOLOv8n 모델을 ONNX나 TensorRT와 같은 다른 형식으로 내보냅니다. 내보내기 인수에 대한 전체 목록은 아래 '인수' 섹션을 참조하세요.

!!! 예시 ""

    === "Python"

        ```python
        from ultralytics import YOLO

        # 모델을 불러오기
        model = YOLO('yolov8n.pt')  # 공식 모델을 불러오기
        model = YOLO('path/to/best.pt')  # 사용자 지정 훈련 모델을 불러오기

        # 모델을 내보내기
        model.export(format='onnx')
        ```
    === "CLI"

        ```bash
        yolo export model=yolov8n.pt format=onnx  # 공식 모델을 내보내기
        yolo export model=path/to/best.pt format=onnx  # 사용자 지정 훈련 모델을 내보내기
        ```

## 인수

YOLO 모델의 내보내기 설정은 다른 환경이나 플랫폼에서 모델을 사용하기 위해 저장 또는 내보내기할 때 사용하는 다양한 구성 및 옵션을 의미합니다. 이러한 설정은 모델의 성능, 크기 및 다양한 시스템과의 호환성에 영향을 미칠 수 있습니다. 일반적인 YOLO 내보내기 설정에는 내보낸 모델 파일의 형식(예: ONNX, TensorFlow SavedModel), 모델이 실행될 장치(예: CPU, GPU) 및 마스크 또는 상자당 여러 라벨과 같은 추가 기능의 포함 여부 등이 있습니다. 모델이 사용되는 특정 작업과 대상 환경 또는 플랫폼의 요구 사항이나 제약 사항에 따라 내보내기 과정에 영향을 미치는 다른 요소들도 있을 수 있습니다. 내보낸 모델이 의도한 용도로 최적화되어 있고 대상 환경에서 효과적으로 사용할 수 있도록 이러한 설정을 세심하게 고려하고 구성하는 것이 중요합니다.

| 키           | 값               | 설명                                          |
|-------------|-----------------|---------------------------------------------|
| `format`    | `'torchscript'` | 내보낼 형식                                      |
| `imgsz`     | `640`           | 스칼라 또는 (h, w) 리스트 형식의 이미지 크기, 예: (640, 480) |
| `keras`     | `False`         | TF SavedModel 내보내기에 Keras 사용                |
| `optimize`  | `False`         | TorchScript: 모바일 최적화                        |
| `half`      | `False`         | FP16 양자화                                    |
| `int8`      | `False`         | INT8 양자화                                    |
| `dynamic`   | `False`         | ONNX/TensorRT: 동적 축                         |
| `simplify`  | `False`         | ONNX/TensorRT: 모델 단순화                       |
| `opset`     | `None`          | ONNX: opset 버전 (선택적, 기본값은 최신)               |
| `workspace` | `4`             | TensorRT: 작업공간 크기 (GB)                      |
| `nms`       | `False`         | CoreML: NMS 추가                              |

## 내보내기 형식

아래 표에는 사용 가능한 YOLOv8 내보내기 형식이 나와 있습니다. `format` 인수를 사용하여 어떤 형식으로든 내보낼 수 있습니다. 예: `format='onnx'` 또는 `format='engine'`.

| 형식                                                                 | `format` 인수   | 모델                        | 메타데이터 | 인수                                                  |
|--------------------------------------------------------------------|---------------|---------------------------|-------|-----------------------------------------------------|
| [PyTorch](https://pytorch.org/)                                    | -             | `yolov8n.pt`              | ✅     | -                                                   |
| [TorchScript](https://pytorch.org/docs/stable/jit.html)            | `torchscript` | `yolov8n.torchscript`     | ✅     | `imgsz`, `optimize`                                 |
| [ONNX](https://onnx.ai/)                                           | `onnx`        | `yolov8n.onnx`            | ✅     | `imgsz`, `half`, `dynamic`, `simplify`, `opset`     |
| [OpenVINO](https://docs.openvino.ai/latest/index.html)             | `openvino`    | `yolov8n_openvino_model/` | ✅     | `imgsz`, `half`                                     |
| [TensorRT](https://developer.nvidia.com/tensorrt)                  | `engine`      | `yolov8n.engine`          | ✅     | `imgsz`, `half`, `dynamic`, `simplify`, `workspace` |
| [CoreML](https://github.com/apple/coremltools)                     | `coreml`      | `yolov8n.mlpackage`       | ✅     | `imgsz`, `half`, `int8`, `nms`                      |
| [TF SavedModel](https://www.tensorflow.org/guide/saved_model)      | `saved_model` | `yolov8n_saved_model/`    | ✅     | `imgsz`, `keras`                                    |
| [TF GraphDef](https://www.tensorflow.org/api_docs/python/tf/Graph) | `pb`          | `yolov8n.pb`              | ❌     | `imgsz`                                             |
| [TF Lite](https://www.tensorflow.org/lite)                         | `tflite`      | `yolov8n.tflite`          | ✅     | `imgsz`, `half`, `int8`                             |
| [TF Edge TPU](https://coral.ai/docs/edgetpu/models-intro/)         | `edgetpu`     | `yolov8n_edgetpu.tflite`  | ✅     | `imgsz`                                             |
| [TF.js](https://www.tensorflow.org/js)                             | `tfjs`        | `yolov8n_web_model/`      | ✅     | `imgsz`                                             |
| [PaddlePaddle](https://github.com/PaddlePaddle)                    | `paddle`      | `yolov8n_paddle_model/`   | ✅     | `imgsz`                                             |
| [ncnn](https://github.com/Tencent/ncnn)                            | `ncnn`        | `yolov8n_ncnn_model/`     | ✅     | `imgsz`, `half`                                     |
