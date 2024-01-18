---
comments: true
description: YOLOv8의 다양한 내보내기 형식에 걸쳐 속도 및 정확성을 프로파일링하는 방법을 알아보고, mAP50-95, accuracy_top5 메트릭 및 기타에 대한 통찰을 얻으십시오.
keywords: Ultralytics, YOLOv8, 벤치마킹, 속도 프로파일링, 정확도 프로파일링, mAP50-95, accuracy_top5, ONNX, OpenVINO, TensorRT, YOLO 내보내기 형식
---

# Ultralytics YOLO를 사용한 모델 벤치마킹

<img width="1024" src="https://github.com/ultralytics/assets/raw/main/yolov8/banner-integrations.png" alt="Ultralytics YOLO 생태계 및 통합">

## 소개

모델을 학습하고 검증한 후, 다음으로 논리적인 단계는 다양한 실제 상황에서의 성능을 평가하는 것입니다. Ultralytics YOLOv8의 벤치마크 모드는 다양한 내보내기 형식에서 모델의 속도와 정확도를 평가하는 강력한 프레임워크를 제공하여 이와 같은 목적을 수행하는 역할을 합니다.

## 벤치마킹이 왜 중요한가요?

- **정보에 기반한 결정:** 속도와 정확도 사이의 타협점에 대한 통찰력을 얻을 수 있습니다.
- **자원 배분:** 다양한 하드웨어에서 각기 다른 내보내기 형식의 성능을 이해합니다.
- **최적화:** 특정 사용 사례에 가장 적합한 내보내기 형식을 알아냅니다.
- **비용 효율성:** 벤치마크 결과에 기반하여 하드웨어 자원을 보다 효율적으로 사용합니다.

### 벤치마크 모드의 주요 메트릭

- **mAP50-95:** 객체 인식, 세분화, 자세 추정에 사용됩니다.
- **accuracy_top5:** 이미지 분류에 사용됩니다.
- **추론 시간:** 각 이미지 당 밀리초로 측정된 시간입니다.

### 지원되는 내보내기 형식

- **ONNX:** CPU 성능 최적화를 위함
- **TensorRT:** GPU 효율성을 극대화하기 위함
- **OpenVINO:** 인텔 하드웨어 최적화를 위함
- **CoreML, TensorFlow SavedModel, 그 외:** 다양한 배포 요구 사항을 위함.

!!! Tip "팁"

    * CPU 속도 향상을 위해 ONNX 또는 OpenVINO로 내보내기.
    * GPU 속도 향상을 위해 TensorRT로 내보내기.

## 사용 예제

YOLOv8n 벤치마킹을 ONNX, TensorRT 등 모든 지원되는 내보내기 형식에 대해 실행합니다. 완벽한 내보내기 인수 목록을 보려면 아래의 인수 섹션을 참조하세요.

!!! Example "예제"

    === "파이썬"

        ```python
        from ultralytics.utils.benchmarks import benchmark

        # GPU에서 벤치마킹
        benchmark(model='yolov8n.pt', data='coco8.yaml', imgsz=640, half=False, device=0)
        ```
    === "CLI"

        ```bash
        yolo benchmark model=yolov8n.pt data='coco8.yaml' imgsz=640 half=False device=0
        ```

## 인수

`model`, `data`, `imgsz`, `half`, `device`, `verbose`와 같은 인수들은 사용자들이 벤치마킹을 특정 필요에 맞게 조정하고 쉽게 다른 내보내기 형식의 성능을 비교할 수 있도록 유연성을 제공합니다.

| 키         | 값       | 설명                                                       |
|-----------|---------|----------------------------------------------------------|
| `model`   | `None`  | 모델 파일 경로, 예: yolov8n.pt, yolov8n.yaml                    |
| `data`    | `None`  | 벤치마킹 데이터 세트를 참조하는 YAML 경로 ('val' 레이블 아래)                 |
| `imgsz`   | `640`   | 스칼라 또는 (h, w) 리스트 형태의 이미지 크기, 예: (640, 480)              |
| `half`    | `False` | FP16 양자화                                                 |
| `int8`    | `False` | INT8 양자화                                                 |
| `device`  | `None`  | 실행할 기기, 예: CUDA device=0 혹은 device=0,1,2,3 또는 device=cpu |
| `verbose` | `False` | 오류 시 계속하지 않음 (bool), 또는 val 하한 임계값 (float)               |

## 내보내기 형식

벤치마크는 아래에 나와있는 가능한 모든 내보내기 형식에서 자동으로 실행을 시도합니다.

| 형식                                                                 | `format` 인자   | 모델                        | 메타데이터 | 인수                                                  |
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

전체 `export` 세부 정보는 [Export](https://docs.ultralytics.com/modes/export/) 페이지에서 확인할 수 있습니다.
