---
comments: true
description: Ultralytics 공식 YOLOv8 문서입니다. 모델 훈련, 검증, 예측 및 다양한 형식으로 모델 내보내기 방법을 배우십시오. 세부적인 성능 통계를 포함합니다.
keywords: YOLOv8, Ultralytics, 객체 감지, 사전 훈련된 모델, 훈련, 검증, 예측, 모델 내보내기, COCO, ImageNet, PyTorch, ONNX, CoreML
---

# 객체 감지

<img width="1024" src="https://user-images.githubusercontent.com/26833433/243418624-5785cb93-74c9-4541-9179-d5c6782d491a.png" alt="객체 감지 예제">

객체 감지는 이미지 또는 비디오 스트림 내의 객체의 위치와 클래스를 식별하는 작업입니다.

객체 감지기의 출력은 이미지 속 객체를 내포하는 경계 상자(bounding box) 세트와 각 상자에 대한 클래스 레이블과 신뢰도 점수를 포함합니다. 장면 내 관심 객체를 식별해야 하지만 객체의 정확한 위치나 정확한 모양을 알 필요가 없을 때 객체 감지가 좋은 선택입니다.

<p align="center">
  <br>
  <iframe width="720" height="405" src="https://www.youtube.com/embed/5ku7npMrW40?si=6HQO1dDXunV8gekh"
    title="YouTube 비디오 플레이어" frameborder="0"
    allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share"
    allowfullscreen>
  </iframe>
  <br>
  <strong>시청하기:</strong> 사전 훈련된 Ultralytics YOLOv8 모델로 객체 감지하기.
</p>

!!! Tip "팁"

    YOLOv8 Detect 모델들은 기본 YOLOv8 모델이며 예를 들어 `yolov8n.pt` 이 [COCO](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/datasets/coco.yaml) 데이터셋에서 사전 훈련되었습니다.

## [모델](https://github.com/ultralytics/ultralytics/tree/main/ultralytics/cfg/models/v8)

여기서는 YOLOv8 사전 훈련된 Detect 모델을 나타냅니다. Detect, Segment, 및 Pose 모델은 [COCO](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/datasets/coco.yaml) 데이터셋에서, Classify 모델은 [ImageNet](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/datasets/ImageNet.yaml) 데이터셋에서 사전 훈련되었습니다.

[모델](https://github.com/ultralytics/ultralytics/tree/main/ultralytics/cfg/models)은 첫 사용 시 Ultralytics의 최신 [릴리즈](https://github.com/ultralytics/assets/releases)에서 자동으로 다운로드됩니다.

| 모델                                                                                   | 크기<br><sup>(픽셀) | mAP<sup>val<br>50-95 | 속도<br><sup>CPU ONNX<br>(ms) | 속도<br><sup>A100 TensorRT<br>(ms) | 파라미터<br><sup>(M) | FLOPs<br><sup>(B) |
|--------------------------------------------------------------------------------------|-----------------|----------------------|-----------------------------|----------------------------------|------------------|-------------------|
| [YOLOv8n](https://github.com/ultralytics/assets/releases/download/v8.1.0/yolov8n.pt) | 640             | 37.3                 | 80.4                        | 0.99                             | 3.2              | 8.7               |
| [YOLOv8s](https://github.com/ultralytics/assets/releases/download/v8.1.0/yolov8s.pt) | 640             | 44.9                 | 128.4                       | 1.20                             | 11.2             | 28.6              |
| [YOLOv8m](https://github.com/ultralytics/assets/releases/download/v8.1.0/yolov8m.pt) | 640             | 50.2                 | 234.7                       | 1.83                             | 25.9             | 78.9              |
| [YOLOv8l](https://github.com/ultralytics/assets/releases/download/v8.1.0/yolov8l.pt) | 640             | 52.9                 | 375.2                       | 2.39                             | 43.7             | 165.2             |
| [YOLOv8x](https://github.com/ultralytics/assets/releases/download/v8.1.0/yolov8x.pt) | 640             | 53.9                 | 479.1                       | 3.53                             | 68.2             | 257.8             |

- **mAP<sup>val</sup>** 값은 [COCO val2017](http://cocodataset.org) 데이터셋에서 단일 모델 단일 스케일을 사용한 값입니다.
  <br>[COCO](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/datasets/coco.yaml) 데이터와 `yolo val detect data=coco.yaml device=0` 명령으로 재현할 수 있습니다.
- **속도**는 [Amazon EC2 P4d](https://aws.amazon.com/ec2/instance-types/p4/) 인스턴스를 사용해 COCO val 이미지들을 평균한 것입니다.
  <br>[COCO128](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/datasets/coco128.yaml) 데이터와 `yolo val detect data=coco128.yaml batch=1 device=0|cpu` 명령으로 재현할 수 있습니다.

## 훈련

COCO128 데이터셋에서 이미지 크기 640으로 YOLOv8n 모델을 100 에포크 동안 훈련합니다. 가능한 모든 인수에 대한 목록은 [설정](/../usage/cfg.md) 페이지에서 확인할 수 있습니다.

!!! Example "예제"

    === "Python"

        ```python
        from ultralytics import YOLO

        # 모델 로드하기
        model = YOLO('yolov8n.yaml')  # YAML에서 새 모델을 빌드합니다.
        model = YOLO('yolov8n.pt')  # 사전 훈련된 모델을 로드합니다(훈련을 위해 권장됩니다).
        model = YOLO('yolov8n.yaml').load('yolov8n.pt')  # YAML에서 빌드하고 가중치를 전달합니다.

        # 모델 훈련하기
        results = model.train(data='coco128.yaml', epochs=100, imgsz=640)
        ```
    === "CLI"

        ```bash
        # YAML에서 새 모델을 빌드하고 처음부터 훈련을 시작합니다.
        yolo detect train data=coco128.yaml model=yolov8n.yaml epochs=100 imgsz=640

        # 사전 훈련된 *.pt 모델로부터 훈련을 시작합니다.
        yolo detect train data=coco128.yaml model=yolov8n.pt epochs=100 imgsz=640

        # YAML에서 새 모델을 빌드하고, 사전 훈련된 가중치를 전달한 후 훈련을 시작합니다.
        yolo detect train data=coco128.yaml model=yolov8n.yaml pretrained=yolov8n.pt epochs=100 imgsz=640
        ```

### 데이터셋 형식

YOLO 감지 데이터셋 형식은 [데이터셋 가이드](../../../datasets/detect/index.md)에서 자세히 볼 수 있습니다. 다른 형식(예: COCO 등)의 기존 데이터셋을 YOLO 형식으로 변환하려면 Ultralytics의 [JSON2YOLO](https://github.com/ultralytics/JSON2YOLO) 도구를 사용하십시오.

## 검증

COCO128 데이터셋에서 훈련된 YOLOv8n 모델의 정확도를 검증합니다. `model`은 훈련 시의 `data`와 인수를 모델 속성으로 보존하기 때문에 인수를 전달할 필요가 없습니다.

!!! Example "예제"

    === "Python"

        ```python
        from ultralytics import YOLO

        # 모델 로드하기
        model = YOLO('yolov8n.pt')  # 공식 모델을 로드합니다.
        model = YOLO('path/to/best.pt')  # 사용자 정의 모델을 로드합니다.

        # 모델 검증하기
        metrics = model.val()  # 데이터셋과 설정을 기억하니 인수는 필요 없습니다.
        metrics.box.map    # map50-95
        metrics.box.map50  # map50
        metrics.box.map75  # map75
        metrics.box.maps   # 각 카테고리의 map50-95가 포함된 리스트입니다.
        ```
    === "CLI"

        ```bash
        yolo detect val model=yolov8n.pt  # 공식 모델 검증하기
        yolo detect val model=path/to/best.pt  # 사용자 정의 모델 검증하기
        ```

## 예측

훈련된 YOLOv8n 모델을 사용하여 이미지에 대한 예측을 수행합니다.

!!! Example "예제"

    === "Python"

        ```python
        from ultralytics import YOLO

        # 모델 로드하기
        model = YOLO('yolov8n.pt')  # 공식 모델을 로드합니다.
        model = YOLO('path/to/best.pt')  # 사용자 정의 모델을 로드합니다.

        # 모델로 예측하기
        results = model('https://ultralytics.com/images/bus.jpg')  # 이미지에 대해 예측합니다.
        ```
    === "CLI"

        ```bash
        yolo detect predict model=yolov8n.pt source='https://ultralytics.com/images/bus.jpg'  # 공식 모델로 예측하기
        yolo detect predict model=path/to/best.pt source='https://ultralytics.com/images/bus.jpg'  # 사용자 정의 모델로 예측하기
        ```

전체 'predict' 모드 세부 사항은 [Predict](https://docs.ultralytics.com/modes/predict/) 페이지에서 확인하세요.

## 내보내기

YOLOv8n 모델을 ONNX, CoreML 등과 같은 다른 형식으로 내보냅니다.

!!! Example "예제"

    === "Python"

        ```python
        from ultralytics import YOLO

        # 모델 로드하기
        model = YOLO('yolov8n.pt')  # 공식 모델을 로드합니다.
        model = YOLO('path/to/best.pt')  # 사용자 정의 모델을 로드합니다.

        # 모델 내보내기
        model.export(format='onnx')
        ```
    === "CLI"

        ```bash
        yolo export model=yolov8n.pt format=onnx  # 공식 모델 내보내기
        yolo export model=path/to/best.pt format=onnx  # 사용자 정의 모델 내보내기
        ```

사용 가능한 YOLOv8 내보내기 형식은 아래 표에 나와 있습니다. 내보내기 완료 후 사용 예시는 모델에 대해 보여줍니다.

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

전체 'export' 세부 사항은 [Export](https://docs.ultralytics.com/modes/export/) 페이지에서 확인하세요.
