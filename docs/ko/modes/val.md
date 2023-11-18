---
comments: true
description: YOLOv8 모델 검증 가이드. 검증 설정 및 측정 항목을 사용하여 YOLO 모델의 성능을 평가하는 방법에 대해 알아보세요. Python 및 CLI 예제가 포함되어 있습니다.
keywords: Ultralytics, YOLO 문서, YOLOv8, 검증, 모델 평가, 하이퍼파라미터, 정확도, 측정 항목, Python, CLI
---

# Ultralytics YOLO로 모델 검증하기

<img width="1024" src="https://github.com/ultralytics/assets/raw/main/yolov8/banner-integrations.png" alt="Ultralytics YOLO 생태계 및 통합">

## 도입

검증은 훈련된 모델의 품질을 평가할 수 있게 해주는 기계학습 파이프라인에서 중요한 단계입니다. Ultralytics YOLOv8의 Val 모드는 모델의 객체 탐지 성능을 평가하기 위한 강력한 도구 및 측정 항목 모음을 제공합니다. 이 가이드는 Val 모드를 효과적으로 사용하여 모델의 정확성과 신뢰성을 보장하는 방법에 대한 완벽한 리소스 역할을 합니다.

## 왜 Ultralytics YOLO로 검증을 해야 할까요?

YOLOv8의 Val 모드를 사용하는 이점은 다음과 같습니다:

- **정밀도:** mAP50, mAP75, mAP50-95와 같은 정확한 측정 항목으로 모델을 종합적으로 평가합니다.
- **편의성:** 훈련 설정을 기억하는 내장 기능을 활용하여 검증 절차를 단순화합니다.
- **유연성:** 같거나 다른 데이터셋과 이미지 크기로 모델을 검증할 수 있습니다.
- **하이퍼파라미터 튜닝:** 검증 측정 항목을 사용하여 모델의 성능을 더 잘 조율합니다.

### Val 모드의 주요 기능

YOLOv8의 Val 모드가 제공하는 주목할 만한 기능들은 다음과 같습니다:

- **자동화된 설정:** 모델은 훈련 구성을 기억하여 간단하게 검증이 가능합니다.
- **멀티-메트릭 지원:** 다양한 정확도 측정 항목을 기반으로 모델을 평가합니다.
- **CLI 및 Python API:** 검증을 위해 명령 줄 인터페이스 또는 Python API 중에서 선택할 수 있습니다.
- **데이터 호환성:** 훈련 단계에서 사용된 데이터셋과 사용자 정의 데이터셋 모두와 원활하게 작동합니다.

!!! Tip "팁"

    * YOLOv8 모델은 훈련 설정을 자동으로 기억하므로 `yolo val model=yolov8n.pt`나 `model('yolov8n.pt').val()`만으로 같은 이미지 크기와 원본 데이터셋에서 쉽게 검증할 수 있습니다.

## 사용 예제

COCO128 데이터셋에서 훈련된 YOLOv8n 모델의 정확도를 검증합니다. `모델`은 훈련 `데이터`와 인자를 모델 속성으로 유지하므로 인자가 필요 없습니다. 전체 내보내기 인자 목록은 아래의 인자 섹션을 참고하세요.

!!! Example "예제"

    === "Python"

        ```python
        from ultralytics import YOLO

        # 모델 로드
        model = YOLO('yolov8n.pt')  # 공식 모델을 로드합니다
        model = YOLO('path/to/best.pt')  # 사용자 정의 모델을 로드합니다

        # 모델 검증
        metrics = model.val()  # 인자가 필요 없음, 데이터셋과 설정이 기억됩니다
        metrics.box.map    # map50-95
        metrics.box.map50  # map50
        metrics.box.map75  # map75
        metrics.box.maps   # 각 카테고리의 map50-95가 포함된 목록
        ```
    === "CLI"

        ```bash
        yolo detect val model=yolov8n.pt  # 공식 모델 검증
        yolo detect val model=path/to/best.pt  # 사용자 정의 모델 검증
        ```

## 인자

YOLO 모델의 검증 설정은 모델의 성능을 검증 데이터셋에서 평가하기 위한 다양한 하이퍼파라미터 및 구성을 의미합니다. 이러한 설정은 모델의 성능, 속도, 정확성에 영향을 미칠 수 있습니다. 일반적인 YOLO 검증 설정에는 배치 크기, 훈련 중 검증이 수행되는 빈도 및 모델 성능을 평가하기 위해 사용되는 측정 항목이 포함됩니다. 검증 과정에 영향을 줄 수 있는 다른 요소로는 검증 데이터셋의 크기와 구성 및 모델이 사용되는 구체적인 작업이 있습니다. 모델이 검증 데이터셋에서 잘 수행되고 있고 과적합을 감지하고 방지하기 위해서는 이러한 설정을 신중하게 조정하고 실험하는 것이 중요합니다.

| Key           | Value   | Description                                       |
|---------------|---------|---------------------------------------------------|
| `data`        | `None`  | 데이터 파일 경로 예: coco128.yaml                         |
| `imgsz`       | `640`   | 입력 이미지의 크기를 정수로 지정                                |
| `batch`       | `16`    | 배치 당 이미지 수 (-1은 AutoBatch에 해당)                    |
| `save_json`   | `False` | 결과를 JSON 파일로 저장                                   |
| `save_hybrid` | `False` | 라벨의 하이브리드 버전(라벨 + 추가 예측)을 저장                      |
| `conf`        | `0.001` | 탐지를 위한 객체 신뢰도 임계값                                 |
| `iou`         | `0.6`   | NMS 용 교차 영역과 합친 영역(IoU)의 임계값                      |
| `max_det`     | `300`   | 이미지 당 최대 탐지 개수                                    |
| `half`        | `True`  | 반정밀도(FP16) 사용                                     |
| `device`      | `None`  | 사용할 장치 예: cuda의 device=0/1/2/3이나 device=cpu       |
| `dnn`         | `False` | ONNX 추론에 OpenCV DNN 사용                            |
| `plots`       | `False` | 훈련 중 플롯 표시                                        |
| `rect`        | `False` | 최소한의 패딩을 위해 각 배치가 직사각형 val로 조정됨                   |
| `split`       | `val`   | 검증을 위해 사용되는 데이터셋 분할, 예: 'val', 'test', 혹은 'train' |
|
