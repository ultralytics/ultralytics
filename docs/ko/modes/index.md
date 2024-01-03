---
comments: true
description: YOLOv8를 통해 트레이닝부터 추적까지, Ultralytics에 최적화된 모든 기능을 활용하세요. 지원되는 각 모드에 대한 통찰력과 예시를 포함하여 검증, 내보내기, 벤치마킹까지 이해하실 수 있습니다.
keywords: Ultralytics, YOLOv8, 머신러닝, 객체탐지, 트레이닝, 검증, 예측, 내보내기, 추적, 벤치마킹
---

# Ultralytics YOLOv8 모드

<img width="1024" src="https://github.com/ultralytics/assets/raw/main/yolov8/banner-integrations.png" alt="Ultralytics YOLO 생태계 및 통합">

## 서론

Ultralytics YOLOv8는 단순한 객체 탐지 모델이 아닙니다; 데이터 수집에서 모델 트레이닝, 검증, 배포, 실세계 추적에 이르기까지 머신러닝 모델의 전체 생애주기를 커버하기 위해 설계된 다재다능한 프레임워크입니다. 각각의 모드는 특정 목적을 위해 섬세하게 구성되며, 다양한 작업 및 사용 사례에 필요한 유연성과 효율성을 제공합니다.

<p align="center">
  <br>
  <iframe width="720" height="405" src="https://www.youtube.com/embed/j8uQc0qB91s?si=dhnGKgqvs7nPgeaM"
    title="YouTube video player" frameborder="0"
    allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share"
    allowfullscreen>
  </iframe>
  <br>
  <strong>시청하기:</strong> Ultralytics 모드 튜토리얼: 트레이닝, 검증, 예측, 내보내기 및 벤치마킹.
</p>

### 모드 요약

YOLOv8이 지원하는 **모드**를 이해하는 것은 모델을 최대한 활용하기 위해 필수적입니다:

- **Train** 모드: 사용자 맞춤 또는 사전 로드된 데이터셋 위에서 모델을 튜닝합니다.
- **Val** 모드: 트레이닝 후 모델 성능을 검증하기 위한 체크포인트.
- **Predict** 모드: 실세계 데이터에서 모델의 예측력을 발휘합니다.
- **Export** 모드: 다양한 포맷으로 모델을 배포 준비 상태로 만듭니다.
- **Track** 모드: 객체 탐지 모델을 실시간 추적 애플리케이션으로 확장합니다.
- **Benchmark** 모드: 다양한 배포 환경에서 모델의 속도와 정확도를 분석합니다.

이 포괄적인 가이드는 각 모드에 대한 개요와 실제 인사이트를 제공하여 YOLOv8의 전체 잠재력을 활용할 수 있도록 도와줍니다.

## [Train](train.md)

Train 모드는 사용자 맞춤 데이터셋 위에서 YOLOv8 모델을 트레이닝하기 위해 사용됩니다. 이 모드에서는 지정된 데이터셋과 하이퍼파라미터를 사용하여 모델을 트레이닝합니다. 트레이닝 과정에서 모델의 파라미터를 최적화하여 이미지 내 객체의 클래스와 위치를 정확히 예측할 수 있도록 합니다.

[Train 예시](train.md){ .md-button }

## [Val](val.md)

Val 모드는 트레이닝된 YOLOv8 모델을 검증하기 위해 사용됩니다. 이 모드에서는 모델을 검증 세트에서 평가하여 정확도 및 일반화 성능을 측정합니다. 이 모드는 모델의 하이퍼파라미터를 조정하고 성능을 개선하는데 사용할 수 있습니다.

[Val 예시](val.md){ .md-button }

## [Predict](predict.md)

Predict 모드는 트레이닝된 YOLOv8 모델을 사용하여 새 이미지 또는 비디오에서 예측을 수행하기 위해 사용됩니다. 이 모드에서는 체크포인트 파일에서 모델을 로드하고, 사용자가 이미지나 비디오를 제공하여 추론을 수행합니다. 모델은 입력 이미지 또는 비디오에서 객체의 클래스와 위치를 예측합니다.

[Predict 예시](predict.md){ .md-button }

## [Export](export.md)

Export 모드는 배포를 위해 YOLOv8 모델을 내보낼 수 있는 포맷으로 변환하기 위해 사용됩니다. 이 모드에서는 모델을 다른 소프트웨어 어플리케이션 또는 하드웨어 기기에서 사용할 수 있는 포맷으로 변환합니다. 이 모드는 모델을 생산 환경으로 배포하는데 유용합니다.

[Export 예시](export.md){ .md-button }

## [Track](track.md)

Track 모드는 실시간으로 YOLOv8 모델을 사용하여 객체를 추적하기 위해 사용됩니다. 이 모드에서는 체크포인트 파일에서 모델을 로드하고, 사용자가 실시간 비디오 스트림을 제공하여 실시간 객체 추적을 수행합니다. 이 모드는 감시 시스템이나 자율 주행 차량 같은 애플리케이션에 유용합니다.

[Track 예시](track.md){ .md-button }

## [Benchmark](benchmark.md)

Benchmark 모드는 YOLOv8의 다양한 내보내기 포맷에 대한 속도와 정확도를 프로파일링하기 위해 사용됩니다. 벤치마크는 내보낸 포맷의 크기, 그리고 객체 탐지, 세분화 및 포즈에 대한 `mAP50-95` 메트릭 또는 분류에 대한 `accuracy_top5` 메트릭, 그리고 ONNX, OpenVINO, TensorRT 등 다양한 내보내기 포맷에서의 이미지당 추론 시간을 밀리초로 제공합니다. 이 정보는 속도와 정확도에 대한 특정 사용 사례 요구 사항에 기반하여 최적의 내보내기 포맷을 선택하는 데 도움이 될 수 있습니다.

[Benchmark 예시](benchmark.md){ .md-button }
