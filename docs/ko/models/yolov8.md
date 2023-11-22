---
comments: true
description: 최신 버전의 실시간 객체 탐지기인 YOLOv8의 흥미로운 기능을 살펴보세요! 고급 아키텍처, 사전 훈련된 모델, 정확도와 속도 사이의 최적의 균형 등에 대해 알아보면, 다양한 객체 탐지 작업에 YOLOv8이 완벽한 선택임을 알 수 있습니다.
keywords: YOLOv8, Ultralytics, 실시간 객체 탐지기, 사전 훈련된 모델, 문서, 객체 탐지, YOLO 시리즈, 고급 아키텍처, 정확도, 속도
---

# YOLOv8

## 개요

YOLOv8는 실시간 객체 탐지기인 YOLO 시리즈의 최신 버전으로, 정확성과 속도 측면에서 최첨단 성능을 제공합니다. 이전 YOLO 버전의 기술적 발전을 더욱 발전시킨 YOLOv8은 새로운 기능과 최적화를 도입하여 다양한 응용 분야에서 다양한 객체 탐지 작업에 이상적인 선택지가 됩니다.

![Ultralytics YOLOv8](https://raw.githubusercontent.com/ultralytics/assets/main/yolov8/yolo-comparison-plots.png)

## 주요 기능

- **고급 백본 및 네크 아키텍처**: YOLOv8은 최첨단의 백본 및 네크 아키텍처를 사용하여 향상된 특징 추출 및 객체 탐지 성능을 제공합니다.
- **앵커 없는 분할 Ultralytics 헤드**: YOLOv8은 앵커를 사용하는 방법보다 정확성과 효율성이 뛰어난 앵커 없는 분할 Ultralytics 헤드를 채택합니다.
- **최적의 정확도-속도 균형**: 정확도와 속도 사이에 최적의 균형을 유지하기 위한 YOLOv8은 다양한 응용 분야에서 실시간 객체 탐지 작업에 맞습니다.
- **다양한 사전 훈련된 모델**: YOLOv8은 다양한 작업과 성능 요구 사항에 대한 사전 훈련된 모델을 제공하여 특정 사용 사례에 적합한 모델을 쉽게 찾을 수 있습니다.

## 지원되는 작업

| 모델 유형       | 사전 훈련된 가중치                                                                                                          | 작업      |
|-------------|---------------------------------------------------------------------------------------------------------------------|---------|
| YOLOv8      | `yolov8n.pt`, `yolov8s.pt`, `yolov8m.pt`, `yolov8l.pt`, `yolov8x.pt`                                                | 탐지      |
| YOLOv8-seg  | `yolov8n-seg.pt`, `yolov8s-seg.pt`, `yolov8m-seg.pt`, `yolov8l-seg.pt`, `yolov8x-seg.pt`                            | 인스턴스 분할 |
| YOLOv8-pose | `yolov8n-pose.pt`, `yolov8s-pose.pt`, `yolov8m-pose.pt`, `yolov8l-pose.pt`, `yolov8x-pose.pt`, `yolov8x-pose-p6.pt` | 포즈/키포인트 |
| YOLOv8-cls  | `yolov8n-cls.pt`, `yolov8s-cls.pt`, `yolov8m-cls.pt`, `yolov8l-cls.pt`, `yolov8x-cls.pt`                            | 분류      |

## 지원되는 모드

| 모드 | 지원 |
|----|----|
| 추론 | ✅  |
| 검증 | ✅  |
| 훈련 | ✅  |

!!! 성능

    === "탐지 (COCO)"

        | 모델                                                                                | 크기<br><sup>(픽셀) | mAP<sup>val<br>50-95 | 속도<br><sup>CPU ONNX<br>(밀리초) | 속도<br><sup>A100 TensorRT<br>(밀리초) | 파라미터<br><sup>(백만 개) | FLOPs<br><sup>(십억 개) |
        | ------------------------------------------------------------------------------------ | --------------------- | -------------------- | ------------------------------ | ----------------------------------- | ------------------ | ----------------- |
        | [YOLOv8n](https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n.pt) | 640                   | 37.3                 | 80.4                           | 0.99                                | 3.2                | 8.7               |
        | [YOLOv8s](https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8s.pt) | 640                   | 44.9                 | 128.4                          | 1.20                                | 11.2               | 28.6              |
        | [YOLOv8m](https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8m.pt) | 640                   | 50.2                 | 234.7                          | 1.83                                | 25.9               | 78.9              |
        | [YOLOv8l](https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8l.pt) | 640                   | 52.9                 | 375.2                          | 2.39                                | 43.7               | 165.2             |
        | [YOLOv8x](https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8x.pt) | 640                   | 53.9                 | 479.1                          | 3.53                                | 68.2               | 257.8             |

    === "탐지 (Open Images V7)"

        [Open Image V7](https://docs.ultralytics.com/datasets/detect/open-images-v7/)에서 훈련된 이러한 모델을 사용한 사용 예는 [Detection Docs](https://docs.ultralytics.com/tasks/detect/)를 참조하세요. 이 예는 600개의 사전 훈련된 클래스를 포함합니다.

        | 모델                                                                                     | 크기<br><sup>(픽셀) | mAP<sup>val<br>50-95 | 속도<br><sup>CPU ONNX<br>(밀리초) | A100 TensorRT<br>(ms) | 파라미터<br><sup>(백만 개) | FLOPs<br><sup>(십억 개) |
        | ----------------------------------------------------------------------------------------- | --------------------- | -------------------- | ------------------------------ | ----------------------------------- | ------------------ | ----------------- |
        | [YOLOv8n](https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n-oiv7.pt) | 640                   | 18.4                 | 142.4                          | 1.21                                | 3.5                | 10.5              |
        | [YOLOv8s](https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8s-oiv7.pt) | 640                   | 27.7                 | 183.1                          | 1.40                                | 11.4               | 29.7              |
        | [YOLOv8m](https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8m-oiv7.pt) | 640                   | 33.6                 | 408.5                          | 2.26                                | 26.2               | 80.6              |
        | [YOLOv8l](https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8l-oiv7.pt) | 640                   | 34.9                 | 596.9                          | 2.43                                | 44.1               | 167.4             |
        | [YOLOv8x](https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8x-oiv7.pt) | 640                   | 36.3                 | 860.6                          | 3.56                                | 68.7               | 260.6             |

    === "분할 (COCO)"

        | 모델                                                                                        | 크기<br><sup>(픽셀) | mAP<sup>box<br>50-95 | mAP<sup>mask<br>50-95 | CPU ONNX<br>(밀리초) | A100 TensorRT<br>(밀리초) | 파라미터<br><sup>(백만 개) | FLOPs<br><sup>(십억 개) |
        | -------------------------------------------------------------------------------------------- | --------------------- | -------------------- | --------------------- | ------------------------------ | ----------------------------------- | ------------------ | ----------------- |
        | [YOLOv8n-seg](https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n-seg.pt) | 640                   | 36.7                 | 30.5                  | 96.1                           | 1.21                                | 3.4                | 12.6              |
        | [YOLOv8s-seg](https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8s-seg.pt) | 640                   | 44.6                 | 36.8                  | 155.7                          | 1.47                                | 11.8               | 42.6              |
        | [YOLOv8m-seg](https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8m-seg.pt) | 640                   | 49.9                 | 40.8                  | 317.0                          | 2.18                                | 27.3               | 110.2             |
        | [YOLOv8l-seg](https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8l-seg.pt) | 640                   | 52.3                 | 42.6                  | 572.4                          | 2.79                                | 46.0               | 220.5             |
        | [YOLOv8x-seg](https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8x-seg.pt) | 640                   | 53.4                 | 43.4                  | 712.1                          | 4.02                                | 71.8               | 344.1             |

    === "분류 (ImageNet)"

        | 모델                                                                                        | 크기<br><sup>(픽셀) | 정확도<br><sup>상위1 | 정확도<br><sup>상위5 | CPU ONNX<br>(밀리초) | A100 TensorRT<br>(밀리초) | 파라미터<br><sup>(백만 개) | FLOPs<br><sup>(백만 개) |
        | -------------------------------------------------------------------------------------------- | --------------------- | ---------------- | ---------------- | ------------------------------ | ----------------------------------- | ------------------ | ------------------------ |
        | [YOLOv8n-cls](https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n-cls.pt) | 224                   | 66.6             | 87.0             | 12.9                           | 0.31                                | 2.7                | 4.3                      |
        | [YOLOv8s-cls](https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8s-cls.pt) | 224                   | 72.3             | 91.1             | 23.4                           | 0.35                                | 6.4                | 13.5                     |
        | [YOLOv8m-cls](https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8m-cls.pt) | 224                   | 76.4             | 93.2             | 85.4                           | 0.62                                | 17.0               | 42.7                     |
        | [YOLOv8l-cls](https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8l-cls.pt) | 224                   | 78.0             | 94.1             | 163.0                          | 0.87                                | 37.5               | 99.7                     |
        | [YOLOv8x-cls](https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8x-cls.pt) | 224                   | 78.4             | 94.3             | 232.0                          | 1.01                                | 57.4               | 154.8                    |

    === "포즈 (COCO)"

        | 모델                                                                                                | 크기<br><sup>(픽셀) | mAP<sup>pose<br>50-95 | mAP<sup>pose<br>50 | CPU ONNX<br>(밀리초) | A100 TensorRT<br>(밀리초) | 파라미터<br><sup>(백만 개) | FLOPs<br><sup>(십억 개) |
        | ---------------------------------------------------------------------------------------------------- | --------------------- | --------------------- | ------------------ | ------------------------------ | ----------------------------------- | ------------------ | ----------------- |
        | [YOLOv8n-pose](https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n-pose.pt)       | 640                   | 50.4                  | 80.1               | 131.8                          | 1.18                                | 3.3                | 9.2               |
        | [YOLOv8s-pose](https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8s-pose.pt)       | 640                   | 60.0                  | 86.2               | 233.2                          | 1.42                                | 11.6               | 30.2              |
        | [YOLOv8m-pose](https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8m-pose.pt)       | 640                   | 65.0                  | 88.8               | 456.3                          | 2.00                                | 26.4               | 81.0              |
        | [YOLOv8l-pose](https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8l-pose.pt)       | 640                   | 67.6                  | 90.0               | 784.5                          | 2.59                                | 44.4               | 168.6             |
        | [YOLOv8x-pose](https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8x-pose.pt)       | 640                   | 69.2                  | 90.2               | 1607.1                         | 3.73                                | 69.4               | 263.2             |
        | [YOLOv8x-pose-p6](https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8x-pose-p6.pt) | 1280                  | 71.6                  | 91.2               | 4088.7                         | 10.04                               | 99.1               | 1066.4            |

## 사용 방법

실시간 객체 탐지 작업에 YOLOv8을 사용하려면 Ultralytics 파이썬 패키지를 사용할 수 있습니다. 다음은 YOLOv8 모델을 인퍼런스하기위한 예시 코드 조각입니다.

!!! 예시 ""

    이 예시는 YOLOv8을 위한 간단한 인퍼런스 코드를 제공합니다. 결과 처리를 포함한 더 많은 옵션에 대해서는 [Predict](../modes/predict.md) 모드를 참조하세요. 다른 모드를 사용하여 YOLOv8을 사용하는 방법은 [Train](../modes/train.md), [Val](../modes/val.md) 및 [Export](../modes/export.md)를 참조하세요.

    === "파이썬"

        PyTorch 사전 훈련 `*.pt` 모델 및 구성 `*.yaml` 파일을 python에서 `YOLO()` 클래스에 전달하여 모델 인스턴스를 생성할 수 있습니다:

        ```python
        from ultralytics import YOLO

        # COCO 사전 훈련 YOLOv8n 모델 로드
        model = YOLO('yolov8n.pt')

        # 모델 정보 표시 (선택 사항)
        model.info()

        # COCO8 예제 데이터셋으로 모델 훈련 (100 epoch)
        results = model.train(data='coco8.yaml', epochs=100, imgsz=640)

        # 'bus.jpg' 이미지에서 YOLOv8n 모델로 인퍼런스 실행
        results = model('path/to/bus.jpg')
        ```

    === "CLI"

        CLI 명령을 사용하여 모델을 직접 실행할 수 있습니다:

        ```bash
        # COCO 사전 훈련 YOLOv8n 모델 로드하고 COCO8 예제 데이터셋에서 100 epoch 동안 훈련
        yolo train model=yolov8n.pt data=coco8.yaml epochs=100 imgsz=640

        # COCO 사전 훈련 YOLOv8n 모델 로드하고 'bus.jpg' 이미지에서 인퍼런스 실행
        yolo predict model=yolov8n.pt source=path/to/bus.jpg
        ```

## 인용 및 감사의 글

본 저장소의 YOLOv8 모델 또는 기타 소프트웨어를 작업에서 사용하신 경우, 다음과 같은 형식으로 인용해주시기 바랍니다:

!!! 노트 ""

    === "BibTeX"

        ```bibtex
        @software{yolov8_ultralytics,
          author = {Glenn Jocher and Ayush Chaurasia and Jing Qiu},
          title = {Ultralytics YOLOv8},
          version = {8.0.0},
          year = {2023},
          url = {https://github.com/ultralytics/ultralytics},
          orcid = {0000-0001-5950-6979, 0000-0002-7603-6750, 0000-0003-3783-7069},
          license = {AGPL-3.0}
        }
        ```

DOI는 아직 발표되지 않았으며, 이용은 AGPL-3.0 라이선스에 따릅니다.
