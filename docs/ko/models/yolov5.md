---
comments: true
description: YOLOv5u는 YOLOv5 모델의 개선된 정확도-속도 절충 모델로, 다양한 객체 감지 작업에 대한 사전 훈련된 모델을 제공합니다.
keywords: YOLOv5u, 객체 감지, 사전 훈련된 모델, Ultralytics, 추론, 검증, YOLOv5, YOLOv8, 앵커 없음, 객체 여부 없음, 실시간 응용, 머신 러닝
---

# YOLOv5

## 개요

YOLOv5u는 객체 감지 기법에서의 진보를 나타냅니다. Ultralytics에서 개발한 [YOLOv5](https://github.com/ultralytics/yolov5) 모델의 기본 아키텍처를 기반으로 한 YOLOv5u는 [YOLOv8](yolov8.md) 모델에서 도입된 앵커 없음, 객체 여부 없음 분리 헤드(head) 기능을 통합합니다. 이러한 적응으로 인해 모델의 아키텍처가 개선되어, 객체 감지 작업의 정확도와 속도 절충을 더욱 향상시킵니다. 경험적 결과와 해당 기능을 고려할 때, YOLOv5u는 연구 및 실제 응용 모두에서 견고한 솔루션을 찾고 있는 사용자들에게 효율적인 대안을 제공합니다.

![Ultralytics YOLOv5](https://raw.githubusercontent.com/ultralytics/assets/main/yolov5/v70/splash.png)

## 주요 기능

- **앵커 없는 분리 Ultralytics 헤드:** 기존의 객체 감지 모델은 사전 정의된 앵커 박스를 사용하여 객체의 위치를 예측합니다. 그러나 YOLOv5u는 이 방식을 현대화합니다. 앵커 없는 분리 Ultralytics 헤드를 도입함으로써 더욱 유연하고 적응적인 감지 메커니즘을 보장하여 다양한 시나리오에서 성능을 향상시킵니다.

- **정확도-속도 절충의 최적화:** 속도와 정확도는 종종 상충하는 관계에 있습니다. 그러나 YOLOv5u는 이러한 절충을 도전합니다. 실시간 탐지를 보장하면서도 정확도를 희생하지 않는 균형을 제시합니다. 이 기능은 자율주행 차량, 로봇 공학, 실시간 비디오 분석 등 신속한 응답을 요구하는 응용 프로그램에서 특히 중요합니다.

- **다양한 사전 훈련된 모델:** 다른 작업에는 다른 도구 세트가 필요하다는 것을 이해하는 YOLOv5u는 다양한 사전 훈련된 모델을 제공합니다. 추론, 검증 또는 훈련에 집중하고 있는지 여부에 관계없이 맞춤형 모델이 기다리고 있습니다. 이 다양성은 일반적인 솔루션이 아닌 독특한 도전 과제에 대해 특별히 세밀하게 조정된 모델을 사용하고 있다는 것을 보장합니다.

## 지원되는 작업 및 모드

разнобойacionales of YOLOv5u 모델은 다양한 사전 훈련된 가중치로 [객체 감지](../tasks/detect.md) 작업에서 뛰어난 성능을 발휘합니다. 이들은 개발부터 배포까지 다양한 응용 프로그램에 적합한 다양한 모드를 지원합니다.

| 모델 유형   | 사전 훈련된 가중치                                                                                                                  | 작업                          | 추론 | 검증 | 훈련 | 내보내기 |
|---------|-----------------------------------------------------------------------------------------------------------------------------|-----------------------------|----|----|----|------|
| YOLOv5u | `yolov5nu`, `yolov5su`, `yolov5mu`, `yolov5lu`, `yolov5xu`, `yolov5n6u`, `yolov5s6u`, `yolov5m6u`, `yolov5l6u`, `yolov5x6u` | [객체 감지](../tasks/detect.md) | ✅  | ✅  | ✅  | ✅    |

이 표는 YOLOv5u 모델의 다양한 변형을 상세히 보여주며, 객체 감지 작업에서의 적용 가능성과 [추론](../modes/predict.md), [검증](../modes/val.md), [훈련](../modes/train.md), [내보내기](../modes/export.md)와 같은 다양한 작업 모드의 지원을 강조합니다. 이러한 포괄적인 지원을 통해 사용자는 다양한 객체 감지 시나리오에서 YOLOv5u 모델의 기능을 완전히 활용할 수 있습니다.

## 성능 지표

!!! 성능

    === "감지"

    [COCO](https://docs.ultralytics.com/datasets/detect/coco/)에서 학습된 이러한 모델을 사용한 사용 예제는 [감지 문서](https://docs.ultralytics.com/tasks/detect/)를 참조하세요. 이 문서에는 80개의 사전 훈련된 클래스를 포함합니다.

    | 모델                                                                                       | YAML                                                                                                           | 크기<br><sup>(픽셀) | mAP<sup>val<br>50-95 | 속도<br><sup>CPU ONNX<br>(ms) | 속도<br><sup>A100 TensorRT<br>(ms) | 매개변수<br><sup>(M) | FLOPs<br><sup>(B) |
    |---------------------------------------------------------------------------------------------|----------------------------------------------------------------------------------------------------------------|-----------------------|----------------------|--------------------------------|-------------------------------------|--------------------|-------------------|
    | [yolov5nu.pt](https://github.com/ultralytics/assets/releases/download/v8.1.0/yolov5nu.pt)   | [yolov5n.yaml](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/models/v5/yolov5.yaml)     | 640                   | 34.3                 | 73.6                           | 1.06                                | 2.6                | 7.7               |
    | [yolov5su.pt](https://github.com/ultralytics/assets/releases/download/v8.1.0/yolov5su.pt)   | [yolov5s.yaml](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/models/v5/yolov5.yaml)     | 640                   | 43.0                 | 120.7                          | 1.27                                | 9.1                | 24.0              |
    | [yolov5mu.pt](https://github.com/ultralytics/assets/releases/download/v8.1.0/yolov5mu.pt)   | [yolov5m.yaml](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/models/v5/yolov5.yaml)     | 640                   | 49.0                 | 233.9                          | 1.86                                | 25.1               | 64.2              |
    | [yolov5lu.pt](https://github.com/ultralytics/assets/releases/download/v8.1.0/yolov5lu.pt)   | [yolov5l.yaml](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/models/v5/yolov5.yaml)     | 640                   | 52.2                 | 408.4                          | 2.50                                | 53.2               | 135.0             |
    | [yolov5xu.pt](https://github.com/ultralytics/assets/releases/download/v8.1.0/yolov5xu.pt)   | [yolov5x.yaml](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/models/v5/yolov5.yaml)     | 640                   | 53.2                 | 763.2                          | 3.81                                | 97.2               | 246.4             |
    |                                                                                             |                                                                                                                |                       |                      |                                |                                     |                    |                   |
    | [yolov5n6u.pt](https://github.com/ultralytics/assets/releases/download/v8.1.0/yolov5n6u.pt) | [yolov5n6.yaml](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/models/v5/yolov5-p6.yaml) | 1280                  | 42.1                 | 211.0                          | 1.83                                | 4.3                | 7.8               |
    | [yolov5s6u.pt](https://github.com/ultralytics/assets/releases/download/v8.1.0/yolov5s6u.pt) | [yolov5s6.yaml](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/models/v5/yolov5-p6.yaml) | 1280                  | 48.6                 | 422.6                          | 2.34                                | 15.3               | 24.6              |
    | [yolov5m6u.pt](https://github.com/ultralytics/assets/releases/download/v8.1.0/yolov5m6u.pt) | [yolov5m6.yaml](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/models/v5/yolov5-p6.yaml) | 1280                  | 53.6                 | 810.9                          | 4.36                                | 41.2               | 65.7              |
    | [yolov5l6u.pt](https://github.com/ultralytics/assets/releases/download/v8.1.0/yolov5l6u.pt) | [yolov5l6.yaml](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/models/v5/yolov5-p6.yaml) | 1280                  | 55.7                 | 1470.9                         | 5.47                                | 86.1               | 137.4             |
    | [yolov5x6u.pt](https://github.com/ultralytics/assets/releases/download/v8.1.0/yolov5x6u.pt) | [yolov5x6.yaml](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/models/v5/yolov5-p6.yaml) | 1280                  | 56.8                 | 2436.5                         | 8.98                                | 155.4              | 250.7             |

## 사용 예제

이 예제는 간단한 YOLOv5 훈련 및 추론 예제를 제공합니다. 이와 기타 [모드](../modes/index.md)의 자세한 설명은 [Predict](../modes/predict.md), [Train](../modes/train.md), [Val](../modes/val.md) 및 [Export](../modes/export.md) 문서 페이지를 참조하세요.

!!! Example "예제"

    === "Python"

        Python에서 `YOLO()` 클래스로 `*.pt` 사전 훈련된 모델과 구성 `*.yaml` 파일을 전달하여 모델 인스턴스를 만들 수 있습니다.

        ```python
        from ultralytics import YOLO

        # COCO 사전 훈련된 YOLOv5n 모델 로드
        model = YOLO('yolov5n.pt')

        # 모델 정보 표시 (선택 사항)
        model.info()

        # COCO8 예제 데이터셋을 사용하여 모델을 100번 에포크로 훈련
        results = model.train(data='coco8.yaml', epochs=100, imgsz=640)

        # 'bus.jpg' 이미지에 대해 YOLOv5n 모델로 추론 실행
        results = model('path/to/bus.jpg')
        ```

    === "CLI"

        CLI 명령을 사용하여 모델을 직접 실행할 수 있습니다.

        ```bash
        # COCO 사전 훈련된 YOLOv5n 모델 로드 및 COCO8 예제 데이터셋을 사용하여 모델을 100번 에포크로 훈련
        yolo train model=yolov5n.pt data=coco8.yaml epochs=100 imgsz=640

        # COCO 사전 훈련된 YOLOv5n 모델 로드 및 'bus.jpg' 이미지에서 추론 실행
        yolo predict model=yolov5n.pt source=path/to/bus.jpg
        ```

## 인용 및 감사의 글

연구에서 YOLOv5 또는 YOLOv5u를 사용하는 경우 Ultralytics YOLOv5 리포지토리를 다음과 같이 인용하세요.

!!! Quote ""

    === "BibTeX"
        ```bibtex
        @software{yolov5,
          title = {Ultralytics YOLOv5},
          author = {Glenn Jocher},
          year = {2020},
          version = {7.0},
          license = {AGPL-3.0},
          url = {https://github.com/ultralytics/yolov5},
          doi = {10.5281/zenodo.3908559},
          orcid = {0000-0001-5950-6979}
        }
        ```

YOLOv5 모델은 [AGPL-3.0](https://github.com/ultralytics/ultralytics/blob/main/LICENSE) 및 [Enterprise](https://ultralytics.com/license) 라이선스로 제공됩니다.
