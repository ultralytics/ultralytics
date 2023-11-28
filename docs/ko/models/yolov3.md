---
comments: true
description: YOLOv3, YOLOv3-Ultralytics 및 YOLOv3u에 대한 개요를 얻으세요. 물체 탐지를 위한 주요 기능, 사용법 및 지원 작업에 대해 알아보세요.
keywords: YOLOv3, YOLOv3-Ultralytics, YOLOv3u, 물체 탐지, 추론, 훈련, Ultralytics
---

# YOLOv3, YOLOv3-Ultralytics 및 YOLOv3u

## 개요

이 문서는 세 가지 밀접하게 관련된 물체 탐지 모델인 [YOLOv3](https://pjreddie.com/darknet/yolo/), [YOLOv3-Ultralytics](https://github.com/ultralytics/yolov3) 및 [YOLOv3u](https://github.com/ultralytics/ultralytics)에 대한 개요를 제공합니다.

1. **YOLOv3:** 이것은 You Only Look Once (YOLO) 물체 탐지 알고리즘의 세 번째 버전입니다. Joseph Redmon이 처음 개발한 YOLOv3는 다중 스케일 예측 및 세 가지 다른 크기의 탐지 커널과 같은 기능을 도입하여 이전 모델보다 향상됐습니다.

2. **YOLOv3-Ultralytics:** 이것은 Ultralytics의 YOLOv3 모델 구현입니다. 이 모델은 원본 YOLOv3 아키텍처를 복제하며 더 많은 사전 훈련 모델 및 쉬운 사용자 정의 옵션과 같은 추가 기능을 제공합니다.

3. **YOLOv3u:** 이것은 YOLOv8 모델에서 사용되는 앵커 없이 물체 없음 분리 헤드를 통합한 YOLOv3-Ultralytics의 업데이트된 버전입니다. YOLOv3u는 YOLOv3와 동일한 백본 및 네크 아키텍처를 유지하지만 YOLOv8에서 업데이트된 탐지 헤드를 사용합니다.

![Ultralytics YOLOv3](https://raw.githubusercontent.com/ultralytics/assets/main/yolov3/banner-yolov3.png)

## 주요 기능

- **YOLOv3:** 이 모델은 탐지를 위해 13x13, 26x26 및 52x52의 세 가지 다른 크기의 탐지 커널을 활용하는 세 가지 다른 스케일을 도입했습니다. 이는 다양한 크기의 객체에 대한 탐지 정확도를 크게 향상시켰습니다. 또한 YOLOv3은 각 경계 상자에 대한 다중 레이블 예측과 더 나은 특징 추출기 네트워크와 같은 기능을 추가했습니다.

- **YOLOv3-Ultralytics:** Ultralytics의 YOLOv3 구현은 원본 모델과 동일한 성능을 제공하지만 더 많은 사전 훈련 모델, 추가적인 훈련 방법 및 쉬운 사용자 정의 옵션을 제공합니다. 이로써 실제 응용 분야에 대해 더 다양하고 사용자 친화적인 모델이 됩니다.

- **YOLOv3u:** 이 업데이트된 모델은 YOLOv8의 앵커 없음, 물체 없는 분리 헤드를 통합합니다. 미리 정의된 앵커 박스 및 물체 점수가 필요 없어진 이 탐지 헤드 설계는 다양한 크기와 모양의 객체를 탐지하는 능력을 향상시킬 수 있습니다. 이로써 YOLOv3u는 물체 탐지 작업에 대해 더 견고하고 정확한 모델이 됩니다.

## 지원되는 작업 및 모드

YOLOv3, YOLOv3-Ultralytics 및 YOLOv3u 시리즈는 물체 탐지 작업을 위해 특별히 설계되었습니다. 이러한 모델은 정확성과 속도를 균형있게 유지하여 다양한 실제 시나리오에서 효과적으로 사용될 수 있습니다. 각 버전은 독특한 기능과 최적화를 제공하여 다양한 응용 분야에 적합합니다.

세 가지 모델은 [추론](../modes/predict.md), [유효성 검사](../modes/val.md), [훈련](../modes/train.md) 및 [내보내기](../modes/export.md)와 같은 포괄적인 모드를 지원하여 효과적인 물체 탐지를 위한 완벽한 도구 세트를 제공합니다.

| 모델 유형              | 지원되는 작업                     | 추론 | 유효성 검사 | 훈련 | 내보내기 |
|--------------------|-----------------------------|----|--------|----|------|
| YOLOv3             | [물체 탐지](../tasks/detect.md) | ✅  | ✅      | ✅  | ✅    |
| YOLOv3-Ultralytics | [물체 탐지](../tasks/detect.md) | ✅  | ✅      | ✅  | ✅    |
| YOLOv3u            | [물체 탐지](../tasks/detect.md) | ✅  | ✅      | ✅  | ✅    |

이 표는 각 YOLOv3 버전의 기능을 한 눈에 보여주며, 물체 탐지 워크플로우의 다양한 작업 및 운영 모드에 대해 다양성과 적합성을 강조합니다.

## 사용 예제

다음 예제는 간단한 YOLOv3 훈련 및 추론 예제를 제공합니다. 이와 다른 [모드](../modes/index.md)의 전체 설명은 [Predict](../modes/predict.md), [Train](../modes/train.md), [Val](../modes/val.md) 및 [Export](../modes/export.md) 문서 페이지를 참조하세요.

!!! Example "예제"

    === "Python"

        Python에서 PyTorch 사전 훈련된 `*.pt` 모델 및 설정 `*.yaml` 파일을 YOLO() 클래스에 전달하여 모델 인스턴스를 만들 수 있습니다.

        ```python
        from ultralytics import YOLO

        # COCO 사전 훈련된 YOLOv3n 모델 로드
        model = YOLO('yolov3n.pt')

        # 모델 정보 표시 (선택 사항)
        model.info()

        # COCO8 예제 데이터셋에서 100 epoch 동안 모델 훈련
        results = model.train(data='coco8.yaml', epochs=100, imgsz=640)

        # YOLOv3n 모델로 'bus.jpg' 이미지에 추론 실행
        results = model('path/to/bus.jpg')
        ```

    === "CLI"

        CLI 명령어를 사용하여 모델을 직접 실행할 수 있습니다.

        ```bash
        # COCO 사전 훈련된 YOLOv3n 모델 로드하고 COCO8 예제 데이터셋에서 100 epoch 동안 훈련
        yolo train model=yolov3n.pt data=coco8.yaml epochs=100 imgsz=640

        # COCO 사전 훈련된 YOLOv3n 모델 로드하고 'bus.jpg' 이미지에 추론 실행
        yolo predict model=yolov3n.pt source=path/to/bus.jpg
        ```

## 인용 및 감사의 글

본인의 연구에서 YOLOv3를 사용한다면, 원본 YOLO 논문과 Ultralytics YOLOv3 저장소를 인용해 주십시오.

!!! Quote ""

    === "BibTeX"

        ```bibtex
        @article{redmon2018yolov3,
          title={YOLOv3: An Incremental Improvement},
          author={Redmon, Joseph and Farhadi, Ali},
          journal={arXiv preprint arXiv:1804.02767},
          year={2018}
        }
        ```

Joseph Redmon과 Ali Farhadi에게 원본 YOLOv3 개발에 대한 감사의 글을 전합니다.
