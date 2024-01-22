---
comments: true
description: YOLO-NAS는 우수한 물체 감지 모델로서 자세한 설명서를 탐색해보세요. Ultralytics Python API를 사용한 기능, 사전 훈련된 모델, 사용법 등을 자세히 알아보세요.
keywords: YOLO-NAS, Deci AI, 물체 감지, 딥러닝, 신경 아키텍처 검색, Ultralytics Python API, YOLO 모델, 사전 훈련된 모델, 양자화, 최적화, COCO, Objects365, Roboflow 100
---

# YOLO-NAS

## 개요

Deci AI에서 개발한 YOLO-NAS는 원래의 YOLO 모델의 한계를 해결하기 위해 고도의 신경 아키텍처 검색(Neural Architecture Search) 기술로 만들어진 혁신적인 물체 감지 기반 모델입니다. 양자화 지원과 정확성-지연 트레이드오프의 중요한 개선을 통해 YOLO-NAS는 물체 감지 분야에서 주목할 만한 성능 향상을 제공합니다.

![모델 예시 이미지](https://learnopencv.com/wp-content/uploads/2023/05/yolo-nas_COCO_map_metrics.png)
**YOLO-NAS 개요.** YOLO-NAS는 양자화 관련 블록과 선택적 양자화를 적용하여 최적의 성능을 달성합니다. 모델은 INT8 양자화 버전으로 변환될 때 최소한의 정확도 감소를 경험하므로 다른 모델들과 비교했을 때 상당한 개선을 이끌어냅니다. 이러한 혁신은 예측할 수 없는 물체 감지 능력과 높은 성능을 가진 우수한 아키텍처로 이어집니다.

### 주요 기능

- **양자화 친화적인 기본 블록**: YOLO-NAS는 이전 YOLO 모델의 한계 중 하나인 양자화에 적합한 새로운 기본 블록을 도입합니다.
- **정교한 훈련과 양자화**: YOLO-NAS는 고급 훈련 방식과 훈련 후 양자화를 활용하여 성능을 향상시킵니다.
- **AutoNAC 최적화와 사전 훈련**: YOLO-NAS는 AutoNAC 최적화를 활용하며 COCO, Objects365, Roboflow 100과 같은 유명한 데이터셋에서 사전 훈련됩니다. 이를 통해 YOLO-NAS는 본격적인 프로덕션 환경에서의 물체 감지 작업에 매우 적합합니다.

## 사전 훈련된 모델

Ultralytics가 제공하는 사전 훈련된 YOLO-NAS 모델로 다음 세대의 물체 감지 기술의 힘을 체험해 보세요. 이러한 모델은 속도와 정확성 측면에서 최고의 성능을 제공하기 위해 설계되었습니다. 특정 요구에 맞게 다양한 옵션 중 선택하세요:

| 모델               | mAP   | 지연 시간 (밀리초) |
|------------------|-------|-------------|
| YOLO-NAS S       | 47.5  | 3.21        |
| YOLO-NAS M       | 51.55 | 5.85        |
| YOLO-NAS L       | 52.22 | 7.87        |
| YOLO-NAS S INT-8 | 47.03 | 2.36        |
| YOLO-NAS M INT-8 | 51.0  | 3.78        |
| YOLO-NAS L INT-8 | 52.1  | 4.78        |

각 모델 변형은 평균 평균 정밀도(mAP)와 지연 시간 간의 균형을 제공하여 물체 감지 작업을 성능과 속도 모두 최적화할 수 있도록 합니다.

## 사용 예시

Ultralytics는 YOLO-NAS 모델을 `ultralytics` Python 패키지를 통해 Python 애플리케이션에 쉽게 통합할 수 있도록 지원합니다. 이 패키지는 프로세스를 간소화하기 위한 사용자 친화적인 Python API를 제공합니다.

다음 예시에서는 추론과 유효성 검사를 위해 `ultralytics` 패키지와 함께 YOLO-NAS 모델을 사용하는 방법을 보여줍니다:

### 추론과 유효성 검사 예시

이 예시에서는 COCO8 데이터셋에서 YOLO-NAS-s 모델을 유효성 검사합니다.

!!! Example "예제"

    이 예시에서는 YOLO-NAS를 위한 간단한 추론 및 유효성 검사 코드를 제공합니다. 추론 결과를 처리하기 위한 방법은 [예측](../modes/predict.md) 모드를 참조하세요. 추가 모드에서 YOLO-NAS를 사용하는 방법은 [Val](../modes/val.md) 및 [Export](../modes/export.md)를 참조하세요. `ultralytics` 패키지에서 YOLO-NAS의 훈련은 지원하지 않습니다.

    === "Python"

        PyTorch 사전 훈련된 `*.pt` 모델 파일을 `NAS()` 클래스에 전달하여 Python에서 모델 인스턴스를 생성할 수 있습니다:

        ```python
        from ultralytics import NAS

        # COCO 사전 훈련된 YOLO-NAS-s 모델 로드
        model = NAS('yolo_nas_s.pt')

        # 모델 정보 표시 (선택 사항)
        model.info()

        # COCO8 예제 데이터셋에서 모델 유효성 검사
        results = model.val(data='coco8.yaml')

        # YOLO-NAS-s 모델로 'bus.jpg' 이미지에 추론 실행
        results = model('path/to/bus.jpg')
        ```

    === "CLI"

        CLI 명령을 사용하여 모델을 직접 실행할 수 있습니다:

        ```bash
        # COCO 사전 훈련된 YOLO-NAS-s 모델로 COCO8 예제 데이터셋의 성능 유효성 검사
        yolo val model=yolo_nas_s.pt data=coco8.yaml

        # COCO 사전 훈련된 YOLO-NAS-s 모델로 'bus.jpg' 이미지에 추론 실행
        yolo predict model=yolo_nas_s.pt source=path/to/bus.jpg
        ```

## 지원되는 작업 및 모드

YOLO-NAS 모델은 Small (s), Medium (m) 및 Large (l) 세 가지 변형이 있습니다. 각 변형은 다른 계산 및 성능 요구 사항을 충족시키기 위해 설계되었습니다:

- **YOLO-NAS-s**: 계산 자원이 제한되고 효율성이 중요한 환경에 최적화되었습니다.
- **YOLO-NAS-m**: 더 높은 정확성을 가지는 일반적인 물체 감지 작업에 적합한 균형잡힌 모델입니다.
- **YOLO-NAS-l**: 계산 자원이 제한되지 않는 환경에서 가장 높은 정확성이 필요한 시나리오에 맞게 설계되었습니다.

아래는 각 모델에 대한 자세한 개요로, 사전 훈련된 가중치, 지원하는 작업, 다양한 작동 모드와의 호환성에 대한 링크가 제공됩니다.

| 모델 유형      | 사전 훈련된 가중치                                                                                    | 지원되는 작업                     | 추론 | 유효성 검사 | 훈련 | 내보내기 |
|------------|-----------------------------------------------------------------------------------------------|-----------------------------|----|--------|----|------|
| YOLO-NAS-s | [yolo_nas_s.pt](https://github.com/ultralytics/assets/releases/download/v8.1.0/yolo_nas_s.pt) | [물체 감지](../tasks/detect.md) | ✅  | ✅      | ❌  | ✅    |
| YOLO-NAS-m | [yolo_nas_m.pt](https://github.com/ultralytics/assets/releases/download/v8.1.0/yolo_nas_m.pt) | [물체 감지](../tasks/detect.md) | ✅  | ✅      | ❌  | ✅    |
| YOLO-NAS-l | [yolo_nas_l.pt](https://github.com/ultralytics/assets/releases/download/v8.1.0/yolo_nas_l.pt) | [물체 감지](../tasks/detect.md) | ✅  | ✅      | ❌  | ✅    |

## 인용 및 감사의 말씀

YOLO-NAS를 연구 또는 개발 작업에 활용한 경우 SuperGradients를 인용해 주세요.

!!! Quote ""

    === "BibTeX"

        ```bibtex
        @misc{supergradients,
              doi = {10.5281/ZENODO.7789328},
              url = {https://zenodo.org/record/7789328},
              author = {Aharon,  Shay and {Louis-Dupont} and {Ofri Masad} and Yurkova,  Kate and {Lotem Fridman} and {Lkdci} and Khvedchenya,  Eugene and Rubin,  Ran and Bagrov,  Natan and Tymchenko,  Borys and Keren,  Tomer and Zhilko,  Alexander and {Eran-Deci}},
              title = {Super-Gradients},
              publisher = {GitHub},
              journal = {GitHub repository},
              year = {2021},
        }
        ```

Deci AI의 [SuperGradients](https://github.com/Deci-AI/super-gradients/) 팀에게 컴퓨터 비전 커뮤니티를 위해 이 가치 있는 자료를 만들고 유지 관리한 데 대해 감사의 말씀을 전합니다. 혁신적인 아키텍처와 우수한 물체 감지 능력을 갖춘 YOLO-NAS가 개발자와 연구자에게 중요한 도구가 될 것으로 기대합니다.
