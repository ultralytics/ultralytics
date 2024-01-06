---
comments: true
description: 최첨단 물체 감지(오브젝트 디텍션) 모델인 'Meituan YOLOv6'을 알아보세요. 속도와 정확도 사이의 균형을 유지하는 이 모델은 실시간 애플리케이션에 인기 있는 선택입니다. 이 모델은 BiC(Bi-directional Concatenation) 모듈, AAT(Anchor-Aided Training) 전략, COCO 데이터셋에서 최첨단 정확도를 실현하기 위한 개선된 백본(backbone) 및 네크(neck) 설계 등에 대한 여러 주목할만한 향상 사항을 도입하고 있습니다.
keywords: Meituan YOLOv6, 오브젝트 디텍션, Ultralytics, YOLOv6 문서, Bi-directional Concatenation, Anchor-Aided Training, 사전 훈련 모델, 실시간 애플리케이션
---

# Meituan YOLOv6

## 개요

[Meituan](https://about.meituan.com/) YOLOv6은 속도와 정확도 사이에서 현저한 균형을 제공하는 최첨단 물체 감지기입니다. 이 모델은 Bi-directional Concatenation(BiC) 모듈, Anchor-Aided Training(AAT) 전략, 그리고 COCO 데이터셋에서 최첨단 정확도를 실현하기 위한 개선된 백본(backbone) 및 네크(neck) 디자인 등, 아키텍처와 훈련 방식에 대한 여러 주목할만한 향상 사항을 제공합니다.

![Meituan YOLOv6](https://user-images.githubusercontent.com/26833433/240750495-4da954ce-8b3b-41c4-8afd-ddb74361d3c2.png)
![모델 예시 이미지](https://user-images.githubusercontent.com/26833433/240750557-3e9ec4f0-0598-49a8-83ea-f33c91eb6d68.png)
**YOLOv6 개요**. 아키텍처 다이어그램으로, 다시 설계된 네트워크 구성 요소와 훈련 전략이 중요한 성능 개선을 이끈 모습을 보여줍니다. (a) YOLOv6의 네크(neck) (N과 S 표시)입니다. M/L의 경우, RepBlocks은 CSPStackRep으로 대체됩니다. (b) BiC 모듈의 구조입니다. (c) SimCSPSPPF 블록입니다. ([출처](https://arxiv.org/pdf/2301.05586.pdf)).

### 주요 특징

- **Bi-directional Concatenation (BiC) 모듈**: YOLOv6은 감지기(neck)에 BiC 모듈을 도입하여 위치 신호(localization signals)를 강화하고 성능을 향상시키는데, 속도 저하가 거의 없습니다.
- **Anchor-Aided Training (AAT) 전략**: 이 모델은 추론 효율을 저하시키지 않고 앵커 기반(anchor-based)과 앵커 없음(anchor-free) 패러다임의 이점을 모두 누릴 수 있도록 AAT를 제안합니다.
- **개선된 백본 및 네크 디자인**: YOLOv6을 백본과 네크에 추가적인 단계를 포함하여 깊게 만들어 COCO 데이터셋에서 최첨단 성능을 달성합니다.
- **셀프 디스틸레이션 전략**: YOLOv6의 작은 모델 성능을 강화하기 위해 새로운 셀프 디스틸레이션 전략이 도입되었습니다. 이는 훈련 중 보조 회귀 브랜치를 강화하고 추론 중에는 이를 제거하여 성능 저하를 방지합니다.

## 성능 메트릭

YOLOv6은 다양한 스케일의 사전 훈련 모델을 제공합니다:

- YOLOv6-N: NVIDIA Tesla T4 GPU에서 1187 FPS로 COCO val2017에서 37.5% AP.
- YOLOv6-S: 484 FPS로 45.0% AP.
- YOLOv6-M: 226 FPS로 50.0% AP.
- YOLOv6-L: 116 FPS로 52.8% AP.
- YOLOv6-L6: 실시간에서 최첨단 정확성.

또한, YOLOv6은 다양한 정밀도에 대한 양자화 모델과 모바일 플랫폼에 최적화된 모델도 제공합니다.

## 사용 예시

다음은 간단한 YOLOv6 훈련 및 추론 예시입니다. 이 외에도 [Predict](../modes/predict.md), [Train](../modes/train.md), [Val](../modes/val.md), [Export](../modes/export.md) 문서 페이지에서 자세한 내용을 확인할 수 있습니다.

!!! Example "예제"

    === "Python"

        `*.pt` 사전 훈련된 PyTorch 모델과 구성 `*.yaml` 파일을 `YOLO()` 클래스에 전달하여 파이썬에서 모델 인스턴스를 만들 수 있습니다:

        ```python
        from ultralytics import YOLO

        # YOLOv6n 모델을 처음부터 만듭니다
        model = YOLO('yolov6n.yaml')

        # 모델 정보를 표시합니다 (선택 사항)
        model.info()

        # COCO8 예시 데이터셋으로 모델을 100 에폭 동안 훈련합니다
        results = model.train(data='coco8.yaml', epochs=100, imgsz=640)

        # YOLOv6n 모델로 'bus.jpg' 이미지에서 추론을 실행합니다
        results = model('path/to/bus.jpg')
        ```

    === "CLI"

        CLI 명령을 사용하여 모델을 직접 실행할 수 있습니다:

        ```bash
        # 처음부터 YOLOv6n 모델을 만들고 COCO8 예시 데이터셋으로 100 에폭 동안 훈련합니다
        yolo train model=yolov6n.yaml data=coco8.yaml epochs=100 imgsz=640

        # 처음부터 YOLOv6n 모델을 만들고 'bus.jpg' 이미지에서 추론을 실행합니다
        yolo predict model=yolov6n.yaml source=path/to/bus.jpg
        ```

## 지원되는 작업 및 모드

YOLOv6 시리즈는 높은 성능의 [오브젝트 디텍션](../tasks/detect.md)을 위해 최적화된 다양한 모델을 제공합니다. 이 모델들은 다양한 계산 요구 사항과 정확도 요구 사항에 맞추어 다용도로 사용할 수 있습니다.

| 모델 유형     | 사전 훈련 가중치      | 지원되는 작업                        | 추론 | 검증 | 훈련 | 익스포트 |
|-----------|----------------|--------------------------------|----|----|----|------|
| YOLOv6-N  | `yolov6-n.pt`  | [오브젝트 디텍션](../tasks/detect.md) | ✅  | ✅  | ✅  | ✅    |
| YOLOv6-S  | `yolov6-s.pt`  | [오브젝트 디텍션](../tasks/detect.md) | ✅  | ✅  | ✅  | ✅    |
| YOLOv6-M  | `yolov6-m.pt`  | [오브젝트 디텍션](../tasks/detect.md) | ✅  | ✅  | ✅  | ✅    |
| YOLOv6-L  | `yolov6-l.pt`  | [오브젝트 디텍션](../tasks/detect.md) | ✅  | ✅  | ✅  | ✅    |
| YOLOv6-L6 | `yolov6-l6.pt` | [오브젝트 디텍션](../tasks/detect.md) | ✅  | ✅  | ✅  | ✅    |

이 표는 YOLOv6 모델의 다양한 변형에 대한 자세한 개요를 제공하며, 오브젝트 디텍션 작업과 [추론](../modes/predict.md), [검증](../modes/val.md), [훈련](../modes/train.md), [익스포트](../modes/export.md)와 같은 다양한 운영 모드와의 호환성을 강조합니다. 이러한 포괄적인 지원을 통해 사용자들은 다양한 오브젝트 디텍션 시나리오에서 YOLOv6 모델의 기능을 최대한 활용할 수 있습니다.

## 인용 및 감사의 글

실시간 물체 감지 분야에서의 중요한 기여에 대해 작성자들에게 감사의 말씀을 전합니다:

!!! Quote ""

    === "BibTeX"

        ```bibtex
        @misc{li2023yolov6,
              title={YOLOv6 v3.0: A Full-Scale Reloading},
              author={Chuyi Li and Lulu Li and Yifei Geng and Hongliang Jiang and Meng Cheng and Bo Zhang and Zaidan Ke and Xiaoming Xu and Xiangxiang Chu},
              year={2023},
              eprint={2301.05586},
              archivePrefix={arXiv},
              primaryClass={cs.CV}
        }
        ```

    YOLOv6 원본 논문은 [arXiv](https://arxiv.org/abs/2301.05586)에서 찾을 수 있습니다. 작성자들이 자신의 작업을 공개하지 않았으며, 코드는 [GitHub](https://github.com/meituan/YOLOv6)에서 액세스할 수 있습니다. 우리는 그들의 노력과 업계 발전을 위해 노력해 널리 알려져 있게 한 저자들에게 감사의 말씀을 전합니다.
