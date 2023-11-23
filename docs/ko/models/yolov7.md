---
comments: true
description: YOLOv7은 실시간 객체 검출기로, 뛰어난 속도, 강력한 정확성, 독특한 trainable bag-of-freebies 최적화에 대해 알아봅니다.
keywords: YOLOv7, 실시간 객체 검출기, 최첨단, Ultralytics, MS COCO 데이터셋, 모델 재파라미터화, 동적 라벨 할당, 확장 스케일, 복합 스케일
---

# YOLOv7: Trainable Bag-of-Freebies

YOLOv7은 5 FPS에서 160 FPS까지의 범위에서 알려진 모든 객체 검출기를 속도와 정확성에서 능가하는 최첨단 실시간 객체 검출기입니다. 이 모델은 GPU V100에서 30 FPS 이상을 달성하여, 알려진 실시간 객체 검출기 중 가장 높은 정확도(56.8% AP)를 보여줍니다. 게다가, YOLOv7은 다른 객체 검출기인 YOLOR, YOLOX, Scaled-YOLOv4, YOLOv5 등에 비해 속도와 정확성 면에서 더 뛰어납니다. 이 모델은 다른 데이터셋이나 사전 학습된 가중치를 사용하지 않고 MS COCO 데이터셋에서 처음부터 훈련되었습니다. YOLOv7의 소스 코드는 GitHub에서 확인할 수 있습니다.

![YOLOv7와 최첨단 객체 검출기 비교](https://github.com/ultralytics/ultralytics/assets/26833433/5e1e0420-8122-4c79-b8d0-2860aa79af92)
**최첨단 객체 검출기 비교**. 표 2의 결과에서 볼 수 있듯이, 제안된 방법은 최상의 속도-정확성 균형을 가지고 있습니다. YOLOv7-tiny-SiLU를 YOLOv5-N(r6.1)과 비교해보면, 저희 방법은 약 127 fps가 더 빠르고 AP에서 10.7% 정확도가 향상됩니다. 게다가, YOLOv7은 161 fps 프레임 속도에서 51.4% AP를 달성하는 반면, PPYOLOE-L은 동일한 AP에서 78 fps 프레임 속도만 갖습니다. 매개 변수 사용 측면에서 YOLOv7는 PPYOLOE-L의 41%를 줄입니다. YOLOv7-X를 114 fps의 추론 속도로 YOLOv5-L(r6.1)의 99 fps 추론 속도와 비교하면, YOLOv7-X는 AP를 3.9% 향상시킵니다. YOLOv7-X를 유사한 스케일의 YOLOv5-X(r6.1)와 비교하면, YOLOv7-X의 추론 속도가 31 fps 더 빨라집니다. 또한, 매개 변수 및 계산의 양 측면에서 YOLOv7-X는 YOLOv5-X(r6.1)과 비교하여 매개 변수 22%와 계산 8%를 줄이고 AP를 2.2% 향상시킵니다 ([출처](https://arxiv.org/pdf/2207.02696.pdf)).

## 개요

실시간 객체 검출은 다중 객체 추적, 자율 주행, 로봇 공학 및 의료 이미지 분석을 비롯한 많은 컴퓨터 비전 시스템의 중요한 구성 요소입니다. 최근 몇 년간 실시간 객체 검출 개발은 효율적인 구조 설계와 다양한 CPU, GPU 및 신경 처리 장치(NPU)의 추론 속도 향상에 초점을 맞추고 있습니다. YOLOv7은 모바일 GPU와 GPU 장치를 모두 지원하여 엣지부터 클라우드까지 다양한 환경에서 사용할 수 있습니다.

기존의 실시간 객체 검출기가 아키텍처 최적화에 중점을 둔 것과는 달리, YOLOv7은 훈련 과정 최적화에 초점을 두고 있습니다. 이는 추론 비용을 증가시키지 않고 객체 검출의 정확도를 향상시키는 모듈과 최적화 방법을 포함한 "trainable bag-of-freebies" 개념을 도입합니다.

## 주요 기능

YOLOv7은 다음과 같은 주요 기능을 도입합니다:

1. **모델 재파라미터화**: YOLOv7은 그래디언트 전파 경로 개념을 이용한 다른 네트워크의 레이어에 적용 가능한 전략인 계획된 재파라미터화 모델을 제안합니다.

2. **동적 라벨 할당**: 다중 출력 레이어 모델의 훈련에서는 "다른 브랜치의 출력에 대해 동적 타깃을 어떻게 할당할 것인가?"라는 새로운 문제가 발생합니다. 이를 해결하기 위해 YOLOv7은 coarse-to-fine 리드 가이드 라벨 할당이라는 새로운 라벨 할당 방법을 도입합니다.

3. **확장 및 복합 스케일링**: YOLOv7은 매개 변수와 계산을 효과적으로 활용할 수 있는 실시간 객체 검출기를 위한 "확장" 및 "복합 스케일링" 방법을 제안합니다.

4. **효율성**: YOLOv7이 제안한 방법은 최첨단 실시간 객체 검출기의 매개 변수 약 40%, 계산 약 50%를 효과적으로 줄일 수 있으며, 더 빠른 추론 속도와 더 높은 검출 정확도를 달성할 수 있습니다.

## 사용 예시

기술 시점에서 Ultralytics은 현재 YOLOv7 모델을 지원하지 않습니다. 따라서 YOLOv7을 사용하려는 사용자는 YOLOv7 GitHub 저장소의 설치 및 사용 지침을 직접 참조해야 합니다.

YOLOv7을 사용하는 일반적인 단계에 대해 간략히 설명해 드리겠습니다:

1. YOLOv7 GitHub 저장소를 방문합니다: [https://github.com/WongKinYiu/yolov7](https://github.com/WongKinYiu/yolov7).

2. 설치에 대한 README 파일에서 제공하는 지침을 따릅니다. 일반적으로 저장소를 복제하고 필요한 종속성을 설치하고 필요한 환경 변수를 설정하는 것이 포함됩니다.

3. 설치가 완료되면 저장소에서 제공하는 사용 지침에 따라 모델을 훈련하고 사용할 수 있습니다. 이는 데이터셋을 준비하고 모델 매개 변수를 구성하고 모델을 훈련한 다음 훈련된 모델을 사용하여 객체 검출을 수행하는 것을 일반적으로 포함합니다.

특정 단계는 사용 사례와 YOLOv7 저장소의 현재 상태에 따라 달라질 수 있습니다. 따라서 YOLOv7 GitHub 저장소에서 제공하는 지침을 직접 참조하는 것이 권장됩니다.

YOLOv7을 지원하게 되면, Ultralytics의 사용 예시를 포함하여 이 문서를 업데이트하기 위해 최선을 다하겠습니다.

## 인용 및 감사의 글

실시간 객체 검출 분야에서의 중요한 기여로 인해 YOLOv7의 저자들에게 감사의 말씀을 전하고자 합니다:

!!! Quote ""

    === "BibTeX"

        ```bibtex
        @article{wang2022yolov7,
          title={{YOLOv7}: Trainable bag-of-freebies sets new state-of-the-art for real-time object detectors},
          author={Wang, Chien-Yao and Bochkovskiy, Alexey and Liao, Hong-Yuan Mark},
          journal={arXiv preprint arXiv:2207.02696},
          year={2022}
        }
        ```

원본 YOLOv7 논문은 [arXiv](https://arxiv.org/pdf/2207.02696.pdf)에서 찾을 수 있습니다. 저자들은 작업을 공개적으로 사용 가능하게 하였고, 코드베이스는 [GitHub](https://github.com/WongKinYiu/yolov7)에서 확인할 수 있습니다. 저희는 이들이 해당 분야의 발전에 기여하고 작업을 폭넓은 커뮤니티에게 공개 가능하게 한 노력에 감사드립니다.
