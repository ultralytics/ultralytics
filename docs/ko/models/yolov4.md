---
comments: true
description: YOLOv4에 대한 상세 가이드를 살펴보세요. 최신 실시간 객체 감지기의 아키텍처 하이라이트, 혁신적인 기능 및 응용 예제를 이해하세요.
keywords: ultralytics, YOLOv4, 객체 감지, 신경망, 실시간 감지, 객체 감지기, 기계 학습
---

# YOLOv4: 높은 속도와 정밀도를 갖는 객체 감지

Ultralytics YOLOv4 문서 페이지에 오신 것을 환영합니다. YOLOv4는 아키텍처 및 알고리즘 개선으로 실시간 객체 감지의 최적 속도와 정확도를 제공하는 최신 객체 감지기입니다. 2020년에 Alexey Bochkovskiy가 [https://github.com/AlexeyAB/darknet](https://github.com/AlexeyAB/darknet)에서 출시되었습니다. YOLOv4는 많은 응용 분야에서 우수한 선택입니다.

![YOLOv4 아키텍처 다이어그램](https://user-images.githubusercontent.com/26833433/246185689-530b7fe8-737b-4bb0-b5dd-de10ef5aface.png)
**YOLOv4 아키텍처 다이어그램**. YOLOv4의 복잡한 네트워크 설계를 보여줍니다. 최적의 실시간 객체 감지를 위해 백본, 넥 및 헤드 구성 요소와 이들의 상호 연결된 레이어가 포함되어 있습니다.

## 소개

YOLOv4는 You Only Look Once의 4번째 버전을 의미합니다. 이전 YOLO 버전인 [YOLOv3](yolov3.md) 및 기타 객체 감지 모델의 한계를 극복하기 위해 개발된 실시간 객체 감지 모델입니다. 다른 합성곱 신경망(Convolutional Neural Network, CNN) 기반 객체 감지기와는 달리 YOLOv4는 추천 시스템뿐만 아니라 독립적인 프로세스 관리 및 인적 감소에도 적용할 수 있습니다. 이는 일반적인 그래픽 처리 장치(Graphics Processing Unit, GPU)에서 작동함으로써 저렴한 가격에 대량 사용을 가능하게 합니다. 또한, 훈련을 위해 하나의 GPU만 필요합니다.

## 아키텍처

YOLOv4는 성능을 최적화하기 위해 여러 혁신적인 기능을 사용합니다. 이에는 Weighted-Residual-Connections (WRC), Cross-Stage-Partial-connections (CSP), Cross mini-Batch Normalization (CmBN), Self-adversarial-training (SAT), Mish-activation, Mosaic data augmentation, DropBlock regularization 및 CIoU loss가 포함됩니다. 이러한 기능들은 최첨단 결과를 달성하기 위해 결합되었습니다.

일반적인 객체 감지기는 입력, 백본, 넥 및 헤드와 같은 여러 부분으로 구성됩니다. YOLOv4의 백본은 ImageNet에서 사전 훈련되며, 객체의 클래스 및 경계 상자를 예측하는 데 사용됩니다. 백본은 VGG, ResNet, ResNeXt 또는 DenseNet과 같은 여러 모델에서 가져올 수 있습니다. 객체 감지기의 넥 부분은 다양한 단계에서 피처 맵을 수집하는 데 사용되며, 일반적으로 여러 하향 경로 및 여러 상향 경로를 포함합니다. 헤드 부분은 최종 객체 감지 및 분류에 사용됩니다.

## 베고 오브 프리비스

YOLOv4는 학습 중 모델의 정확성을 향상시키는 기법인 "베고 오브 프리비스"를 사용하기도 합니다. 데이터 증강은 객체 감지에서 주로 사용되는 베고 오브 프리비스 기법으로, 입력 이미지의 다양성을 높여 모델의 견고성을 향상시킵니다. 데이터 증강의 몇 가지 예는 화질 왜곡(이미지의 밝기, 대조도, 색상, 채도 및 노이즈 조정) 및 기하학적 왜곡(임의의 스케일링, 크롭, 뒤집기, 회전 추가)입니다. 이러한 기술은 모델이 다양한 유형의 이미지에 대해 더 잘 일반화되도록 돕습니다.

## 기능 및 성능

YOLOv4는 객체 감지의 최적 속도와 정확도를 위해 설계되었습니다. YOLOv4의 아키텍처에는 백본으로 CSPDarknet53, 넥으로 PANet, 감지 헤드로 YOLOv3가 포함되어 있습니다. 이 설계를 통해 YOLOv4는 뛰어난 속도로 객체 감지를 수행하며, 실시간 응용 프로그램에 적합합니다. YOLOv4는 객체 감지 벤치마크에서 최첨단 결과를 달성하고 정확도 면에서도 뛰어난 성능을 보입니다.

## 사용 예제

작성 시점 기준으로 Ultralytics는 현재 YOLOv4 모델을 지원하지 않습니다. 따라서 YOLOv4를 사용하려는 사용자는 YOLOv4 GitHub 저장소의 설치 및 사용 지침을 직접 참조해야 합니다.

다음은 YOLOv4를 사용하는 일반적인 단계에 대한 간략한 개요입니다:

1. YOLOv4 GitHub 저장소를 방문하세요: [https://github.com/AlexeyAB/darknet](https://github.com/AlexeyAB/darknet).

2. 설치에 대한 README 파일에 제공된 지침을 따르세요. 일반적으로 저장소를 클론하고 필요한 종속성을 설치하고 필요한 환경 변수를 설정하는 과정을 포함합니다.

3. 설치가 완료되면, 저장소에서 제공하는 사용 지침에 따라 모델을 훈련하고 사용할 수 있습니다. 이는 일반적으로 데이터셋을 준비하고 모델 매개변수를 설정하고 모델을 훈련한 다음 훈련된 모델을 사용하여 객체 감지를 수행하는 것을 포함합니다.

특정 단계는 사용 사례와 YOLOv4 저장소의 현재 상태에 따라 다를 수 있습니다. 따라서 YOLOv4 GitHub 저장소에서 제공되는 지침을 직접 참조하는 것이 강력히 권장됩니다.

YOLOv4의 지원이 구현되면 Ultralytics를 위한 사용 예제로 이 문서를 업데이트하기 위해 노력하겠습니다.

## 결론

YOLOv4는 속도와 정확도의 균형을 이루는 강력하고 효율적인 객체 감지 모델입니다. 학습 중 특정 기법 및 베고 오브 프리비스 기법의 사용으로 실시간 객체 감지 작업에서 탁월한 성능을 발휘합니다. 일반적인 GPU를 가진 사용자 누구나 사용하고 훈련할 수 있어 다양한 응용 분야에 접근 가능하고 실용적입니다.

## 인용 및 감사의 글

실시간 객체 감지 분야에서 중요한 기여를 한 YOLOv4 저자들에게 감사드립니다:

!!! Quote ""

    === "BibTeX"

        ```bibtex
        @misc{bochkovskiy2020yolov4,
              title={YOLOv4: Optimal Speed and Accuracy of Object Detection},
              author={Alexey Bochkovskiy and Chien-Yao Wang and Hong-Yuan Mark Liao},
              year={2020},
              eprint={2004.10934},
              archivePrefix={arXiv},
              primaryClass={cs.CV}
        }
        ```

원본 YOLOv4 논문은 [arXiv](https://arxiv.org/pdf/2004.10934.pdf)에서 확인할 수 있습니다. 저자들은 자신들의 작업을 일반에 공개하고 코드베이스는 [GitHub](https://github.com/AlexeyAB/darknet)에서 액세스할 수 있도록 했습니다. 저자들의 노력과 널리 알려진 커뮤니티에 작업을 제공해 준 사항을 감사히 여깁니다.
