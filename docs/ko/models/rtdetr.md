---
comments: true
description: 비둘기(Baidu)가 개발한 RT-DETR은 비전 트랜스포머(Vision Transformers)를 기반으로 한 실시간 객체 검출기로, 사전 훈련된 모델을 사용하여 시간지연이 없는 고성능을 제공합니다.
keywords: RT-DETR, 비둘기, 비전 트랜스포머, 객체 검출, 실시간 성능, CUDA, TensorRT, IoU-aware query selection, Ultralytics, 파이썬 API, PaddlePaddle
---

# 비둘기의 RT-DETR: 비전 트랜스포머 기반 실시간 객체 검출기

## 개요

비둘기(Baidu)가 개발한 Real-Time Detection Transformer(RT-DETR)은 고정밀도를 유지하면서 실시간 성능을 제공하는 첨단 엔드 투 엔드 객체 검출기입니다. 비전 트랜스포머(Vision Transformers, ViT)의 성능을 활용하여, 다중 스케일 특징을 효율적으로 처리할 수 있도록 인트라 스케일 상호 작용과 크로스 스케일 퓨전을 분리합니다. RT-DETR은 다양한 디코더 레이어를 사용하여 추론 속도를 유연하게 조정할 수 있으므로 재훈련 없이 실시간 객체 검출에 적용하기에 매우 적합합니다. 이 모델은 CUDA와 TensorRT와 같은 가속화된 백엔드에서 많은 다른 실시간 객체 검출기보다 뛰어난 성능을 발휘합니다.

![모델 예시 이미지](https://user-images.githubusercontent.com/26833433/238963168-90e8483f-90aa-4eb6-a5e1-0d408b23dd33.png)
**비둘기의 RT-DETR 개요** 비둘기의 RT-DETR 모델 구조 다이어그램은 백본 네트워크의 마지막 세 단계 {S3, S4, S5}를 인코더의 입력으로 보여줍니다. 효율적인 하이브리드 인코더는 인트라스케일 특징 상호 작용(AIFI, intrascale feature interaction)과 크로스 스케일 특징 퓨전 모듈(CCFM, cross-scale feature-fusion module)을 통해 다중 스케일 특징을 이미지 특징의 시퀀스로 변환합니다. IoU-aware query selection은 디코더에 대한 초기 객체 쿼리로 작동하기 위해 일정한 수의 이미지 특징을 선택하는 데 사용됩니다. 마지막으로, 보조 예측 헤드와 함께 디코더는 객체 쿼리를 반복하여 박스와 신뢰도 점수를 최적화합니다. ([원문](https://arxiv.org/pdf/2304.08069.pdf) 참조).

### 주요 기능

- **효율적인 하이브리드 인코더:** 비둘기의 RT-DETR은 다중 스케일 특징을 인트라 스케일 상호 작용과 크로스 스케일 퓨전을 분리하여 처리하는 효율적인 하이브리드 인코더를 사용합니다. 이 독특한 비전 트랜스포머 기반 디자인은 계산 비용을 줄이고 실시간 객체 검출이 가능하도록 합니다.
- **IoU-aware 쿼리 선택:** 비둘기의 RT-DETR은 IoU-aware 쿼리 선택을 사용하여 개체 쿼리 초기화를 개선합니다. 이를 통해 모델은 장면에서 가장 관련성 있는 개체에 집중하며 검출 정확도를 향상시킵니다.
- **융통성 있는 추론 속도 조정:** 비둘기의 RT-DETR은 훈련 없이 다른 디코더 레이어를 사용하여 추론 속도를 유연하게 조정할 수 있습니다. 이러한 적응성은 다양한 실시간 객체 검출 시나리오에서 실용적인 응용을 용이하게 합니다.

## 사전 훈련된 모델

Ultralytics의 파이썬 API는 다양한 스케일의 사전 훈련된 PaddlePaddle RT-DETR 모델을 제공합니다:

- RT-DETR-L: COCO val2017에서 53.0% AP, T4 GPU에서 114 FPS
- RT-DETR-X: COCO val2017에서 54.8% AP, T4 GPU에서 74 FPS

## 사용 예시

이 예시는 간단한 RT-DETRR 훈련 및 추론 예시를 제공합니다. [Predict](../modes/predict.md), [Train](../modes/train.md), [Val](../modes/val.md), [Export](../modes/export.md) 등의 자세한 문서는 [Predict](../modes/predict.md), [Train](../modes/train.md), [Val](../modes/val.md), [Export](../modes/export.md) 문서 페이지를 참조하십시오.

!!! 예시

    === "파이썬"

        ```python
        from ultralytics import RTDETR

        # COCO 사전 훈련된 RT-DETR-l 모델 로드
        model = RTDETR('rtdetr-l.pt')

        # 모델 정보 표시 (선택 사항)
        model.info()

        # COCO8 예제 데이터셋에 대해 100 epoch 동안 모델 훈련
        results = model.train(data='coco8.yaml', epochs=100, imgsz=640)

        # 'bus.jpg' 이미지에서 RT-DETR-l 모델로 추론 실행
        results = model('path/to/bus.jpg')
        ```

    === "CLI"

        ```bash
        # COCO 사전 훈련된 RT-DETR-l 모델 로드하고 COCO8 예제 데이터셋에 대해 100 epoch 동안 훈련
        yolo train model=rtdetr-l.pt data=coco8.yaml epochs=100 imgsz=640

        # COCO 사전 훈련된 RT-DETR-l 모델 로드하고 'bus.jpg' 이미지에서 추론 실행
        yolo predict model=rtdetr-l.pt source=path/to/bus.jpg
        ```

## 지원되는 작업 및 모드

이 테이블은 각 모델의 유형, 특정 사전 훈련 가중치, 각 모델이 지원하는 작업 및 [모드](../modes/train.md), [Val](../modes/val.md), [Predict](../modes/predict.md), [Export](../modes/export.md)와 같은 다양한 모드를 나타내는 ✅ 이모지로 표시된 모드를 지원합니다.

| 모델 유형               | 사전 훈련 가중치     | 지원되는 작업                     | 추론 | 검증 | 훈련 | 출력 |
|---------------------|---------------|-----------------------------|----|----|----|----|
| RT-DETR Large       | `rtdetr-l.pt` | [객체 검출](../tasks/detect.md) | ✅  | ✅  | ✅  | ✅  |
| RT-DETR Extra-Large | `rtdetr-x.pt` | [객체 검출](../tasks/detect.md) | ✅  | ✅  | ✅  | ✅  |

## 인용 및 감사의 말

만약 연구나 개발 작업에서 비둘기(Baidu)의 RT-DETR을 사용한다면, [원래 논문을](https://arxiv.org/abs/2304.08069) 인용해주시기 바랍니다:

!!! Quote ""

    === "BibTeX"

        ```bibtex
        @misc{lv2023detrs,
              title={DETRs Beat YOLOs on Real-time Object Detection},
              author={Wenyu Lv and Shangliang Xu and Yian Zhao and Guanzhong Wang and Jinman Wei and Cheng Cui and Yuning Du and Qingqing Dang and Yi Liu},
              year={2023},
              eprint={2304.08069},
              archivePrefix={arXiv},
              primaryClass={cs.CV}
        }
        ```

컴퓨터 비전 커뮤니티에게 귀중한 자료인 비전 트랜스포머 기반 실시간 객체 검출기인 비둘기(Baidu)의 RT-DETR을 만들고 유지하기 위해 비둘기와 [PaddlePaddle](https://github.com/PaddlePaddle/PaddleDetection) 팀에게 감사의 인사를 전합니다.

*Keywords: RT-DETR, Transformer, ViT, 비전 트랜스포머, 비둘기 RT-DETR, PaddlePaddle, Paddle Paddle RT-DETR, 실시간 객체 검출, 비전 트랜스포머 기반 객체 검출, 사전 훈련된 PaddlePaddle RT-DETR 모델, 비둘기 RT-DETR 사용법, Ultralytics 파이썬 API*
