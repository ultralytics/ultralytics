---
comments: true
description: 얼트라리얼리틱스(Ultralytics)의 최첨단 이미지 세분화 모델인 Segment Anything Model(SAM)에 대해 알아보세요. 해당 모델은 실시간 이미지 세분화를 가능하게 하며, 프롬프트를 이용한 세분화, 제로샷 성능 및 사용법에 대해 알아봅니다.
keywords: 얼트라리얼리틱스, 이미지 세분화, Segment Anything Model, SAM, SA-1B 데이터셋, 실시간 성능, 제로샷 전이, 객체 감지, 이미지 분석, 머신 러닝
---

# Segment Anything Model (SAM)

Segment Anything Model(SAM) 을 어서 오세요. 이 혁신적인 모델은 프롬프트 기반의 실시간 세분화를 통해 세분화 분야에서 새로운 기준을 세웠습니다.

## SAM 소개: Segment Anything Model의 소개

Segment Anything Model(SAM)은 프롬프트 기반의 세분화를 가능하게 하는 뛰어난 이미지 세분화 모델입니다. SAM은 이미지 세분석 작업에서 독창성을 보여주는 Segment Anything 이니셔티브의 핵심을 형성하고 있으며, 이미지 세분화를 위한 새로운 모델, 작업 및 데이터셋을 소개하는 혁신적인 프로젝트입니다.

SAM의 고급설계는 모델이 기존 지식 없이도 새로운 이미지 분포 및 작업에 대응할 수 있는 기능인 제로샷 전이를 보여줍니다. 1,100만 개의 정교하게 선별된 이미지에 분포된 10억 개 이상의 마스크를 포함한 SA-1B 데이터셋으로 학습된 SAM은 많은 경우에 전적으로 감독된 학습 결과를 능가하는 인상적인 제로샷 성능을 보여줍니다.

![데이터셋 샘플 이미지](https://user-images.githubusercontent.com/26833433/238056229-0e8ffbeb-f81a-477e-a490-aff3d82fd8ce.jpg)
새롭게 도입된 SA-1B 데이터셋에서 오버레이된 마스크를 포함한 예시 이미지입니다. SA-1B는 다양한 고해상도의 이미지를 라이선스 보호하며 사생활을 보호하고 있으며, 1,100만 개의 고품질 세분화 마스크를 가지고 있습니다. 이러한 마스크는 SAM에 의해 자동으로 주석이 달렸으며, 인간 평가 및 다양한 실험을 통해 높은 품질과 다양성을 갖추었음이 검증되었습니다. 시각화를 위해 이미지는 이미지 당 평균 100개의 마스크로 그룹화되었습니다.

## Segment Anything Model (SAM)의 주요 기능

- **프롬프트 기반 세분화 작업:** SAM은 프롬프트 기반의 세분화 작업을 위해 설계되어, 공간 또는 텍스트 단서를 이용하여 개체를 식별합니다.
- **고급설계:** Segment Anything Model은 강력한 이미지 인코더, 프롬프트 인코더 및 가벼운 마스크 디코더를 사용합니다. 이 독특한 아키텍처는 유연한 프롬프팅, 실시간 마스크 계산 및 세분화 작업에서의 모호성 인식을 가능케 합니다.
- **SA-1B 데이터셋:** Segment Anything 프로젝트에서 소개된 SA-1B 데이터셋은 1,100만 개의 이미지에 10억 개 이상의 세분화 마스크를 가지고 있습니다. 이는 지금까지 가장 큰 세분화 데이터셋으로, SAM에게 다양하고 대규모의 학습 데이터를 제공합니다.
- **제로샷 성능:** SAM은 다양한 세분화 작업에서 뛰어난 제로샷 성능을 보여주므로, 프롬프트 엔지니어링의 필요성을 최소화하고 다양한 응용 프로그램에 즉시 사용할 수 있는 도구입니다.

Segment Anything Model 및 SA-1B 데이터셋에 대한 자세한 내용은 [Segment Anything 웹사이트](https://segment-anything.com)와 연구 논문 [Segment Anything](https://arxiv.org/abs/2304.02643)을 참조해 주세요.

## 사용 가능한 모델, 지원하는 작업 및 운영 모드

아래 표는 사용 가능한 모델과 해당 모델의 사전 훈련 가중치, 지원하는 작업 및 [추론](../modes/predict.md), [검증](../modes/val.md), [훈련](../modes/train.md) 및 [내보내기](../modes/export.md)와 같은 다른 운영 모드와의 호환성을 나타냅니다. 지원되는 모드는 ✅ 이모지로, 지원되지 않는 모드는 ❌ 이모지로 표시되었습니다.

| 모델 유형     | 사전 훈련 가중치  | 지원 작업                           | 추론 | 검증 | 훈련 | 내보내기 |
|-----------|------------|---------------------------------|----|----|----|------|
| SAM base  | `sam_b.pt` | [인스턴스 세분화](../tasks/segment.md) | ✅  | ❌  | ❌  | ✅    |
| SAM large | `sam_l.pt` | [인스턴스 세분화](../tasks/segment.md) | ✅  | ❌  | ❌  | ✅    |

## SAM 사용 방법: 이미지 세분화에서의 다재다능함과 강력함

Segment Anything Model은 훈련 데이터를 초월하는 다양한 하위 작업에 대해서도 사용될 수 있습니다. 이에는 가장자리 검출, 객체 제안 생성, 인스턴스 세분장 및 초기 텍스트-마스크 예측 등이 포함됩니다. SAM은 프롬프팅 엔지니어링을 통해 새로운 작업 및 데이터 분포에 빠르게 적응할 수 있으므로, 이미지 세분화에 대한 다재다능하고 강력한 도구로 사용될 수 있습니다.

### SAM 예측 예제

!!! Example "프롬프트를 이용한 세분화"

    주어진 프롬프트로 이미지 세분화를 실행합니다.

    === "파이썬"

        ```python
        from ultralytics import SAM

        # 모델 로드
        model = SAM('sam_b.pt')

        # 모델 정보 표시 (선택 사항)
        model.info()

        # bboxes 프롬프트로 추론 실행
        model('ultralytics/assets/zidane.jpg', bboxes=[439, 437, 524, 709])

        # points 프롬프트로 추론 실행
        model('ultralytics/assets/zidane.jpg', points=[900, 370], labels=[1])
        ```

!!! Example "전체 이미지 세분화"

    전체 이미지 세분화를 실행합니다.

    === "파이썬"

        ```python
        from ultralytics import SAM

        # 모델 로드
        model = SAM('sam_b.pt')

        # 모델 정보 표시 (선택 사항)
        model.info()

        # 추론 실행
        model('path/to/image.jpg')
        ```

    === "CLI"

        ```bash
        # SAM 모델로 추론 실행
        yolo predict model=sam_b.pt source=path/to/image.jpg
        ```

- 여기서 전체 이미지 세분화는 프롬프트(bboxes/points/masks)를 전달하지 않으면 실행됩니다.

!!! Example "SAMPredictor 예제"

    이미지를 설정하고 이미지 인코더를 여러번 실행하지 않고 여러번 프롬프트 추론을 실행할 수 있습니다.

    === "프롬프트 추론"

        ```python
        from ultralytics.models.sam import Predictor as SAMPredictor

        # SAMPredictor 생성
        overrides = dict(conf=0.25, task='segment', mode='predict', imgsz=1024, model="mobile_sam.pt")
        predictor = SAMPredictor(overrides=overrides)

        # 이미지 설정
        predictor.set_image("ultralytics/assets/zidane.jpg")  # 이미지 파일로 설정
        predictor.set_image(cv2.imread("ultralytics/assets/zidane.jpg"))  # np.ndarray로 설정
        results = predictor(bboxes=[439, 437, 524, 709])
        results = predictor(points=[900, 370], labels=[1])

        # 이미지 리셋
        predictor.reset_image()
        ```

    추가 인수로 전체 이미지를 세분화합니다.

    === "전체 이미지 세분화"

        ```python
        from ultralytics.models.sam import Predictor as SAMPredictor

        # SAMPredictor 생성
        overrides = dict(conf=0.25, task='segment', mode='predict', imgsz=1024, model="mobile_sam.pt")
        predictor = SAMPredictor(overrides=overrides)

        # 추가 인수로 이미지 세분화
        results = predictor(source="ultralytics/assets/zidane.jpg", crop_n_layers=1, points_stride=64)
        ```

- `전체 이미지 세분화`에 대한 자세한 추가 인수는 [`Predictor/generate` 참조](../../../reference/models/sam/predict.md)를 참조하세요.

## YOLOv8과의 SAM 비교

여기서는 Meta의 가장 작은 SAM 모델인 SAM-b를 얼트라리얼리틱스의 가장 작은 세분화 모델, [YOLOv8n-seg](../tasks/segment.md),과 비교합니다:

| 모델                                             | 크기                    | 파라미터                 | 속도 (CPU)               |
|------------------------------------------------|-----------------------|----------------------|------------------------|
| Meta's SAM-b                                   | 358 MB                | 94.7 M               | 51096 ms/im            |
| [MobileSAM](mobile-sam.md)                     | 40.7 MB               | 10.1 M               | 46122 ms/im            |
| [FastSAM-s](fast-sam.md) with YOLOv8 backbone  | 23.7 MB               | 11.8 M               | 115 ms/im              |
| Ultralytics [YOLOv8n-seg](../tasks/segment.md) | **6.7 MB** (53.4배 작음) | **3.4 M** (27.9배 적음) | **59 ms/im** (866배 빠름) |

이 비교는 모델 크기 및 속도에 대한 상당한 차이를 보여줍니다. SAM은 자동으로 세분화하는 독특한 기능을 제공하지만, 작은 크기와 높은 처리 속도로 인해 YOLOv8 세분화 모델과 직접 경쟁하지는 않습니다.

이 테스트는 2023년 애플 M2 맥북(16GB RAM)에서 수행되었습니다. 이 테스트를 재현하려면:

!!! Example "예제"

    === "파이썬"
        ```python
        from ultralytics import FastSAM, SAM, YOLO

        # SAM-b 프로파일링
        model = SAM('sam_b.pt')
        model.info()
        model('ultralytics/assets')

        # MobileSAM 프로파일링
        model = SAM('mobile_sam.pt')
        model.info()
        model('ultralytics/assets')

        # FastSAM-s 프로파일링
        model = FastSAM('FastSAM-s.pt')
        model.info()
        model('ultralytics/assets')

        # YOLOv8n-seg 프로파일링
        model = YOLO('yolov8n-seg.pt')
        model.info()
        model('ultralytics/assets')
        ```

## 자동 주석: 세분화 데이터셋을 위한 신속한 경로

자동 주석은 SAM의 핵심 기능으로, 미리 훈련된 탐지 모델을 사용하여 [세분화 데이터셋](https://docs.ultralytics.com/datasets/segment)을 생성할 수 있습니다. 이 기능을 사용하면 번거롭고 시간이 오래 걸리는 수작업 주석 작업을 건너뛰고 대량의 이미지를 신속하게 정확하게 주석을 달 수 있습니다.

### 탐지 모델을 사용하여 세분화 데이터셋 생성하기

Ultralytics 프레임워크를 사용하여 미리 훈련된 탐지 및 SAM 세분화 모델과 함께 데이터셋을 자동으로 주석할 수 있습니다. 아래와 같이 `auto_annotate` 함수를 사용하세요:

!!! Example "예제"

    === "파이썬"
        ```python
        from ultralytics.data.annotator import auto_annotate

        auto_annotate(data="path/to/images", det_model="yolov8x.pt", sam_model='sam_b.pt')
        ```

| 인수         | 유형              | 설명                                                                | 기본값          |
|------------|-----------------|-------------------------------------------------------------------|--------------|
| data       | 문자열             | 주석을 달 이미지가 포함된 폴더 경로.                                             |              |
| det_model  | 문자열, 선택사항       | 미리 훈련된 YOLO 탐지 모델. 기본값은 'yolov8x.pt'.                             | 'yolov8x.pt' |
| sam_model  | 문자열, 선택사항       | 미리 훈련된 SAM 세분화 모델. 기본값은 'sam_b.pt'.                               | 'sam_b.pt'   |
| device     | 문자열, 선택사항       | 모델을 실행할 디바이스. 기본값은 빈 문자열 (CPU 또는 사용 가능한 GPU 사용).                  |              |
| output_dir | 문자열, None, 선택사항 | 주석이 포함된 결과를 저장할 디렉토리 경로. 기본값은 'data'와 같은 디렉토리 내부의 'labels' 폴더입니다. | None         |

`auto_annotate` 함수는 이미지 경로를 입력으로 받아, 입력한 미리 훈련된 탐지와 SAM 세분화 모델, 이 함수를 실행할 디바이스 및 주석이 포함된 결과를 저장할 디렉토리 경로를 선택적으로 지정할 수 있는 기능을 제공합니다.

미리 훈련된 모델을 사용한 자동 주석 기능을 활용하면 높은 품질의 세분화 데이터셋을 생성하는 데 소요되는 시간과 노력을 크게 줄일 수 있습니다. 이 기능은 특히 대량의 이미지 컬렉션을 다루는 연구원과 개발자에게 유용하며, 수작업 주석 대신 모델 개발과 평가에 집중할 수 있습니다.

## 인용 및 감사의 말

귀하의 연구 또는 개발 작업에 SAM이 유용하게 사용된 경우, 저희 논문을 인용해 주시기 바랍니다:

!!! Quote ""

    === "BibTeX"

        ```bibtex
        @misc{kirillov2023segment,
              title={Segment Anything},
              author={Alexander Kirillov and Eric Mintun and Nikhila Ravi and Hanzi Mao and Chloe Rolland and Laura Gustafson and Tete Xiao and Spencer Whitehead and Alexander C. Berg and Wan-Yen Lo and Piotr Dollár and Ross Girshick},
              year={2023},
              eprint={2304.02643},
              archivePrefix={arXiv},
              primaryClass={cs.CV}
        }
        ```

모델 개발과 알고리즘 개발을 위한 귀중한 리소스를 만들고 유지 관리하는 Meta AI에게 감사의 말씀을 드립니다.

*keywords: Segment Anything, Segment Anything Model, SAM, Meta SAM, 이미지 세분화, 프롬프트 기반 세분화, 제로샷 성능, SA-1B 데이터셋, 고급설계, 자동 주석, 얼트라리얼리틱스, 사전 훈련 모델, SAM base, SAM large, 인스턴스 세분화, 컴퓨터 비전, 인공 지능, 머신 러닝, 데이터 주석, 세분화 마스크, 탐지 모델, YOLO 탐지 모델, bibtex, Meta AI.*
