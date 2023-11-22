---
comments: true
description: Ultralytics 프레임워크에서 MobileSAM을 다운로드하고 테스트하는 방법, MobileSAM의 구현 방식, 원본 SAM과의 비교, 모바일 애플리케이션 향상 등에 대해 자세히 알아보세요. 오늘부터 모바일 애플리케이션을 개선하세요.
keywords: MobileSAM, Ultralytics, SAM, 모바일 애플리케이션, Arxiv, GPU, API, 이미지 인코더, 마스크 디코더, 모델 다운로드, 테스트 방법
---

![MobileSAM 로고](https://github.com/ChaoningZhang/MobileSAM/blob/master/assets/logo2.png?raw=true)

# Mobile Segment Anything (MobileSAM)

MobileSAM 논문은 이제 [arXiv](https://arxiv.org/pdf/2306.14289.pdf)에서 사용할 수 있습니다.

MobileSAM을 CPU에서 실행하는 데모는 이 [데모 링크](https://huggingface.co/spaces/dhkim2810/MobileSAM)에서 확인할 수 있습니다. Mac i5 CPU에서의 성능은 약 3초입니다. Hugging Face 데모에서는 인터페이스와 낮은 성능의 CPU가 느린 응답으로 이어지지만, 여전히 효과적으로 작동합니다.

MobileSAM은 [Grounding-SAM](https://github.com/IDEA-Research/Grounded-Segment-Anything), [AnyLabeling](https://github.com/vietanhdev/anylabeling) 및 [Segment Anything in 3D](https://github.com/Jumpat/SegmentAnythingin3D)를 비롯한 여러 프로젝트에서 구현되었습니다.

MobileSAM은 1%의 원본 이미지로 구성된 100k 데이터셋에서 하루 이내에 단일 GPU로 학습됩니다. 이 학습을 위한 코드는 나중에 공개될 예정입니다.

## 사용 가능한 모델, 지원되는 작업 및 작동 모드

이 표에서는 사용 가능한 모델과 해당 모델에 대한 사전 훈련 가중치, 지원되는 작업, [Inference](../modes/predict.md), [Validation](../modes/val.md), [Training](../modes/train.md) 및 [Export](../modes/export.md)와 같은 다양한 작동 모드의 호환성을 나타냅니다. 지원되는 모드는 ✅ 이모지로 표시되고, 지원되지 않는 모드는 ❌ 이모지로 표시됩니다.

| 모델 유형     | 사전 훈련 가중치       | 지원되는 작업                            | Inference | Validation | Training | Export |
|-----------|-----------------|------------------------------------|-----------|------------|----------|--------|
| MobileSAM | `mobile_sam.pt` | [인스턴스 세그멘테이션](../tasks/segment.md) | ✅         | ❌          | ❌        | ✅      |

## SAM에서 MobileSAM으로의 적응

MobileSAM은 원본 SAM과 동일한 파이프라인을 유지하므로, 원본의 전처리, 후처리 및 모든 다른 인터페이스를 통합했습니다. 따라서 현재 원본 SAM을 사용 중인 경우, MobileSAM으로 전환하는 데 최소한의 노력이 필요합니다.

MobileSAM은 원본 SAM과 비교 가능한 성능을 발휘하며, 이미지 인코더만 변경되었습니다. 구체적으로, 원본의 무거운 ViT-H 인코더 (632M)를 더 작은 Tiny-ViT (5M)로 대체했습니다. 단일 GPU에서 MobileSAM은 이미지 당 약 12ms의 작업 시간이 소요됩니다. 이미지 인코더에는 8ms가 소요되고, 마스크 디코더에는 4ms가 소요됩니다.

다음 표는 ViT 기반 이미지 인코더를 비교합니다:

| 이미지 인코더 | 원본 SAM | MobileSAM |
|---------|--------|-----------|
| 매개변수    | 611M   | 5M        |
| 속도      | 452ms  | 8ms       |

원본 SAM과 MobileSAM은 동일한 프롬프트 가이드 마스크 디코더를 사용합니다:

| 마스크 디코더 | 원본 SAM | MobileSAM |
|---------|--------|-----------|
| 매개변수    | 3.876M | 3.876M    |
| 속도      | 4ms    | 4ms       |

전체 파이프라인의 비교는 다음과 같습니다:

| 전체 파이프라인 (인코더+디코더) | 원본 SAM | MobileSAM |
|--------------------|--------|-----------|
| 매개변수               | 615M   | 9.66M     |
| 속도                 | 456ms  | 12ms      |

MobileSAM과 원본 SAM의 성능은 포인트 및 박스를 사용한 프롬프트를 통해 확인할 수 있습니다.

![포인트 프롬프트가 있는 이미지](https://raw.githubusercontent.com/ChaoningZhang/MobileSAM/master/assets/mask_box.jpg?raw=true)

![박스 프롬프트가 있는 이미지](https://github.com/ChaoningZhang/MobileSAM/blob/master/assets/logo2.png?raw=true)

MobileSAM은 우수한 성능을 자랑하며, 현재의 FastSAM보다 약 5배 작고 7배 빠릅니다. 자세한 내용은 [MobileSAM 프로젝트 페이지](https://github.com/ChaoningZhang/MobileSAM)에서 확인할 수 있습니다.

## Ultralytics에서 MobileSAM 테스트

원본 SAM과 마찬가지로, 포인트 및 박스 프롬프트 모드를 포함한 Ultralytics에서 간단한 테스트 방법을 제공합니다.

### 모델 다운로드

모델을 [여기](https://github.com/ChaoningZhang/MobileSAM/blob/master/weights/mobile_sam.pt)에서 다운로드할 수 있습니다.

### 포인트 프롬프트

!!! Example "예제"

    === "Python"
        ```python
        from ultralytics import SAM

        # 모델 불러오기
        model = SAM('mobile_sam.pt')

        # 포인트 프롬프트를 기반으로 세그먼트 예측
        model.predict('ultralytics/assets/zidane.jpg', points=[900, 370], labels=[1])
        ```

### 박스 프롬프트

!!! Example "예제"

    === "Python"
        ```python
        from ultralytics import SAM

        # 모델 불러오기
        model = SAM('mobile_sam.pt')

        # 박스 프롬프트를 기반으로 세그먼트 예측
        model.predict('ultralytics/assets/zidane.jpg', bboxes=[439, 437, 524, 709])
        ```

`MobileSAM`과 `SAM`은 동일한 API를 사용하여 구현되었습니다. 더 많은 사용법에 대해서는 [SAM 페이지](sam.md)를 참조하세요.

## 인용 및 감사의 글

MobileSAM이 연구 또는 개발에 유용하게 사용된 경우, 다음의 논문을 인용해 주시기 바랍니다:

!!! Quote ""

    === "BibTeX"

        ```bibtex
        @article{mobile_sam,
          title={Faster Segment Anything: Towards Lightweight SAM for Mobile Applications},
          author={Zhang, Chaoning and Han, Dongshen and Qiao, Yu and Kim, Jung Uk and Bae, Sung Ho and Lee, Seungkyu and Hong, Choong Seon},
          journal={arXiv preprint arXiv:2306.14289},
          year={2023}
        }
