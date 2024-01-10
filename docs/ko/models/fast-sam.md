---
comments: true
description: FastSAM은 이미지에서 실시간 객체 분할을 위한 CNN 기반 솔루션으로, 향상된 사용자 상호작용, 계산 효율성, 다양한 비전 작업에 대응할 수 있는 특징을 갖고 있습니다.
keywords: FastSAM, 머신러닝, CNN 기반 솔루션, 객체 분할, 실시간 솔루션, Ultralytics, 비전 작업, 이미지 처리, 산업 응용, 사용자 상호작용
---

# Fast Segment Anything Model (FastSAM)

Fast Segment Anything Model (FastSAM)은 Segment Anything 작업을 위한 새로운 실시간 CNN 기반 솔루션입니다. 이 작업은 다양한 사용자 상호작용 프롬프트에 따라 이미지 내의 모든 객체를 분할하는 것을 목표로 합니다. FastSAM은 계산 요구 사항을 크게 줄이면서 경쟁력 있는 성능을 유지하기 때문에 다양한 비전 작업에 실용적인 선택지가 될 수 있습니다.

![Fast Segment Anything Model (FastSAM) 아키텍처 개요](https://user-images.githubusercontent.com/26833433/248551984-d98f0f6d-7535-45d0-b380-2e1440b52ad7.jpg)

## 개요

FastSAM은 계산 리소스 요구 사항이 큰 Transformer 모델인 Segment Anything Model (SAM)의 한계를 해결하기 위해 설계되었습니다. FastSAM은 Segment Anything 작업을 두 단계로 분리한 방식을 채택합니다: 모든 인스턴스 분할과 프롬프트로 인한 영역 선택. 첫 번째 단계에서는 [YOLOv8-seg](../tasks/segment.md)를 사용하여 이미지의 모든 인스턴스의 분할 마스크를 생성합니다. 두 번째 단계에서는 프롬프트에 해당하는 관심 영역을 출력합니다.

## 주요 특징

1. **실시간 솔루션**: CNN의 계산 효율성을 활용하여 FastSAM은 Segment Anything 작업에 대한 실시간 솔루션을 제공하며, 빠른 결과가 필요한 산업 응용에 가치가 있습니다.

2. **효율성과 성능**: FastSAM은 성능 품질을 희생하지 않고 계산과 리소스 요구 사항을 크게 줄입니다. SAM과 비교해 유사한 성능을 달성하면서 계산 리소스를 크게 줄여 실시간 응용이 가능해집니다.

3. **프롬프트 안내 분할**: FastSAM은 다양한 사용자 상호작용 프롬프트에 따라 이미지 내의 모든 객체를 분할할 수 있으므로 다양한 시나리오에서 유연성과 적응성을 제공합니다.

4. **YOLOv8-seg 기반**: FastSAM은 [YOLOv8-seg](../tasks/segment.md)를 기반으로 한 것으로, 인스턴스 분할 브랜치가 장착된 객체 감지기입니다. 이를 통해 이미지의 모든 인스턴스의 분할 마스크를 효과적으로 생성할 수 있습니다.

5. **벤치마크에서 경쟁 결과**: MS COCO에서의 객체 제안 작업에서 FastSAM은 [SAM](sam.md)에 비해 단일 NVIDIA RTX 3090에서 훨씬 더 빠른 속도로 높은 점수를 달성하여 효율성과 능력을 입증했습니다.

6. **실용적인 응용**: FastSAM은 현재 방법보다 수십 배 또는 수백 배 더 빠른 속도로 여러 비전 작업의 신속한 솔루션을 제공하여 실질적인 적용 가능성을 제시합니다.

7. **모델 압축 가능성**: FastSAM은 구조에 인공 사전을 도입하여 계산 비용을 크게 줄일 수 있는 경로를 보여주어 일반 비전 작업에 대한 대형 모델 아키텍처에 대한 새로운 가능성을 열어줍니다.

## 사용 가능한 모델, 지원되는 작업 및 운영 모드

이 표는 사용 가능한 모델과 해당하는 사전 훈련 가중치, 지원하는 작업 및 [Inference](../modes/predict.md), [Validation](../modes/val.md), [Training](../modes/train.md), [Export](../modes/export.md)와 같은 다른 운영 모드에 대한 호환성을 나타내며, 지원되는 모드는 ✅ 이모지로, 지원되지 않는 모드는 ❌ 이모지로 표시됩니다.

| 모델 유형     | 사전 훈련 가중치      | 지원되는 작업                        | Inference | Validation | Training | Export |
|-----------|----------------|--------------------------------|-----------|------------|----------|--------|
| FastSAM-s | `FastSAM-s.pt` | [인스턴스 분할](../tasks/segment.md) | ✅         | ❌          | ❌        | ✅      |
| FastSAM-x | `FastSAM-x.pt` | [인스턴스 분할](../tasks/segment.md) | ✅         | ❌          | ❌        | ✅      |

## 사용 예시

FastSAM 모델을 Python 애플리케이션에 쉽게 통합할 수 있습니다. Ultralytics는 개발을 간소화하기 위해 사용자 친화적인 Python API 및 CLI 명령을 제공합니다.

### 예측 사용법

이미지에서 객체 검출을 수행하려면 다음과 같이 `predict` 메서드를 사용합니다:

!!! Example "예제"

    === "Python"
        ```python
        from ultralytics import FastSAM
        from ultralytics.models.fastsam import FastSAMPrompt

        # 추론 소스 정의
        source = 'path/to/bus.jpg'

        # FastSAM 모델 생성
        model = FastSAM('FastSAM-s.pt')  # 또는 FastSAM-x.pt

        # 이미지에 대한 추론 실행
        everything_results = model(source, device='cpu', retina_masks=True, imgsz=1024, conf=0.4, iou=0.9)

        # Prompt Process 객체 준비
        prompt_process = FastSAMPrompt(source, everything_results, device='cpu')

        # 모든 프롬프트
        ann = prompt_process.everything_prompt()

        # 바운딩 박스의 기본 모양은 [0,0,0,0]에서 [x1,y1,x2,y2]로 변경
        ann = prompt_process.box_prompt(bbox=[200, 200, 300, 300])

        # 텍스트 프롬프트
        ann = prompt_process.text_prompt(text='a photo of a dog')

        # 포인트 프롬프트
        # 기본 포인트 [[0,0]] [[x1,y1],[x2,y2]]
        # 기본 포인트 레이블 [0] [1,0] 0:배경, 1:전경
        ann = prompt_process.point_prompt(points=[[200, 200]], pointlabel=[1])
        prompt_process.plot(annotations=ann, output='./')
        ```

    === "CLI"
        ```bash
        # FastSAM 모델 로드 및 모든 것을 세분화하여 추출
        yolo segment predict model=FastSAM-s.pt source=path/to/bus.jpg imgsz=640
        ```

이 코드 조각은 사전 훈련된 모델을 로드하고 이미지에 대한 예측을 실행하는 간편함을 보여줍니다.

### 검증 사용법

데이터셋에서 모델을 검증하는 방법은 다음과 같습니다:

!!! Example "예제"

    === "Python"
        ```python
        from ultralytics import FastSAM

        # FastSAM 모델 생성
        model = FastSAM('FastSAM-s.pt')  # 또는 FastSAM-x.pt

        # 모델 검증
        results = model.val(data='coco8-seg.yaml')
        ```

    === "CLI"
        ```bash
        # FastSAM 모델 로드 및 이미지 크기 640에서 COCO8 예제 데이터셋에 대해 유효성 검사
        yolo segment val model=FastSAM-s.pt data=coco8.yaml imgsz=640
        ```

FastSAM은 단일 클래스 객체의 감지와 분할만 지원합니다. 이는 모든 객체를 동일한 클래스로 인식하고 분할한다는 의미입니다. 따라서 데이터셋을 준비할 때 모든 객체 카테고리 ID를 0으로 변환해야 합니다.

## FastSAM 공식 사용법

FastSAM은 [https://github.com/CASIA-IVA-Lab/FastSAM](https://github.com/CASIA-IVA-Lab/FastSAM) 저장소에서 직접 사용할 수도 있습니다. FastSAM을 사용하기 위해 수행할 일반적인 단계를 간단히 소개합니다:

### 설치

1. FastSAM 저장소를 복제합니다:
   ```shell
   git clone https://github.com/CASIA-IVA-Lab/FastSAM.git
   ```

2. Python 3.9로 Conda 환경을 생성하고 활성화합니다:
   ```shell
   conda create -n FastSAM python=3.9
   conda activate FastSAM
   ```

3. 복제한 저장소로 이동하여 필요한 패키지를 설치합니다:
   ```shell
   cd FastSAM
   pip install -r requirements.txt
   ```

4. CLIP 모델을 설치합니다:
   ```shell
   pip install git+https://github.com/openai/CLIP.git
   ```

### 예시 사용법

1. [모델 체크포인트](https://drive.google.com/file/d/1m1sjY4ihXBU1fZXdQ-Xdj-mDltW-2Rqv/view?usp=sharing)를 다운로드합니다.

2. FastSAM을 추론하기 위해 다음과 같이 사용합니다. 예시 명령어:

    - 이미지에서 모든 것을 세분화:
      ```shell
      python Inference.py --model_path ./weights/FastSAM.pt --img_path ./images/dogs.jpg
      ```

    - 텍스트 프롬프트를 사용하여 특정 객체를 세분화:
      ```shell
      python Inference.py --model_path ./weights/FastSAM.pt --img_path ./images/dogs.jpg --text_prompt "the yellow dog"
      ```

    - 바운딩 박스 내의 객체를 세분화 (xywh 형식으로 상자 좌표 제공):
      ```shell
      python Inference.py --model_path ./weights/FastSAM.pt --img_path ./images/dogs.jpg --box_prompt "[570,200,230,400]"
      ```

    - 특정 지점 근처의 객체를 세분화:
      ```shell
      python Inference.py --model_path ./weights/FastSAM.pt --img_path ./images/dogs.jpg --point_prompt "[[520,360],[620,300]]" --point_label "[1,0]"
      ```

또한, FastSAM을 [Colab 데모](https://colab.research.google.com/drive/1oX14f6IneGGw612WgVlAiy91UHwFAvr9?usp=sharing) 또는 [HuggingFace 웹 데모](https://huggingface.co/spaces/An-619/FastSAM)에서 시각적인 경험으로 시도해 볼 수 있습니다.

## 인용 및 감사의 말씀

FastSAM의 실시간 인스턴스 분할 분야에 대한 혁신적인 기여를 위해 FastSAM 저자들에게 감사의 말씀을 전합니다:

!!! Quote ""

    === "BibTeX"

      ```bibtex
      @misc{zhao2023fast,
            title={Fast Segment Anything},
            author={Xu Zhao and Wenchao Ding and Yongqi An and Yinglong Du and Tao Yu and Min Li and Ming Tang and Jinqiao Wang},
            year={2023},
            eprint={2306.12156},
            archivePrefix={arXiv},
            primaryClass={cs.CV}
      }
      ```

FastSAM 원본 논문은 [arXiv](https://arxiv.org/abs/2306.12156)에서 찾을 수 있습니다. 저자들은 자신들의 작업을 공개적으로 제공하였으며, 코드베이스는 [GitHub](https://github.com/CASIA-IVA-Lab/FastSAM)에서 이용할 수 있습니다. 저자들의 노력에 감사드리며 저작물을 더 폭넓은 커뮤니티에 알리기 위한 기여를 기대합니다.
