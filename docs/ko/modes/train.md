---
comments: true
description: YOLOv8 모델을 Ultralytics YOLO를 사용하여 훈련하는 단계별 가이드로, 단일 GPU 및 다중 GPU 훈련의 예제 포함
keywords: Ultralytics, YOLOv8, YOLO, 객체 감지, 훈련 모드, 사용자 정의 데이터셋, GPU 훈련, 다중 GPU, 하이퍼파라미터, CLI 예제, Python 예제
---

# Ultralytics YOLO와 함께 하는 모델 훈련

<img width="1024" src="https://github.com/ultralytics/assets/raw/main/yolov8/banner-integrations.png" alt="Ultralytics YOLO 생태계 및 통합">

## 소개

딥러닝 모델을 훈련한다는 것은 모델에 데이터를 공급하고 그것이 정확한 예측을 할 수 있도록 매개변수를 조정하는 과정을 말합니다. Ultralytics YOLOv8의 훈련 모드는 현대 하드웨어 기능을 완전히 활용하여 객체 감지 모델의 효과적이고 효율적인 훈련을 위해 설계되었습니다. 이 가이드는 YOLOv8의 강력한 기능 세트를 사용하여 자체 모델을 훈련하는 데 필요한 모든 세부 정보를 다루는 것을 목표로 합니다.

<p align="center">
  <br>
  <iframe width="720" height="405" src="https://www.youtube.com/embed/LNwODJXcvt4?si=7n1UvGRLSd9p5wKs"
    title="YouTube video player" frameborder="0"
    allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share"
    allowfullscreen>
  </iframe>
  <br>
  <strong>시청하기:</strong> Google Colab에서 여러분의 사용자 정의 데이터셋으로 YOLOv8 모델을 훈련하는 방법.
</p>

## Ultralytics YOLO로 훈련하는 이유?

YOLOv8의 훈련 모드를 선택하는 데는 몇 가지 설득력 있는 이유가 있습니다:

- **효율성:** 단일 GPU 설정이든 여러 GPU로 확장하든, 하드웨어를 최대한 활용하세요.
- **다양성:** COCO, VOC, ImageNet과 같은 기존의 데이터셋뿐만 아니라 사용자 정의 데이터셋으로도 훈련 가능.
- **사용자 친화적:** 간단하면서도 강력한 CLI 및 Python 인터페이스를 통한 직관적인 훈련 경험 제공.
- **하이퍼파라미터 유연성:** 모델의 성능을 미세 조정할 수 있는 다양하게 구성 가능한 하이퍼파라미터.

### 훈련 모드의 주요 기능

다음은 YOLOv8의 훈련 모드의 주요 기능 중 일부입니다:

- **자동 데이터셋 다운로드:** COCO, VOC, ImageNet과 같은 표준 데이터셋들은 첫 사용시 자동으로 다운로드됩니다.
- **다중 GPU 지원:** 여러 GPU에 걸쳐 훈련 노력을 빠르게 확대하기 위한 규모있는 훈련 지원.
- **하이퍼파라미터 구성:** YAML 구성 파일이나 CLI 인수를 통해 하이퍼파라미터 수정 가능.
- **시각화 및 모니터링:** 훈련 지표의 실시간 추적 및 학습 과정의 시각화로 더 나은 인사이트 제공.

!!! Tip "팁"

    * YOLOv8 데이터셋들은 첫 사용시 자동으로 다운로드됩니다, 예: `yolo train data=coco.yaml`

## 사용 예제

COCO128 데이터셋에서 YOLOv8n을 이미지 크기 640으로 100 에포크 동안 훈련합니다. 훈련 장치는 `device` 인수를 사용하여 지정할 수 있습니다. 인수를 전달하지 않으면 사용 가능한 경우 GPU `device=0`이, 아니면 `device=cpu`가 사용됩니다. 전체 훈련 인수 목록은 아래 Arguments 섹션을 참조하세요.

!!! Example "단일 GPU 및 CPU 훈련 예제"

    장치는 자동으로 결정됩니다. GPU가 사용 가능하면 사용되며, 그렇지 않으면 CPU에서 훈련이 시작됩니다.

    === "Python"

        ```python
        from ultralytics import YOLO

        # 모델을 로드하세요.
        model = YOLO('yolov8n.yaml')  # YAML에서 새 모델 구축
        model = YOLO('yolov8n.pt')  # 사전 훈련된 모델 로드 (훈련을 위해 권장됨)
        model = YOLO('yolov8n.yaml').load('yolov8n.pt')  # YAML에서 구축 및 가중치 전달

        # 모델을 훈련합니다.
        results = model.train(data='coco128.yaml', epochs=100, imgsz=640)
        ```

    === "CLI"

        ```bash
        # YAML에서 새 모델을 구축하고 처음부터 훈련을 시작하세요.
        yolo detect train data=coco128.yaml model=yolov8n.yaml epochs=100 imgsz=640

        # 사전 훈련된 *.pt 모델에서 훈련을 시작하세요.
        yolo detect train data=coco128.yaml model=yolov8n.pt epochs=100 imgsz=640

        # YAML에서 새 모델을 구축하고, 사전 훈련된 가중치를 전달하고 훈련을 시작하세요.
        yolo detect train data=coco128.yaml model=yolov8n.yaml pretrained=yolov8n.pt epochs=100 imgsz=640
        ```

### 다중 GPU 훈련

다중 GPU 훈련을 통해 사용 가능한 하드웨어 리소스를 더 효율적으로 활용할 수 있습니다. 이 기능은 Python API와 명령행 인터페이스 모두를 통해 사용할 수 있습니다. 다중 GPU 훈련을 활성화하려면 사용하려는 GPU 장치 ID를 지정하세요.

!!! Example "다중 GPU 훈련 예제"

    2개의 GPU, CUDA 장치 0과 1로 훈련하려면 다음 명령을 사용하세요. 필요에 따라 추가 GPU로 확장하세요.

    === "Python"

        ```python
        from ultralytics import YOLO

        # 모델을 로드하세요.
        model = YOLO('yolov8n.pt')  # 사전 훈련된 모델 로드 (훈련 추천됨)

        # 2개의 GPU로 모델을 훈련합니다.
        results = model.train(data='coco128.yaml', epochs=100, imgsz=640, device=[0, 1])
        ```

    === "CLI"

        ```bash
        # 사전 훈련된 *.pt 모델로부터 시작하여 GPU 0과 1을 사용하여 훈련합니다.
        yolo detect train data=coco128.yaml model=yolov8n.pt epochs=100 imgsz=640 device=0,1
        ```

### Apple M1 및 M2 MPS 훈련

Ultralytics YOLO 모델에 통합된 Apple M1 및 M2 칩들에 대한 지원을 통해 Apple의 강력한 Metal Performance Shaders (MPS) 프레임워크를 활용하여 장치에서 모델을 훈련할 수 있습니다. MPS는 Apple 사용자 지정 실리콘에서 컴퓨터 및 이미지 처리 작업을 실행하는 고성능 방법을 제공합니다.

Apple M1 및 M2 칩에서 훈련을 활성화하려면, 훈련 과정을 시작할 때 장치로 'mps'를 지정해야 합니다. 아래는 Python 및 명령행 인터페이스를 통해 이를 수행할 수 있는 예제입니다:

!!! Example "MPS 훈련 예제"

    === "Python"

        ```python
        from ultralytics import YOLO

        # 모델을 로드하세요.
        model = YOLO('yolov8n.pt')  # 사전 훈련된 모델 로드 (훈련 추천됨)

        # 2개의 GPU로 모델을 훈련합니다.
        results = model.train(data='coco128.yaml', epochs=100, imgsz=640, device='mps')
        ```

    === "CLI"

        ```bash
        # 사전 훈련된 *.pt 모델을 사용하여 mps 장치에서 훈련을 시작합니다.
        yolo detect train data=coco128.yaml model=yolov8n.pt epochs=100 imgsz=640 device=mps
        ```

M1/M2 칩의 연산력을 활용하면서 훈련 작업을 더 효율적으로 처리할 수 있습니다. 자세한 지침과 고급 설정 옵션을 원하신다면 [PyTorch MPS 문서](https://pytorch.org/docs/stable/notes/mps.html)를 참조하세요.

### 중단된 훈련 이어나가기

이전에 저장된 상태에서 훈련을 이어나가는 기능은 딥러닝 모델을 다룰 때 중요한 기능입니다. 이 기능은 훈련 과정이 예기치 않게 중단되었거나 새로운 데이터로 모델을 계속 훈련하거나 더 많은 에포크 동안 훈련을 진행하고 싶을 때 유용합니다.

훈련을 재개할 때, Ultralytics YOLO는 마지막으로 저장된 모델에서 가중치를 로드하고 옵티마이저 상태, 학습률 스케줄러, 에포크 번호도 복원합니다. 이를 통해 훈련 과정을 중단된 지점부터 이어갈 수 있습니다.

Ultralytics YOLO에서 `train` 메서드 호출 시 `resume` 인수를 `True`로 설정하고 부분적으로 훈련된 모델 가중치가 포함된 `.pt` 파일의 경로를 지정하면 훈련을 이어나갈 수 있습니다.
