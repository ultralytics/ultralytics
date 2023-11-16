---
comments: true
description: pip, conda, git 및 Docker를 사용하여 Ultralytics을 설치하는 다양한 방법을 탐색해 보세요. Ultralytics을 명령줄 인터페이스 또는 Python 프로젝트 내에서 사용하는 방법을 알아보세요.
keywords: Ultralytics 설치, pip를 이용한 Ultralytics 설치, Docker를 이용한 Ultralytics 설치, Ultralytics 명령줄 인터페이스, Ultralytics Python 인터페이스
---

## Ultralytics 설치하기

Ultralytics는 pip, conda, Docker를 포함한 다양한 설치 방법을 제공합니다. `ultralytics` pip 패키지를 이용해 가장 안정적인 최신 버전의 YOLOv8을 설치하거나 [Ultralytics GitHub 저장소](https://github.com/ultralytics/ultralytics)를 복제하여 가장 최신 버전을 받아볼 수 있습니다. Docker를 이용하면 패키지를 로컬에 설치하지 않고 격리된 컨테이너에서 실행할 수 있습니다.

!!! example "설치하기"

    === "Pip 설치하기 (권장)"
        pip을 사용하여 `ultralytics` 패키지를 설치하거나, `pip install -U ultralytics`를 실행하여 기존 설치를 업데이트하세요. Python Package Index(PyPI)에서 `ultralytics` 패키지에 대한 자세한 내용을 확인하세요: [https://pypi.org/project/ultralytics/](https://pypi.org/project/ultralytics/).

        [![PyPI 버전](https://badge.fury.io/py/ultralytics.svg)](https://badge.fury.io/py/ultralytics) [![다운로드](https://static.pepy.tech/badge/ultralytics)](https://pepy.tech/project/ultralytics)

        ```bash
        # PyPI에서 ultralytics 패키지 설치하기
        pip install ultralytics
        ```

        GitHub [저장소](https://github.com/ultralytics/ultralytics)에서 직접 `ultralytics` 패키지를 설치할 수도 있습니다. 최신 개발 버전이 필요한 경우 유용할 수 있습니다. 시스템에 Git 명령줄 도구가 설치되어 있는지 확인하세요. `@main` 명령어는 `main` 브랜치를 설치하며, `@my-branch`로 변경하거나 `main` 브랜치를 기본으로 사용하려면 아예 제거하면 됩니다.

        ```bash
        # GitHub에서 ultralytics 패키지 설치하기
        pip install git+https://github.com/ultralytics/ultralytics.git@main
        ```


    === "Conda 설치하기"
        pip의 대안으로 사용할 수 있는 또 다른 패키지 관리자인 Conda를 통해서도 설치할 수 있습니다. [https://anaconda.org/conda-forge/ultralytics](https://anaconda.org/conda-forge/ultralytics)에서 Anaconda에 대한 자세한 정보를 확인하세요. Conda 패키지를 업데이트하는 Ultralytics feedstock 저장소는 [https://github.com/conda-forge/ultralytics-feedstock/](https://github.com/conda-forge/ultralytics-feedstock/)에 있습니다.


        [![Conda 레시피](https://img.shields.io/badge/recipe-ultralytics-green.svg)](https://anaconda.org/conda-forge/ultralytics) [![Conda 다운로드](https://img.shields.io/conda/dn/conda-forge/ultralytics.svg)](https://anaconda.org/conda-forge/ultralytics) [![Conda 버전](https://img.shields.io/conda/vn/conda-forge/ultralytics.svg)](https://anaconda.org/conda-forge/ultralytics) [![Conda 플랫폼](https://img.shields.io/conda/pn/conda-forge/ultralytics.svg)](https://anaconda.org/conda-forge/ultralytics)

        ```bash
        # conda를 사용하여 ultralytics 패키지 설치하기
        conda install -c conda-forge ultralytics
        ```

        !!! note

            CUDA 환경에서 설치하는 경우 일반적으로 `ultralytics`, `pytorch` 및 `pytorch-cuda`를 동일한 명령어로 설치하여 Conda 패키지 관리자가 충돌을 해결하도록 하거나, 필요한 경우 CPU 전용 `pytorch` 패키지를 덮어쓸 수 있도록 `pytorch-cuda`를 마지막에 설치하는 것이 좋습니다.
            ```bash
            # Conda를 사용하여 모든 패키지 함께 설치하기
            conda install -c pytorch -c nvidia -c conda-forge pytorch torchvision pytorch-cuda=11.8 ultralytics
            ```

        ### Conda Docker 이미지

        Ultralytics Conda Docker 이미지들도 [DockerHub](https://hub.docker.com/r/ultralytics/ultralytics)에서 사용할 수 있습니다. 이 이미지들은 [Miniconda3](https://docs.conda.io/projects/miniconda/en/latest/)를 기반으로 하며, Conda 환경에서 `ultralytics`를 사용하기 위한 간단한 방법입니다.

        ```bash
        # 이미지 이름을 변수로 설정하기
        t=ultralytics/ultralytics:latest-conda

        # Docker Hub에서 최신 ultralytics 이미지 가져오기
        sudo docker pull $t

        # GPU 지원으로 ultralytics 이미지를 컨테이너에서 실행하기
        sudo docker run -it --ipc=host --gpus all $t  # 모든 GPU 사용
        sudo docker run -it --ipc=host --gpus '"device=2,3"' $t  # 특정 GPU 지정
        ```


    === "Git 복제하기"
        개발에 기여하거나 최신 소스 코드를 실험해 보고 싶다면 `ultralytics` 저장소를 복제하세요. 복제한 후 해당 디렉토리로 이동하여 pip을 이용해 편집 가능 모드 `-e`로 패키지를 설치합니다.
        ```bash
        # ultralytics 저장소 복제하기
        git clone https://github.com/ultralytics/ultralytics

        # 복제한 디렉토리로 이동하기
        cd ultralytics

        # 개발을 위한 편집 가능 모드로 패키지 설치하기
        pip install -e .
        ```

    === "Docker 사용하기"

        Docker를 사용하면 `ultralytics` 패키지를 격리된 컨테이너에서 원활하게 실행할 수 있으며, 다양한 환경에서 일관된 성능을 보장합니다. [Docker Hub](https://hub.docker.com/r/ultralytics/ultralytics)의 공식 `ultralytics` 이미지 중 하나를 선택함으로써 로컬 설치의 복잡함을 피하고 검증된 작업 환경에 접근할 수 있습니다. Ultralytics은 서로 다른 플랫폼과 사용 사례에 대해 높은 호환성과 효율성을 제공하기 위해 5가지 주요 Docker 이미지를 제공합니다:

        <a href="https://hub.docker.com/r/ultralytics/ultralytics"><img src="https://img.shields.io/docker/pulls/ultralytics/ultralytics?logo=docker" alt="Docker Pulls"></a>

        - **Dockerfile:** 트레이닝에 추천되는 GPU 이미지입니다.
        - **Dockerfile-arm64:** Raspberry Pi와 같은 ARM64 기반 플랫폼에 배포하기에 최적화된 ARM64 아키텍처용입니다.
        - **Dockerfile-cpu:** GPU가 없는 환경에서 인퍼런스에 적합한 Ubuntu 기반 CPU 전용 버전입니다.
        - **Dockerfile-jetson:** NVIDIA Jetson 장치에 최적화된 GPU 지원을 통합한 버전입니다.
        - **Dockerfile-python:** 가볍게 애플리케이션을 위해 필요한 종속성과 Python만 있는 최소한의 이미지입니다.
        - **Dockerfile-conda:** Miniconda3를 기반으로 하며 ultralytics 패키지의 conda 설치를 포함하고 있습니다.

        아래의 명령어로 최신 이미지를 받고 실행할 수 있습니다:

        ```bash
        # 이미지 이름을 변수로 설정하기
        t=ultralytics/ultralytics:latest

        # Docker Hub에서 최신 ultralytics 이미지 가져오기
        sudo docker pull $t

        # GPU 지원으로 ultralytics 이미지를 컨테이너에서 실행하기
        sudo docker run -it --ipc=host --gpus all $t  # 모든 GPU 사용
        sudo docker run -it --ipc=host --gpus '"device=2,3"' $t  # 특정 GPU 지정
        ```

        위 명령어는 최신 `ultralytics` 이미지로 Docker 컨테이너를 초기화합니다. `-it` 플래그는 pseudo-TTY를 할당하고 표준 입력을 유지하여 컨테이너와 상호 작용할 수 있게 해줍니다. `--ipc=host` 플래그는 프로세스 간 메모리 공유에 필요한 IPC(Inter-Process Communication) 네임스페이스를 호스트로 설정합니다. `--gpus all` 플래그는 컨테이너 내에서 사용 가능한 모든 GPU에 대한 접근을 활성화하는데, GPU 계산이 필요한 작업에 중요합니다.

        참고: 로컬 기계의 파일을 컨테이너 내에서 작업하기 위해서는 로컬 디렉토리를 컨테이너에 마운트하는 데 Docker 볼륨을 사용하세요:

        ```bash
        # 로컬 디렉토리를 컨테이너 내부 디렉토리에 마운트하기
        sudo docker run -it --ipc=host --gpus all -v /path/on/host:/path/in/container $t
        ```

        `/path/on/host`를 로컬 기계의 디렉토리 경로로, `/path/in/container`를 컨테이너 내부에서 원하는 경로로 변경하여 접근할 수 있게 하세요.

        Docker 사용에 대한 고급 기능은 [Ultralytics Docker 가이드](https://docs.ultralytics.com/guides/docker-quickstart/)에서 더 탐구해보세요.

`ultralytics`의 종속성 목록은 [requirements.txt](https://github.com/ultralytics/ultralytics/blob/main/requirements.txt) 파일에서 확인할 수 있습니다. 위 예제에서는 모든 필요한 종속성을 설치합니다.

<p align="center">
  <br>
  <iframe width="720" height="405" src="https://www.youtube.com/embed/_a7cVL9hqnk"
    title="YouTube video player" frameborder="0"
    allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share"
    allowfullscreen>
  </iframe>
  <br>
  <strong>Watch:</strong> Ultralytics YOLO Quick Start Guide
</p>

!!! tip "팁"

    PyTorch 설치 요구사항은 운영 체제와 CUDA 요구사항에 따라 다르므로 [https://pytorch.org/get-started/locally](https://pytorch.org/get-started/locally)의 지침에 따라 PyTorch를 먼저 설치하는 것이 권장됩니다.

    <a href="https://pytorch.org/get-started/locally/">
        <img width="800" alt="PyTorch 설치 지침" src="https://user-images.githubusercontent.com/26833433/228650108-ab0ec98a-b328-4f40-a40d-95355e8a84e3.png">
    </a>

## 명령줄 인터페이스(CLI)로 Ultralytics 사용하기

Ultralytics 명령줄 인터페이스(CLI)는 Python 환경이 필요 없이 단일 라인 명령어를 통해 작업을 쉽게 실행할 수 있도록 합니다. CLI는 커스터마이징이나 Python 코드가 필요 없습니다. `yolo` 명령어를 이용해 터미널에서 모든 작업을 실행할 수 있습니다. 명령줄에서 YOLOv8을 사용하는 방법에 대해 더 알아보려면 [CLI 가이드](/../usage/cli.md)를 참고하세요.

!!! example

    === "문법"

        Ultralytics `yolo` 명령어는 다음과 같은 문법을 사용합니다:
        ```bash
        yolo TASK MODE ARGS

        여기서   TASK (선택적)은 [detect, segment, classify] 중 하나
                MODE (필수)는 [train, val, predict, export, track] 중 하나
                ARGS (선택적)은 'imgsz=320'과 같이 기본값을 재정의하는 'arg=value' 쌍을 아무 개수나 지정할 수 있습니다.
        ```
        모든 ARGS는 전체 [구성 가이드](/../usage/cfg.md)에서 또는 `yolo cfg`로 확인할 수 있습니다

    === "Train"

        10 에포크 동안 초기 학습률 0.01로 감지 모델을 훈련합니다.
        ```bash
        yolo train data=coco128.yaml model=yolov8n.pt epochs=10 lr0=0.01
        ```

    === "Predict"

        이전 훈련된 세분화 모델을 사용하여 이미지 크기 320으로 YouTube 동영상을 예측합니다:
        ```bash
        yolo predict model=yolov8n-seg.pt source='https://youtu.be/LNwODJXcvt4' imgsz=320
        ```

    === "Val"

        배치 크기 1와 이미지 크기 640으로 이전 훈련된 감지 모델을 검증합니다:
        ```bash
        yolo val model=yolov8n.pt data=coco128.yaml batch=1 imgsz=640
        ```

    === "Export"

        YOLOv8n 분류 모델을 ONNX 형식으로 내보냅니다. 이미지 크기는 224x128입니다 (TASK 필요 없음).
        ```bash
        yolo export model=yolov8n-cls.pt format=onnx imgsz=224,128
        ```

    === "특별"

        버전 확인, 설정 보기, 검사 실행 등을 위한 특별 명령어를 실행하세요:
        ```bash
        yolo help
        yolo checks
        yolo version
        yolo settings
        yolo copy-cfg
        yolo cfg
        ```

!!! warning "주의"

    모든 인수는 `arg=val`쌍으로 전달되어야 하며, 각 쌍 사이에는 공백으로 구분해야 합니다. 인수 접두사로 `--`를 사용하거나 인수 사이에 쉼표 `,`를 사용해서는 안 됩니다.

    - `yolo predict model=yolov8n.pt imgsz=640 conf=0.25` &nbsp; ✅
    - `yolo predict model yolov8n.pt imgsz 640 conf 0.25` &nbsp; ❌
    - `yolo predict --model yolov8n.pt --imgsz 640 --conf 0.25` &nbsp; ❌

[CLI 가이드](/../usage/cli.md){ .md-button .md-button--primary}
