---
comments: true
description: Ultralytics YOLOv8을 완벽하게 탐구하는 가이드로, 고속 및 정확성이 특징인 객체 탐지 및 이미지 분할 모델입니다. 설치, 예측, 훈련 튜토리얼 등이 포함되어 있습니다.
keywords: Ultralytics, YOLOv8, 객체 탐지, 이미지 분할, 기계 학습, 딥러닝, 컴퓨터 비전, YOLOv8 설치, YOLOv8 예측, YOLOv8 훈련, YOLO 역사, YOLO 라이센스
---

<div align="center">
  <p>
    <a href="https://yolovision.ultralytics.com" target="_blank">
    <img width="1024" src="https://raw.githubusercontent.com/ultralytics/assets/main/yolov8/banner-yolov8.png" alt="Ultralytics YOLO 배너"></a>
  </p>
  <a href="https://github.com/ultralytics"><img src="https://github.com/ultralytics/assets/raw/main/social/logo-social-github.png" width="3%" alt="Ultralytics GitHub"></a>
  <img src="https://github.com/ultralytics/assets/raw/main/social/logo-transparent.png" width="3%" alt="space">
  <a href="https://www.linkedin.com/company/ultralytics/"><img src="https://github.com/ultralytics/assets/raw/main/social/logo-social-linkedin.png" width="3%" alt="Ultralytics LinkedIn"></a>
  <img src="https://github.com/ultralytics/assets/raw/main/social/logo-transparent.png" width="3%" alt="space">
  <a href="https://twitter.com/ultralytics"><img src="https://github.com/ultralytics/assets/raw/main/social/logo-social-twitter.png" width="3%" alt="Ultralytics Twitter"></a>
  <img src="https://github.com/ultralytics/assets/raw/main/social/logo-transparent.png" width="3%" alt="space">
  <a href="https://youtube.com/ultralytics"><img src="https://github.com/ultralytics/assets/raw/main/social/logo-social-youtube.png" width="3%" alt="Ultralytics YouTube"></a>
  <img src="https://github.com/ultralytics/assets/raw/main/social/logo-transparent.png" width="3%" alt="space">
  <a href="https://www.tiktok.com/@ultralytics"><img src="https://github.com/ultralytics/assets/raw/main/social/logo-social-tiktok.png" width="3%" alt="Ultralytics TikTok"></a>
  <img src="https://github.com/ultralytics/assets/raw/main/social/logo-transparent.png" width="3%" alt="space">
  <a href="https://www.instagram.com/ultralytics/"><img src="https://github.com/ultralytics/assets/raw/main/social/logo-social-instagram.png" width="3%" alt="Ultralytics Instagram"></a>
  <img src="https://github.com/ultralytics/assets/raw/main/social/logo-transparent.png" width="3%" alt="space">
  <a href="https://ultralytics.com/discord"><img src="https://github.com/ultralytics/assets/raw/main/social/logo-social-discord.png" width="3%" alt="Ultralytics Discord"></a>
  <br>
  <br>
  <a href="https://github.com/ultralytics/ultralytics/actions/workflows/ci.yaml"><img src="https://github.com/ultralytics/ultralytics/actions/workflows/ci.yaml/badge.svg" alt="Ultralytics CI"></a>
  <a href="https://codecov.io/github/ultralytics/ultralytics"><img src="https://codecov.io/github/ultralytics/ultralytics/branch/main/graph/badge.svg?token=HHW7IIVFVY" alt="Ultralytics 코드 커버리지"></a>
  <a href="https://zenodo.org/badge/latestdoi/264818686"><img src="https://zenodo.org/badge/264818686.svg" alt="YOLOv8 인용"></a>
  <a href="https://hub.docker.com/r/ultralytics/ultralytics"><img src="https://img.shields.io/docker/pulls/ultralytics/ultralytics?logo=docker" alt="Docker 당기기"></a>
  <a href="https://ultralytics.com/discord"><img alt="Discord" src="https://img.shields.io/discord/1089800235347353640?logo=discord&logoColor=white&label=Discord&color=blue"></a>
  <br>
  <a href="https://console.paperspace.com/github/ultralytics/ultralytics"><img src="https://assets.paperspace.io/img/gradient-badge.svg" alt="Run on Gradient"></a>
  <a href="https://colab.research.google.com/github/ultralytics/ultralytics/blob/main/examples/tutorial.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"></a>
  <a href="https://www.kaggle.com/ultralytics/yolov8"><img src="https://kaggle.com/static/images/open-in-kaggle.svg" alt="Open In Kaggle"></a>
</div>

Ultralytics의 최신 버전인 [YOLOv8](https://github.com/ultralytics/ultralytics)을 소개합니다. 이 모델은 딥러닝과 컴퓨터 비전의 최신 발전을 바탕으로 구축되었으며, 속도와 정확성 면에서 뛰어난 성능을 제공합니다. 간결한 설계로 인해 다양한 애플리케이션에 적합하며, 엣지 디바이스에서부터 클라우드 API에 이르기까지 다양한 하드웨어 플랫폼에 쉽게 적응 가능합니다.

YOLOv8 문서를 탐구하여, 그 기능과 능력을 이해하고 활용할 수 있도록 돕는 종합적인 자원입니다. 기계 학습 분야에서 경험이 많건, 새롭게 시작하는 이들이건, 이 허브는 YOLOv8의 잠재력을 극대화하기 위해 설계되었습니다.

!!! Note "노트"

    🚧 다국어 문서는 현재 제작 중이며, 이를 개선하기 위해 노력하고 있습니다. 인내해 주셔서 감사합니다! 🙏

## 시작하기

- **설치** `ultralytics`를 pip으로 설치하고 몇 분 만에 시작하세요 &nbsp; [:material-clock-fast: 시작하기](quickstart.md){ .md-button }
- **예측** YOLOv8로 새로운 이미지와 비디오를 감지하세요 &nbsp; [:octicons-image-16: 이미지에서 예측하기](modes/predict.md){ .md-button }
- **훈련** 새로운 YOLOv8 모델을 사용자의 맞춤 데이터셋으로 훈련하세요 &nbsp; [:fontawesome-solid-brain: 모델 훈련하기](modes/train.md){ .md-button }
- **탐험** 세분화, 분류, 자세 인식, 추적과 같은 YOLOv8 작업 &nbsp; [:material-magnify-expand: 작업 탐험하기](tasks/index.md){ .md-button }

<p align="center">
  <br>
  <iframe width="720" height="405" src="https://www.youtube.com/embed/LNwODJXcvt4?si=7n1UvGRLSd9p5wKs"
    title="YouTube 비디오 플레이어" frameborder="0"
    allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share"
    allowfullscreen>
  </iframe>
  <br>
  <strong>시청하기:</strong> 사용자의 맞춤 데이터셋으로 YOLOv8 모델을 훈련하는 방법을 <a href="https://colab.research.google.com/github/ultralytics/ultralytics/blob/main/examples/tutorial.ipynb" target="_blank">Google Colab</a>에서 알아보세요.
</p>

## YOLO: 간단한 역사

[YOLO](https://arxiv.org/abs/1506.02640) (You Only Look Once, 단 한 번의 검사)는 워싱턴 대학교의 Joseph Redmon과 Ali Farhadi가 개발한 인기 있는 객체 탐지 및 이미지 분할 모델입니다. 2015년에 출시된 YOLO는 그 빠른 속도와 정확성으로 인해 빠르게 인기를 얻었습니다.

- [YOLOv2](https://arxiv.org/abs/1612.08242)는 2016년에 공개되었으며 배치 정규화, 앵커 박스, 차원 클러스터를 통합하여 원본 모델을 개선했습니다.
- [YOLOv3](https://pjreddie.com/media/files/papers/YOLOv3.pdf)는 2018년에 출시되어 더 효율적인 백본 네트워크, 복수 앵커 및 공간 피라미드 풀링을 사용하여 모델의 성능을 더욱 향상시켰습니다.
- [YOLOv4](https://arxiv.org/abs/2004.10934)는 2020년에 나와서 모자이크 데이터 증가, 새로운 앵커-프리 탐지 헤드, 새로운 손실 함수와 같은 혁신을 도입했습니다.
- [YOLOv5](https://github.com/ultralytics/yolov5)는 모델의 성능을 더욱 향상시키고 하이퍼파라미터 최적화, 통합 실험 추적 및 인기 있는 수출 형식으로의 자동 수출과 같은 새로운 기능을 추가했습니다.
- [YOLOv6](https://github.com/meituan/YOLOv6)는 2022년에 [Meituan](https://about.meituan.com/)에 의해 오픈 소스화되었으며, 이 회사의 자율 배달 로봇에서 사용되고 있습니다.
- [YOLOv7](https://github.com/WongKinYiu/yolov7)는 COCO 키포인트 데이터셋에서의 자세 추정과 같은 추가 작업을 추가했습니다.
- [YOLOv8](https://github.com/ultralytics/ultralytics)은 Ultralytics에서 출시한 YOLO의 최신 버전입니다. 첨단 상태 기술 모델로서, YOLOv8은 이전 버전들의 성공을 기반으로 새로운 기능과 개선 사항을 도입하여 성능, 유연성 및 효율성을 향상시켰습니다. YOLOv8은 [탐지](tasks/detect.md), [분할](tasks/segment.md), [자세 추정](tasks/pose.md), [추적](modes/track.md), [분류](tasks/classify.md)를 포함하여 다양한 비전 AI 작업을 지원합니다. 이러한 다재다능함은 사용자들이 다양한 애플리케이션과 도메인 전반에 걸쳐 YOLOv8의 능력을 활용할 수 있도록 합니다.

## YOLO 라이센스: Ultralytics YOLO는 어떻게 라이센스가 부여되나요?

Ultralytics는 다양한 사용 사례에 맞춰 두 가지 라이선스 옵션을 제공합니다:

- **AGPL-3.0 라이선스**: 이 [OSI 승인](https://opensource.org/licenses/) 오픈 소스 라이선스는 학생 및 애호가에게 이상적입니다. 오픈 협력과 지식 공유를 촉진합니다. 자세한 내용은 [라이선스](https://github.com/ultralytics/ultralytics/blob/main/LICENSE) 파일을 참조하세요.
- **기업 라이선스**: 상업적 사용을 위해 설계된 이 라이선스는 Ultralytics 소프트웨어 및 AI 모델을 상업적 제품 및 서비스에 원활하게 통합할 수 있게 하여 AGPL-3.0의 오픈 소스 요건을 우회할 수 있습니다. 상업적 제공물에 솔루션을 내장하는 시나리오에 관여하는 경우 [Ultralytics 라이선싱](https://ultralytics.com/license)을 통해 문의하시기 바랍니다.

우리의 라이선스 전략은 오픈 소스 프로젝트에 대한 개선 사항이 커뮤니티에 되돌아가도록 보장하려는 것입니다. 우리는 오픈 소스의 원칙을 가슴 깊이 새기고 있으며, 우리의 기여가 모두에게 유용한 방식으로 활용되고 확장될 수 있도록 보장하는 것이 우리의 사명입니다.❤️
