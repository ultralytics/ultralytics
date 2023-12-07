---
comments: true
description: Ultralytics에서 지원하는 YOLO 계열, SAM, MobileSAM, FastSAM, YOLO-NAS, RT-DETR 모델의 다양한 범위를 탐색하고 CLI 및 Python 사용 예시를 통해 시작해 보세요.
keywords: Ultralytics, 문서화, YOLO, SAM, MobileSAM, FastSAM, YOLO-NAS, RT-DETR, 모델, 아키텍처, Python, CLI
---

# Ultralytics에서 지원하는 모델

Ultralytics 모델 문서에 오신 것을 환영합니다! 저희는 [객체 검출](../tasks/detect.md), [인스턴스 분할](../tasks/segment.md), [이미지 분류](../tasks/classify.md), [자세 추정](../tasks/pose.md), [다중 객체 추적](../modes/track.md) 등 특정 작업에 맞춤화된 다양한 모델을 지원합니다. Ultralytics에 모델 아키텍처를 기여하고자 한다면, 저희의 [기여 가이드](../../help/contributing.md)를 확인해 주세요.

!!! Note "노트"

    🚧 현재 다국어 문서화 작업이 진행 중이며 문서를 개선하기 위해 열심히 작업하고 있습니다. 기다려 주셔서 감사합니다! 🙏

## 주요 모델

여기 몇 가지 주요 모델을 소개합니다:

1. **[YOLOv3](../../models/yolov3.md)**: Joseph Redmon에 의해 처음 제안된 YOLO 모델 계열의 세 번째 버전으로, 효율적인 실시간 객체 검출 능력으로 알려져 있습니다.
2. **[YOLOv4](../../models/yolov4.md)**: 2020년 Alexey Bochkovskiy에 의해 발표된 YOLOv3의 다크넷 기반 업데이트 버전입니다.
3. **[YOLOv5](../../models/yolov5.md)**: Ultralytics에 의해 개선된 YOLO 아키텍처 버전으로, 이전 버전들과 비교해 더 나은 성능 및 속도 저하를 제공합니다.
4. **[YOLOv6](../../models/yolov6.md)**: 2022년 [Meituan](https://about.meituan.com/)에 의해 발표되었으며, 회사의 자율 배송 로봇에 많이 사용되고 있습니다.
5. **[YOLOv7](../../models/yolov7.md)**: YOLOv4의 저자들에 의해 2022년에 발표된 업데이트된 YOLO 모델입니다.
6. **[YOLOv8](../../models/yolov8.md)**: YOLO 계열의 최신 버전으로, 인스턴스 분할, 자세/키포인트 추정 및 분류 등 향상된 기능을 제공합니다.
7. **[Segment Anything Model (SAM)](../../models/sam.md)**: Meta의 Segment Anything Model (SAM)입니다.
8. **[Mobile Segment Anything Model (MobileSAM)](../../models/mobile-sam.md)**: 경희대학교에 의한 모바일 애플리케이션용 MobileSAM입니다.
9. **[Fast Segment Anything Model (FastSAM)](../../models/fast-sam.md)**: 중국 과학원 자동화 연구소의 영상 및 비디오 분석 그룹에 의한 FastSAM입니다.
10. **[YOLO-NAS](../../models/yolo-nas.md)**: YOLO Neural Architecture Search (NAS) 모델입니다.
11. **[Realtime Detection Transformers (RT-DETR)](../../models/rtdetr.md)**: Baidu의 PaddlePaddle Realtime Detection Transformer (RT-DETR) 모델입니다.

<p align="center">
  <br>
  <iframe width="720" height="405" src="https://www.youtube.com/embed/MWq1UxqTClU?si=nHAW-lYDzrz68jR0"
    title="YouTube video player" frameborder="0"
    allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share"
    allowfullscreen>
  </iframe>
  <br>
  <strong>보기:</strong> 몇 줄의 코드로 Ultralytics YOLO 모델을 실행하세요.
</p>

## 시작하기: 사용 예시

!!! Example "예제"

    === "Python"

        PyTorch로 사전훈련된 `*.pt` 모델과 구성 `*.yaml` 파일은 Python에서 `YOLO()`, `SAM()`, `NAS()` 및 `RTDETR()` 클래스에 전달하여 모델 인스턴스를 생성할 수 있습니다:

        ```python
        from ultralytics import YOLO

        # COCO 사전훈련된 YOLOv8n 모델을 로드
        model = YOLO('yolov8n.pt')

        # 모델 정보 표시 (선택 사항)
        model.info()

        # COCO8 예시 데이터셋에서 YOLOv8n 모델로 100 에포크 동안 훈련
        results = model.train(data='coco8.yaml', epochs=100, imgsz=640)

        # 'bus.jpg' 이미지에 YOLOv8n 모델로 추론 실행
        results = model('path/to/bus.jpg')
        ```

    === "CLI"

        모델을 직접 실행하기 위한 CLI 커맨드도 제공됩니다:

        ```bash
        # COCO 사전훈련된 YOLOv8n 모델을 로드하고 COCO8 예시 데이터셋에서 100 에포크 동안 훈련
        yolo train model=yolov8n.pt data=coco8.yaml epochs=100 imgsz=640

        # COCO 사전훈련된 YOLOv8n 모델을 로드하고 'bus.jpg' 이미지에서 추론 실행
        yolo predict model=yolov8n.pt source=path/to/bus.jpg
        ```

## 새 모델 기여하기

Ultralytics에 모델을 기여하고 싶으신가요? 훌륭합니다! 저희는 항상 모델 포트폴리오를 확장하는 것에 개방적입니다.

1. **저장소를 포크하세요**: [Ultralytics GitHub 저장소](https://github.com/ultralytics/ultralytics)를 포크하며 시작하세요.

2. **포크를 클론하세요**: 로컬 기기에 포크한 저장소를 클론하고 작업할 새 브랜치를 만드세요.

3. **모델 구현하기**: 저희의 [기여 가이드](../../help/contributing.md)에 제시된 코딩 표준과 가이드라인을 따라 모델을 추가하세요.

4. **철저히 테스트하기**: 파이프라인의 일부로서 뿐만 아니라 독립적으로도 모델을 철저히 테스트하세요.

5. **풀 리퀘스트 생성하기**: 모델에 만족하게 되면, 리뷰를 위해 메인 저장소로 풀 리퀘스트를 생성하세요.

6. **코드 리뷰 및 병합**: 리뷰 이후, 모델이 저희의 기준을 만족한다면 메인 저장소로 병합될 것입니다.

자세한 단계는 저희의 [기여 가이드](../../help/contributing.md)를 참조하세요.
