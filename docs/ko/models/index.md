---
comments: true
description: Ultralytics가 지원하는 다양한 YOLO 계열 모델, SAM, MobileSAM, FastSAM, YOLO-NAS, RT-DETR에 대해 알아보고 CLI와 Python 사용 예제를 통해 시작하세요.
keywords: Ultralytics, 문서화, YOLO, SAM, MobileSAM, FastSAM, YOLO-NAS, RT-DETR, 모델, 아키텍처, Python, CLI
---

# Ultralytics가 지원하는 모델들

Ultralytics 모델 문서에 오신 것을 환영합니다! 우리는 [객체 감지](../tasks/detect.md), [인스턴스 분할](../tasks/segment.md), [이미지 분류](../tasks/classify.md), [자세 추정](../tasks/pose.md), [다중 객체 추적](../modes/track.md)과 같은 특정 작업에 맞춰진 다양한 범위의 모델을 지원합니다. Ultralytics에 모델 아키텍처를 기여하고 싶다면, [기여 가이드](../../help/contributing.md)를 확인해 보세요.

!!! Note "주의사항"

    🚧 현재 다양한 언어로 된 문서 작업이 진행 중이며, 이를 개선하기 위해 열심히 노력하고 있습니다. 인내해 주셔서 감사합니다! 🙏

## 주요 모델들

다음은 지원되는 핵심 모델 목록입니다:

1. **[YOLOv3](yolov3.md)**: Joseph Redmon에 의해 최초로 만들어진 YOLO 모델 패밀리의 세 번째 버전으로, 효율적인 실시간 객체 감지 능력으로 알려져 있습니다.
2. **[YOLOv4](yolov4.md)**: 2020년 Alexey Bochkovskiy가 발표한 YOLOv3의 다크넷 기반 업데이트 버전입니다.
3. **[YOLOv5](yolov5.md)**: Ultralytics에 의해 향상된 YOLO 아키텍처로, 이전 버전들에 비해 더 나은 성능과 속도 트레이드오프를 제공합니다.
4. **[YOLOv6](yolov6.md)**: [미투안](https://about.meituan.com/)에서 2022년에 발표하여, 회사의 자율 주행 배달 로봇에서 많이 사용되고 있습니다.
5. **[YOLOv7](yolov7.md)**: YOLOv4의 저자들에 의해 2022년에 업데이트된 YOLO 모델들입니다.
6. **[YOLOv8](yolov8.md) 새로운 🚀**: YOLO 패밀리의 최신 버전으로, 인스턴스 분할, 자세/키포인트 추정, 분류 등 향상된 기능을 제공합니다.
7. **[Segment Anything Model (SAM)](sam.md)**: 메타의 Segment Anything Model (SAM)입니다.
8. **[Mobile Segment Anything Model (MobileSAM)](mobile-sam.md)**: 경희대학교에서 모바일 어플리케이션을 위해 개발한 MobileSAM입니다.
9. **[Fast Segment Anything Model (FastSAM)](fast-sam.md)**: 중국 과학원 자동화 연구소의 이미지 및 비디오 분석 그룹에 의해 개발된 FastSAM입니다.
10. **[YOLO-NAS](yolo-nas.md)**: YOLO Neural Architecture Search (NAS) 모델들입니다.
11. **[Realtime Detection Transformers (RT-DETR)](rtdetr.md)**: 바이두의 PaddlePaddle Realtime Detection Transformer (RT-DETR) 모델들입니다.

<p align="center">
  <br>
  <iframe width="720" height="405" src="https://www.youtube.com/embed/MWq1UxqTClU?si=nHAW-lYDzrz68jR0"
    title="YouTube video player" frameborder="0"
    allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share"
    allowfullscreen>
  </iframe>
  <br>
  <strong>시청하기:</strong> 몇 줄의 코드로 Ultralytics YOLO 모델을 실행하세요.
</p>

## 시작하기: 사용 예제

이 예제는 YOLO 학습과 추론에 대한 간단한 예제를 제공합니다. 이에 대한 전체 문서는 [예측](../modes/predict.md), [학습](../modes/train.md), [검증](../modes/val.md), [내보내기](../modes/export.md) 문서 페이지에서 확인할 수 있습니다.

아래 예제는 객체 감지를 위한 YOLOv8 [감지](../tasks/detect.md) 모델에 대한 것입니다. 추가적으로 지원되는 작업들은 [분할](../tasks/segment.md), [분류](../tasks/classify.md), [자세](../tasks/pose.md) 문서를 참조하세요.

!!! Example "예제"

    === "Python"

        PyTorch로 사전 학습된 `*.pt` 모델들과 구성 `*.yaml` 파일들은 `YOLO()`, `SAM()`, `NAS()`, `RTDETR()` 클래스에 전달하여 파이썬에서 모델 인스턴스를 생성할 수 있습니다:

        ```python
        from ultralytics import YOLO

        # COCO로 사전 학습된 YOLOv8n 모델 불러오기
        model = YOLO('yolov8n.pt')

        # 모델 정보 표시 (선택사항)
        model.info()

        # COCO8 예제 데이터셋에 대해 100 에포크 동안 모델 학습
        results = model.train(data='coco8.yaml', epochs=100, imgsz=640)

        # 'bus.jpg' 이미지에 대한 YOLOv8n 모델 추론 실행
        results = model('path/to/bus.jpg')
        ```

    === "CLI"

        모델을 직접 실행하기 위한 CLI 명령어가 제공됩니다:

        ```bash
        # COCO로 사전 학습된 YOLOv8n 모델을 불러와 COCO8 예제 데이터셋에서 100 에포크 동안 학습
        yolo train model=yolov8n.pt data=coco8.yaml epochs=100 imgsz=640

        # COCO로 사전 학습된 YOLOv8n 모델을 불러와 'bus.jpg' 이미지에 대한 추론 실행
        yolo predict model=yolov8n.pt source=path/to/bus.jpg
        ```

## 새로운 모델 기여하기

Ultralytics에 여러분의 모델을 기여하고 싶으신가요? 훌륭합니다! 우리는 항상 모델 포트폴리오를 확장하는 것에 열려 있습니다.

1. **저장소 포크하기**: [Ultralytics GitHub 저장소](https://github.com/ultralytics/ultralytics)를 포크하여 시작합니다.

2. **포크 복제하기**: 포크한 저장소를 로컬 기계에 복제하고 새로운 브랜치를 생성하여 작업합니다.

3. **모델 구현하기**: 우리의 [기여 가이드](../../help/contributing.md)에 제공된 코딩 표준 및 가이드라인을 따라 모델을 추가합니다.

4. **철저히 테스트하기**: 독립적으로뿐만 아니라 파이프라인의 일부로도 모델을 철저히 테스트해야 합니다.

5. **풀 리퀘스트 생성하기**: 모델에 만족하게 되면, 리뷰를 위해 메인 저장소에 풀 리퀘스트를 생성합니다.

6. **코드 리뷰 & 병합**: 리뷰 후, 여러분의 모델이 우리 기준에 부합한다면 메인 저장소에 병합됩니다.

자세한 단계는 [기여 가이드](../../help/contributing.md)를 참조해주십시오.
