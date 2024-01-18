---
comments: true
description: YOLOv8는 실시간 객체 탐지기인 YOLO 시리즈의 최신 버전으로, 최신 아키텍처, 사전 훈련된 모델 및 정확도와 속도의 최적 균형을 제공함으로써 다양한 객체 탐지 작업에 적합한 선택지입니다.
keywords: YOLOv8, Ultralytics, 실시간 객체 탐지기, 사전 훈련된 모델, 문서, 객체 탐지, YOLO 시리즈, 고급 아키텍처, 정확도, 속도
---

# YOLOv8

## 개요

YOLOv8는 실시간 객체 탐지기인 YOLO 시리즈의 최신 버전으로, 정확도와 속도면에서 최신 기술을 제공합니다. 이전 YOLO 버전의 기술적 발전을 바탕으로, YOLOv8은 새로운 기능과 최적화를 도입하여 다양한 응용 분야에서 다양한 객체 탐지 작업에 이상적인 선택지가 됩니다.

![Ultralytics YOLOv8](https://raw.githubusercontent.com/ultralytics/assets/main/yolov8/yolo-comparison-plots.png)

## 주요 기능

- **고급 백본 및 넥 아키텍처:** YOLOv8은 최첨단 백본 및 넥 아키텍처를 사용하여 향상된 특징 추출과 객체 탐지 성능을 제공합니다.
- **앵커 없는 분리된 Ultralytics 헤드:** YOLOv8은 앵커 기반 접근법에 비해 더 나은 정확도와 효율적인 탐지 과정을 위한 앵커 없는 분리된 Ultralytics 헤드를 채택합니다.
- **정확도와 속도의 최적 균형화:** 정확도와 속도의 최적 균형을 유지하는 것에 초점을 맞춘 YOLOv8은 다양한 응용 분야에서 실시간 객체 탐지 작업에 적합합니다.
- **다양한 사전 훈련된 모델:** YOLOv8은 다양한 작업과 성능 요건에 맞는 사전 훈련된 모델을 제공하여 특정 사용 사례에 적합한 모델을 쉽게 찾을 수 있도록 합니다.

## 지원하는 작업 및 모드

YOLOv8 시리즈는 컴퓨터 비전의 특정 작업을 위해 특화된 다양한 모델을 제공합니다. 이러한 모델은 객체 탐지부터 인스턴스 세그멘테이션, 포즈/키포인트 탐지, 분류와 같은 복잡한 작업까지 다양한 요구 사항을 충족시킬 수 있도록 설계되었습니다.

각 YOLOv8 시리즈의 변형은 해당 작업에 최적화되어 고성능과 정확도를 제공합니다. 또한 이러한 모델은 [추론](../modes/predict.md), [검증](../modes/val.md), [훈련](../modes/train.md) 및 [내보내기](../modes/export.md)와 같은 다양한 운영 모드와 호환되어 배포 및 개발의 다양한 단계에서 사용할 수 있도록 지원합니다.

| 모델          | 파일 이름                                                                                                          | 작업                                 | 추론 | 검증 | 훈련 | 내보내기 |
|-------------|----------------------------------------------------------------------------------------------------------------|------------------------------------|----|----|----|------|
| YOLOv8      | `yolov8n.pt` `yolov8s.pt` `yolov8m.pt` `yolov8l.pt` `yolov8x.pt`                                               | [객체 탐지](../tasks/detect.md)        | ✅  | ✅  | ✅  | ✅    |
| YOLOv8-seg  | `yolov8n-seg.pt` `yolov8s-seg.pt` `yolov8m-seg.pt` `yolov8l-seg.pt` `yolov8x-seg.pt`                           | [인스턴스 세그멘테이션](../tasks/segment.md) | ✅  | ✅  | ✅  | ✅    |
| YOLOv8-pose | `yolov8n-pose.pt` `yolov8s-pose.pt` `yolov8m-pose.pt` `yolov8l-pose.pt` `yolov8x-pose.pt` `yolov8x-pose-p6.pt` | [포즈/키포인트](../tasks/pose.md)        | ✅  | ✅  | ✅  | ✅    |
| YOLOv8-cls  | `yolov8n-cls.pt` `yolov8s-cls.pt` `yolov8m-cls.pt` `yolov8l-cls.pt` `yolov8x-cls.pt`                           | [분류](../tasks/classify.md)         | ✅  | ✅  | ✅  | ✅    |

이 표는 YOLOv8 모델의 변형을 개요로 제공하며, 특정 작업에서의 적용 가능성 및 추론, 검증, 훈련 및 내보내기와 같은 다양한 운영 모드와의 호환성을 강조합니다. 이는 컴퓨터 비전의 다양한 응용 분야에 적합한 YOLOv8 시리즈의 다재다능함과 견고성을 보여줍니다.

## 성능 지표

!!! 성능

    === "객체 탐지 (COCO)"

        [COCO](https://docs.ultralytics.com/datasets/detect/coco/)에서 사전 훈련된 이러한 모델을 사용한 사용 예제에 대한 정보는 [객체 탐지 문서](https://docs.ultralytics.com/tasks/detect/)를 참조하십시오. 이는 80개의 사전 훈련된 클래스를 포함합니다.

        | 모델                                                                                | 크기<br><sup>(픽셀) | mAP<br>val 50-95    | 속도<br><sup>CPU ONNX<br>(밀리초) | 속도<br><sup>A100 TensorRT<br>(밀리초) | 매개변수<br><sup>(백만) | FLOPs<br><sup>(십억) |
        | ------------------------------------------------------------------------------------ | --------------------- | -------------------- | ------------------------------ | ----------------------------------- | ------------------- | ----------------- |
        | [YOLOv8n](https://github.com/ultralytics/assets/releases/download/v8.1.0/yolov8n.pt) | 640                   | 37.3                 | 80.4                           | 0.99                                | 3.2                 | 8.7               |
        | [YOLOv8s](https://github.com/ultralytics/assets/releases/download/v8.1.0/yolov8s.pt) | 640                   | 44.9                 | 128.4                          | 1.20                                | 11.2                | 28.6              |
        | [YOLOv8m](https://github.com/ultralytics/assets/releases/download/v8.1.0/yolov8m.pt) | 640                   | 50.2                 | 234.7                          | 1.83                                | 25.9                | 78.9              |
        | [YOLOv8l](https://github.com/ultralytics/assets/releases/download/v8.1.0/yolov8l.pt) | 640                   | 52.9                 | 375.2                          | 2.39                                | 43.7                | 165.2             |
        | [YOLOv8x](https://github.com/ultralytics/assets/releases/download/v8.1.0/yolov8x.pt) | 640                   | 53.9                 | 479.1                          | 3.53                                | 68.2                | 257.8             |

    === "객체 탐지 (Open Images V7)"

        [Open Images V7](https://docs.ultralytics.com/datasets/detect/open-images-v7/)에서 사전 훈련된 이러한 모델을 사용한 사용 예제에 대한 정보는 [객체 탐지 문서](https://docs.ultralytics.com/tasks/detect/)를 참조하십시오. 이는 600개의 사전 훈련된 클래스를 포함합니다.

        | 모델                                                                                     | 크기<br><sup>(픽셀) | mAP<br>val 50-95    | 속도<br><sup>CPU ONNX<br>(밀리초) | 속도<br><sup>A100 TensorRT<br>(밀리초) | 매개변수<br><sup>(백만) | FLOPs<br><sup>(십억) |
        | ----------------------------------------------------------------------------------------- | --------------------- | -------------------- | ------------------------------ | ----------------------------------- | ------------------- | ----------------- |
        | [YOLOv8n-oiv7](https://github.com/ultralytics/assets/releases/download/v8.1.0/yolov8n-oiv7.pt) | 640                   | 18.4                 | 142.4                          | 1.21                                | 3.5                 | 10.5              |
        | [YOLOv8s-oiv7](https://github.com/ultralytics/assets/releases/download/v8.1.0/yolov8s-oiv7.pt) | 640                   | 27.7                 | 183.1                          | 1.40                                | 11.4                | 29.7              |
        | [YOLOv8m-oiv7](https://github.com/ultralytics/assets/releases/download/v8.1.0/yolov8m-oiv7.pt) | 640                   | 33.6                 | 408.5                          | 2.26                                | 26.2                | 80.6              |
        | [YOLOv8l-oiv7](https://github.com/ultralytics/assets/releases/download/v8.1.0/yolov8l-oiv7.pt) | 640                   | 34.9                 | 596.9                          | 2.43                                | 44.1                | 167.4             |
        | [YOLOv8x-oiv7](https://github.com/ultralytics/assets/releases/download/v8.1.0/yolov8x-oiv7.pt) | 640                   | 36.3                 | 860.6                          | 3.56                                | 68.7                | 260.6             |

    === "세그멘테이션 (COCO)"

        [COCO](https://docs.ultralytics.com/datasets/segment/coco/)에서 사전 훈련된 이러한 모델을 사용한 사용 예제에 대한 정보는 [세그멘테이션 문서](https://docs.ultralytics.com/tasks/segment/)를 참조하십시오. 이는 80개의 사전 훈련된 클래스를 포함합니다.

        | 모델                                                                                        | 크기<br><sup>(픽셀) | mAP<br>box 50-95    | mAP<br>mask 50-95 | 속도<br><sup>CPU ONNX<br>(밀리초) | 속도<br><sup>A100 TensorRT<br>(밀리초) | 매개변수<br><sup>(백만) | FLOPs<br><sup>(십억) |
        | -------------------------------------------------------------------------------------------- | --------------------- | -------------------- | ------------------ | ------------------------------ | ----------------------------------- | ------------------- | ----------------- |
        | [YOLOv8n-seg](https://github.com/ultralytics/assets/releases/download/v8.1.0/yolov8n-seg.pt) | 640                   | 36.7                 | 30.5               | 96.1                           | 1.21                                | 3.4                 | 12.6              |
        | [YOLOv8s-seg](https://github.com/ultralytics/assets/releases/download/v8.1.0/yolov8s-seg.pt) | 640                   | 44.6                 | 36.8               | 155.7                          | 1.47                                | 11.8                | 42.6              |
        | [YOLOv8m-seg](https://github.com/ultralytics/assets/releases/download/v8.1.0/yolov8m-seg.pt) | 640                   | 49.9                 | 40.8               | 317.0                          | 2.18                                | 27.3                | 110.2             |
        | [YOLOv8l-seg](https://github.com/ultralytics/assets/releases/download/v8.1.0/yolov8l-seg.pt) | 640                   | 52.3                 | 42.6               | 572.4                          | 2.79                                | 46.0                | 220.5             |
        | [YOLOv8x-seg](https://github.com/ultralytics/assets/releases/download/v8.1.0/yolov8x-seg.pt) | 640                   | 53.4                 | 43.4               | 712.1                          | 4.02                                | 71.8                | 344.1             |

    === "분류 (ImageNet)"

        [ImageNet](https://docs.ultralytics.com/datasets/classify/imagenet/)에서 사전 훈련된 이러한 모델을 사용한 사용 예제에 대한 정보는 [분류 문서](https://docs.ultralytics.com/tasks/classify/)를 참조하십시오. 이는 1000개의 사전 훈련된 클래스를 포함합니다.

        | 모델                                                                                        | 크기<br><sup>(픽셀) | top1 정확도        | top5 정확도       | 속도<br><sup>CPU ONNX<br>(밀리초) | 속도<br><sup>A100 TensorRT<br>(밀리초) | 매개변수<br><sup>(백만) | FLOPs<br><sup>(백만) at 640 |
        | -------------------------------------------------------------------------------------------- | --------------------- | -------------------- | ----------------- | ------------------------------ | ----------------------------------- | ------------------- | ------------------------ |
        | [YOLOv8n-cls](https://github.com/ultralytics/assets/releases/download/v8.1.0/yolov8n-cls.pt) | 224                   | 66.6               | 87.0              | 12.9                           | 0.31                                | 2.7                 | 4.3                      |
        | [YOLOv8s-cls](https://github.com/ultralytics/assets/releases/download/v8.1.0/yolov8s-cls.pt) | 224                   | 72.3               | 91.1              | 23.4                           | 0.35                                | 6.4                 | 13.5                     |
        | [YOLOv8m-cls](https://github.com/ultralytics/assets/releases/download/v8.1.0/yolov8m-cls.pt) | 224                   | 76.4               | 93.2              | 85.4                           | 0.62                                | 17.0                | 42.7                     |
        | [YOLOv8l-cls](https://github.com/ultralytics/assets/releases/download/v8.1.0/yolov8l-cls.pt) | 224                   | 78.0               | 94.1              | 163.0                          | 0.87                                | 37.5                | 99.7                     |
        | [YOLOv8x-cls](https://github.com/ultralytics/assets/releases/download/v8.1.0/yolov8x-cls.pt) | 224                   | 78.4               | 94.3              | 232.0                          | 1.01                                | 57.4                | 154.8                    |

    === "포즈 (COCO)"

        [COCO](https://docs.ultralytics.com/datasets/pose/coco/)에서 사전 훈련된 이러한 모델을 사용한 사용 예제에 대한 정보는 [포즈 문서](https://docs.ultralytics.com/tasks/pose/)를 참조하십시오. 이는 'person' 하나의 사전 훈련된 클래스를 포함합니다.

        | 모델                                                                                                | 크기<br><sup>(픽셀) | mAP<br>pose 50-95    | mAP<br>pose 50   | 속도<br><sup>CPU ONNX<br>(밀리초) | 속도<br><sup>A100 TensorRT<br>(밀리초) | 매개변수<br><sup>(백만) | FLOPs<br><sup>(십억) |
        | ---------------------------------------------------------------------------------------------------- | --------------------- | -------------------- | ----------------- | ------------------------------ | ----------------------------------- | ------------------- | ----------------- |
        | [YOLOv8n-pose](https://github.com/ultralytics/assets/releases/download/v8.1.0/yolov8n-pose.pt)       | 640                   | 50.4                 | 80.1              | 131.8                           | 1.18                                | 3.3                 | 9.2               |
        | [YOLOv8s-pose](https://github.com/ultralytics/assets/releases/download/v8.1.0/yolov8s-pose.pt)       | 640                   | 60.0                 | 86.2              | 233.2                           | 1.42                                | 11.6                | 30.2              |
        | [YOLOv8m-pose](https://github.com/ultralytics/assets/releases/download/v8.1.0/yolov8m-pose.pt)       | 640                   | 65.0                 | 88.8              | 456.3                           | 2.00                                | 26.4                | 81.0              |
        | [YOLOv8l-pose](https://github.com/ultralytics/assets/releases/download/v8.1.0/yolov8l-pose.pt)       | 640                   | 67.6                 | 90.0              | 784.5                           | 2.59                                | 44.4                | 168.6             |
        | [YOLOv8x-pose](https://github.com/ultralytics/assets/releases/download/v8.1.0/yolov8x-pose.pt)       | 640                   | 69.2                 | 90.2              | 1607.1                          | 3.73                                | 69.4                | 263.2             |
        | [YOLOv8x-pose-p6](https://github.com/ultralytics/assets/releases/download/v8.1.0/yolov8x-pose-p6.pt) | 1280                  | 71.6                 | 91.2              | 4088.7                          | 10.04                               | 99.1                | 1066.4            |

## 사용 예제

이 예제는 간단한 YOLOv8 훈련 및 추론 예제를 제공합니다. 이와 다른 [모드](../modes/index.md)의 전체 문서는 [예측](../modes/predict.md), [훈련](../modes/train.md), [검증](../modes/val.md) 및 [내보내기](../modes/export.md) 문서를 참조하십시오.

아래 예제는 객체 탐지를 위한 YOLOv8 [Detect](../tasks/detect.md) 모델을 대상으로 합니다. 추가 지원되는 작업에 대해서는 [Segment](../tasks/segment.md), [Classify](../tasks/classify.md) 및 [Pose](../tasks/pose.md) 문서를 참조하십시오.

!!! 예시

    === "Python"

        python에서 PyTorch 사전 훈련된 `*.pt` 모델 및 구성 `*.yaml` 파일을 `YOLO()` 클래스에 전달하여 모델 인스턴스를 만들 수 있습니다.

        ```python
        from ultralytics import YOLO

        # COCO 사전 훈련된 YOLOv8n 모델 로드
        model = YOLO('yolov8n.pt')

        # 모델 정보 표시 (옵션)
        model.info()

        # COCO8 예제 데이터셋에서 모델을 100회 에포크로 훈련
        results = model.train(data='coco8.yaml', epochs=100, imgsz=640)

        # 'bus.jpg' 이미지에서 YOLOv8n 모델로 추론 실행
        results = model('path/to/bus.jpg')
        ```

    === "CLI"

        CLI 명령을 사용하면 직접 모델을 실행할 수 있습니다.

        ```bash
        # COCO 사전 훈련된 YOLOv8n 모델을 로드하고 COCO8 예제 데이터셋에서 100회 에포크로 훈련
        yolo train model=yolov8n.pt data=coco8.yaml epochs=100 imgsz=640

        # COCO 사전 훈련된 YOLOv8n 모델을 로드하고 'bus.jpg' 이미지에서 추론 실행
        yolo predict model=yolov8n.pt source=path/to/bus.jpg
        ```

## 인용 및 감사의 글

이 저장소에서 YOLOv8 모델이나 다른 소프트웨어를 사용한 작업에 대해서는 다음 형식으로 인용해주시기 바랍니다.

!!! Quote ""

    === "BibTeX"

        ```bibtex
        @software{yolov8_ultralytics,
          author = {Glenn Jocher and Ayush Chaurasia and Jing Qiu},
          title = {Ultralytics YOLOv8},
          version = {8.0.0},
          year = {2023},
          url = {https://github.com/ultralytics/ultralytics},
          orcid = {0000-0001-5950-6979, 0000-0002-7603-6750, 0000-0003-3783-7069},
          license = {AGPL-3.0}
        }
        ```

DOI는 준비 중이며 이용 가능해지면 인용문에 추가될 예정입니다. YOLOv8 모델은 [AGPL-3.0](https://github.com/ultralytics/ultralytics/blob/main/LICENSE) 및 [Enterprise](https://ultralytics.com/license) 라이센스에 따라 제공됩니다.
