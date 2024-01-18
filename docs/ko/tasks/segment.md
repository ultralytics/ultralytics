---
comments: true
description: Ultralytics YOLO를 이용한 인스턴스 세그멘테이션 모델 사용법 배우기. 훈련, 검증, 이미지 예측 및 모델 수출에 대한 지침.
keywords: yolov8, 인스턴스 세그멘테이션, Ultralytics, COCO 데이터셋, 이미지 세그멘테이션, 객체 탐지, 모델 훈련, 모델 검증, 이미지 예측, 모델 수출
---

# 인스턴스 세그멘테이션

<img width="1024" src="https://user-images.githubusercontent.com/26833433/243418644-7df320b8-098d-47f1-85c5-26604d761286.png" alt="인스턴스 세그멘테이션 예시">

인스턴스 세그멘테이션은 객체 탐지를 한 단계 더 발전시켜 이미지에서 각각의 개별 객체를 식별하고 이미지의 나머지 부분에서 분리하는 기술입니다.

인스턴스 세그멘테이션 모델의 출력은 이미지의 각 객체를 윤곽하는 마스크나 윤곽 선뿐만 아니라 각 객체에 대한 클래스 레이블과 신뢰도 점수로 구성됩니다. 객체들이 이미지 안에서 어디에 있는지 뿐만 아니라 그들의 정확한 형태가 무엇인지 알아야 할 때 인스턴스 세그멘테이션이 유용합니다.

<p align="center">
  <br>
  <iframe width="720" height="405" src="https://www.youtube.com/embed/o4Zd-IeMlSY?si=37nusCzDTd74Obsp"
    title="YouTube 비디오 플레이어" frameborder="0"
    allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share"
    allowfullscreen>
  </iframe>
  <br>
  <strong>시청하기:</strong> Python에서 사전 훈련된 Ultralytics YOLOv8 모델로 세그멘테이션 실행.
</p>

!!! Tip "팁"

    YOLOv8 Segment 모델은 '-seg' 접미사를 사용하며 즉, `yolov8n-seg.pt`와 같이 [COCO](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/datasets/coco.yaml) 데이터셋에 사전 훈련되어 있습니다.

## [모델](https://github.com/ultralytics/ultralytics/tree/main/ultralytics/cfg/models/v8)

여기에는 YOLOv8 사전 훈련 세그먼트 모델들이 나열되어 있습니다. Detect, Segment, Pose 모델들은 [COCO](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/datasets/coco.yaml) 데이터셋에 사전 훈련되어 있으며, Classify 모델들은 [ImageNet](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/datasets/ImageNet.yaml) 데이터셋에 사전 훈련되어 있습니다.

[모델](https://github.com/ultralytics/ultralytics/tree/main/ultralytics/cfg/models)은 첫 사용 시 Ultralytics의 최신 [릴리스](https://github.com/ultralytics/assets/releases)에서 자동으로 다운로드 됩니다.

| 모델                                                                                           | 크기<br><sup>(픽셀) | mAP<sup>박스<br>50-95 | mAP<sup>마스크<br>50-95 | 속도<br><sup>CPU ONNX<br>(밀리초) | 속도<br><sup>A100 TensorRT<br>(밀리초) | 매개변수<br><sup>(M) | FLOPs<br><sup>(B) |
|----------------------------------------------------------------------------------------------|-----------------|---------------------|----------------------|------------------------------|-----------------------------------|------------------|-------------------|
| [YOLOv8n-seg](https://github.com/ultralytics/assets/releases/download/v8.1.0/yolov8n-seg.pt) | 640             | 36.7                | 30.5                 | 96.1                         | 1.21                              | 3.4              | 12.6              |
| [YOLOv8s-seg](https://github.com/ultralytics/assets/releases/download/v8.1.0/yolov8s-seg.pt) | 640             | 44.6                | 36.8                 | 155.7                        | 1.47                              | 11.8             | 42.6              |
| [YOLOv8m-seg](https://github.com/ultralytics/assets/releases/download/v8.1.0/yolov8m-seg.pt) | 640             | 49.9                | 40.8                 | 317.0                        | 2.18                              | 27.3             | 110.2             |
| [YOLOv8l-seg](https://github.com/ultralytics/assets/releases/download/v8.1.0/yolov8l-seg.pt) | 640             | 52.3                | 42.6                 | 572.4                        | 2.79                              | 46.0             | 220.5             |
| [YOLOv8x-seg](https://github.com/ultralytics/assets/releases/download/v8.1.0/yolov8x-seg.pt) | 640             | 53.4                | 43.4                 | 712.1                        | 4.02                              | 71.8             | 344.1             |

- **mAP<sup>val</sup>** 값들은 [COCO val2017](https://cocodataset.org) 데이터셋에서 단일 모델 단일 스케일로 얻은 값입니다.
  <br>복제는 `yolo val segment data=coco.yaml device=0` 명령어로 실행할 수 있습니다.
- **속도**는 [Amazon EC2 P4d](https://aws.amazon.com/ec2/instance-types/p4/) 인스턴스를 이용하여 COCO 검증 이미지로 평균 내었습니다.
  <br>복제는 `yolo val segment data=coco128-seg.yaml batch=1 device=0|cpu` 명령어로 실행할 수 있습니다.

## 훈련

COCO128-seg 데이터셋에서 이미지 크기 640으로 YOLOv8n-seg을 100 에포크 동안 훈련합니다. 가능한 모든 인자 목록은 [설정](/../usage/cfg.md) 페이지에서 확인할 수 있습니다.

!!! Example "예제"

    === "파이썬"

        ```python
        from ultralytics import YOLO

        # 모델을 불러옵니다
        model = YOLO('yolov8n-seg.yaml')  # YAML에서 새로운 모델을 구성
        model = YOLO('yolov8n-seg.pt')    # 사전 훈련된 모델을 불러옴 (훈련에 추천)
        model = YOLO('yolov8n-seg.yaml').load('yolov8n.pt') # YAML에서 구성하고 가중치를 전달

        # 모델을 훈련시킵니다
        results = model.train(data='coco128-seg.yaml', epochs=100, imgsz=640)
        ```
    === "CLI"

        ```bash
        # YAML에서 새로운 모델을 구성하고 처음부터 훈련을 시작합니다
        yolo segment train data=coco128-seg.yaml model=yolov8n-seg.yaml epochs=100 imgsz=640

        # 사전 훈련된 *.pt 모델로 부터 훈련을 시작합니다
        yolo segment train data=coco128-seg.yaml model=yolov8n-seg.pt epochs=100 imgsz=640

        # YAML에서 새로운 모델을 구성하고 사전 훈련된 가중치를 전달한 뒤 훈련을 시작합니다
        yolo segment train data=coco128-seg.yaml model=yolov8n-seg.yaml pretrained=yolov8n-seg.pt epochs=100 imgsz=640
        ```

### 데이터셋 형식

YOLO 세그멘테이션 데이터셋 형식은 [데이터셋 가이드](../../../datasets/segment/index.md)에서 자세히 확인할 수 있습니다. 기존 데이터셋 (COCO 등)을 YOLO 형식으로 변환하려면 Ultralytics의 [JSON2YOLO](https://github.com/ultralytics/JSON2YOLO) 도구를 이용하세요.

## 검증

COCO128-seg 데이터셋에서 훈련된 YOLOv8n-seg 모델의 정확도를 검증합니다. 모델은 훈련할 때의 `data`와 인자를 모델 속성으로 기억하기 때문에 별도의 인자를 전달할 필요가 없습니다.

!!! Example "예제"

    === "파이썬"

        ```python
        from ultralytics import YOLO

        # 모델을 불러옵니다
        model = YOLO('yolov8n-seg.pt')    # 공식 모델을 불러옴
        model = YOLO('path/to/best.pt')    # 커스텀 모델을 불러옴

        # 모델을 검증합니다
        metrics = model.val()  # 데이터셋과 설정이 기억되어 있어 인자가 필요 없습니다
        metrics.box.map    # map50-95(B)
        metrics.box.map50  # map50(B)
        metrics.box.map75  # map75(B)
        metrics.box.maps   # 각 카테고리별 map50-95(B) 리스트
        metrics.seg.map    # map50-95(M)
        metrics.seg.map50  # map50(M)
        metrics.seg.map75  # map75(M)
        metrics.seg.maps   # 각 카테고리별 map50-95(M) 리스트
        ```
    === "CLI"

        ```bash
        yolo segment val model=yolov8n-seg.pt  # 공식 모델로 검증
        yolo segment val model=path/to/best.pt  # 커스텀 모델로 검증
        ```

## 예측

훈련된 YOLOv8n-seg 모델을 사용하여 이미지에 대한 예측을 실행합니다.

!!! Example "예제"

    === "파이썬"

        ```python
        from ultralytics import YOLO

        # 모델을 불러옵니다
        model = YOLO('yolov8n-seg.pt')    # 공식 모델을 불러옴
        model = YOLO('path/to/best.pt')    # 커스텀 모델을 불러옴

        # 모델로 예측을 진행합니다
        results = model('https://ultralytics.com/images/bus.jpg')  # 이미지에 대한 예측
        ```
    === "CLI"

        ```bash
        yolo segment predict model=yolov8n-seg.pt source='https://ultralytics.com/images/bus.jpg'  # 공식 모델로 예측 실행
        yolo segment predict model=path/to/best.pt source='https://ultralytics.com/images/bus.jpg'  # 커스텀 모델로 예측 실행
        ```

`predict` 모드의 전체 세부 사항은 [예측](https://docs.ultralytics.com/modes/predict/) 페이지에서 확인할 수 있습니다.

## 수출

ONNX, CoreML 등과 같은 다른 형식으로 YOLOv8n-seg 모델을 수출합니다.

!!! Example "예제"

    === "파이썬"

        ```python
        from ultralytics import YOLO

        # 모델을 불러옵니다
        model = YOLO('yolov8n-seg.pt')    # 공식 모델을 불러옴
        model = YOLO('path/to/best.pt')    # 커스텀 훈련 모델을 불러옴

        # 모델을 수출합니다
        model.export(format='onnx')
        ```
    === "CLI"

        ```bash
        yolo export model=yolov8n-seg.pt format=onnx  # 공식 모델을 수출합니다
        yolo export model=path/to/best.pt format=onnx  # 커스텀 훈련 모델을 수출합니다
        ```

아래 표에 나열된 것은 가능한 YOLOv8-seg 수출 형식입니다. 수출 완료 후 모델 사용 예는 모델을 직접 예측하거나 검증할 때 사용할 수 있습니다.

| 형식                                                                 | `format` 인자   | 모델                            | 메타데이터 | 인자                                                  |
|--------------------------------------------------------------------|---------------|-------------------------------|-------|-----------------------------------------------------|
| [PyTorch](https://pytorch.org/)                                    | -             | `yolov8n-seg.pt`              | ✅     | -                                                   |
| [TorchScript](https://pytorch.org/docs/stable/jit.html)            | `torchscript` | `yolov8n-seg.torchscript`     | ✅     | `imgsz`, `optimize`                                 |
| [ONNX](https://onnx.ai/)                                           | `onnx`        | `yolov8n-seg.onnx`            | ✅     | `imgsz`, `half`, `dynamic`, `simplify`, `opset`     |
| [OpenVINO](https://docs.openvino.ai/latest/index.html)             | `openvino`    | `yolov8n-seg_openvino_model/` | ✅     | `imgsz`, `half`                                     |
| [TensorRT](https://developer.nvidia.com/tensorrt)                  | `engine`      | `yolov8n-seg.engine`          | ✅     | `imgsz`, `half`, `dynamic`, `simplify`, `workspace` |
| [CoreML](https://github.com/apple/coremltools)                     | `coreml`      | `yolov8n-seg.mlpackage`       | ✅     | `imgsz`, `half`, `int8`, `nms`                      |
| [TF SavedModel](https://www.tensorflow.org/guide/saved_model)      | `saved_model` | `yolov8n-seg_saved_model/`    | ✅     | `imgsz`, `keras`                                    |
| [TF GraphDef](https://www.tensorflow.org/api_docs/python/tf/Graph) | `pb`          | `yolov8n-seg.pb`              | ❌     | `imgsz`                                             |
| [TF Lite](https://www.tensorflow.org/lite)                         | `tflite`      | `yolov8n-seg.tflite`          | ✅     | `imgsz`, `half`, `int8`                             |
| [TF Edge TPU](https://coral.ai/docs/edgetpu/models-intro/)         | `edgetpu`     | `yolov8n-seg_edgetpu.tflite`  | ✅     | `imgsz`                                             |
| [TF.js](https://www.tensorflow.org/js)                             | `tfjs`        | `yolov8n-seg_web_model/`      | ✅     | `imgsz`                                             |
| [PaddlePaddle](https://github.com/PaddlePaddle)                    | `paddle`      | `yolov8n-seg_paddle_model/`   | ✅     | `imgsz`                                             |
| [ncnn](https://github.com/Tencent/ncnn)                            | `ncnn`        | `yolov8n-seg_ncnn_model/`     | ✅     | `imgsz`, `half`                                     |

`export`의 전체 세부 사항은 [수출](https://docs.ultralytics.com/modes/export/) 페이지에서 확인할 수 있습니다.
