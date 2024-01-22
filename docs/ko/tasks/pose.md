---
comments: true
description: Ultralytics YOLOv8을 사용하여 포즈 추정 작업을 수행하는 방법을 알아보세요. 미리 학습된 모델을 찾고, 학습, 검증, 예측, 내보내기 등을 진행하는 방법을 배울 수 있습니다.
keywords: Ultralytics, YOLO, YOLOv8, 포즈 추정, 키포인트 검출, 객체 검출, 미리 학습된 모델, 기계 학습, 인공 지능
---

# 포즈 추정

<img width="1024" src="https://user-images.githubusercontent.com/26833433/243418616-9811ac0b-a4a7-452a-8aba-484ba32bb4a8.png" alt="포즈 추정 예시">

포즈 추정은 이미지 내 특정 점들의 위치를 식별하는 작업입니다. 이러한 점들은 보통 관절, 표식, 또는 기타 구별 가능한 특징으로 나타나는 키포인트입니다. 키포인트의 위치는 대개 2D `[x, y]` 또는 3D `[x, y, visible]` 좌표의 집합으로 표현됩니다.

포즈 추정 모델의 출력은 이미지 속 객체 상의 키포인트를 나타내는 점들의 집합과 각 점의 신뢰도 점수를 포함합니다. 포즈 추정은 장면 속 객체의 구체적인 부분을 식별하고, 서로 관련된 위치를 파악해야 할 때 좋은 선택입니다.

<p align="center">
  <br>
  <iframe width="720" height="405" src="https://www.youtube.com/embed/Y28xXQmju64?si=pCY4ZwejZFu6Z4kZ"
    title="YouTube video player" frameborder="0"
    allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share"
    allowfullscreen>
  </iframe>
  <br>
  <strong>시청하기:</strong> Ultralytics YOLOv8을 이용한 포즈 추정.
</p>

!!! Tip "팁"

    YOLOv8 _pose_ 모델은 `-pose` 접미사가 붙습니다. 예: `yolov8n-pose.pt`. 이 모델들은 [COCO keypoints](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/datasets/coco-pose.yaml) 데이터셋으로 학습되었으며 포즈 추정 작업에 적합합니다.

## [모델](https://github.com/ultralytics/ultralytics/tree/main/ultralytics/cfg/models/v8)

여기에 표시된 YOLOv8 미리 학습된 포즈 모델을 확인하세요. Detect, Segment 및 Pose 모델은 [COCO](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/datasets/coco.yaml) 데이터셋으로 미리 학습되며, Classify 모델은 [ImageNet](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/datasets/ImageNet.yaml) 데이터셋으로 미리 학습됩니다.

[모델](https://github.com/ultralytics/ultralytics/tree/main/ultralytics/cfg/models)은 첫 사용 시 Ultralytics [릴리스](https://github.com/ultralytics/assets/releases)에서 자동으로 다운로드됩니다.

| 모델                                                                                                   | 크기<br><sup>(픽셀) | mAP<sup>포즈<br>50-95 | mAP<sup>포즈<br>50 | 속도<br><sup>CPU ONNX<br>(ms) | 속도<br><sup>A100 TensorRT<br>(ms) | 파라미터<br><sup>(M) | FLOPs<br><sup>(B) |
|------------------------------------------------------------------------------------------------------|-----------------|---------------------|------------------|-----------------------------|----------------------------------|------------------|-------------------|
| [YOLOv8n-pose](https://github.com/ultralytics/assets/releases/download/v8.1.0/yolov8n-pose.pt)       | 640             | 50.4                | 80.1             | 131.8                       | 1.18                             | 3.3              | 9.2               |
| [YOLOv8s-pose](https://github.com/ultralytics/assets/releases/download/v8.1.0/yolov8s-pose.pt)       | 640             | 60.0                | 86.2             | 233.2                       | 1.42                             | 11.6             | 30.2              |
| [YOLOv8m-pose](https://github.com/ultralytics/assets/releases/download/v8.1.0/yolov8m-pose.pt)       | 640             | 65.0                | 88.8             | 456.3                       | 2.00                             | 26.4             | 81.0              |
| [YOLOv8l-pose](https://github.com/ultralytics/assets/releases/download/v8.1.0/yolov8l-pose.pt)       | 640             | 67.6                | 90.0             | 784.5                       | 2.59                             | 44.4             | 168.6             |
| [YOLOv8x-pose](https://github.com/ultralytics/assets/releases/download/v8.1.0/yolov8x-pose.pt)       | 640             | 69.2                | 90.2             | 1607.1                      | 3.73                             | 69.4             | 263.2             |
| [YOLOv8x-pose-p6](https://github.com/ultralytics/assets/releases/download/v8.1.0/yolov8x-pose-p6.pt) | 1280            | 71.6                | 91.2             | 4088.7                      | 10.04                            | 99.1             | 1066.4            |

- **mAP<sup>val</sup>** 값은 [COCO Keypoints val2017](https://cocodataset.org) 데이터셋에서 단일 모델 단일 규모를 기준으로 합니다.
  <br>재현하려면 `yolo val pose data=coco-pose.yaml device=0`을 사용하세요.
- **속도**는 [Amazon EC2 P4d](https://aws.amazon.com/ec2/instance-types/p4/) 인스턴스를 사용하여 COCO val 이미지 평균입니다.
  <br>재현하려면 `yolo val pose data=coco8-pose.yaml batch=1 device=0|cpu`를 사용하세요.

## 학습

COCO128-pose 데이터셋에서 YOLOv8-pose 모델 학습하기.

!!! Example "예제"

    === "Python"

        ```python
        from ultralytics import YOLO

        # 모델 불러오기
        model = YOLO('yolov8n-pose.yaml')  # YAML에서 새로운 모델 구축
        model = YOLO('yolov8n-pose.pt')    # 사전 학습된 모델 불러오기 (학습에 추천)
        model = YOLO('yolov8n-pose.yaml').load('yolov8n-pose.pt')  # YAML에서 구축하고 가중치 전달

        # 모델 학습
        results = model.train(data='coco8-pose.yaml', epochs=100, imgsz=640)
        ```
    === "CLI"

        ```bash
        # YAML에서 새로운 모델 구축하고 처음부터 학습 시작
        yolo pose train data=coco8-pose.yaml model=yolov8n-pose.yaml epochs=100 imgsz=640

        # 사전 학습된 *.pt 모델로부터 학습 시작
        yolo pose train data=coco8-pose.yaml model=yolov8n-pose.pt epochs=100 imgsz=640

        # YAML에서 새로운 모델 구축하고 사전 학습된 가중치를 전달하여 학습 시작
        yolo pose train data=coco8-pose.yaml model=yolov8n-pose.yaml pretrained=yolov8n-pose.pt epochs=100 imgsz=640
        ```

### 데이터셋 형식

YOLO 포즈 데이터셋 형식에 대한 자세한 내용은 [데이터셋 가이드](../../../datasets/pose/index.md)에서 찾아볼 수 있습니다. 기존 데이터셋을 다른 형식(예: COCO 등)에서 YOLO 형식으로 변환하려면 Ultralytics의 [JSON2YOLO](https://github.com/ultralytics/JSON2YOLO) 도구를 사용하세요.

## 검증

학습된 YOLOv8n-pose 모델의 정확도를 COCO128-pose 데이터셋에서 검증하기. 모델은 학습 `data` 및 인수를 모델 속성으로 유지하기 때문에 인수를 전달할 필요가 없습니다.

!!! Example "예제"

    === "Python"

        ```python
        from ultralytics import YOLO

        # 모델 불러오기
        model = YOLO('yolov8n-pose.pt')  # 공식 모델 불러오기
        model = YOLO('path/to/best.pt')  # 사용자 모델 불러오기

        # 모델 검증
        metrics = model.val()  # 데이터셋 및 설정을 기억하므로 인수 필요 없음
        metrics.box.map    # map50-95
        metrics.box.map50  # map50
        metrics.box.map75  # map75
        metrics.box.maps   # 각 범주의 map50-95를 포함하는 리스트
        ```
    === "CLI"

        ```bash
        yolo pose val model=yolov8n-pose.pt  # 공식 모델 검증
        yolo pose val model=path/to/best.pt  # 사용자 모델 검증
        ```

## 예측

학습된 YOLOv8n-pose 모델을 사용하여 이미지에 대한 예측 수행하기.

!!! Example "예제"

    === "Python"

        ```python
        from ultralytics import YOLO

        # 모델 불러오기
        model = YOLO('yolov8n-pose.pt')  # 공식 모델 불러오기
        model = YOLO('path/to/best.pt')  # 사용자 모델 불러오기

        # 모델로 예측하기
        results = model('https://ultralytics.com/images/bus.jpg')  # 이미지에서 예측
        ```
    === "CLI"

        ```bash
        yolo pose predict model=yolov8n-pose.pt source='https://ultralytics.com/images/bus.jpg'  # 공식 모델로 예측
        yolo pose predict model=path/to/best.pt source='https://ultralytics.com/images/bus.jpg'  # 사용자 모델로 예측
        ```

`predict` 모드의 전체 세부 정보는 [예측](https://docs.ultralytics.com/modes/predict/) 페이지에서 확인하세요.

## 내보내기

YOLOv8n 포즈 모델을 ONNX, CoreML 등 다른 형식으로 내보내기.

!!! Example "예제"

    === "Python"

        ```python
        from ultralytics import YOLO

        # 모델 불러오기
        model = YOLO('yolov8n-pose.pt')  # 공식 모델 불러오기
        model = YOLO('path/to/best.pt')  # 사용자 학습 모델 불러오기

        # 모델 내보내기
        model.export(format='onnx')
        ```
    === "CLI"

        ```bash
        yolo export model=yolov8n-pose.pt format=onnx  # 공식 모델 내보내기
        yolo export model=path/to/best.pt format=onnx  # 사용자 학습 모델 내보내기
        ```

YOLOv8-pose 내보내기 가능한 형식은 아래 표에 나열되어 있습니다. 내보낸 모델에서 직접 예측 또는 검증이 가능합니다, 예: `yolo predict model=yolov8n-pose.onnx`. 내보내기가 완료된 후 모델 사용 예제가 표시됩니다.

| 형식                                                                 | `format` 인수   | 모델                             | 메타데이터 | 인수                                                  |
|--------------------------------------------------------------------|---------------|--------------------------------|-------|-----------------------------------------------------|
| [PyTorch](https://pytorch.org/)                                    | -             | `yolov8n-pose.pt`              | ✅     | -                                                   |
| [TorchScript](https://pytorch.org/docs/stable/jit.html)            | `torchscript` | `yolov8n-pose.torchscript`     | ✅     | `imgsz`, `optimize`                                 |
| [ONNX](https://onnx.ai/)                                           | `onnx`        | `yolov8n-pose.onnx`            | ✅     | `imgsz`, `half`, `dynamic`, `simplify`, `opset`     |
| [OpenVINO](https://docs.openvino.ai/latest/index.html)             | `openvino`    | `yolov8n-pose_openvino_model/` | ✅     | `imgsz`, `half`                                     |
| [TensorRT](https://developer.nvidia.com/tensorrt)                  | `engine`      | `yolov8n-pose.engine`          | ✅     | `imgsz`, `half`, `dynamic`, `simplify`, `workspace` |
| [CoreML](https://github.com/apple/coremltools)                     | `coreml`      | `yolov8n-pose.mlpackage`       | ✅     | `imgsz`, `half`, `int8`, `nms`                      |
| [TF SavedModel](https://www.tensorflow.org/guide/saved_model)      | `saved_model` | `yolov8n-pose_saved_model/`    | ✅     | `imgsz`, `keras`                                    |
| [TF GraphDef](https://www.tensorflow.org/api_docs/python/tf/Graph) | `pb`          | `yolov8n-pose.pb`              | ❌     | `imgsz`                                             |
| [TF Lite](https://www.tensorflow.org/lite)                         | `tflite`      | `yolov8n-pose.tflite`          | ✅     | `imgsz`, `half`, `int8`                             |
| [TF Edge TPU](https://coral.ai/docs/edgetpu/models-intro/)         | `edgetpu`     | `yolov8n-pose_edgetpu.tflite`  | ✅     | `imgsz`                                             |
| [TF.js](https://www.tensorflow.org/js)                             | `tfjs`        | `yolov8n-pose_web_model/`      | ✅     | `imgsz`                                             |
| [PaddlePaddle](https://github.com/PaddlePaddle)                    | `paddle`      | `yolov8n-pose_paddle_model/`   | ✅     | `imgsz`                                             |
| [ncnn](https://github.com/Tencent/ncnn)                            | `ncnn`        | `yolov8n-pose_ncnn_model/`     | ✅     | `imgsz`, `half`                                     |

`export`의 전체 세부 정보는 [내보내기](https://docs.ultralytics.com/modes/export/) 페이지에서 확인하세요.
