---
comments: true
description: YOLOv8 분류 모델에 대한 이미지 분류 정보를 알아보세요. 사전 훈련된 모델 목록과 모델 학습, 검증, 예측, 내보내기 방법에 대한 자세한 정보를 확인하실 수 있습니다.
keywords: Ultralytics, YOLOv8, 이미지 분류, 사전 훈련된 모델, YOLOv8n-cls, 학습, 검증, 예측, 모델 내보내기
---

# 이미지 분류

<img width="1024" src="https://user-images.githubusercontent.com/26833433/243418606-adf35c62-2e11-405d-84c6-b84e7d013804.png" alt="Image classification examples">

이미지 분류는 가장 단순한 세 가지 작업 중 하나로, 전체 이미지를 미리 정의된 클래스 집합 중 하나로 분류하는 작업입니다.

이미지 분류기의 출력은 단일 클래스 라벨과 신뢰도 점수입니다. 이미지 분류는 클래스의 이미지만 알고 싶고 해당 클래스의 객체가 어디에 위치하고 있는지 또는 그 정확한 형태가 무엇인지 알 필요가 없을 때 유용합니다.

!!! Tip "팁"

    YOLOv8 분류 모델은 `-cls` 접미사를 사용합니다. 예: `yolov8n-cls.pt`이며, [ImageNet](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/datasets/ImageNet.yaml)에서 사전 훈련되었습니다.

## [모델](https://github.com/ultralytics/ultralytics/tree/main/ultralytics/cfg/models/v8)

여기에는 사전 훈련된 YOLOv8 분류 모델이 표시됩니다. Detect, Segment 및 Pose 모델은 [COCO](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/datasets/coco.yaml) 데이터셋에서 사전 훈련되고, 분류 모델은 [ImageNet](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/datasets/ImageNet.yaml) 데이터셋에서 사전 훈련됩니다.

[모델](https://github.com/ultralytics/ultralytics/tree/main/ultralytics/cfg/models)은 첫 사용 시 최신 Ultralytics [릴리스](https://github.com/ultralytics/assets/releases)에서 자동으로 다운로드됩니다.

| 모델                                                                                           | 크기<br><sup>(픽셀) | 정확도<br><sup>top1 | 정확도<br><sup>top5 | 속도<br><sup>CPU ONNX<br>(ms) | 속도<br><sup>A100 TensorRT<br>(ms) | 매개변수<br><sup>(M) | FLOPs<br><sup>(B) at 640 |
|----------------------------------------------------------------------------------------------|-----------------|------------------|------------------|-----------------------------|----------------------------------|------------------|--------------------------|
| [YOLOv8n-cls](https://github.com/ultralytics/assets/releases/download/v8.1.0/yolov8n-cls.pt) | 224             | 66.6             | 87.0             | 12.9                        | 0.31                             | 2.7              | 4.3                      |
| [YOLOv8s-cls](https://github.com/ultralytics/assets/releases/download/v8.1.0/yolov8s-cls.pt) | 224             | 72.3             | 91.1             | 23.4                        | 0.35                             | 6.4              | 13.5                     |
| [YOLOv8m-cls](https://github.com/ultralytics/assets/releases/download/v8.1.0/yolov8m-cls.pt) | 224             | 76.4             | 93.2             | 85.4                        | 0.62                             | 17.0             | 42.7                     |
| [YOLOv8l-cls](https://github.com/ultralytics/assets/releases/download/v8.1.0/yolov8l-cls.pt) | 224             | 78.0             | 94.1             | 163.0                       | 0.87                             | 37.5             | 99.7                     |
| [YOLOv8x-cls](https://github.com/ultralytics/assets/releases/download/v8.1.0/yolov8x-cls.pt) | 224             | 78.4             | 94.3             | 232.0                       | 1.01                             | 57.4             | 154.8                    |

- **정확도** 값은 [ImageNet](https://www.image-net.org/) 데이터셋 검증 세트에서의 모델 정확도입니다.
  <br>[ImageNet](https://www.image-net.org/)에서 재현 가능합니다: `yolo val classify data=path/to/ImageNet device=0`
- **속도**는 [Amazon EC2 P4d](https://aws.amazon.com/ec2/instance-types/p4/) 인스턴스를 사용해 ImageNet 검증 이미지들의 평균 속도입니다.
  <br>[ImageNet](https://www.image-net.org/)에서 재현 가능합니다: `yolo val classify data=path/to/ImageNet batch=1 device=0|cpu`

## 학습

YOLOv8n-cls 모델을 MNIST160 데이터셋에서 100 에포크 동안 학습시키고 이미지 크기는 64로 설정합니다. 가능한 모든 인자는 [설정](/../usage/cfg.md) 페이지에서 확인할 수 있습니다.

!!! Example "예제"

    === "Python"

        ```python
        from ultralytics import YOLO

        # 모델 불러오기
        model = YOLO('yolov8n-cls.yaml')  # YAML에서 새 모델 구축
        model = YOLO('yolov8n-cls.pt')  # 사전 훈련된 모델 불러오기 (학습용 추천)
        model = YOLO('yolov8n-cls.yaml').load('yolov8n-cls.pt')  # YAML로 구축하고 가중치 전송

        # 모델 학습
        result = model.train(data='mnist160', epochs=100, imgsz=64)
        ```

    === "CLI"

        ```bash
        # YAML에서 새 모델을 구축하고 처음부터 학습 시작
        yolo classify train data=mnist160 model=yolov8n-cls.yaml epochs=100 imgsz=64

        # 사전 훈련된 *.pt 모델에서 학습 시작
        yolo classify train data=mnist160 model=yolov8n-cls.pt epochs=100 imgsz=64

        # YAML에서 새 모델을 구축하고 사전 훈련된 가중치를 전송한 뒤 학습 시작
        yolo classify train data=mnist160 model=yolov8n-cls.yaml pretrained=yolov8n-cls.pt epochs=100 imgsz=64
        ```

### 데이터셋 형식

YOLO 분류 데이터셋 형식은 [데이터셋 가이드](../../../datasets/classify/index.md)에서 자세히 확인할 수 있습니다.

## 검증

학습된 YOLOv8n-cls 모델의 정확도를 MNIST160 데이터셋에서 검증합니다. `model`은 모델 속성으로 훈련 시 `data` 및 인자를 유지하므로 추가 인자를 전달할 필요가 없습니다.

!!! Example "예제"

    === "Python"

        ```python
        from ultralytics import YOLO

        # 모델 불러오기
        model = YOLO('yolov8n-cls.pt')  # 공식 모델 불러오기
        model = YOLO('path/to/best.pt')  # 사용자 모델 불러오기

        # 모델 검증
        metrics = model.val()  # 추가 인자 불필요, 데이터셋 및 설정 기억함
        metrics.top1   # top1 정확도
        metrics.top5   # top5 정확도
        ```
    === "CLI"

        ```bash
        yolo classify val model=yolov8n-cls.pt  # 공식 모델 검증
        yolo classify val model=path/to/best.pt  # 사용자 모델 검증
        ```

## 예측

학습된 YOLOv8n-cls 모델을 사용하여 이미지에 대한 예측을 실행합니다.

!!! Example "예제"

    === "Python"

        ```python
        from ultralytics import YOLO

        # 모델 불러오기
        model = YOLO('yolov8n-cls.pt')  # 공식 모델 불러오기
        model = YOLO('path/to/best.pt')  # 사용자 모델 불러오기

        # 예측 실행
        results = model('https://ultralytics.com/images/bus.jpg')  # 이미지에 대한 예측 실행
        ```
    === "CLI"

        ```bash
        yolo classify predict model=yolov8n-cls.pt source='https://ultralytics.com/images/bus.jpg'  # 공식 모델로 예측 실행
        yolo classify predict model=path/to/best.pt source='https://ultralytics.com/images/bus.jpg'  # 사용자 모델로 예측 실행
        ```

자세한 `predict` 모드 정보는 [예측](https://docs.ultralytics.com/modes/predict/) 페이지에서 확인하세요.

## 내보내기

YOLOv8n-cls 모델을 ONNX, CoreML 등과 같은 다른 형식으로 내보냅니다.

!!! Example "예제"

    === "Python"

        ```python
        from ultralytics import YOLO

        # 모델 불러오기
        model = YOLO('yolov8n-cls.pt')  # 공식 모델 불러오기
        model = YOLO('path/to/best.pt')  # 사용자 훈련 모델 불러오기

        # 모델 내보내기
        model.export(format='onnx')
        ```
    === "CLI"

        ```bash
        yolo export model=yolov8n-cls.pt format=onnx  # 공식 모델 내보내기
        yolo export model=path/to/best.pt format=onnx  # 사용자 훈련 모델 내보내기
        ```

아래 표에 사용 가능한 YOLOv8-cls 내보내기 형식이 나와 있습니다. 내보낸 모델에서 바로 예측하거나 검증할 수 있습니다. 즉, `yolo predict model=yolov8n-cls.onnx`를 사용할 수 있습니다. 내보내기가 완료된 후 모델에 대한 사용 예제들이 표시됩니다.

| 형식                                                                 | `format` 인자   | 모델                            | 메타데이터 | 인자                                                  |
|--------------------------------------------------------------------|---------------|-------------------------------|-------|-----------------------------------------------------|
| [PyTorch](https://pytorch.org/)                                    | -             | `yolov8n-cls.pt`              | ✅     | -                                                   |
| [TorchScript](https://pytorch.org/docs/stable/jit.html)            | `torchscript` | `yolov8n-cls.torchscript`     | ✅     | `imgsz`, `optimize`                                 |
| [ONNX](https://onnx.ai/)                                           | `onnx`        | `yolov8n-cls.onnx`            | ✅     | `imgsz`, `half`, `dynamic`, `simplify`, `opset`     |
| [OpenVINO](https://docs.openvino.ai/latest/index.html)             | `openvino`    | `yolov8n-cls_openvino_model/` | ✅     | `imgsz`, `half`                                     |
| [TensorRT](https://developer.nvidia.com/tensorrt)                  | `engine`      | `yolov8n-cls.engine`          | ✅     | `imgsz`, `half`, `dynamic`, `simplify`, `workspace` |
| [CoreML](https://github.com/apple/coremltools)                     | `coreml`      | `yolov8n-cls.mlpackage`       | ✅     | `imgsz`, `half`, `int8`, `nms`                      |
| [TF SavedModel](https://www.tensorflow.org/guide/saved_model)      | `saved_model` | `yolov8n-cls_saved_model/`    | ✅     | `imgsz`, `keras`                                    |
| [TF GraphDef](https://www.tensorflow.org/api_docs/python/tf/Graph) | `pb`          | `yolov8n-cls.pb`              | ❌     | `imgsz`                                             |
| [TF Lite](https://www.tensorflow.org/lite)                         | `tflite`      | `yolov8n-cls.tflite`          | ✅     | `imgsz`, `half`, `int8`                             |
| [TF Edge TPU](https://coral.ai/docs/edgetpu/models-intro/)         | `edgetpu`     | `yolov8n-cls_edgetpu.tflite`  | ✅     | `imgsz`                                             |
| [TF.js](https://www.tensorflow.org/js)                             | `tfjs`        | `yolov8n-cls_web_model/`      | ✅     | `imgsz`                                             |
| [PaddlePaddle](https://github.com/PaddlePaddle)                    | `paddle`      | `yolov8n-cls_paddle_model/`   | ✅     | `imgsz`                                             |
| [ncnn](https://github.com/Tencent/ncnn)                            | `ncnn`        | `yolov8n-cls_ncnn_model/`     | ✅     | `imgsz`, `half`                                     |

자세한 `export` 정보는 [내보내기](https://docs.ultralytics.com/modes/export/) 페이지에서 확인하세요.
