---
comments: true
description: YOLOv8 分類モデルについての画像分類。事前トレーニングされたモデルのリストとモデルのトレーニング、検証、予測、エクスポート方法の詳細情報を学びます。
keywords: Ultralytics, YOLOv8, 画像分類, 事前トレーニングされたモデル, YOLOv8n-cls, トレーニング, 検証, 予測, モデルエクスポート
---

# 画像分類

<img width="1024" src="https://user-images.githubusercontent.com/26833433/243418606-adf35c62-2e11-405d-84c6-b84e7d013804.png" alt="画像分類の例">

画像分類は3つのタスクの中で最も単純で、1枚の画像をあらかじめ定義されたクラスのセットに分類します。

画像分類器の出力は単一のクラスラベルと信頼度スコアです。画像がどのクラスに属しているかのみを知る必要があり、クラスのオブジェクトがどこにあるか、その正確な形状は必要としない場合に画像分類が役立ちます。

!!! Tip "ヒント"

    YOLOv8 分類モデルは `-cls` 接尾辞を使用します。例: `yolov8n-cls.pt` これらは [ImageNet](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/datasets/ImageNet.yaml) で事前にトレーニングされています。

## [モデル](https://github.com/ultralytics/ultralytics/tree/main/ultralytics/cfg/models/v8)

ここに事前トレーニングされた YOLOv8 分類モデルが表示されています。検出、セグメンテーション、ポーズモデルは [COCO](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/datasets/coco.yaml) データセットで事前にトレーニングされていますが、分類モデルは [ImageNet](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/datasets/ImageNet.yaml) で事前にトレーニングされています。

[モデル](https://github.com/ultralytics/ultralytics/tree/main/ultralytics/cfg/models) は初回使用時に Ultralytics の最新 [リリース](https://github.com/ultralytics/assets/releases) から自動的にダウンロードされます。

| モデル                                                                                          | サイズ<br><sup>(ピクセル) | 正確性<br><sup>トップ1 | 正確性<br><sup>トップ5 | スピード<br><sup>CPU ONNX<br>(ms) | スピード<br><sup>A100 TensorRT<br>(ms) | パラメータ<br><sup>(M) | FLOPs<br><sup>(B) at 640 |
|----------------------------------------------------------------------------------------------|--------------------|------------------|------------------|-------------------------------|------------------------------------|-------------------|--------------------------|
| [YOLOv8n-cls](https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n-cls.pt) | 224                | 66.6             | 87.0             | 12.9                          | 0.31                               | 2.7               | 4.3                      |
| [YOLOv8s-cls](https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8s-cls.pt) | 224                | 72.3             | 91.1             | 23.4                          | 0.35                               | 6.4               | 13.5                     |
| [YOLOv8m-cls](https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8m-cls.pt) | 224                | 76.4             | 93.2             | 85.4                          | 0.62                               | 17.0              | 42.7                     |
| [YOLOv8l-cls](https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8l-cls.pt) | 224                | 78.0             | 94.1             | 163.0                         | 0.87                               | 37.5              | 99.7                     |
| [YOLOv8x-cls](https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8x-cls.pt) | 224                | 78.4             | 94.3             | 232.0                         | 1.01                               | 57.4              | 154.8                    |

- **正確性** の値は [ImageNet](https://www.image-net.org/) データセットの検証セットでのモデルの正確性です。
  <br>再現するには `yolo val classify data=path/to/ImageNet device=0`
- **スピード** は [Amazon EC2 P4d](https://aws.amazon.com/ec2/instance-types/p4/) インスタンスを使用して ImageNet 検証画像を平均化したものです。
  <br>再現するには `yolo val classify data=path/to/ImageNet batch=1 device=0|cpu`

## トレーニング

画像サイズ64で100エポックにわたってMNIST160データセットにYOLOv8n-clsをトレーニングします。利用可能な引数の完全なリストについては、[設定](/../usage/cfg.md) ページを参照してください。

!!! Example "例"

    === "Python"

        ```python
        from ultralytics import YOLO

        # モデルをロードする
        model = YOLO('yolov8n-cls.yaml')  # YAMLから新しいモデルをビルド
        model = YOLO('yolov8n-cls.pt')  # 事前にトレーニングされたモデルをロード（トレーニングに推奨）
        model = YOLO('yolov8n-cls.yaml').load('yolov8n-cls.pt')  # YAMLからビルドしてウェイトを転送

        # モデルをトレーニングする
        results = model.train(data='mnist160', epochs=100, imgsz=64)
        ```

    === "CLI"

        ```bash
        # YAMLから新しいモデルをビルドし、ゼロからトレーニングを開始
        yolo classify train data=mnist160 model=yolov8n-cls.yaml epochs=100 imgsz=64

        # 事前にトレーニングされた *.pt モデルからトレーニングを開始
        yolo classify train data=mnist160 model=yolov8n-cls.pt epochs=100 imgsz=64

        # YAMLから新しいモデルをビルドし、事前トレーニングされたウェイトを転送してトレーニングを開始
        yolo classify train data=mnist160 model=yolov8n-cls.yaml pretrained=yolov8n-cls.pt epochs=100 imgsz=64
        ```

### データセットフォーマット

YOLO分類データセットのフォーマットの詳細は [データセットガイド](../../../datasets/classify/index.md) にあります。

## 検証

MNIST160データセットでトレーニング済みのYOLOv8n-clsモデルの正確性を検証します。引数は必要ありません。`model` はトレーニング時の `data` および引数をモデル属性として保持しています。

!!! Example "例"

    === "Python"

        ```python
        from ultralytics import YOLO

        # モデルをロードする
        model = YOLO('yolov8n-cls.pt')  # 公式モデルをロード
        model = YOLO('path/to/best.pt')  # カスタムモデルをロード

        # モデルを検証する
        metrics = model.val()  # 引数不要、データセットと設定は記憶されている
        metrics.top1   # トップ1の正確性
        metrics.top5   # トップ5の正確性
        ```
    === "CLI"

        ```bash
        yolo classify val model=yolov8n-cls.pt  # 公式モデルを検証
        yolo classify val model=path/to/best.pt  # カスタムモデルを検証
        ```

## 予測

トレーニング済みのYOLOv8n-clsモデルを使用して、画像に対する予測を実行します。

!!! Example "例"

    === "Python"

        ```python
        from ultralytics import YOLO

        # モデルをロードする
        model = YOLO('yolov8n-cls.pt')  # 公式モデルをロード
        model = YOLO('path/to/best.pt')  # カスタムモデルをロード

        # モデルで予測する
        results = model('https://ultralytics.com/images/bus.jpg')  # 画像で予測
        ```
    === "CLI"

        ```bash
        yolo classify predict model=yolov8n-cls.pt source='https://ultralytics.com/images/bus.jpg'  # 公式モデルで予測
        yolo classify predict model=path/to/best.pt source='https://ultralytics.com/images/bus.jpg'  # カスタムモデルで予測
        ```

`predict` モードの完全な詳細は [予測](https://docs.ultralytics.com/modes/predict/) ページを参照してください。

## エクスポート

YOLOv8n-clsモデルをONNX、CoreMLなどの異なる形式にエクスポートします。

!!! Example "例"

    === "Python"

        ```python
        from ultralytics import YOLO

        # モデルをロードする
        model = YOLO('yolov8n-cls.pt')  # 公式モデルをロード
        model = YOLO('path/to/best.pt')  # カスタムトレーニングされたモデルをロード

        # モデルをエクスポートする
        model.export(format='onnx')
        ```
    === "CLI"

        ```bash
        yolo export model=yolov8n-cls.pt format=onnx  # 公式モデルをエクスポート
        yolo export model=path/to/best.pt format=onnx  # カスタムトレーニングされたモデルをエクスポート
        ```

利用可能な YOLOv8-cls エクスポート形式は以下の表にあります。エクスポートされたモデルで直接予測または検証が可能です、例: `yolo predict model=yolov8n-cls.onnx`。エクスポート完了後、モデルの使用例が表示されます。

| 形式                                                                 | `format` 引数   | モデル                           | メタデータ | 引数                                                  |
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

`export` の詳細は [エクスポート](https://docs.ultralytics.com/modes/export/) ページを参照してください。
