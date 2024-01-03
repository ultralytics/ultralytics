---
comments: true
description: Ultralytics YOLOv8を使用してポーズ推定タスクを行う方法を学びます。事前トレーニング済みのモデルを見つけ、トレーニング、検証、予測、独自のエクスポートを行います。
keywords: Ultralytics, YOLO, YOLOv8, ポーズ推定, キーポイント検出, 物体検出, 事前トレーニング済みモデル, 機械学習, 人工知能
---

# ポーズ推定

<img width="1024" src="https://user-images.githubusercontent.com/26833433/243418616-9811ac0b-a4a7-452a-8aba-484ba32bb4a8.png" alt="ポーズ推定例">

ポーズ推定は、通常キーポイントとして参照される画像内の特定の点の位置を識別するタスクです。キーポイントは、関節、ランドマーク、またはその他の特徴的な特徴など、対象物のさまざまな部分を表すことができます。キーポイントの位置は、通常2Dの `[x, y]` または3D `[x, y, visible]` 座標のセットとして表されます。

ポーズ推定モデルの出力は、画像内のオブジェクト上のキーポイントを表す一連の点であり、通常は各点の信頼スコアを伴います。ポーズ推定は、シーン内のオブジェクトの特定の部分と、それらが互いに対して位置する場所を特定する必要がある場合に適しています。

<p align="center">
  <br>
  <iframe width="720" height="405" src="https://www.youtube.com/embed/Y28xXQmju64?si=pCY4ZwejZFu6Z4kZ"
    title="YouTube video player" frameborder="0"
    allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share"
    allowfullscreen>
  </iframe>
  <br>
  <strong>視聴:</strong> Ultralytics YOLOv8によるポーズ推定。
</p>

!!! Tip "ヒント"

    YOLOv8 _pose_ モデルは `-pose` サフィックスを使用します。例：`yolov8n-pose.pt`。これらのモデルは [COCOキーポイント](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/datasets/coco-pose.yaml) データセットでトレーニングされ、多様なポーズ推定タスクに適しています。

## [モデル](https://github.com/ultralytics/ultralytics/tree/main/ultralytics/cfg/models/v8)

YOLOv8事前トレーニング済みポーズモデルはこちらです。Detect, Segment, Poseモデルは [COCO](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/datasets/coco.yaml) データセットで、Classifyモデルは [ImageNet](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/datasets/ImageNet.yaml) データセットで事前トレーニングされています。

[モデル](https://github.com/ultralytics/ultralytics/tree/main/ultralytics/cfg/models)は最新のUltralytics [リリース](https://github.com/ultralytics/assets/releases)から最初の使用時に自動的にダウンロードされます。

| モデル                                                                                                  | サイズ<br><sup>(ピクセル) | mAP<sup>ポーズ<br>50-95 | mAP<sup>ポーズ<br>50 | 速度<br><sup>CPU ONNX<br>(ミリ秒) | 速度<br><sup>A100 TensorRT<br>(ミリ秒) | パラメータ<br><sup>(M) | FLOPs<br><sup>(B) |
|------------------------------------------------------------------------------------------------------|--------------------|----------------------|-------------------|------------------------------|-----------------------------------|-------------------|-------------------|
| [YOLOv8n-pose](https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n-pose.pt)       | 640                | 50.4                 | 80.1              | 131.8                        | 1.18                              | 3.3               | 9.2               |
| [YOLOv8s-pose](https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8s-pose.pt)       | 640                | 60.0                 | 86.2              | 233.2                        | 1.42                              | 11.6              | 30.2              |
| [YOLOv8m-pose](https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8m-pose.pt)       | 640                | 65.0                 | 88.8              | 456.3                        | 2.00                              | 26.4              | 81.0              |
| [YOLOv8l-pose](https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8l-pose.pt)       | 640                | 67.6                 | 90.0              | 784.5                        | 2.59                              | 44.4              | 168.6             |
| [YOLOv8x-pose](https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8x-pose.pt)       | 640                | 69.2                 | 90.2              | 1607.1                       | 3.73                              | 69.4              | 263.2             |
| [YOLOv8x-pose-p6](https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8x-pose-p6.pt) | 1280               | 71.6                 | 91.2              | 4088.7                       | 10.04                             | 99.1              | 1066.4            |

- **mAP<sup>val</sup>** の値は、[COCO Keypoints val2017](http://cocodataset.org)データセットでの単一モデル単一スケールに対するものです。
  <br>再現方法 `yolo val pose data=coco-pose.yaml device=0`
- **速度** は [Amazon EC2 P4d](https://aws.amazon.com/ec2/instance-types/p4/)インスタンスを使用したCOCO val画像の平均です。
  <br>再現方法 `yolo val pose data=coco8-pose.yaml batch=1 device=0|cpu`

## トレーニング

COCO128-poseデータセットでYOLOv8-poseモデルをトレーニングします。

!!! Example "例"

    === "Python"

        ```python
        from ultralytics import YOLO

        # モデルをロード
        model = YOLO('yolov8n-pose.yaml')  # 新しいモデルをYAMLからビルド
        model = YOLO('yolov8n-pose.pt')    # 事前トレーニング済みのモデルをロード（トレーニング用に推奨）
        model = YOLO('yolov8n-pose.yaml').load('yolov8n-pose.pt')  # YAMLからビルドして重みを転送

        # モデルのトレーニング
        results = model.train(data='coco8-pose.yaml', epochs=100, imgsz=640)
        ```
    === "CLI"

        ```bash
        # YAMLから新しいモデルをビルドし、最初からトレーニングを開始
        yolo pose train data=coco8-pose.yaml model=yolov8n-pose.yaml epochs=100 imgsz=640

        # 事前トレーニング済みの*.ptモデルからトレーニングを開始
        yolo pose train data=coco8-pose.yaml model=yolov8n-pose.pt epochs=100 imgsz=640

        # YAMLから新しいモデルをビルド、事前トレーニング済みの重みを転送してトレーニングを開始
        yolo pose train data=coco8-pose.yaml model=yolov8n-pose.yaml pretrained=yolov8n-pose.pt epochs=100 imgsz=640
        ```

### データセットフォーマット

YOLOポーズデータセットフォーマットの詳細は、[データセットガイド](../../../datasets/pose/index.md)に記載されています。既存のデータセットを他のフォーマット（COCOなど）からYOLOフォーマットに変換するには、Ultralyticsの[JSON2YOLO](https://github.com/ultralytics/JSON2YOLO) ツールをご使用ください。

## Val

COCO128-poseデータセットでトレーニングされたYOLOv8n-poseモデルの精度を検証します。引数は必要なく、`model`にはトレーニング`data`と引数がモデル属性として保持されます。

!!! Example "例"

    === "Python"

        ```python
        from ultralytics import YOLO

        # モデルをロード
        model = YOLO('yolov8n-pose.pt')  # 公式モデルをロード
        model = YOLO('path/to/best.pt')   # カスタムモデルをロード

        # モデルを検証
        metrics = model.val()  # データセットや設定は記録されているため引数は不要
        metrics.box.map    # map50-95
        metrics.box.map50  # map50
        metrics.box.map75  # map75
        metrics.box.maps   # 各カテゴリのmap50-95が含まれるリスト
        ```
    === "CLI"

        ```bash
        yolo pose val model=yolov8n-pose.pt  # 公式モデルを検証
        yolo pose val model=path/to/best.pt  # カスタムモデルを検証
        ```

## Predict

トレーニング済みのYOLOv8n-poseモデルを使用して画像を予測します。

!!! Example "例"

    === "Python"

        ```python
        from ultralytics import YOLO

        # モデルをロード
        model = YOLO('yolov8n-pose.pt')  # 公式モデルをロード
        model = YOLO('path/to/best.pt')   # カスタムモデルをロード

        # モデルで予測
        results = model('https://ultralytics.com/images/bus.jpg')  # 画像に予測を実行
        ```
    === "CLI"

        ```bash
        yolo pose predict model=yolov8n-pose.pt source='https://ultralytics.com/images/bus.jpg'  # 公式モデルで予測
        yolo pose predict model=path/to/best.pt source='https://ultralytics.com/images/bus.jpg'  # カスタムモデルで予測
        ```

`predict`モードの詳細を[Predict](https://docs.ultralytics.com/modes/predict/)ページでご覧いただけます。

## Export

YOLOv8n PoseモデルをONNX、CoreMLなどの異なるフォーマットにエクスポートします。

!!! Example "例"

    === "Python"

        ```python
        from ultralytics import YOLO

        # モデルをロード
        model = YOLO('yolov8n-pose.pt')  # 公式モデルをロード
        model = YOLO('path/to/best.pt')   # カスタムトレーニング済みモデルをロード

        # モデルをエクスポート
        model.export(format='onnx')
        ```
    === "CLI"

        ```bash
        yolo export model=yolov8n-pose.pt format=onnx  # 公式モデルをエクスポート
        yolo export model=path/to/best.pt format=onnx  # カスタムトレーニング済みモデルをエクスポート
        ```

利用可能なYOLOv8-poseエクスポートフォーマットは以下の表に示されており、エクスポート完了後にお使いのモデルに関する使用例が示されます。

| フォーマット                                                             | `format`引数    | モデル                            | メタデータ | 引数                                                  |
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

`export`の詳細は[Export](https://docs.ultralytics.com/modes/export/)ページでご覧いただけます。
