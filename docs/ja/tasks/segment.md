---
comments: true
description: Ultralytics YOLOを使用してインスタンスセグメンテーションモデルを使いこなす方法を学びましょう。トレーニング、バリデーション、画像予測、モデルエクスポートに関する指示が含まれています。
keywords: yolov8, インスタンスセグメンテーション, Ultralytics, COCOデータセット, 画像セグメンテーション, オブジェクト検出, モデルトレーニング, モデルバリデーション, 画像予測, モデルエクスポート
---

# インスタンスセグメンテーション

<img width="1024" src="https://user-images.githubusercontent.com/26833433/243418644-7df320b8-098d-47f1-85c5-26604d761286.png" alt="インスタンスセグメンテーションの例">

インスタンスセグメンテーションはオブジェクト検出を一歩進めており、画像内の個々のオブジェクトを識別し、それらを画像の残りの部分からセグメント化します。

インスタンスセグメンテーションモデルの出力は、画像内の各オブジェクトを概説するマスクまたは輪郭のセットであり、各オブジェクトにはクラスラベルと信頼スコアが含まれています。オブジェクトの位置だけでなく、その正確な形状を知る必要がある場合に、インスタンスセグメンテーションが役立ちます。

<p align="center">
  <br>
  <iframe width="720" height="405" src="https://www.youtube.com/embed/o4Zd-IeMlSY?si=37nusCzDTd74Obsp"
    title="YouTubeのビデオプレーヤー" frameborder="0"
    allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share"
    allowfullscreen>
  </iframe>
  <br>
  <strong>視聴:</strong> Pythonで事前トレーニング済みのUltralytics YOLOv8モデルでセグメンテーションを実行する。
</p>

!!! Tip "ヒント"

    YOLOv8セグメントモデルは`-seg`サフィックスを使用し、つまり`yolov8n-seg.pt`などは[COCO](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/datasets/coco.yaml)で事前トレーニングされています。

## [モデル](https://github.com/ultralytics/ultralytics/tree/main/ultralytics/cfg/models/v8)

ここでは、事前トレーニングされたYOLOv8セグメントモデルが示されています。Detect、Segment、Poseモデルは[COCO](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/datasets/coco.yaml)データセットで事前トレーニングされている一方、Classifyモデルは[ImageNet](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/datasets/ImageNet.yaml)データセットで事前トレーニングされています。

[モデル](https://github.com/ultralytics/ultralytics/tree/main/ultralytics/cfg/models)は初回使用時に最新のUltralytics[リリース](https://github.com/ultralytics/assets/releases)から自動的にダウンロードされます。

| モデル                                                                                          | サイズ<br><sup>(ピクセル) | mAP<sup>box<br>50-95 | mAP<sup>mask<br>50-95 | スピード<br><sup>CPU ONNX<br>(ms) | スピード<br><sup>A100 TensorRT<br>(ms) | パラメータ<br><sup>(M) | FLOPs<br><sup>(B) |
|----------------------------------------------------------------------------------------------|--------------------|----------------------|-----------------------|-------------------------------|------------------------------------|-------------------|-------------------|
| [YOLOv8n-seg](https://github.com/ultralytics/assets/releases/download/v8.1.0/yolov8n-seg.pt) | 640                | 36.7                 | 30.5                  | 96.1                          | 1.21                               | 3.4               | 12.6              |
| [YOLOv8s-seg](https://github.com/ultralytics/assets/releases/download/v8.1.0/yolov8s-seg.pt) | 640                | 44.6                 | 36.8                  | 155.7                         | 1.47                               | 11.8              | 42.6              |
| [YOLOv8m-seg](https://github.com/ultralytics/assets/releases/download/v8.1.0/yolov8m-seg.pt) | 640                | 49.9                 | 40.8                  | 317.0                         | 2.18                               | 27.3              | 110.2             |
| [YOLOv8l-seg](https://github.com/ultralytics/assets/releases/download/v8.1.0/yolov8l-seg.pt) | 640                | 52.3                 | 42.6                  | 572.4                         | 2.79                               | 46.0              | 220.5             |
| [YOLOv8x-seg](https://github.com/ultralytics/assets/releases/download/v8.1.0/yolov8x-seg.pt) | 640                | 53.4                 | 43.4                  | 712.1                         | 4.02                               | 71.8              | 344.1             |

- **mAP<sup>val</sup>**の値は[COCO val2017](https://cocodataset.org)データセットでの単一モデル単一スケールの値です。
  <br>再現するには `yolo val segment data=coco.yaml device=0`
- **スピード**は[Amazon EC2 P4d](https://aws.amazon.com/ec2/instance-types/p4/)インスタンスを使用してCOCO val画像で平均化されます。
  <br>再現するには `yolo val segment data=coco128-seg.yaml batch=1 device=0|cpu`

## トレーニング

COCO128-segデータセットで、画像サイズ640でYOLOv8n-segを100エポックトレーニングします。利用可能な全ての引数については、[コンフィギュレーション](/../usage/cfg.md)ページを参照してください。

!!! Example "例"

    === "Python"

        ```python
        from ultralytics import YOLO

        # モデルをロード
        model = YOLO('yolov8n-seg.yaml')  # YAMLから新しいモデルをビルド
        model = YOLO('yolov8n-seg.pt')  # 事前トレーニングされたモデルをロード(トレーニングに推奨)
        model = YOLO('yolov8n-seg.yaml').load('yolov8n.pt')  # YAMLからビルドしウェイトを移行

        # モデルをトレーニング
        results = model.train(data='coco128-seg.yaml', epochs=100, imgsz=640)
        ```
    === "CLI"

        ```bash
        # YAMLから新しいモデルをビルドしゼロからトレーニングを開始
        yolo segment train data=coco128-seg.yaml model=yolov8n-seg.yaml epochs=100 imgsz=640

        # 事前トレーニング済みの*.ptモデルからトレーニングを開始
        yolo segment train data=coco128-seg.yaml model=yolov8n-seg.pt epochs=100 imgsz=640

        # YAMLから新しいモデルをビルドし、事前トレーニング済みウェイトを移行してトレーニングを開始
        yolo segment train data=coco128-seg.yaml model=yolov8n-seg.yaml pretrained=yolov8n-seg.pt epochs=100 imgsz=640
        ```

### データセットフォーマット

YOLOセグメンテーションデータセットのフォーマットの詳細は、[データセットガイド](../../../datasets/segment/index.md)で見つけることができます。既存のデータセットを他のフォーマット(例えばCOCOなど)からYOLOフォーマットに変換するには、Ultralyticsの[JSON2YOLO](https://github.com/ultralytics/JSON2YOLO)ツールを使用してください。

## 評価

訓練されたYOLOv8n-segモデルの精度をCOCO128-segデータセットで検証します。引数は必要ありません、なぜなら`model`はモデル属性としてトレーニング`data`と引数を保持しているからです。

!!! Example "例"

    === "Python"

        ```python
        from ultralytics import YOLO

        # モデルをロード
        model = YOLO('yolov8n-seg.pt')  # 公式モデルをロード
        model = YOLO('path/to/best.pt')  # カスタムモデルをロード

        # モデルを評価
        metrics = model.val()  # 引数は必要なし、データセットと設定は記憶している
        metrics.box.map    # map50-95(B)
        metrics.box.map50  # map50(B)
        metrics.box.map75  # map75(B)
        metrics.box.maps   # 各カテゴリのmap50-95(B)のリスト
        metrics.seg.map    # map50-95(M)
        metrics.seg.map50  # map50(M)
        metrics.seg.map75  # map75(M)
        metrics.seg.maps   # 各カテゴリのmap50-95(M)のリスト
        ```
    === "CLI"

        ```bash
        yolo segment val model=yolov8n-seg.pt  # 公式モデルを評価
        yolo segment val model=path/to/best.pt  # カスタムモデルを評価
        ```

## 予測

訓練されたYOLOv8n-segモデルを使用して画像の予測を実行します。

!!! Example "例"

    === "Python"

        ```python
        from ultralytics import YOLO

        # モデルをロード
        model = YOLO('yolov8n-seg.pt')  # 公式モデルをロード
        model = YOLO('path/to/best.pt')  # カスタムモデルをロード

        # モデルで予測
        results = model('https://ultralytics.com/images/bus.jpg')  # 画像で予測
        ```
    === "CLI"

        ```bash
        yolo segment predict model=yolov8n-seg.pt source='https://ultralytics.com/images/bus.jpg'  # 公式モデルで予測
        yolo segment predict model=path/to/best.pt source='https://ultralytics.com/images/bus.jpg'  # カスタムモデルで予測
        ```

`predict`モードの完全な詳細は、[予測](https://docs.ultralytics.com/modes/predict/)ページにて確認できます。

## エクスポート

YOLOv8n-segモデルをONNX、CoreMLなどの別の形式にエクスポートします。

!!! Example "例"

    === "Python"

        ```python
        from ultralytics import YOLO

        # モデルをロード
        model = YOLO('yolov8n-seg.pt')  # 公式モデルをロード
        model = YOLO('path/to/best.pt')  # カスタムトレーニングされたモデルをロード

        # モデルをエクスポート
        model.export(format='onnx')
        ```
    === "CLI"

        ```bash
        yolo export model=yolov8n-seg.pt format=onnx  # 公式モデルをエクスポート
        yolo export model=path/to/best.pt format=onnx  # カスタムトレーニングされたモデルをエクスポート
        ```

ご利用可能なYOLOv8-segエクスポート形式は以下の表に示されています。エクスポートされたモデルに直接予測または評価が可能です、つまり `yolo predict model=yolov8n-seg.onnx`。エクスポートが完了した後に、モデルの使用例が表示されます。

| 形式                                                                 | `format`引数    | モデル                           | メタデータ | 引数                                                  |
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
