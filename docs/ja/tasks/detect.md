---
comments: true
description: Ultralyticsの公式ドキュメント YOLOv8。モデルのトレーニング、検証、予測、そして様々なフォーマットでのモデルエクスポート方法を学ぶ。詳細なパフォーマンス統計も含む。
keywords: YOLOv8, Ultralytics, 物体検出, 事前訓練済みモデル, トレーニング, 検証, 予測, モデルエクスポート, COCO, ImageNet, PyTorch, ONNX, CoreML
---

# 物体検出

<img width="1024" src="https://user-images.githubusercontent.com/26833433/243418624-5785cb93-74c9-4541-9179-d5c6782d491a.png" alt="物体検出の例">

物体検出とは、画像やビデオストリーム内の物体の位置とクラスを特定するタスクです。

物体検出器の出力は、画像内の物体を囲む一連のバウンディングボックスであり、各ボックスにはクラスラベルと信頼度スコアが付けられます。シーン内の関心対象を識別する必要があるが、その物体の正確な位置や形状までは必要ない場合に、物体検出が適しています。

<p align="center">
  <br>
  <iframe width="720" height="405" src="https://www.youtube.com/embed/5ku7npMrW40?si=6HQO1dDXunV8gekh"
    title="YouTube video player" frameborder="0"
    allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share"
    allowfullscreen>
  </iframe>
  <br>
  <strong>視聴する:</strong> Ultralyticsの事前訓練済みYOLOv8モデルによる物体検出。
</p>

!!! Tip "ヒント"

    YOLOv8 Detectモデルは、デフォルトのYOLOv8モデル、つまり`yolov8n.pt`であり、[COCO](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/datasets/coco.yaml)で事前訓練されています。

## [モデル](https://github.com/ultralytics/ultralytics/tree/main/ultralytics/cfg/models/v8)

事前訓練されたYOLOv8 Detectモデルがこちらに示されます。Detect, Segment, Poseモデルは[COCO](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/datasets/coco.yaml)データセットで、Classifyモデルは[ImageNet](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/datasets/ImageNet.yaml)データセットで事前訓練されています。

[モデル](https://github.com/ultralytics/ultralytics/tree/main/ultralytics/cfg/models)は、最初の使用時にUltralyticsの最新の[リリース](https://github.com/ultralytics/assets/releases)から自動的にダウンロードされます。

| モデル                                                                                  | サイズ<br><sup>(ピクセル) | mAP<sup>val<br>50-95 | 速度<br><sup>CPU ONNX<br>(ms) | 速度<br><sup>A100 TensorRT<br>(ms) | パラメータ数<br><sup>(M) | FLOPs<br><sup>(B) |
|--------------------------------------------------------------------------------------|--------------------|----------------------|-----------------------------|----------------------------------|--------------------|-------------------|
| [YOLOv8n](https://github.com/ultralytics/assets/releases/download/v8.1.0/yolov8n.pt) | 640                | 37.3                 | 80.4                        | 0.99                             | 3.2                | 8.7               |
| [YOLOv8s](https://github.com/ultralytics/assets/releases/download/v8.1.0/yolov8s.pt) | 640                | 44.9                 | 128.4                       | 1.20                             | 11.2               | 28.6              |
| [YOLOv8m](https://github.com/ultralytics/assets/releases/download/v8.1.0/yolov8m.pt) | 640                | 50.2                 | 234.7                       | 1.83                             | 25.9               | 78.9              |
| [YOLOv8l](https://github.com/ultralytics/assets/releases/download/v8.1.0/yolov8l.pt) | 640                | 52.9                 | 375.2                       | 2.39                             | 43.7               | 165.2             |
| [YOLOv8x](https://github.com/ultralytics/assets/releases/download/v8.1.0/yolov8x.pt) | 640                | 53.9                 | 479.1                       | 3.53                             | 68.2               | 257.8             |

- **mAP<sup>val</sup>** の値は[COCO val2017](https://cocodataset.org)データセットにおいて、単一モデル単一スケールでのものです。
  <br>再現方法: `yolo val detect data=coco.yaml device=0`
- **速度** は[Amazon EC2 P4d](https://aws.amazon.com/ec2/instance-types/p4/)インスタンスを使用してCOCO val画像に対して平均化されたものです。
  <br>再現方法: `yolo val detect data=coco128.yaml batch=1 device=0|cpu`

## トレーニング

YOLOv8nを画像サイズ640でCOCO128データセットに対して100エポックでトレーニングします。使用可能な引数の完全なリストについては、[設定](/../usage/cfg.md)ページをご覧ください。

!!! Example "例"

    === "Python"

        ```python
        from ultralytics import YOLO

        # モデルをロードする
        model = YOLO('yolov8n.yaml')  # YAMLから新しいモデルを構築
        model = YOLO('yolov8n.pt')    # 事前訓練済みモデルをロード（トレーニングに推奨）
        model = YOLO('yolov8n.yaml').load('yolov8n.pt')  # YAMLから構築し、重みを転送

        # モデルをトレーニングする
        results = model.train(data='coco128.yaml', epochs=100, imgsz=640)
        ```
    === "CLI"

        ```bash
        # YAMLから新しいモデルを作成し、ゼロからトレーニングを開始
        yolo detect train data=coco128.yaml model=yolov8n.yaml epochs=100 imgsz=640

        # 事前訓練済みの*.ptモデルからトレーニングを開始
        yolo detect train data=coco128.yaml model=yolov8n.pt epochs=100 imgsz=640

        # YAMLから新しいモデルを作成し、事前訓練済みの重みを転送してトレーニングを開始
        yolo detect train data=coco128.yaml model=yolov8n.yaml pretrained=yolov8n.pt epochs=100 imgsz=640
        ```

### データセットの形式

YOLO検出データセットの形式の詳細は、[データセットガイド](../../../datasets/detect/index.md)に記載されています。他の形式（COCO等）からYOLO形式に既存のデータセットを変換するには、Ultralyticsの[JSON2YOLO](https://github.com/ultralytics/JSON2YOLO)ツールをご利用ください。

## 検証

トレーニングされたYOLOv8nモデルの精度をCOCO128データセットで検証します。引数は不要で、モデルはトレーニングの`data`と引数をモデル属性として保持しています。

!!! Example "例"

    === "Python"

        ```python
        from ultralytics import YOLO

        # モデルをロードする
        model = YOLO('yolov8n.pt')          # 公式モデルをロード
        model = YOLO('パス/ベスト.pt')     # カスタムモデルをロード

        # モデルを検証する
        metrics = model.val()             # 引数不要、データセットと設定は記憶されている
        metrics.box.map                   # map50-95
        metrics.box.map50                 # map50
        metrics.box.map75                 # map75
        metrics.box.maps                  # 各カテゴリのmap50-95を含むリスト
        ```
    === "CLI"

        ```bash
        yolo detect val model=yolov8n.pt                 # 公式モデルを検証
        yolo detect val model=パス/ベスト.pt            # カスタムモデルを検証
        ```

## 予測

トレーニングされたYOLOv8nモデルを使用して画像の予測を実行します。

!!! Example "例"

    === "Python"

        ```python
        from ultralytics import YOLO

        # モデルをロードする
        model = YOLO('yolov8n.pt')          # 公式モデルをロード
        model = YOLO('パス/ベスト.pt')     # カスタムモデルをロード

        # モデルで予測
        results = model('https://ultralytics.com/images/bus.jpg')  # 画像の予測実行
        ```
    === "CLI"

        ```bash
        yolo detect predict model=yolov8n.pt source='https://ultralytics.com/images/bus.jpg'  # 公式モデルで予測
        yolo detect predict model=パス/ベスト.pt source='https://ultralytics.com/images/bus.jpg'  # カスタムモデルで予測
        ```

`predict`モードの詳細は、[Predict](https://docs.ultralytics.com/modes/predict/)ページで全て見ることができます。

## エクスポート

YOLOv8nモデルをONNX、CoreMLなどの異なるフォーマットにエクスポートします。

!!! Example "例"

    === "Python"

        ```python
        from ultralytics import YOLO

        # モデルをロード
        model = YOLO('yolov8n.pt')          # 公式モデルをロード
        model = YOLO('パス/ベスト.pt')     # カスタムトレーニングモデルをロード

        # モデルをエクスポート
        model.export(format='onnx')
        ```
    === "CLI"

        ```bash
        yolo export model=yolov8n.pt format=onnx  # 公式モデルをエクスポート
        yolo export model=パス/ベスト.pt format=onnx  # カスタムトレーニングモデルをエクスポート
        ```

YOLOv8エクスポート可能なフォーマットのテーブルは以下です。エクスポート完了後に、エクスポートされたモデルで直接予測または検証が可能です。つまり、`yolo predict model=yolov8n.onnx` です。使用例はエクスポート完了後にモデルに表示されます。

| フォーマット                                                             | `format`引数    | モデル                       | メタデータ | 引数                                                  |
|--------------------------------------------------------------------|---------------|---------------------------|-------|-----------------------------------------------------|
| [PyTorch](https://pytorch.org/)                                    | -             | `yolov8n.pt`              | ✅     | -                                                   |
| [TorchScript](https://pytorch.org/docs/stable/jit.html)            | `torchscript` | `yolov8n.torchscript`     | ✅     | `imgsz`, `optimize`                                 |
| [ONNX](https://onnx.ai/)                                           | `onnx`        | `yolov8n.onnx`            | ✅     | `imgsz`, `half`, `dynamic`, `simplify`, `opset`     |
| [OpenVINO](https://docs.openvino.ai/latest/index.html)             | `openvino`    | `yolov8n_openvino_model/` | ✅     | `imgsz`, `half`                                     |
| [TensorRT](https://developer.nvidia.com/tensorrt)                  | `engine`      | `yolov8n.engine`          | ✅     | `imgsz`, `half`, `dynamic`, `simplify`, `workspace` |
| [CoreML](https://github.com/apple/coremltools)                     | `coreml`      | `yolov8n.mlpackage`       | ✅     | `imgsz`, `half`, `int8`, `nms`                      |
| [TF SavedModel](https://www.tensorflow.org/guide/saved_model)      | `saved_model` | `yolov8n_saved_model/`    | ✅     | `imgsz`, `keras`                                    |
| [TF GraphDef](https://www.tensorflow.org/api_docs/python/tf/Graph) | `pb`          | `yolov8n.pb`              | ❌     | `imgsz`                                             |
| [TF Lite](https://www.tensorflow.org/lite)                         | `tflite`      | `yolov8n.tflite`          | ✅     | `imgsz`, `half`, `int8`                             |
| [TF Edge TPU](https://coral.ai/docs/edgetpu/models-intro/)         | `edgetpu`     | `yolov8n_edgetpu.tflite`  | ✅     | `imgsz`                                             |
| [TF.js](https://www.tensorflow.org/js)                             | `tfjs`        | `yolov8n_web_model/`      | ✅     | `imgsz`                                             |
| [PaddlePaddle](https://github.com/PaddlePaddle)                    | `paddle`      | `yolov8n_paddle_model/`   | ✅     | `imgsz`                                             |
| [ncnn](https://github.com/Tencent/ncnn)                            | `ncnn`        | `yolov8n_ncnn_model/`     | ✅     | `imgsz`, `half`                                     |

`export`の詳細は、[Export](https://docs.ultralytics.com/modes/export/)ページで全て見ることができます。
