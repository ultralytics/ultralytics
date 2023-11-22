---
comments: true
description: 最新バージョンのリアルタイム物体検出器であるYOLOv8のスリリングな機能をご紹介！最新のアーキテクチャ、事前学習済みモデル、および正確性と速度の最適なバランスが、YOLOv8を物体検出タスクに最適な選択肢にします。
keywords: YOLOv8, Ultralytics, リアルタイム物体検出器, 事前学習済みモデル, ドキュメンテーション, 物体検出, YOLOシリーズ, 高度なアーキテクチャ, 正確性, 速度
---

# YOLOv8

## 概要

YOLOv8は、リアルタイム物体検出のYOLOシリーズの最新版であり、正確性と速度の面で最先端のパフォーマンスを提供します。以前のYOLOバージョンの進歩に基づき、YOLOv8は新機能と最適化を導入し、さまざまなアプリケーションの物体検出タスクに理想的な選択肢となります。

![Ultralytics YOLOv8](https://raw.githubusercontent.com/ultralytics/assets/main/yolov8/yolo-comparison-plots.png)

## 主な特徴

- **高度なバックボーンおよびネックアーキテクチャ：** YOLOv8は、最先端のバックボーンおよびネックアーキテクチャを使用しており、特徴抽出と物体検出のパフォーマンスが向上しています。
- **アンカーフリーの分割Ultralyticsヘッド：** YOLOv8はアンカーベースのアプローチと比較して、アンカーフリーの分割Ultralyticsヘッドを採用しています。これにより、より高い精度とより効率的な検出が可能です。
- **正確性と速度の最適なトレードオフ：** 正確性と速度の最適なバランスを維持することを重視したYOLOv8は、さまざまなアプリケーション領域でのリアルタイム物体検出タスクに適しています。
- **さまざまな事前学習済みモデル：** YOLOv8にはさまざまな事前学習済みモデルが用意されており、さまざまなタスクとパフォーマンス要件に対応しています。特定のユースケースに適したモデルを簡単に見つけることができます。

## サポートされるタスク

| モデルの種類      | 事前学習済み重み                                                                                                            | タスク             |
|-------------|---------------------------------------------------------------------------------------------------------------------|-----------------|
| YOLOv8      | `yolov8n.pt`, `yolov8s.pt`, `yolov8m.pt`, `yolov8l.pt`, `yolov8x.pt`                                                | 物体検出            |
| YOLOv8-seg  | `yolov8n-seg.pt`, `yolov8s-seg.pt`, `yolov8m-seg.pt`, `yolov8l-seg.pt`, `yolov8x-seg.pt`                            | インスタンスセグメンテーション |
| YOLOv8-pose | `yolov8n-pose.pt`, `yolov8s-pose.pt`, `yolov8m-pose.pt`, `yolov8l-pose.pt`, `yolov8x-pose.pt`, `yolov8x-pose-p6.pt` | ポーズ/キーポイント      |
| YOLOv8-cls  | `yolov8n-cls.pt`, `yolov8s-cls.pt`, `yolov8m-cls.pt`, `yolov8l-cls.pt`, `yolov8x-cls.pt`                            | 分類              |

## サポートされるモード

| モード     | サポート |
|---------|------|
| 推論      | ✅    |
| バリデーション | ✅    |
| トレーニング  | ✅    |

!!! パフォーマンス

    === "物体検出（COCO）"

        | モデル                                                                                 | サイズ<br><sup>(ピクセル) | mAP<sup>val<br>50-95 | 速度<br><sup>CPU ONNX<br>(ミリ秒) | 速度<br><sup>A100 TensorRT<br>(ミリ秒) | パラメータ<br><sup>(百万) | FLOPs<br><sup>(十億) |
        | ------------------------------------------------------------------------------------ | --------------------- | -------------------- | ------------------------------ | ----------------------------------- | ------------------ | ----------------- |
        | [YOLOv8n](https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n.pt) | 640                   | 37.3                 | 80.4                           | 0.99                                | 3.2                | 8.7               |
        | [YOLOv8s](https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8s.pt) | 640                   | 44.9                 | 128.4                          | 1.20                                | 11.2               | 28.6              |
        | [YOLOv8m](https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8m.pt) | 640                   | 50.2                 | 234.7                          | 1.83                                | 25.9               | 78.9              |
        | [YOLOv8l](https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8l.pt) | 640                   | 52.9                 | 375.2                          | 2.39                                | 43.7               | 165.2             |
        | [YOLOv8x](https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8x.pt) | 640                   | 53.9                 | 479.1                          | 3.53                                | 68.2               | 257.8             |

    === "物体検出（Open Images V7）"

        これらのモデルは、[Open Image V7](https://docs.ultralytics.com/datasets/detect/open-images-v7/)でトレーニングされたもので、600の事前学習済みクラスが含まれています。使用例については、[Detection Docs](https://docs.ultralytics.com/tasks/detect/)を参照してください。

        | モデル                                                                                     | サイズ<br><sup>(ピクセル) | mAP<sup>val<br>50-95 | 速度<br><sup>CPU ONNX<br>(ミリ秒) | 速度<br><sup>A100 TensorRT<br>(ミリ秒) | パラメータ<br><sup>(百万) | FLOPs<br><sup>(十億) |
        | ----------------------------------------------------------------------------------------- | --------------------- | -------------------- | ------------------------------ | ----------------------------------- | ------------------ | ----------------- |
        | [YOLOv8n](https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n-oiv7.pt) | 640                   | 18.4                 | 142.4                          | 1.21                                | 3.5                | 10.5              |
        | [YOLOv8s](https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8s-oiv7.pt) | 640                   | 27.7                 | 183.1                          | 1.40                                | 11.4               | 29.7              |
        | [YOLOv8m](https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8m-oiv7.pt) | 640                   | 33.6                 | 408.5                          | 2.26                                | 26.2               | 80.6              |
        | [YOLOv8l](https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8l-oiv7.pt) | 640                   | 34.9                 | 596.9                          | 2.43                                | 44.1               | 167.4             |
        | [YOLOv8x](https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8x-oiv7.pt) | 640                   | 36.3                 | 860.6                          | 3.56                                | 68.7               | 260.6             |

    === "セグメンテーション（COCO）"

        | モデル                                                                                        | サイズ<br><sup>(ピクセル) | mAP<sup>box<br>50-95 | mAP<sup>mask<br>50-95 | 速度<br><sup>CPU ONNX<br>(ミリ秒) | 速度<br><sup>A100 TensorRT<br>(ミリ秒) | パラメータ<br><sup>(百万) | FLOPs<br><sup>(十億) |
        | -------------------------------------------------------------------------------------------- | --------------------- | -------------------- | --------------------- | ------------------------------ | ----------------------------------- | ------------------ | ----------------- |
        | [YOLOv8n-seg](https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n-seg.pt) | 640                   | 36.7                 | 30.5                  | 96.1                           | 1.21                                | 3.4                | 12.6              |
        | [YOLOv8s-seg](https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8s-seg.pt) | 640                   | 44.6                 | 36.8                  | 155.7                          | 1.47                                | 11.8               | 42.6              |
        | [YOLOv8m-seg](https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8m-seg.pt) | 640                   | 49.9                 | 40.8                  | 317.0                          | 2.18                                | 27.3               | 110.2             |
        | [YOLOv8l-seg](https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8l-seg.pt) | 640                   | 52.3                 | 42.6                  | 572.4                          | 2.79                                | 46.0               | 220.5             |
        | [YOLOv8x-seg](https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8x-seg.pt) | 640                   | 53.4                 | 43.4                  | 712.1                          | 4.02                                | 71.8               | 344.1             |

    === "分類（ImageNet）"

        | モデル                                                                                        | サイズ<br><sup>(ピクセル) | acc<br><sup>top1 | acc<br><sup>top5 | 速度<br><sup>CPU ONNX<br>(ミリ秒) | 速度<br><sup>A100 TensorRT<br>(ミリ秒) | パラメータ<br><sup>(百万) | FLOPs<br><sup>(640ピクセル) |
        | -------------------------------------------------------------------------------------------- | --------------------- | ---------------- | ---------------- | ------------------------------ | ----------------------------------- | ------------------ | ------------------------ |
        | [YOLOv8n-cls](https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n-cls.pt) | 224                   | 66.6             | 87.0             | 12.9                           | 0.31                                | 2.7                | 4.3                      |
        | [YOLOv8s-cls](https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8s-cls.pt) | 224                   | 72.3             | 91.1             | 23.4                           | 0.35                                | 6.4                | 13.5                     |
        | [YOLOv8m-cls](https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8m-cls.pt) | 224                   | 76.4             | 93.2             | 85.4                           | 0.62                                | 17.0               | 42.7                     |
        | [YOLOv8l-cls](https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8l-cls.pt) | 224                   | 78.0             | 94.1             | 163.0                          | 0.87                                | 37.5               | 99.7                     |
        | [YOLOv8x-cls](https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8x-cls.pt) | 224                   | 78.4             | 94.3             | 232.0                          | 1.01                                | 57.4               | 154.8                    |

    === "ポーズ（COCO）"

        | モデル                                                                                                | サイズ<br><sup>(ピクセル) | mAP<sup>pose<br>50-95 | mAP<sup>pose<br>50 | 速度<br><sup>CPU ONNX<br>(ミリ秒) | 速度<br><sup>A100 TensorRT<br>(ミリ秒) | パラメータ<br><sup>(百万) | FLOPs<br><sup>(十億) |
        | ---------------------------------------------------------------------------------------------------- | --------------------- | --------------------- | ------------------ | ------------------------------ | ----------------------------------- | ------------------ | ----------------- |
        | [YOLOv8n-pose](https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n-pose.pt)       | 640                   | 50.4                  | 80.1               | 131.8                          | 1.18                                | 3.3                | 9.2               |
        | [YOLOv8s-pose](https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8s-pose.pt)       | 640                   | 60.0                  | 86.2               | 233.2                          | 1.42                                | 11.6               | 30.2              |
        | [YOLOv8m-pose](https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8m-pose.pt)       | 640                   | 65.0                  | 88.8               | 456.3                          | 2.00                                | 26.4               | 81.0              |
        | [YOLOv8l-pose](https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8l-pose.pt)       | 640                   | 67.6                  | 90.0               | 784.5                          | 2.59                                | 44.4               | 168.6             |
        | [YOLOv8x-pose](https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8x-pose.pt)       | 640                   | 69.2                  | 90.2               | 1607.1                         | 3.73                                | 69.4               | 263.2             |
        | [YOLOv8x-pose-p6](https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8x-pose-p6.pt) | 1280                  | 71.6                  | 91.2               | 4088.7                         | 10.04                               | 99.1               | 1066.4            |

## 使用方法

Ultralyticsのpipパッケージを使用して、物体検出タスクにYOLOv8を使用することができます。以下は、推論のためにYOLOv8モデルを使用するコードのサンプルです。

!!! 例 ""

    この例では、YOLOv8の単純な推論コードを提供します。推論結果の処理など、より詳細なオプションについては、[Predict](../modes/predict.md)モードを参照してください。他のモードでYOLOv8を使用する方法については、[Train](../modes/train.md)、[Val](../modes/val.md)、および[Export](../modes/export.md)を参照してください。

    === "Python"

        PyTorchの事前学習済みの `*.pt` モデルと設定の `*.yaml` ファイルは、`YOLO()` クラスに渡すことで、pythonでモデルのインスタンスを作成できます。

        ```python
        from ultralytics import YOLO

        # COCO事前学習済みのYOLOv8nモデルをロードします
        model = YOLO('yolov8n.pt')

        # モデルの情報を表示します（オプション）
        model.info()

        # COCO8の例のデータセットでモデルを100エポックトレーニングします
        results = model.train(data='coco8.yaml', epochs=100, imgsz=640)

        # YOLOv8nモデルを使用して'bus.jpg'イメージ上で推論を実行します
        results = model('path/to/bus.jpg')
        ```

    === "CLI"

        CLIコマンドを使用すると、直接モデルを実行できます。

        ```bash
        # COCO事前学習済みのYOLOv8nモデルをロードし、COCO8の例のデータセットで100エポックトレーニングします
        yolo train model=yolov8n.pt data=coco8.yaml epochs=100 imgsz=640

        # COCO事前学習済みのYOLOv8nモデルをロードし、'bus.jpg'イメージ上で推論を実行します
        yolo predict model=yolov8n.pt source=path/to/bus.jpg
        ```

## 引用および謝辞

このリポジトリからYOLOv8モデルまたはその他のソフトウェアを使用する場合は、以下の形式で引用してください。

!!! Note ""

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

DOIは保留中であり、利用規約に従ってソフトウェアを使用しています。
