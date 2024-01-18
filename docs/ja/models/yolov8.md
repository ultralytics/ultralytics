---
comments: true
description: YOLOv8の魅力的な機能を探索しましょう。これは当社のリアルタイムオブジェクト検出器の最新バージョンです！高度なアーキテクチャ、事前学習済みモデル、そして精度と速度の最適なバランスがYOLOv8を完璧なオブジェクト検出タスクの選択肢とします。
keywords: YOLOv8, Ultralytics, リアルタイムオブジェクト検出器, 事前学習済みモデル, ドキュメント, オブジェクト検出, YOLOシリーズ, 高度なアーキテクチャ, 精度, 速度
---

# YOLOv8

## 概要

YOLOv8は、リアルタイムオブジェクト検出器のYOLOシリーズの最新版であり、精度と速度において最先端の性能を提供します。以前のYOLOバージョンの進化を基にして、YOLOv8は新機能と最適化を導入し、様々なオブジェクト検出タスクにおいて広範なアプリケーションに適した選択肢となります。

![Ultralytics YOLOv8](https://raw.githubusercontent.com/ultralytics/assets/main/yolov8/yolo-comparison-plots.png)

## 主な特徴

- **高度なバックボーンおよびネックアーキテクチャ:** YOLOv8は最先端のバックボーンとネックアーキテクチャを採用しており、特徴抽出とオブジェクト検出のパフォーマンスが向上しています。
- **アンカーフリーな分割Ultralyticsヘッド:** YOLOv8はアンカーベースの方法と比べて、アンカーフリーな分割Ultralyticsヘッドを採用し、より高い精度と効率的な検出プロセスに貢献しています。
- **最適な精度と速度のトレードオフの最適化:** 精度と速度の最適なバランスを保ちながら、YOLOv8は多岐にわたるリアルタイムオブジェクト検出タスクに適しています。
- **さまざまな事前学習済みモデル:** YOLOv8はさまざまなタスクやパフォーマンス要件に対応する事前学習済みモデルを提供しており、特定のユースケースに適したモデルを簡単に見つけることができます。

## サポートされるタスクとモード

YOLOv8シリーズは、コンピュータビジョンのさまざまなタスクに特化したさまざまなモデルを提供しています。これらのモデルは、オブジェクト検出からインスタンスセグメンテーション、ポーズ/キーポイントの検出、分類などのより複雑なタスクまで、さまざまな要件に対応するように設計されています。

YOLOv8シリーズの各バリアントは、それぞれのタスクに最適化されており、高いパフォーマンスと精度を提供します。また、これらのモデルは、推論、検証、トレーニング、エクスポートといったさまざまな運用モードと互換性があり、展開と開発のさまざまな段階で使用することができます。

| モデル         | ファイル名                                                                                                          | タスク                                    | 推論 | 検証 | トレーニング | エクスポート |
|-------------|----------------------------------------------------------------------------------------------------------------|----------------------------------------|----|----|--------|--------|
| YOLOv8      | `yolov8n.pt` `yolov8s.pt` `yolov8m.pt` `yolov8l.pt` `yolov8x.pt`                                               | [検出](../tasks/detect.md)               | ✅  | ✅  | ✅      | ✅      |
| YOLOv8-seg  | `yolov8n-seg.pt` `yolov8s-seg.pt` `yolov8m-seg.pt` `yolov8l-seg.pt` `yolov8x-seg.pt`                           | [インスタンスセグメンテーション](../tasks/segment.md) | ✅  | ✅  | ✅      | ✅      |
| YOLOv8-pose | `yolov8n-pose.pt` `yolov8s-pose.pt` `yolov8m-pose.pt` `yolov8l-pose.pt` `yolov8x-pose.pt` `yolov8x-pose-p6.pt` | [ポーズ/キーポイント](../tasks/pose.md)         | ✅  | ✅  | ✅      | ✅      |
| YOLOv8-cls  | `yolov8n-cls.pt` `yolov8s-cls.pt` `yolov8m-cls.pt` `yolov8l-cls.pt` `yolov8x-cls.pt`                           | [分類](../tasks/classify.md)             | ✅  | ✅  | ✅      | ✅      |

この表は、YOLOv8モデルのバリアントの概要を提供し、特定のタスクでの適用性と推論、検証、トレーニング、エクスポートなどのさまざまな運用モードとの互換性を示しています。これにより、YOLOv8シリーズの多様性と堅牢性が示されており、コンピュータビジョンのさまざまなアプリケーションに適しています。

## 性能メトリクス

!!! パフォーマンス

    === "検出（COCO）"

        [COCO](https://docs.ultralytics.com/datasets/detect/coco/)でトレーニングされたこれらのモデルを使用した使用例については、[検出ドキュメント](https://docs.ultralytics.com/tasks/detect/)を参照してください。これらのモデルには80の事前学習済みクラスが含まれています。

        | モデル                                                                                | サイズ（ピクセル） | mAP<sup>val<br>50-95 | スピード（CPU ONNX<br>（ミリ秒）） | スピード（A100 TensorRT<br>（ミリ秒）） | パラメータ数（百万） | FLOPs（10億） |
        | ------------------------------------------------------------------------------------ | --------------------- | -------------------- | ------------------------------ | ----------------------------------- | ------------------ | ----------------- |
        | [YOLOv8n](https://github.com/ultralytics/assets/releases/download/v8.1.0/yolov8n.pt) | 640                   | 37.3                 | 80.4                           | 0.99                                | 3.2                | 8.7               |
        | [YOLOv8s](https://github.com/ultralytics/assets/releases/download/v8.1.0/yolov8s.pt) | 640                   | 44.9                 | 128.4                          | 1.20                                | 11.2               | 28.6              |
        | [YOLOv8m](https://github.com/ultralytics/assets/releases/download/v8.1.0/yolov8m.pt) | 640                   | 50.2                 | 234.7                          | 1.83                                | 25.9               | 78.9              |
        | [YOLOv8l](https://github.com/ultralytics/assets/releases/download/v8.1.0/yolov8l.pt) | 640                   | 52.9                 | 375.2                          | 2.39                                | 43.7               | 165.2             |
        | [YOLOv8x](https://github.com/ultralytics/assets/releases/download/v8.1.0/yolov8x.pt) | 640                   | 53.9                 | 479.1                          | 3.53                                | 68.2               | 257.8             |

    === "検出（Open Images V7）"

        [Open Image V7](https://docs.ultralytics.com/datasets/detect/open-images-v7/)でトレーニングされたこれらのモデルを使用した使用例については、[検出ドキュメント](https://docs.ultralytics.com/tasks/detect/)を参照してください。これらのモデルには600の事前訓練クラスが含まれています。

        | モデル                                                                                     | サイズ（ピクセル） | mAP<sup>val<br>50-95 | スピード（CPU ONNX<br>（ミリ秒）） | スピード（A100 TensorRT<br>（ミリ秒）） | パラメータ数（百万） | FLOPs（10億） |
        | ----------------------------------------------------------------------------------------- | --------------------- | -------------------- | ------------------------------ | ----------------------------------- | ------------------ | ----------------- |
        | [YOLOv8n](https://github.com/ultralytics/assets/releases/download/v8.1.0/yolov8n-oiv7.pt) | 640                   | 18.4                 | 142.4                          | 1.21                                | 3.5                | 10.5              |
        | [YOLOv8s](https://github.com/ultralytics/assets/releases/download/v8.1.0/yolov8s-oiv7.pt) | 640                   | 27.7                 | 183.1                          | 1.40                                | 11.4               | 29.7              |
        | [YOLOv8m](https://github.com/ultralytics/assets/releases/download/v8.1.0/yolov8m-oiv7.pt) | 640                   | 33.6                 | 408.5                          | 2.26                                | 26.2               | 80.6              |
        | [YOLOv8l](https://github.com/ultralytics/assets/releases/download/v8.1.0/yolov8l-oiv7.pt) | 640                   | 34.9                 | 596.9                          | 2.43                                | 44.1               | 167.4             |
        | [YOLOv8x](https://github.com/ultralytics/assets/releases/download/v8.1.0/yolov8x-oiv7.pt) | 640                   | 36.3                 | 860.6                          | 3.56                                | 68.7               | 260.6             |

    === "セグメンテーション（COCO）"

        [COCO](https://docs.ultralytics.com/datasets/segment/coco/)でトレーニングされたこれらのモデルを使用した使用例については、[セグメンテーションドキュメント](https://docs.ultralytics.com/tasks/segment/)を参照してください。これらのモデルには80の事前学習済みクラスが含まれています。

        | モデル                                                                                        | サイズ（ピクセル） | mAP<sup>box<br>50-95 | mAP<sup>mask<br>50-95 | スピード（CPU ONNX<br>（ミリ秒）） | スピード（A100 TensorRT<br>（ミリ秒）） | パラメータ数（百万） | FLOPs（10億） |
        | -------------------------------------------------------------------------------------------- | --------------------- | -------------------- | --------------------- | ------------------------------ | ----------------------------------- | ------------------ | ----------------- |
        | [YOLOv8n-seg](https://github.com/ultralytics/assets/releases/download/v8.1.0/yolov8n-seg.pt) | 640                   | 36.7                 | 30.5                  | 96.1                           | 1.21                                | 3.4                | 12.6              |
        | [YOLOv8s-seg](https://github.com/ultralytics/assets/releases/download/v8.1.0/yolov8s-seg.pt) | 640                   | 44.6                 | 36.8                  | 155.7                          | 1.47                                | 11.8               | 42.6              |
        | [YOLOv8m-seg](https://github.com/ultralytics/assets/releases/download/v8.1.0/yolov8m-seg.pt) | 640                   | 49.9                 | 40.8                  | 317.0                          | 2.18                                | 27.3               | 110.2             |
        | [YOLOv8l-seg](https://github.com/ultralytics/assets/releases/download/v8.1.0/yolov8l-seg.pt) | 640                   | 52.3                 | 42.6                  | 572.4                          | 2.79                                | 46.0               | 220.5             |
        | [YOLOv8x-seg](https://github.com/ultralytics/assets/releases/download/v8.1.0/yolov8x-seg.pt) | 640                   | 53.4                 | 43.4                  | 712.1                          | 4.02                                | 71.8               | 344.1             |

    === "分類（ImageNet）"

        [ImageNet](https://docs.ultralytics.com/datasets/classify/imagenet/)でトレーニングされたこれらのモデルを使用した使用例については、[分類ドキュメント](https://docs.ultralytics.com/tasks/classify/)を参照してください。これらのモデルには1000の事前学習済みクラスが含まれています。

        | モデル                                                                                        | サイズ（ピクセル） | top1精度 | top5精度 | スピード（CPU ONNX<br>（ミリ秒）） | スピード（A100 TensorRT<br>（ミリ秒）） | パラメータ数（百万） | FLOPs（640ピクセル時の10億） |
        | -------------------------------------------------------------------------------------------- | --------------------- | --------- | --------- | ------------------------------ | ----------------------------------- | ------------------ | ------------------------ |
        | [YOLOv8n-cls](https://github.com/ultralytics/assets/releases/download/v8.1.0/yolov8n-cls.pt) | 224                   | 66.6      | 87.0      | 12.9                           | 0.31                                | 2.7                | 4.3                      |
        | [YOLOv8s-cls](https://github.com/ultralytics/assets/releases/download/v8.1.0/yolov8s-cls.pt) | 224                   | 72.3      | 91.1      | 23.4                           | 0.35                                | 6.4                | 13.5                     |
        | [YOLOv8m-cls](https://github.com/ultralytics/assets/releases/download/v8.1.0/yolov8m-cls.pt) | 224                   | 76.4      | 93.2      | 85.4                           | 0.62                                | 17.0               | 42.7                     |
        | [YOLOv8l-cls](https://github.com/ultralytics/assets/releases/download/v8.1.0/yolov8l-cls.pt) | 224                   | 78.0      | 94.1      | 163.0                          | 0.87                                | 37.5               | 99.7                     |
        | [YOLOv8x-cls](https://github.com/ultralytics/assets/releases/download/v8.1.0/yolov8x-cls.pt) | 224                   | 78.4      | 94.3      | 232.0                          | 1.01                                | 57.4               | 154.8                    |

    === "ポーズ（COCO）"

        ['person']クラスの1つの事前学習済みクラスを含む、[COCO](https://docs.ultralytics.com/datasets/pose/coco/)でトレーニングされたこれらのモデルを使用した使用例については、[ポーズ推定ドキュメント](https://docs.ultralytics.com/tasks/pose/)を参照してください。

        | モデル                                                                                                | サイズ（ピクセル） | mAP<sup>pose<br>50-95 | mAP<sup>pose<br>50 | スピード（CPU ONNX<br>（ミリ秒）） | スピード（A100 TensorRT<br>（ミリ秒）） | パラメータ数（百万） | FLOPs（10億） |
        | ---------------------------------------------------------------------------------------------------- | --------------------- | --------------------- | ------------------ | ------------------------------ | ----------------------------------- | ------------------ | ----------------- |
        | [YOLOv8n-pose](https://github.com/ultralytics/assets/releases/download/v8.1.0/yolov8n-pose.pt)       | 640                   | 50.4                  | 80.1               | 131.8                          | 1.18                                | 3.3                | 9.2               |
        | [YOLOv8s-pose](https://github.com/ultralytics/assets/releases/download/v8.1.0/yolov8s-pose.pt)       | 640                   | 60.0                  | 86.2               | 233.2                          | 1.42                                | 11.6               | 30.2              |
        | [YOLOv8m-pose](https://github.com/ultralytics/assets/releases/download/v8.1.0/yolov8m-pose.pt)       | 640                   | 65.0                  | 88.8               | 456.3                          | 2.00                                | 26.4               | 81.0              |
        | [YOLOv8l-pose](https://github.com/ultralytics/assets/releases/download/v8.1.0/yolov8l-pose.pt)       | 640                   | 67.6                  | 90.0               | 784.5                          | 2.59                                | 44.4               | 168.6             |
        | [YOLOv8x-pose](https://github.com/ultralytics/assets/releases/download/v8.1.0/yolov8x-pose.pt)       | 640                   | 69.2                  | 90.2               | 1607.1                         | 3.73                                | 69.4               | 263.2             |
        | [YOLOv8x-pose-p6](https://github.com/ultralytics/assets/releases/download/v8.1.0/yolov8x-pose-p6.pt) | 1280                  | 71.6                  | 91.2               | 4088.7                         | 10.04                               | 99.1               | 1066.4            |

## 使用例

この例では、YOLOv8のシンプルなトレーニングと推論の使用例を提供しています。これらとその他のモードに関する完全なドキュメントについては、[Predict](../modes/predict.md)、[Train](../modes/train.md)、[Val](../modes/val.md)、および[Export](../modes/export.md)のドキュメントページを参照してください。

以下の例は、オブジェクト検出のためのYOLOv8 [Detect](../tasks/detect.md)モデル用のものです。他のサポートされるタスクについては、[Segment](../tasks/segment.md)、[Classify](../tasks/classify.md)、および[Pose](../tasks/pose.md)のドキュメントを参照してください。

!!! Example "例"

    === "Python"

        Pythonでの、`.pt`形式の事前学習済みPyTorchモデルおよび設定ファイル`*.yaml`を使用して、`YOLO()`クラスを介してモデルインスタンスを作成する方法の例です：

        ```python
        from ultralytics import YOLO

        # COCOで事前学習済みのYOLOv8nモデルをロード
        model = YOLO('yolov8n.pt')

        # モデルの情報を表示（オプション）
        model.info()

        # COCO8の例データセットで100エポックトレーニングを実行
        results = model.train(data='coco8.yaml', epochs=100, imgsz=640)

        # 'bus.jpg'画像に対してYOLOv8nモデルで推論を実行
        results = model('path/to/bus.jpg')
        ```

    === "CLI"

        直接モデルを実行するためのCLIコマンドも使用できます：

        ```bash
        # COCOで事前学習済みのYOLOv8nモデルをロードし、COCO8の例データセットで100エポックトレーニングを実行
        yolo train model=yolov8n.pt data=coco8.yaml epochs=100 imgsz=640

        # COCOで事前学習済みのYOLOv8nモデルをロードし、'bus.jpg'画像に対して推論を実行
        yolo predict model=yolov8n.pt source=path/to/bus.jpg
        ```

## 引用および謝辞

このリポジトリからYOLOv8モデルまたは他のソフトウェアを使用する場合は、次の形式で引用してください：

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

DOIは準備中であり、利用可能になり次第、引用に追加されます。YOLOv8モデルは[AGPL-3.0](https://github.com/ultralytics/ultralytics/blob/main/LICENSE)および[Enterprise](https://ultralytics.com/license)ライセンスで提供されています。
