---
comments: true
description: YOLOv5uは、改良された精度と速度のトレードオフと、さまざまな物体検出タスク向けの多数の事前トレーニングモデルを備えたYOLOv5モデルの進化バージョンです。
keywords: YOLOv5u, 物体検出, 事前トレーニングモデル, Ultralytics, Inference, Validation, YOLOv5, YOLOv8, アンカーフリー, オブジェクトフリー, リアルタイムアプリケーション, 機械学習
---

# YOLOv5

## 概要

YOLOv5uは、物体検出方法論の進歩を表しています。Ultralyticsが開発した[YOLOv5](https://github.com/ultralytics/yolov5)モデルの基本アーキテクチャを起源とするYOLOv5uは、アンカーフリーでオブジェクトフリーの分割ヘッドを採用しており、以前の[YOLOv8](yolov8.md)モデルで導入された特徴です。この適応により、モデルのアーキテクチャが洗練され、物体検出タスクにおける改善された精度と速度のトレードオフが実現されます。経験的な結果と派生した特徴から明らかなとおり、YOLOv5uは、研究と実用の両方で堅牢なソリューションを求める人々にとって効率的な選択肢です。

![Ultralytics YOLOv5](https://raw.githubusercontent.com/ultralytics/assets/main/yolov5/v70/splash.png)

## 主な特徴

- **アンカーフリーな分割Ultralyticsヘッド：** 伝統的な物体検出モデルは、事前に定義されたアンカーボックスを使用してオブジェクトの位置を予測します。しかし、YOLOv5uはこのアプローチを近代化しています。アンカーフリーな分割Ultralyticsヘッドを採用することで、より柔軟かつ適応性のある検出メカニズムが確保され、さまざまなシナリオでのパフォーマンスが向上します。

- **最適化された精度と速度のトレードオフ：** 速度と精度はしばしば反対の方向に引っ張られます。しかし、YOLOv5uはこのトレードオフに挑戦しています。リアルタイムの検出を確保しながら、精度を損なうことなく、キャリブレーションされたバランスを提供します。この機能は、自動車、ロボット工学、リアルタイムビデオ解析など、迅速な応答を必要とするアプリケーションに特に有用です。

- **さまざまな事前トレーニングモデル：** 異なるタスクには異なるツールセットが必要であることを理解して、YOLOv5uはさまざまな事前トレーニングモデルを提供しています。Inference、Validation、Trainingに焦点を当てていても、ユーザーには待ち受けている特別に調整されたモデルがあります。この多様性により、ワンサイズがすべての解決策ではなく、一意の課題に特化したモデルを使用することができます。

## サポートされるタスクとモード

各種の事前トレーニング済みのYOLOv5uモデルは、[物体検出](../tasks/detect.md)タスクで優れたパフォーマンスを発揮します。[Inference](../modes/predict.md)、[Validation](../modes/val.md)、[Training](../modes/train.md)、および[Export](../modes/export.md)などのさまざまなモードをサポートしているため、開発から展開まで幅広いアプリケーションに適しています。

| モデルの種類  | 事前トレーニング済みの重み                                                                                                               | タスク                        | 推論 | 汎化 | トレーニング | エクスポート |
|---------|-----------------------------------------------------------------------------------------------------------------------------|----------------------------|----|----|--------|--------|
| YOLOv5u | `yolov5nu`, `yolov5su`, `yolov5mu`, `yolov5lu`, `yolov5xu`, `yolov5n6u`, `yolov5s6u`, `yolov5m6u`, `yolov5l6u`, `yolov5x6u` | [物体検出](../tasks/detect.md) | ✅  | ✅  | ✅      | ✅      |

この表では、YOLOv5uモデルのバリアントについて詳細な概要を提供し、物体検出タスクでの適用可能性と、[Inference](../modes/predict.md)、[Validation](../modes/val.md)、[Training](../modes/train.md)、[Export](../modes/export.md)などのさまざまな操作モードのサポートを強調しています。この包括的なサポートにより、ユーザーは広範な物体検出シナリオでYOLOv5uモデルの機能を十分に活用することができます。

## パフォーマンスメトリクス

!!! パフォーマンス

    === "検出"

    [COCO](https://docs.ultralytics.com/datasets/detect/coco/)でトレーニングされたこれらのモデルを使用した使用例については、[検出ドキュメント](https://docs.ultralytics.com/tasks/detect/)を参照してください。これらのモデルには80の事前トレーニングクラスが含まれています。

    | モデル                                                                                     | YAML                                                                                                           | サイズ<br><sup>(pixels) | mAP<sup>val<br>50-95 | 速度<br><sup>CPU ONNX<br>(ms) | 速度<br><sup>A100 TensorRT<br>(ms) | パラメータ<br><sup>(M) | FLOPS<br><sup>(B) |
    |-------------------------------------------------------------------------------------------|----------------------------------------------------------------------------------------------------------------|-----------------------|----------------------|----------------------------|-------------------------------------|----------------------|-------------------|
    | [yolov5nu.pt](https://github.com/ultralytics/assets/releases/download/v8.1.0/yolov5nu.pt) | [yolov5n.yaml](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/models/v5/yolov5.yaml)     | 640                   | 34.3                 | 73.6                       | 1.06                                | 2.6                  | 7.7               |
    | [yolov5su.pt](https://github.com/ultralytics/assets/releases/download/v8.1.0/yolov5su.pt) | [yolov5s.yaml](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/models/v5/yolov5.yaml)     | 640                   | 43.0                 | 120.7                      | 1.27                                | 9.1                  | 24.0              |
    | [yolov5mu.pt](https://github.com/ultralytics/assets/releases/download/v8.1.0/yolov5mu.pt) | [yolov5m.yaml](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/models/v5/yolov5.yaml)     | 640                   | 49.0                 | 233.9                      | 1.86                                | 25.1                 | 64.2              |
    | [yolov5lu.pt](https://github.com/ultralytics/assets/releases/download/v8.1.0/yolov5lu.pt) | [yolov5l.yaml](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/models/v5/yolov5.yaml)     | 640                   | 52.2                 | 408.4                      | 2.50                                | 53.2                 | 135.0             |
    | [yolov5xu.pt](https://github.com/ultralytics/assets/releases/download/v8.1.0/yolov5xu.pt) | [yolov5x.yaml](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/models/v5/yolov5.yaml)     | 640                   | 53.2                 | 763.2                      | 3.81                                | 97.2                 | 246.4             |
    |                                                                                           |                                                                                                                |                       |                      |                            |                                     |                      |                   |
    | [yolov5n6u.pt](https://github.com/ultralytics/assets/releases/download/v8.1.0/yolov5n6u.pt) | [yolov5n6.yaml](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/models/v5/yolov5-p6.yaml) | 1280                  | 42.1                 | 211.0                      | 1.83                                | 4.3                  | 7.8               |
    | [yolov5s6u.pt](https://github.com/ultralytics/assets/releases/download/v8.1.0/yolov5s6u.pt) | [yolov5s6.yaml](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/models/v5/yolov5-p6.yaml) | 1280                  | 48.6                 | 422.6                      | 2.34                                | 15.3                 | 24.6              |
    | [yolov5m6u.pt](https://github.com/ultralytics/assets/releases/download/v8.1.0/yolov5m6u.pt) | [yolov5m6.yaml](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/models/v5/yolov5-p6.yaml) | 1280                  | 53.6                 | 810.9                      | 4.36                                | 41.2                 | 65.7              |
    | [yolov5l6u.pt](https://github.com/ultralytics/assets/releases/download/v8.1.0/yolov5l6u.pt) | [yolov5l6.yaml](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/models/v5/yolov5-p6.yaml) | 1280                  | 55.7                 | 1470.9                     | 5.47                                | 86.1                 | 137.4             |
    | [yolov5x6u.pt](https://github.com/ultralytics/assets/releases/download/v8.1.0/yolov5x6u.pt) | [yolov5x6.yaml](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/models/v5/yolov5-p6.yaml) | 1280                  | 56.8                 | 2436.5                     | 8.98                                | 155.4                | 250.7             |

## 使用例

この例では、単純なYOLOv5のトレーニングと推論の使用例を提供します。これらと他の[モード](../modes/index.md)の完全なドキュメントについては、[Predict](../modes/predict.md)、[Train](../modes/train.md)、[Val](../modes/val.md)、[Export](../modes/export.md)のドキュメントページを参照してください。

!!! Example "例"

    === "Python"

        Pythonでモデルインスタンスを作成するには、PyTorchの事前トレーニング済みの`*.pt`モデルおよび構成`*.yaml`ファイルを`YOLO()`クラスに渡すことができます。

        ```python
        from ultralytics import YOLO

        # COCOで事前トレーニング済みのYOLOv5nモデルをロードする
        model = YOLO('yolov5n.pt')

        # モデル情報を表示する（任意）
        model.info()

        # COCO8の例のデータセットでモデルを100エポックトレーニングする
        results = model.train(data='coco8.yaml', epochs=100, imgsz=640)

        # YOLOv5nモデルを使用して'bus.jpg'画像で推論を実行する
        results = model('path/to/bus.jpg')
        ```

    === "CLI"

        CLIコマンドを使用してモデルを直接実行することもできます。

        ```bash
        # COCOで事前トレーニング済みのYOLOv5nモデルをロードし、COCO8の例のデータセットで100エポックトレーニングする
        yolo train model=yolov5n.pt data=coco8.yaml epochs=100 imgsz=640

        # COCOで事前トレーニング済みのYOLOv5nモデルをロードし、'bus.jpg'画像で推論を実行する
        yolo predict model=yolov5n.pt source=path/to/bus.jpg
        ```

## 引用および謝辞

研究でYOLOv5またはYOLOv5uを使用する場合は、以下のようにUltralytics YOLOv5リポジトリを引用してください：

!!! Quote ""

    === "BibTeX"
        ```bibtex
        @software{yolov5,
          title = {Ultralytics YOLOv5},
          author = {Glenn Jocher},
          year = {2020},
          version = {7.0},
          license = {AGPL-3.0},
          url = {https://github.com/ultralytics/yolov5},
          doi = {10.5281/zenodo.3908559},
          orcid = {0000-0001-5950-6979}
        }
        ```

なお、YOLOv5モデルは[AGPL-3.0](https://github.com/ultralytics/ultralytics/blob/main/LICENSE)および[Enterprise](https://ultralytics.com/license)ライセンスの下で提供されています。
