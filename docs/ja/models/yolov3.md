---
comments: true
description: YOLOv3、YOLOv3-Ultralytics、およびYOLOv3uの概要を把握してください。オブジェクト検出に対するこれらのモデルの主な特徴、使用方法、およびサポートされるタスクについて学びます。
keywords: YOLOv3, YOLOv3-Ultralytics, YOLOv3u, オブジェクト検出, 推論, トレーニング, Ultralytics
---

# YOLOv3、YOLOv3-Ultralytics、およびYOLOv3u

## 概要

このドキュメントでは、[YOLOv3](https://pjreddie.com/darknet/yolo/)、[YOLOv3-Ultralytics](https://github.com/ultralytics/yolov3)、および[YOLOv3u](https://github.com/ultralytics/ultralytics)という3つの関連するオブジェクト検出モデルについて概説します。

1. **YOLOv3:** これはYou Only Look Once (YOLO) オブジェクト検出アルゴリズムの3番目のバージョンです。Joseph Redmonによって最初に開発されたYOLOv3は、マルチスケール予測や3つの異なるサイズの検出カーネルなど、さまざまな機能を導入し、前バージョンよりも性能を向上させました。

2. **YOLOv3-Ultralytics:** これはUltralyticsによるYOLOv3モデルの実装です。オリジナルのYOLOv3アーキテクチャを再現し、より多くの事前学習済みモデルのサポートや簡単なカスタマイズオプションなど、追加の機能を提供します。

3. **YOLOv3u:** これはYOLOv8モデルで使用されるアンカーフリーでオブジェクトネスフリーなスプリットヘッドを組み込んだYOLOv3-Ultralyticsの更新版です。YOLOv3uは、YOLOv3と同じバックボーンとネックアーキテクチャを保持しながら、YOLOv8の更新された検出ヘッドを備えています。

![Ultralytics YOLOv3](https://raw.githubusercontent.com/ultralytics/assets/main/yolov3/banner-yolov3.png)

## 主な特徴

- **YOLOv3:** 3つの異なる検出スケールを使用し、13x13、26x26、および52x52の3つの異なるサイズの検出カーネルを活用しました。これにより、さまざまなサイズのオブジェクトの検出精度が大幅に向上しました。さらに、YOLOv3では各バウンディングボックスの複数のラベル予測や、より良い特徴抽出ネットワークなどの機能が追加されました。

- **YOLOv3-Ultralytics:** UltralyticsのYOLOv3の実装は、オリジナルのモデルと同じ性能を提供しながら、より多くの事前学習済みモデルのサポート、追加のトレーニング方法、および簡単なカスタマイズオプションを提供します。これにより、実用的なアプリケーションにおいてより柔軟で使いやすくなります。

- **YOLOv3u:** この更新されたモデルは、YOLOv8から使用されているアンカーフリーでオブジェクトネスフリーなスプリットヘッドを組み込んでいます。事前定義されたアンカーボックスとオブジェクトネススコアの必要性を排除することで、この検出ヘッドの設計は、さまざまなサイズや形状のオブジェクトの検出能力を向上させることができます。これにより、YOLOv3uはオブジェクト検出タスクにおいてより堅牢で正確なモデルとなります。

## サポートされるタスクとモード

YOLOv3シリーズ、YOLOv3、YOLOv3-Ultralytics、およびYOLOv3uは、オブジェクト検出タスクに特化して設計されています。これらのモデルは、精度と速度のバランスを取りながらさまざまな実世界のシナリオでの効果が高いことで知られています。各バリアントはユニークな機能と最適化を提供し、さまざまなアプリケーションに適しています。

3つのモデルは[推論](../modes/predict.md)、[検証](../modes/val.md)、[トレーニング](../modes/train.md)、および[エクスポート](../modes/export.md)など、幅広いモードをサポートしており、効果的なオブジェクト検出のための完全なツールキットを提供します。

| モデルの種類             | サポートされるタスク                     | 推論 | 検証 | トレーニング | エクスポート |
|--------------------|--------------------------------|----|----|--------|--------|
| YOLOv3             | [オブジェクト検出](../tasks/detect.md) | ✅  | ✅  | ✅      | ✅      |
| YOLOv3-Ultralytics | [オブジェクト検出](../tasks/detect.md) | ✅  | ✅  | ✅      | ✅      |
| YOLOv3u            | [オブジェクト検出](../tasks/detect.md) | ✅  | ✅  | ✅      | ✅      |

この表は、各YOLOv3バリアントの機能を一目で把握するためのもので、オブジェクト検出ワークフローのさまざまなタスクと操作モードに対する多様性と適用性を強調しています。

## 使用例

この例では、YOLOv3の簡単なトレーニングおよび推論の例を提供します。これらおよびその他の[モード](../modes/index.md)の完全なドキュメンテーションについては、[Predict](../modes/predict.md)、[Train](../modes/train.md)、[Val](../modes/val.md)、および[Export](../modes/export.md)のドキュメントページを参照してください。

!!! Example "例"

    === "Python"

        PyTorchの事前学習済み `*.pt` モデルと設定 `*.yaml` ファイルは、`YOLO()` クラスに渡してモデルインスタンスを作成できます。

        ```python
        from ultralytics import YOLO

        # COCOで学習済みのYOLOv3nモデルをロード
        model = YOLO('yolov3n.pt')

        # モデル情報の表示（任意）
        model.info()

        # COCO8のサンプルデータセットでモデルを100エポックトレーニング
        results = model.train(data='coco8.yaml', epochs=100, imgsz=640)

        # YOLOv3nモデルで'bus.jpg'画像に対して推論実行
        results = model('path/to/bus.jpg')
        ```

    === "CLI"

        CLIコマンドを使用して直接モデルを実行できます。

        ```bash
        # COCOで学習済みのYOLOv3nモデルをロードし、COCO8のサンプルデータセットで100エポックトレーニング
        yolo train model=yolov3n.pt data=coco8.yaml epochs=100 imgsz=640

        # COCOで学習済みのYOLOv3nモデルをロードし、'bus.jpg'画像に対して推論実行
        yolo predict model=yolov3n.pt source=path/to/bus.jpg
        ```

## 引用と謝辞

研究でYOLOv3を使用する場合は、元のYOLO論文とUltralyticsのYOLOv3リポジトリを引用してください。

!!! Quote ""

    === "BibTeX"

        ```bibtex
        @article{redmon2018yolov3,
          title={YOLOv3: An Incremental Improvement},
          author={Redmon, Joseph and Farhadi, Ali},
          journal={arXiv preprint arXiv:1804.02767},
          year={2018}
        }
        ```

Joseph RedmonとAli Farhadiには、オリジナルのYOLOv3を開発していただいたことに感謝します。
