---
comments: true
description: UltralyticsがサポートするYOLOファミリー、SAM、MobileSAM、FastSAM、YOLO-NAS、RT-DETRモデルの多様な範囲を探索し、CLIおよびPythonの使用例で始めましょう。
keywords: Ultralytics, ドキュメント, YOLO, SAM, MobileSAM, FastSAM, YOLO-NAS, RT-DETR, モデル, アーキテクチャ, Python, CLI
---

# Ultralyticsがサポートするモデル

Ultralyticsのモデルドキュメントへようこそ！我々は、[オブジェクト検出](../tasks/detect.md)、[インスタンスセグメンテーション](../tasks/segment.md)、[画像分類](../tasks/classify.md)、[ポーズ推定](../tasks/pose.md)、[多対象トラッキング](../modes/track.md)などの特定のタスクに特化した幅広いモデルのサポートを提供しています。Ultralyticsにあなたのモデルアーキテクチャを貢献したい場合は、[貢献ガイド](../../help/contributing.md)を確認してください。

!!! Note "注意"

    🚧 現在、さまざまな言語でのドキュメントを構築中であり、改善に努めています。ご理解ありがとうございます！🙏

## 特集モデル

ここではサポートされている主要なモデルをいくつか紹介します：

1. **[YOLOv3](yolov3.md)**：Joseph RedmonによるYOLOモデルファミリーの第三世代で、効率的なリアルタイムオブジェクト検出能力で知られています。
2. **[YOLOv4](yolov4.md)**：2020年にAlexey BochkovskiyによってリリースされたYOLOv3のdarknetネイティブアップデートです。
3. **[YOLOv5](yolov5.md)**：UltralyticsによるYOLOアーキテクチャの改良版で、以前のバージョンと比較してパフォーマンスと速度のトレードオフが向上しています。
4. **[YOLOv6](yolov6.md)**：2022年に[美団](https://about.meituan.com/)によってリリースされ、同社の多数の自動配送ロボットで使用されています。
5. **[YOLOv7](yolov7.md)**：YOLOv4の著者によって2022年にリリースされたYOLOモデルのアップデートです。
6. **[YOLOv8](yolov8.md) 新機能 🚀**：YOLOファミリーの最新バージョンで、例えばインスタンスセグメンテーション、ポーズ/キーポイント推定、分類などの機能が強化されています。
7. **[Segment Anything Model (SAM)](sam.md)**：MetaのSegment Anything Model (SAM)です。
8. **[Mobile Segment Anything Model (MobileSAM)](mobile-sam.md)**：慶應義塾大学によるモバイルアプリケーションのためのMobileSAMです。
9. **[Fast Segment Anything Model (FastSAM)](fast-sam.md)**：中国科学院自動化研究所、画像及びビデオ解析グループのFastSAMです。
10. **[YOLO-NAS](yolo-nas.md)**：YOLO Neural Architecture Search (NAS)モデルです。
11. **[Realtime Detection Transformers (RT-DETR)](rtdetr.md)**：百度のPaddlePaddle Realtime Detection Transformer (RT-DETR)モデルです。

<p align="center">
  <br>
  <iframe width="720" height="405" src="https://www.youtube.com/embed/MWq1UxqTClU?si=nHAW-lYDzrz68jR0"
    title="YouTube動画プレイヤー" frameborder="0"
    allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share"
    allowfullscreen>
  </iframe>
  <br>
  <strong>視聴:</strong> Ultralytics YOLOモデルをわずか数行のコードで実行します。
</p>

## Getting Started: 使用例

この例は、YOLOのトレーニングと推論の簡単な例を提供します。これらおよびその他の[モード](../modes/index.md)についての完全なドキュメントについては、[Predict](../modes/predict.md)、[Train](../modes/train.md)、[Val](../modes/val.md)、[Export](../modes/export.md)のドキュメントページを参照してください。

以下の例は、オブジェクト検出のためのYOLOv8 [Detect](../tasks/detect.md) モデルについてです。追加のサポートされるタスクについては、[Segment](../tasks/segment.md)、[Classify](../tasks/classify.md)、[Pose](../tasks/pose.md)のドキュメントを参照してください。

!!! Example "例"

    === "Python"

        PyTorchの事前訓練済み`*.pt`モデルや構成`*.yaml`ファイルは、`YOLO()`、`SAM()`、`NAS()`、`RTDETR()`クラスに渡して、Pythonでモデルインスタンスを作成することができます：

        ```python
        from ultralytics import YOLO

        # COCOで事前訓練されたYOLOv8nモデルをロードする
        model = YOLO('yolov8n.pt')

        # モデル情報を表示する（任意）
        model.info()

        # モデルをCOCO8の例示データセットで100エポックトレーニングする
        results = model.train(data='coco8.yaml', epochs=100, imgsz=640)

        # 'bus.jpg'画像でYOLOv8nモデルを用いた推論を実行する
        results = model('path/to/bus.jpg')
        ```

    === "CLI"

        モデルを直接実行するためのCLIコマンドが利用可能です：

        ```bash
        # COCOで事前訓練されたYOLOv8nモデルをロードし、COCO8の例示データセットで100エポックトレーニングする
        yolo train model=yolov8n.pt data=coco8.yaml epochs=100 imgsz=640

        # COCOで事前訓練されたYOLOv8nモデルをロードし、'bus.jpg'画像で推論を実行する
        yolo predict model=yolov8n.pt source=path/to/bus.jpg
        ```

## 新しいモデルの貢献

Ultralyticsにあなたのモデルを貢献することに興味がありますか？素晴らしいです！我々は常にモデルのポートフォリオを拡張することに興味があります。

1. **リポジトリをフォークする**：[Ultralytics GitHubリポジトリ](https://github.com/ultralytics/ultralytics)をフォークすることから始めます。

2. **あなたのフォークをクローンする**：あなたのフォークをローカルマシンにクローンし、作業を行う新しいブランチを作成します。

3. **あなたのモデルを実装する**：[貢献ガイド](../../help/contributing.md)に示されているコーディング規格および指針に従ってモデルを追加します。

4. **徹底的にテストする**：パイプラインの一部としてだけでなく、単独でモデルを厳密にテストすることを確認してください。

5. **プルリクエストを作成する**：モデルに満足したら、レビューのために本リポジトリにプルリクエストを作成します。

6. **コードレビュー＆マージ**：レビュー後、モデルが我々の基準を満たしている場合、本リポジトリにマージされます。

詳細な手順については、[貢献ガイド](../../help/contributing.md)を参照してください。
