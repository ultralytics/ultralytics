---
comments: true
description: UltralyticsがサポートするYOLOファミリー、SAM、MobileSAM、FastSAM、YOLO-NAS、RT-DETRモデルの多様な範囲を探る。CLIとPythonの両方の使用例で始める。
keywords: Ultralytics, ドキュメンテーション, YOLO, SAM, MobileSAM, FastSAM, YOLO-NAS, RT-DETR, モデル, アーキテクチャ, Python, CLI
---

# Ultralyticsによるサポートモデル

Ultralyticsのモデルドキュメンテーションへようこそ！[オブジェクト検出](../tasks/detect.md)、[インスタンスセグメンテーション](../tasks/segment.md)、[画像分類](../tasks/classify.md)、[ポーズ推定](../tasks/pose.md)、[マルチオブジェクトトラッキング](../modes/track.md)など、特定のタスクに適した幅広いモデルをサポートしています。Ultralyticsにあなたのモデルアーキテクチャを寄稿したい場合は、[コントリビューティングガイド](../../help/contributing.md)を確認してください。

!!! Note "ノート"

    🚧 弊社の多言語ドキュメンテーションは現在建設中で、改善に向けて努力しています。ご理解いただきありがとうございます！🙏

## 注目のモデル

以下はサポートされる主要なモデルのいくつかです：

1. **[YOLOv3](../../models/yolov3.md)**: ジョセフ・レッドモンによるYOLOモデルファミリーの第三世代で、効率的なリアルタイムオブジェクト検出能力があります。
2. **[YOLOv4](../../models/yolov4.md)**: YOLOv3へのdarknet-nativeなアップデートで、2020年にアレクセイ・ボチコフスキーが公開しました。
3. **[YOLOv5](../../models/yolov5.md)**: UltralyticsによるYOLOアーキテクチャの改良版で、以前のバージョンと比較してパフォーマンスとスピードのトレードオフが向上しています。
4. **[YOLOv6](../../models/yolov6.md)**: 2022年に[美団](https://about.meituan.com/)によってリリースされ、同社の多くの自治配送ロボットで使用されています。
5. **[YOLOv7](../../models/yolov7.md)**: YOLOv4の作者によって2022年にリリースされた更新されたYOLOモデル。
6. **[YOLOv8](../../models/yolov8.md)**: YOLOファミリーの最新バージョンで、インスタンスセグメンテーション、ポーズ/キーポイント推定、分類などの機能が強化されています。
7. **[Segment Anything Model (SAM)](../../models/sam.md)**: MetaのSegment Anything Model (SAM)です。
8. **[Mobile Segment Anything Model (MobileSAM)](../../models/mobile-sam.md)**: 慶尚大学によるモバイルアプリケーション向けのMobileSAM。
9. **[Fast Segment Anything Model (FastSAM)](../../models/fast-sam.md)**: 中国科学院自動化研究所の画像・映像分析グループによるFastSAM。
10. **[YOLO-NAS](../../models/yolo-nas.md)**: YOLO Neural Architecture Search (NAS)モデル。
11. **[Realtime Detection Transformers (RT-DETR)](../../models/rtdetr.md)**: BaiduのPaddlePaddle Realtime Detection Transformer (RT-DETR)モデル。

<p align="center">
  <br>
  <iframe width="720" height="405" src="https://www.youtube.com/embed/MWq1UxqTClU?si=nHAW-lYDzrz68jR0"
    title="YouTube video player" frameborder="0"
    allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share"
    allowfullscreen>
  </iframe>
  <br>
  <strong>視聴：</strong> Ultralytics YOLOモデルを数行のコードで実行。
</p>

## 入門：使用例

!!! Example "例"

    === "Python"

        PyTorchの事前訓練済み`*.pt`モデルや設定`*.yaml`ファイルを`YOLO()`, `SAM()`, `NAS()`, `RTDETR()`クラスに渡して、Pythonでモデルインスタンスを生成できます：

        ```python
        from ultralytics import YOLO

        # COCO事前訓練済みのYOLOv8nモデルをロード
        model = YOLO('yolov8n.pt')

        # モデル情報の表示（オプション）
        model.info()

        # COCO8の例示データセットでモデルを100エポック訓練
        results = model.train(data='coco8.yaml', epochs=100, imgsz=640)

        # 'bus.jpg'画像上でYOLOv8nモデルによる推論実行
        results = model('path/to/bus.jpg')
        ```

    === "CLI"

        モデルを直接実行するためのCLIコマンドがあります：

        ```bash
        # COCO事前訓練済みのYOLOv8nモデルをロードし、COCO8の例示データセットで100エポック訓練
        yolo train model=yolov8n.pt data=coco8.yaml epochs=100 imgsz=640

        # COCO事前訓練済みのYOLOv8nモデルをロードし、'bus.jpg'画像上で推論実行
        yolo predict model=yolov8n.pt source=path/to/bus.jpg
        ```

## 新しいモデルの提供

Ultralyticsにモデルを提供してみたいですか？素晴らしいことです！私たちは常にモデルのポートフォリオを拡大することに興味があります。

1. **リポジトリをフォークする**：[Ultralytics GitHubリポジトリ](https://github.com/ultralytics/ultralytics)をフォークして始めます。

2. **フォークをクローンする**：フォークをローカルマシンにクローンし、作業用の新しいブランチを作成します。

3. **モデルを実装する**：提供されているコーディング規格とガイドラインに従ってモデルを追加します。

4. **徹底的にテストする**：孤立してもパイプラインの一部としても、モデルを徹底的にテストしてください。

5. **プルリクエストを作成する**：モデルに満足したら、レビューのためにメインリポジトリへのプルリクエストを作成します。

6. **コードレビューとマージ**：レビュー後、モデルが私たちの基準を満たしている場合、メインリポジトリにマージされます。

詳細な手順については、[コントリビューティングガイド](../../help/contributing.md)を参照してください。
