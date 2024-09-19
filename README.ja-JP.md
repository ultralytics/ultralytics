<div align="center">
  <p>
    <a href="https://ultralytics.com/yolov8" target="_blank">
      <img width="100%" src="https://raw.githubusercontent.com/ultralytics/assets/main/yolov8/banner-yolov8.png"></a>
  </p>

[English](README.md) | [简体中文](README.zh-CN.md) | [日本語](README.ja-JP.md)
<br>

<div>
    <a href="https://github.com/ultralytics/ultralytics/actions/workflows/ci.yaml"><img src="https://github.com/ultralytics/ultralytics/actions/workflows/ci.yaml/badge.svg" alt="Ultralytics CI"></a>
    <a href="https://zenodo.org/badge/latestdoi/264818686"><img src="https://zenodo.org/badge/264818686.svg" alt="YOLOv8 Citation"></a>
    <a href="https://hub.docker.com/r/ultralytics/ultralytics"><img src="https://img.shields.io/docker/pulls/ultralytics/ultralytics?logo=docker" alt="Docker Pulls"></a>
    <br>
    <a href="https://console.paperspace.com/github/ultralytics/ultralytics"><img src="https://assets.paperspace.io/img/gradient-badge.svg" alt="Run on Gradient"/></a>
    <a href="https://colab.research.google.com/github/ultralytics/ultralytics/blob/main/examples/tutorial.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"></a>
    <a href="https://www.kaggle.com/ultralytics/yolov8"><img src="https://kaggle.com/static/images/open-in-kaggle.svg" alt="Open In Kaggle"></a>
  </div>
  <br>

[Ultralytics](https://ultralytics.com) [YOLOv8](https://github.com/ultralytics/ultralytics) は、これまでの YOLO バージョンの成功に加え、パフォーマンスと柔軟性をさらに高める新機能と改良を導入した、最先端の (SOTA) モデルです。YOLOv8 は、高速、高精度、使いやすいように設計されており、オブジェクトの検出と追跡、インスタンスのセグメンテーション、画像分類、姿勢推定の幅広いタスクに最適です。

YOLOv8 を最大限に活用するために、ここにあるリソースがお役に立てれば幸いです。詳細については YOLOv8 <a href="https://docs.ultralytics.com/">ドキュメント</a>をご覧ください。サポートについては <a href="https://github.com/ultralytics/ultralytics/issues/new/choose">GitHub</a> で issue を挙げてサポートを受けてください、<a href="https://ultralytics.com/discord">Discord</a> コミュニティで質問やディスカッションができます！

企業向けライセンスのお申し込みは、[Ultralytics Licensing](https://ultralytics.com/license) のフォームにご記入ください。

<img width="100%" src="https://raw.githubusercontent.com/ultralytics/assets/main/yolov8/yolo-comparison-plots.png"></a>

<div align="center">
  <a href="https://github.com/ultralytics" style="text-decoration:none;">
    <img src="https://github.com/ultralytics/assets/raw/main/social/logo-social-github.png" width="2%" alt="" /></a>
  <img src="https://github.com/ultralytics/assets/raw/main/social/logo-transparent.png" width="2%" alt="" />
  <a href="https://www.linkedin.com/company/ultralytics/" style="text-decoration:none;">
    <img src="https://github.com/ultralytics/assets/raw/main/social/logo-social-linkedin.png" width="2%" alt="" /></a>
  <img src="https://github.com/ultralytics/assets/raw/main/social/logo-transparent.png" width="2%" alt="" />
  <a href="https://twitter.com/ultralytics" style="text-decoration:none;">
    <img src="https://github.com/ultralytics/assets/raw/main/social/logo-social-twitter.png" width="2%" alt="" /></a>
  <img src="https://github.com/ultralytics/assets/raw/main/social/logo-transparent.png" width="2%" alt="" />
  <a href="https://youtube.com/ultralytics" style="text-decoration:none;">
    <img src="https://github.com/ultralytics/assets/raw/main/social/logo-social-youtube.png" width="2%" alt="" /></a>
  <img src="https://github.com/ultralytics/assets/raw/main/social/logo-transparent.png" width="2%" alt="" />
  <a href="https://www.tiktok.com/@ultralytics" style="text-decoration:none;">
    <img src="https://github.com/ultralytics/assets/raw/main/social/logo-social-tiktok.png" width="2%" alt="" /></a>
  <img src="https://github.com/ultralytics/assets/raw/main/social/logo-transparent.png" width="2%" alt="" />
  <a href="https://www.instagram.com/ultralytics/" style="text-decoration:none;">
    <img src="https://github.com/ultralytics/assets/raw/main/social/logo-social-instagram.png" width="2%" alt="" /></a>
  <img src="https://github.com/ultralytics/assets/raw/main/social/logo-transparent.png" width="2%" alt="" />
  <a href="https://ultralytics.com/discord" style="text-decoration:none;">
    <img src="https://github.com/ultralytics/assets/blob/main/social/logo-social-discord.png" width="2%" alt="" /></a>
</div>
</div>

## <div align="center">ドキュメント</div>

また、トレーニング、検証、予測、デプロイメントに関する完全なドキュメントは [YOLOv8 ドキュメント](https://docs.ultralytics.com)を参照のこと。

<details open>
<summary>インストール</summary>

Pip は、[**Python>=3.8**](https://www.python.org/) と [**PyTorch>=1.7**](https://pytorch.org/get-started/locally/) の環境に、すべての[要件](https://github.com/ultralytics/ultralytics/blob/main/requirements.txt)を含む ultralytics パッケージをインストールします。

[![PyPI version](https://badge.fury.io/py/ultralytics.svg)](https://badge.fury.io/py/ultralytics) [![Downloads](https://static.pepy.tech/badge/ultralytics)](https://pepy.tech/project/ultralytics)

```bash
pip install ultralytics
```

[Conda](https://anaconda.org/conda-forge/ultralytics)、[Docker](https://hub.docker.com/r/ultralytics/ultralytics)、Git を含む別のインストール方法については、[クイックスタートガイド](https://docs.ultralytics.com/quickstart)を参照してください。

</details>

<details open>
<summary>使用方法</summary>

#### CLI

YOLOv8 は、コマンドラインインターフェース（CLI）で `yolo` コマンドを使って直接使うことができます:

```bash
yolo predict model=yolov8n.pt source='https://ultralytics.com/images/bus.jpg'
```

`yolo` はさまざまなタスクやモードに使用でき、`imgsz=640` のような追加引数を受け付ける。例については、YOLOv8 [CLI ドキュメント](https://docs.ultralytics.com/usage/cli)を参照してください。

#### Python

YOLOv8 は、Python 環境で直接使うこともでき、上記の CLI の例と同じ[引数](https://docs.ultralytics.com/usage/cfg/)を受け付けます:

```python
from ultralytics import YOLO

# モデルのロード
model = YOLO("yolov8n.yaml")  # ゼロから新しいモデルを作る
model = YOLO("yolov8n.pt")  # 事前に学習させたモデルをロードする（トレーニングに推奨）

# モデルを使用
model.train(data="coco128.yaml", epochs=3)  # モデルをトレーニングする
metrics = model.val()  # 検証セットでモデルのパフォーマンスを評価する
results = model("https://ultralytics.com/images/bus.jpg")  # 画像を予測する
path = model.export(format="onnx")  # モデルを ONNX 形式にエクスポートする
```

[モデル](https://github.com/ultralytics/ultralytics/tree/main/ultralytics/cfg/models)は、最新の Ultralytics [リリース](https://github.com/ultralytics/assets/releases)から自動的にダウンロードされます。その他の例については、YOLOv8 [Python ドキュメント](https://docs.ultralytics.com/usage/python)を参照してください。

</details>

## <div align="center">モデル</div>

[COCO](https://docs.ultralytics.com/datasets/detect/coco) データセットで事前学習された YOLOv8 [検出](https://docs.ultralytics.com/tasks/detect)、[セグメンテーション](https://docs.ultralytics.com/tasks/segment)、[ポーズ](https://docs.ultralytics.com/tasks/pose)モデル、および [ImageNet](https://docs.ultralytics.com/datasets/classify/imagenet) データセットで事前学習された YOLOv8 [分類](https://docs.ultralytics.com/tasks/classify)モデルをご利用いただけます。[Track](https://docs.ultralytics.com/modes/track) モードは、すべての検出、セグメンテーション、ポーズモデルで利用可能です。

<img width="1024" src="https://raw.githubusercontent.com/ultralytics/assets/main/im/banner-tasks.png">

すべての[モデル](https://github.com/ultralytics/ultralytics/tree/main/ultralytics/cfg/models)は、初回使用時に最新の Ultralytics [リリース](https://github.com/ultralytics/assets/releases)から自動的にダウンロードされます。

<details open><summary>検出</summary>

これらのモデルの使用例については[検出ドキュメント](https://docs.ultralytics.com/tasks/detect/)を参照のこと。

| モデル                                                                               | サイズ<br><sup>(ピクセル) | mAP<sup>val<br>50-95 | スピード<br><sup>CPU ONNX<br>(ms) | スピード<br><sup>A100 TensorRT<br>(ms) | パラメータ<br><sup>(M) | FLOPs<br><sup>(B) |
| ------------------------------------------------------------------------------------ | ------------------------- | -------------------- | --------------------------------- | -------------------------------------- | ---------------------- | ----------------- |
| [YOLOv8n](https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n.pt) | 640                       | 37.3                 | 80.4                              | 0.99                                   | 3.2                    | 8.7               |
| [YOLOv8s](https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8s.pt) | 640                       | 44.9                 | 128.4                             | 1.20                                   | 11.2                   | 28.6              |
| [YOLOv8m](https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8m.pt) | 640                       | 50.2                 | 234.7                             | 1.83                                   | 25.9                   | 78.9              |
| [YOLOv8l](https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8l.pt) | 640                       | 52.9                 | 375.2                             | 2.39                                   | 43.7                   | 165.2             |
| [YOLOv8x](https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8x.pt) | 640                       | 53.9                 | 479.1                             | 3.53                                   | 68.2                   | 257.8             |

- **mAP<sup>val</sup>** の値は、[COCO val2017](http://cocodataset.org) データセットのシングルモデルシングルスケールのものである。
  <br>`yolo val detect data=coco.yaml device=0` で再現
- [Amazon EC2 P4d](https://aws.amazon.com/ec2/instance-types/p4/) インスタンスを使用し、COCO バル画像を平均した**スピード**。
  <br>`yolo val detect data=coco128.yaml batch=1 device=0|cpu` で再現

</details>

<details><summary>セグメンテーション</summary>

これらのモデルの使用例については[セグメンテーションドキュメント](https://docs.ultralytics.com/tasks/segment/)を参照のこと。

| モデル                                                                                       | サイズ<br><sup>(ピクセル) | mAP<sup>box<br>50-95 | mAP<sup>mask<br>50-95 | スピード<br><sup>CPU ONNX<br>(ms) | スピード<br><sup>A100 TensorRT<br>(ms) | パラメータ<br><sup>(M) | FLOPs<br><sup>(B) |
| -------------------------------------------------------------------------------------------- | ------------------------- | -------------------- | --------------------- | --------------------------------- | -------------------------------------- | ---------------------- | ----------------- |
| [YOLOv8n-seg](https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n-seg.pt) | 640                       | 36.7                 | 30.5                  | 96.1                              | 1.21                                   | 3.4                    | 12.6              |
| [YOLOv8s-seg](https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8s-seg.pt) | 640                       | 44.6                 | 36.8                  | 155.7                             | 1.47                                   | 11.8                   | 42.6              |
| [YOLOv8m-seg](https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8m-seg.pt) | 640                       | 49.9                 | 40.8                  | 317.0                             | 2.18                                   | 27.3                   | 110.2             |
| [YOLOv8l-seg](https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8l-seg.pt) | 640                       | 52.3                 | 42.6                  | 572.4                             | 2.79                                   | 46.0                   | 220.5             |
| [YOLOv8x-seg](https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8x-seg.pt) | 640                       | 53.4                 | 43.4                  | 712.1                             | 4.02                                   | 71.8                   | 344.1             |

- **mAP<sup>val</sup>** の値は、[COCO val2017](http://cocodataset.org) データセットのシングルモデルシングルスケールのものである。
  <br>`yolo val segment data=coco.yaml device=0` で再現
- [Amazon EC2 P4d](https://aws.amazon.com/ec2/instance-types/p4/) インスタンスを使用し、COCO バル画像を平均した**スピード**。
  <br>`yolo val segment data=coco128-seg.yaml batch=1 device=0|cpu` で再現

</details>

<details><summary>分類</summary>

これらのモデルの使用例については[分類ドキュメント](https://docs.ultralytics.com/tasks/classify/)を参照のこと。

| モデル                                                                                       | サイズ<br><sup>(ピクセル) | acc<br><sup>top1 | acc<br><sup>top5 | スピード<br><sup>CPU ONNX<br>(ms) | スピード<br><sup>A100 TensorRT<br>(ms) | パラメータ<br><sup>(M) | FLOPs<br><sup>(B) at 640 |
| -------------------------------------------------------------------------------------------- | ------------------------- | ---------------- | ---------------- | --------------------------------- | -------------------------------------- | ---------------------- | ------------------------ |
| [YOLOv8n-cls](https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n-cls.pt) | 224                       | 66.6             | 87.0             | 12.9                              | 0.31                                   | 2.7                    | 4.3                      |
| [YOLOv8s-cls](https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8s-cls.pt) | 224                       | 72.3             | 91.1             | 23.4                              | 0.35                                   | 6.4                    | 13.5                     |
| [YOLOv8m-cls](https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8m-cls.pt) | 224                       | 76.4             | 93.2             | 85.4                              | 0.62                                   | 17.0                   | 42.7                     |
| [YOLOv8l-cls](https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8l-cls.pt) | 224                       | 78.0             | 94.1             | 163.0                             | 0.87                                   | 37.5                   | 99.7                     |
| [YOLOv8x-cls](https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8x-cls.pt) | 224                       | 78.4             | 94.3             | 232.0                             | 1.01                                   | 57.4                   | 154.8                    |

- **acc** 値は、[ImageNet](https://www.image-net.org/) データセットの検証セットにおけるモデル精度である。
  <br>`yolo val classify data=path/to/ImageNet device=0` で再現
- [Amazon EC2 P4d](https://aws.amazon.com/ec2/instance-types/p4/) インスタンスを使用し、ImageNet の val 画像を平均した**スピード**。
  <br>`yolo val classify data=path/to/ImageNet batch=1 device=0|cpu` で再現

</details>

<details><summary>ポーズ</summary>

これらのモデルの使用例については、[ポーズドキュメント](https://docs.ultralytics.com/tasks/pose)を参照してください。

| モデル                                                                                               | サイズ<br><sup>(ピクセル) | mAP<sup>pose<br>50-95 | mAP<sup>pose<br>50 | スピード<br><sup>CPU ONNX<br>(ms) | スピード<br><sup>A100 TensorRT<br>(ms) | パラメータ<br><sup>(M) | FLOPs<br><sup>(B) |
| ---------------------------------------------------------------------------------------------------- | ------------------------- | --------------------- | ------------------ | --------------------------------- | -------------------------------------- | ---------------------- | ----------------- |
| [YOLOv8n-pose](https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n-pose.pt)       | 640                       | 50.4                  | 80.1               | 131.8                             | 1.18                                   | 3.3                    | 9.2               |
| [YOLOv8s-pose](https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8s-pose.pt)       | 640                       | 60.0                  | 86.2               | 233.2                             | 1.42                                   | 11.6                   | 30.2              |
| [YOLOv8m-pose](https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8m-pose.pt)       | 640                       | 65.0                  | 88.8               | 456.3                             | 2.00                                   | 26.4                   | 81.0              |
| [YOLOv8l-pose](https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8l-pose.pt)       | 640                       | 67.6                  | 90.0               | 784.5                             | 2.59                                   | 44.4                   | 168.6             |
| [YOLOv8x-pose](https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8x-pose.pt)       | 640                       | 69.2                  | 90.2               | 1607.1                            | 3.73                                   | 69.4                   | 263.2             |
| [YOLOv8x-pose-p6](https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8x-pose-p6.pt) | 1280                      | 71.6                  | 91.2               | 4088.7                            | 10.04                                  | 99.1                   | 1066.4            |

- **mAP<sup>val</sup>** の値は、[COCO Keypoints val2017](http://cocodataset.org) データセット上のシングルモデルシングルスケールのものである。
  <br>`yolo val pose data=coco-pose.yaml device=0` で再現
- [Amazon EC2 P4d](https://aws.amazon.com/ec2/instance-types/p4/) インスタンスを使用し、COCO バル画像を平均した**スピード**。
  <br>`yolo val pose data=coco8-pose.yaml batch=1 device=0|cpu` で再現

</details>

## <div align="center">統合</div>

主要な AI プラットフォームとの統合により、Ultralytics の機能を拡張し、データセットのラベリング、トレーニング、可視化、モデル管理などのタスクを強化します。[Roboflow](https://roboflow.com/?ref=ultralytics)、ClearML、[Comet](https://bit.ly/yolov8-readme-comet)、Neural Magic、[OpenVINO](https://docs.ultralytics.com/integrations/openvino) との連携により、Ultralytics がお客様の AI ワークフローをどのように最適化できるかをご覧ください。

<br>
<a href="https://bit.ly/ultralytics_hub" target="_blank">
<img width="100%" src="https://github.com/ultralytics/assets/raw/main/yolov8/banner-integrations.png"></a>
<br>
<br>

<div align="center">
  <a href="https://roboflow.com/?ref=ultralytics">
    <img src="https://github.com/ultralytics/assets/raw/main/partners/logo-roboflow.png" width="10%" /></a>
  <img src="https://github.com/ultralytics/assets/raw/main/social/logo-transparent.png" width="15%" height="0" alt="" />
  <a href="https://cutt.ly/yolov5-readme-clearml">
    <img src="https://github.com/ultralytics/assets/raw/main/partners/logo-clearml.png" width="10%" /></a>
  <img src="https://github.com/ultralytics/assets/raw/main/social/logo-transparent.png" width="15%" height="0" alt="" />
  <a href="https://bit.ly/yolov8-readme-comet">
    <img src="https://github.com/ultralytics/assets/raw/main/partners/logo-comet.png" width="10%" /></a>
  <img src="https://github.com/ultralytics/assets/raw/main/social/logo-transparent.png" width="15%" height="0" alt="" />
  <a href="https://bit.ly/yolov5-neuralmagic">
    <img src="https://github.com/ultralytics/assets/raw/main/partners/logo-neuralmagic.png" width="10%" /></a>
</div>

|                                                                    Roboflow                                                                     |                                                            ClearML ⭐ NEW                                                             |                                                                         Comet ⭐ NEW                                                                         |                                          Neural Magic ⭐ NEW                                           |
| :---------------------------------------------------------------------------------------------------------------------------------------------: | :-----------------------------------------------------------------------------------------------------------------------------------: | :----------------------------------------------------------------------------------------------------------------------------------------------------------: | :----------------------------------------------------------------------------------------------------: |
| [Roboflow](https://roboflow.com/?ref=ultralytics) でのトレーニングのために、カスタムデータセットにラベルを付け、YOLOv8 に直接エクスポートします | [ClearML](https://cutt.ly/yolov5-readme-clearml) (オープンソース!)を使って、YOLOv8 を自動的に追跡、視覚化し、遠隔トレーニングまで行う | 永久無料の [Comet](https://bit.ly/yolov8-readme-comet) は、YOLOv8 モデルの保存、トレーニングの再開、予測値のインタラクティブな可視化とデバッグを可能にします | [Neural Magic DeepSparse](https://bit.ly/yolov5-neuralmagic) で YOLOv8 の推論を最大 6 倍高速に実行する |

## <div align="center">Ultralytics HUB</div>

[Ultralytics HUB](https://bit.ly/ultralytics_hub) ⭐は、データ可視化、YOLOv5 および YOLOv8 🚀モデルのトレーニング、デプロイメントをオールインワンで提供するソリューションで、コーディングなしでシームレスな AI を体験できます。最先端のプラットフォームと使いやすい [Ultralytics App](https://ultralytics.com/app_install) を使って、画像を実用的な洞察に変換し、AI のビジョンを簡単に実現できます。今すぐ**無料**の旅を始めましょう！

<a href="https://bit.ly/ultralytics_hub" target="_blank">
<img width="100%" src="https://github.com/ultralytics/assets/raw/main/im/ultralytics-hub.png"></a>

## <div align="center">コントリビュート</div>

皆様のご意見をお待ちしております！YOLOv5 と YOLOv8 は、私たちのコミュニティからの助けなしには成り立ちません。[コントリビューティングガイド](https://docs.ultralytics.com/help/contributing)をご覧いただき、[アンケート](https://ultralytics.com/survey?utm_source=github&utm_medium=social&utm_campaign=Survey)にご記入の上、ご意見をお寄せください。ご協力いただいた皆様、ありがとうございました！

<!-- SVG image from https://opencollective.com/ultralytics/contributors.svg?width=990 -->

<a href="https://github.com/ultralytics/yolov5/graphs/contributors">
<img width="100%" src="https://github.com/ultralytics/assets/raw/main/im/image-contributors.png"></a>

## <div align="center">ライセンス</div>

Ultralytics は、多様なユースケースに対応するため、2つのライセンスオプションを提供しています:

- **AGPL-3.0 License**: この [OSI 承認](https://opensource.org/licenses/)オープンソースライセンスは、学生や愛好家に理想的で、オープンなコラボレーションと知識の共有を促進します。詳細は [LICENSE](https://github.com/ultralytics/ultralytics/blob/main/LICENSE) ファイルをご覧ください。
- **Enterprise License**: 商用利用のために設計されたこのライセンスは、AGPL-3.0 のオープンソース要件をバイパスして、Ultralytics ソフトウェアと AI モデルを商用商品やサービスにシームレスに統合することを許可します。当社のソリューションを商業的な製品に組み込む場合は、[Ultralytics Licensing](https://ultralytics.com/license) までご連絡ください。

## <div align="center">連絡先</div>

Ultralytics のバグレポートや機能リクエストは [GitHub Issues](https://github.com/ultralytics/ultralytics/issues) を、質問やディスカッションは [Discord](https://ultralytics.com/discord) のコミュニティにご参加ください！

<br>
<div align="center">
  <a href="https://github.com/ultralytics" style="text-decoration:none;">
    <img src="https://github.com/ultralytics/assets/raw/main/social/logo-social-github.png" width="3%" alt="" /></a>
  <img src="https://github.com/ultralytics/assets/raw/main/social/logo-transparent.png" width="3%" alt="" />
  <a href="https://www.linkedin.com/company/ultralytics/" style="text-decoration:none;">
    <img src="https://github.com/ultralytics/assets/raw/main/social/logo-social-linkedin.png" width="3%" alt="" /></a>
  <img src="https://github.com/ultralytics/assets/raw/main/social/logo-transparent.png" width="3%" alt="" />
  <a href="https://twitter.com/ultralytics" style="text-decoration:none;">
    <img src="https://github.com/ultralytics/assets/raw/main/social/logo-social-twitter.png" width="3%" alt="" /></a>
  <img src="https://github.com/ultralytics/assets/raw/main/social/logo-transparent.png" width="3%" alt="" />
  <a href="https://youtube.com/ultralytics" style="text-decoration:none;">
    <img src="https://github.com/ultralytics/assets/raw/main/social/logo-social-youtube.png" width="3%" alt="" /></a>
  <img src="https://github.com/ultralytics/assets/raw/main/social/logo-transparent.png" width="3%" alt="" />
  <a href="https://www.tiktok.com/@ultralytics" style="text-decoration:none;">
    <img src="https://github.com/ultralytics/assets/raw/main/social/logo-social-tiktok.png" width="3%" alt="" /></a>
  <img src="https://github.com/ultralytics/assets/raw/main/social/logo-transparent.png" width="3%" alt="" />
  <a href="https://www.instagram.com/ultralytics/" style="text-decoration:none;">
    <img src="https://github.com/ultralytics/assets/raw/main/social/logo-social-instagram.png" width="3%" alt="" /></a>
  <img src="https://github.com/ultralytics/assets/raw/main/social/logo-transparent.png" width="3%" alt="" />
  <a href="https://ultralytics.com/discord" style="text-decoration:none;">
    <img src="https://github.com/ultralytics/assets/blob/main/social/logo-social-discord.png" width="3%" alt="" /></a>
</div>
