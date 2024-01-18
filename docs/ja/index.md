---
comments: true
description: Ultralytics YOLOv8に関する完全ガイド。高速で高精度なオブジェクト検出・画像セグメンテーションモデル。インストール、予測、トレーニングチュートリアルなど。
keywords: Ultralytics, YOLOv8, オブジェクト検出, 画像セグメンテーション, 機械学習, ディープラーニング, コンピュータビジョン, YOLOv8 インストール, YOLOv8 予測, YOLOv8 トレーニング, YOLO 歴史, YOLO ライセンス
---

<div align="center">
  <p>
    <a href="https://www.ultralytics.com/blog/ultralytics-yolov8-turns-one-a-year-of-breakthroughs-and-innovations" target="_blank">
    <img width="1024" src="https://raw.githubusercontent.com/ultralytics/assets/main/yolov8/banner-yolov8.png" alt="Ultralytics YOLOバナー"></a>
  </p>
  <a href="https://github.com/ultralytics"><img src="https://github.com/ultralytics/assets/raw/main/social/logo-social-github.png" width="3%" alt="Ultralytics GitHub"></a>
  <img src="https://github.com/ultralytics/assets/raw/main/social/logo-transparent.png" width="3%" alt="space">
  <a href="https://www.linkedin.com/company/ultralytics/"><img src="https://github.com/ultralytics/assets/raw/main/social/logo-social-linkedin.png" width="3%" alt="Ultralytics LinkedIn"></a>
  <img src="https://github.com/ultralytics/assets/raw/main/social/logo-transparent.png" width="3%" alt="space">
  <a href="https://twitter.com/ultralytics"><img src="https://github.com/ultralytics/assets/raw/main/social/logo-social-twitter.png" width="3%" alt="Ultralytics Twitter"></a>
  <img src="https://github.com/ultralytics/assets/raw/main/social/logo-transparent.png" width="3%" alt="space">
  <a href="https://youtube.com/ultralytics"><img src="https://github.com/ultralytics/assets/raw/main/social/logo-social-youtube.png" width="3%" alt="Ultralytics YouTube"></a>
  <img src="https://github.com/ultralytics/assets/raw/main/social/logo-transparent.png" width="3%" alt="space">
  <a href="https://www.tiktok.com/@ultralytics"><img src="https://github.com/ultralytics/assets/raw/main/social/logo-social-tiktok.png" width="3%" alt="Ultralytics TikTok"></a>
  <img src="https://github.com/ultralytics/assets/raw/main/social/logo-transparent.png" width="3%" alt="space">
  <a href="https://www.instagram.com/ultralytics/"><img src="https://github.com/ultralytics/assets/raw/main/social/logo-social-instagram.png" width="3%" alt="Ultralytics Instagram"></a>
  <img src="https://github.com/ultralytics/assets/raw/main/social/logo-transparent.png" width="3%" alt="space">
  <a href="https://ultralytics.com/discord"><img src="https://github.com/ultralytics/assets/raw/main/social/logo-social-discord.png" width="3%" alt="Ultralytics Discord"></a>
  <br>
  <br>
  <a href="https://github.com/ultralytics/ultralytics/actions/workflows/ci.yaml"><img src="https://github.com/ultralytics/ultralytics/actions/workflows/ci.yaml/badge.svg" alt="Ultralytics CI"></a>
  <a href="https://codecov.io/github/ultralytics/ultralytics"><img src="https://codecov.io/github/ultralytics/ultralytics/branch/main/graph/badge.svg?token=HHW7IIVFVY" alt="Ultralytics コードカバレッジ"></a>
  <a href="https://zenodo.org/badge/latestdoi/264818686"><img src="https://zenodo.org/badge/264818686.svg" alt="YOLOv8 引用情報"></a>
  <a href="https://hub.docker.com/r/ultralytics/ultralytics"><img src="https://img.shields.io/docker/pulls/ultralytics/ultralytics?logo=docker" alt="Docker プル"></a>
  <a href="https://ultralytics.com/discord"><img alt="Discord" src="https://img.shields.io/discord/1089800235347353640?logo=discord&logoColor=white&label=Discord&color=blue"></a>
  <br>
  <a href="https://console.paperspace.com/github/ultralytics/ultralytics"><img src="https://assets.paperspace.io/img/gradient-badge.svg" alt="Gradient上で実行"></a>
  <a href="https://colab.research.google.com/github/ultralytics/ultralytics/blob/main/examples/tutorial.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Colabで開く"></a>
  <a href="https://www.kaggle.com/ultralytics/yolov8"><img src="https://kaggle.com/static/images/open-in-kaggle.svg" alt="Kaggleで開く"></a>
</div>

全く新しい[Ultralytics](https://ultralytics.com)の[YOLOv8](https://github.com/ultralytics/ultralytics)を紹介します。これは、実時間で動作するオブジェクト検出および画像セグメンテーションモデルの最新バージョンです。YOLOv8は、ディープラーニングとコンピュータビジョンの最先端の進歩に基づいており、速度と精度の面で比類のない性能を提供します。その合理化された設計により、エッジデバイスからクラウドAPIまで、さまざまなアプリケーションやハードウェアプラットフォームへの適応が容易です。

YOLOv8ドキュメントを探索し、その特徴と能力を理解し、活用するための包括的なリソースを提供します。機械学習の経験者であれ、分野の新入りであれ、このハブはあなたのプロジェクトでYOLOv8のポテンシャルを最大限に引き出すことを目指しています。

!!! Note "ノート"

    🚧 多言語ドキュメントは現在作成中であり、改善に努めております。お待ちいただき、ありがとうございます！ 🙏

## はじめに

- pipで`ultralytics`を**インストール**し、数分で稼働 &nbsp; [:material-clock-fast: はじめに](quickstart.md){ .md-button }
- YOLOv8で新しい画像やビデオに**予測** &nbsp; [:octicons-image-16: 画像で予測](modes/predict.md){ .md-button }
- 独自のカスタムデータセットで新しいYOLOv8モデルを**トレーニング** &nbsp; [:fontawesome-solid-brain: モデルをトレーニング](modes/train.md){ .md-button }
- セグメント、クラス分け、ポーズ、トラッキングなどのYOLOv8タスクを**探求** &nbsp; [:material-magnify-expand: タスクを探求](tasks/index.md){ .md-button }

<p align="center">
  <br>
  <iframe width="720" height="405" src="https://www.youtube.com/embed/LNwODJXcvt4?si=7n1UvGRLSd9p5wKs"
    title="YouTubeビデオプレイヤー" frameborder="0"
    allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share"
    allowfullscreen>
  </iframe>
  <br>
  <strong>視聴:</strong> <a href="https://colab.research.google.com/github/ultralytics/ultralytics/blob/main/examples/tutorial.ipynb" target="_blank">Google Colab</a>でカスタムデータセットにYOLOv8モデルをトレーニングする方法。
</p>

## YOLO: 簡単な歴史

[YOLO](https://arxiv.org/abs/1506.02640)（You Only Look Once、一度だけ見る）は、ワシントン大学のJoseph RedmonとAli Farhadiによって開発された、流行のオブジェクト検出および画像セグメンテーションモデルです。2015年に発売されたYOLOは、その高速かつ正確さからすぐに人気を博しました。

- [YOLOv2](https://arxiv.org/abs/1612.08242)は、2016年にリリースされ、バッチ正規化、アンカーボックス、次元クラスタリングを導入し、オリジナルモデルを改善しました。
- [YOLOv3](https://pjreddie.com/media/files/papers/YOLOv3.pdf)は、2018年により効率的なバックボーンネットワーク、複数のアンカー、空間ピラミッドプーリングを使用して、モデルの性能を一段と向上させました。
- [YOLOv4](https://arxiv.org/abs/2004.10934)は2020年にリリースされ、モザイクデータオーギュメンテーション、新しいアンカーフリー検出ヘッド、新しい損失関数などの革新を導入しました。
- [YOLOv5](https://github.com/ultralytics/yolov5)は、モデルの性能をさらに向上させ、ハイパーパラメータ最適化、統合実験トラッキング、一般的なエクスポート形式への自動エクスポートなどの新機能を追加しました。
- [YOLOv6](https://github.com/meituan/YOLOv6)は、2022年に[Meituan](https://about.meituan.com/)によってオープンソース化され、同社の多くの自動配送ロボットで使用されています。
- [YOLOv7](https://github.com/WongKinYiu/yolov7)は、COCOキーポイントデータセット上のポーズ推定などの追加タスクを追加しました。
- [YOLOv8](https://github.com/ultralytics/ultralytics)は、UltralyticsによるYOLOの最新版です。最先端の最新モデルとして、YOLOv8は前バージョンの成功に基づき、性能、柔軟性、効率を向上させる新機能や改善を導入しています。YOLOv8は、[検出](tasks/detect.md)、[セグメンテーション](tasks/segment.md)、[ポーズ推定](tasks/pose.md)、[トラッキング](modes/track.md)、[分類](tasks/classify.md)など、視覚AIタスクの全範囲をサポートしています。この多才性により、ユーザーは多様なアプリケーションとドメインでYOLOv8の機能を活用することができます。

## YOLO ライセンス: UltralyticsのYOLOはどのようにライセンスされていますか？

Ultralyticsは、さまざまなユースケースに対応するために2種類のライセンスオプションを提供しています：

- **AGPL-3.0 ライセンス**: この[OSI認定](https://opensource.org/licenses/)のオープンソースライセンスは、学生や愛好家に理想的であり、オープンなコラボレーションと知識共有を奨励しています。詳細は[ライセンス](https://github.com/ultralytics/ultralytics/blob/main/LICENSE)ファイルをご覧ください。
- **エンタープライズ ライセンス**: 商業用途に設計されたこのライセンスは、UltralyticsのソフトウェアおよびAIモデルを商業商品やサービスにシームレスに統合することを許可し、AGPL-3.0のオープンソース要件をバイパスできます。商業的なオファリングへの組み込みを含むシナリオであれば、[Ultralytics ライセンス](https://ultralytics.com/license)を通じてお問い合わせください。

私たちのライセンス戦略は、オープンソースプロジェクトに対するあらゆる改善がコミュニティに還元されることを確実にするために設計されています。私たちはオープンソースの原則を大切にしており、私たちの貢献が全ての人にとって有益な方法で利用可能であり、さらに拡張されることを保証することを使命としています。❤️
