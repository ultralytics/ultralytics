---
comments: true
description: リアルタイムの画像セグメンテーションを可能にするウルトラリティクスの最先端Segment Anything Model (SAM)を紹介します。SAMのプロンプト可能なセグメンテーション、ゼロショットパフォーマンス、使用方法について学びましょう。
keywords: Ultralytics, 画像セグメンテーション, Segment Anything Model, SAM, SA-1B データセット, リアルタイムパフォーマンス, ゼロショット転送, 物体検出, 画像解析, 機械学習
---

# Segment Anything Model (SAM)

ウルトラリティクスのSegment Anything Model（SAM）へようこそ。この革新的なモデルは、プロンプト可能な画像セグメンテーションを実現し、リアルタイムのパフォーマンスで画期的な成果を上げ、この分野で新たな基準を設定しました。

## SAMの紹介: Segment Anything Model

Segment Anything Model（SAM）は、画像解析タスクにおける柔軟なセグメンテーションを可能にする最先端の画像セグメンテーションモデルです。SAMは、セグメンテーションという新しいモデル、タスク、データセットを導入した画期的なプロジェクト「Segment Anything」の中核をなしています。

SAMの高度な設計により、新しい画像分布やタスクに事前の知識なしで適応するゼロショット転送の機能を持っています。豊富な[SA-1B データセット](https://ai.facebook.com/datasets/segment-anything/)でトレーニングされたSAMは、1億以上のマスクを含む1,100万枚以上の厳選された画像に広がる自動的にアノテーションされたセグメンテーションマスクを備えており、多くの場合、前向きに監督された結果を上回る卓越したゼロショットパフォーマンスを発揮しています。

![データセットサンプルイメージ](https://user-images.githubusercontent.com/26833433/238056229-0e8ffbeb-f81a-477e-a490-aff3d82fd8ce.jpg)
新たに導入されたSA-1Bデータセットからガイドマスクを重畳した例の画像です。SA-1Bには、多様な高解像度のライセンス画像と11億件以上の高品質のセグメンテーションマスクが含まれています。これらのマスクは、SAMによって完全自動的に注釈付けされ、人間の評価と数多くの実験で高品質と多様性が確認されています。画像は可視化のために画像あたりのマスクの数でグループ化されています（平均でおおよそ100個のマスクがあります）。

## Segment Anything Model (SAM)の主な特徴

- **プロンプト可能なセグメンテーションタスク:** SAMは、プロンプト（オブジェクトを特定する空間的なまたはテキスト的な手がかり）から有効なセグメンテーションマスクを生成するように設計されています。
- **高度なアーキテクチャ:** Segment Anything Modelは、強力な画像エンコーダ、プロンプトエンコーダ、軽量のマスクデコーダを採用しています。このユニークなアーキテクチャにより、柔軟なプロンプティング、リアルタイムのマスク計算、セグメンテーションタスクの曖昧さの認識が可能です。
- **SA-1Bデータセット:** Segment Anythingプロジェクトによって導入されたSA-1Bデータセットは、1,100万枚以上の画像に1,000,000,000件以上のマスクを提供します。これまでで最も大規模なセグメンテーションデータセットであり、SAMに多様で大規模なトレーニングデータソースを提供します。
- **ゼロショットパフォーマンス:** SAMは、さまざまなセグメンテーションタスクで優れたゼロショットパフォーマンスを発揮し、プロンプトエンジニアリングの最小限の必要性で多様なアプリケーションに即座に使用できるツールとなります。

Segment Anything ModelおよびSA-1Bデータセットの詳細については、[Segment Anything website](https://segment-anything.com)をご覧いただくか、研究論文[Segment Anything](https://arxiv.org/abs/2304.02643)をご覧ください。

## 使用可能なモデル、サポートされるタスク、および動作モード

このテーブルでは、使用可能なモデルとその特定の事前トレーニング済み重み、サポートされているタスク、およびInference、Validation、Training、Exportなどのさまざまな操作モードに対する互換性を示しています。サポートされるモードは✅の絵文字で表示され、サポートされていないモードは❌の絵文字で表示されます。

| モデルの種類    | 事前トレーニング済みの重み | サポートされているタスク                                                  | Inference | Validation | Training | Export |
|-----------|---------------|---------------------------------------------------------------|-----------|------------|----------|--------|
| SAM base  | `sam_b.pt`    | [Instance Segmentation（インスタンスセグメンテーション）](../tasks/segment.md) | ✅         | ❌          | ❌        | ✅      |
| SAM large | `sam_l.pt`    | [Instance Segmentation（インスタンスセグメンテーション）](../tasks/segment.md) | ✅         | ❌          | ❌        | ✅      |

## SAMの使用方法: 画像セグメンテーションにおける柔軟性とパワー

Segment Anything Modelは、トレーニングデータを超えた多くのダウンストリームタスクに使用することができます。これにはエッジ検出、オブジェクトの提案生成、インスタンスセグメンテーション、および予備的なテキストからマスクへの予測などが含まれます。プロンプトエンジニアリングを使用することで、SAMはゼロショットの方法で新しいタスクとデータ分布にすばやく適応することができ、あらゆる画像セグメンテーションに関する柔軟で強力なツールとなります。

### SAMの予測の例

!!! Example "プロンプトでセグメントする"

    与えられたプロンプトで画像をセグメンテーションします。

    === "Python"

        ```python
        from ultralytics import SAM

        # モデルをロード
        model = SAM('sam_b.pt')

        # モデル情報を表示（オプション）
        model.info()

        # バウンディングボックスのプロンプトで予測を実行
        model('ultralytics/assets/zidane.jpg', bboxes=[439, 437, 524, 709])

        # ポイントのプロンプトで予測を実行
        model('ultralytics/assets/zidane.jpg', points=[900, 370], labels=[1])
        ```

!!! Example "すべてをセグメントする"

    画像全体をセグメンテーションします。

    === "Python"

        ```python
        from ultralytics import SAM

        # モデルをロード
        model = SAM('sam_b.pt')

        # モデル情報を表示（オプション）
        model.info()

        # 予測を実行
        model('path/to/image.jpg')
        ```

    === "CLI"

        ```bash
        # SAMモデルで予測を実行
        yolo predict model=sam_b.pt source=path/to/image.jpg
        ```

- ここでは、プロンプト（バウンディングボックス/ポイント/マスク）を指定しない場合は、画像全体がセグメンテーションされるロジックです。

!!! Example "SAMPredictorの例"

    画像を一度設定し、イメージエンコーダを複数回実行することなく複数回プロンプト推論を実行できます。

    === "プロンプト推論"

        ```python
        from ultralytics.models.sam import Predictor as SAMPredictor

        # SAMPredictorを作成
        overrides = dict(conf=0.25, task='segment', mode='predict', imgsz=1024, model="mobile_sam.pt")
        predictor = SAMPredictor(overrides=overrides)

        # イメージを設定する
        predictor.set_image("ultralytics/assets/zidane.jpg")  # 画像ファイルで設定する
        predictor.set_image(cv2.imread("ultralytics/assets/zidane.jpg"))  # np.ndarrayで設定する
        results = predictor(bboxes=[439, 437, 524, 709])
        results = predictor(points=[900, 370], labels=[1])

        # イメージをリセットする
        predictor.reset_image()
        ```

    追加の引数を指定してすべてのセグメントを設定します。

    === "すべてをセグメントする"

        ```python
        from ultralytics.models.sam import Predictor as SAMPredictor

        # SAMPredictorを作成
        overrides = dict(conf=0.25, task='segment', mode='predict', imgsz=1024, model="mobile_sam.pt")
        predictor = SAMPredictor(overrides=overrides)

        # 追加の引数でセグメント
        results = predictor(source="ultralytics/assets/zidane.jpg", crop_n_layers=1, points_stride=64)
        ```

- `すべてをセグメントする` のための追加の引数の詳細は、[`Predictor/generate` リファレンス](../../../reference/models/sam/predict.md)を参照してください。

## YOLOv8とのSAM比較

ここでは、Metaの最小のSAMモデルであるSAM-bと、Ultralyticsの最小のセグメンテーションモデルである[YOLOv8n-seg](../tasks/segment.md)とを比較します。

| モデル                                            | サイズ                   | パラメータ数               | スピード（CPU）             |
|------------------------------------------------|-----------------------|----------------------|-----------------------|
| MetaのSAM-b                                     | 358 MB                | 94.7 M               | 51096 ms/im           |
| [MobileSAM](mobile-sam.md)                     | 40.7 MB               | 10.1 M               | 46122 ms/im           |
| [FastSAM-s](fast-sam.md) with YOLOv8 backbone  | 23.7 MB               | 11.8 M               | 115 ms/im             |
| Ultralytics [YOLOv8n-seg](../tasks/segment.md) | **6.7 MB** (53.4倍小さい) | **3.4 M** (27.9倍少ない) | **59 ms/im** (866倍速い) |

この比較では、モデルのサイズとスピードの桁違いの違いが示されています。SAMは自動セグメンテーションのユニークな機能を提供しますが、より小さい、より速く、より効率的なYOLOv8セグメントモデルとは競合しません。

テストは、2023年製のApple M2 Macbook、16GBのRAMで実行されました。このテストを再現するには:

!!! Example "例"

    === "Python"
        ```python
        from ultralytics import FastSAM, SAM, YOLO

        # SAM-bのプロファイリング
        model = SAM('sam_b.pt')
        model.info()
        model('ultralytics/assets')

        # MobileSAMのプロファイリング
        model = SAM('mobile_sam.pt')
        model.info()
        model('ultralytics/assets')

        # FastSAM-sのプロファイリング
        model = FastSAM('FastSAM-s.pt')
        model.info()
        model('ultralytics/assets')

        # YOLOv8n-segのプロファイリング
        model = YOLO('yolov8n-seg.pt')
        model.info()
        model('ultralytics/assets')
        ```

## オートアノテーション: セグメンテーションデータセットの迅速な作成方法

オートアノテーションは、SAMの主要な機能の一つであり、事前トレーニング済みの検出モデルを使用して[セグメンテーションデータセット](https://docs.ultralytics.com/datasets/segment)を生成することができます。この機能により、時間のかかる手作業のラベリング作業を回避し、大量の画像の迅速かつ正確な注釈付けが可能になります。

### ディテクションモデルを使用したセグメンテーションデータセットの生成

Ultralyticsフレームワークを使用してデータセットをオートアノテーションするには、以下のように`auto_annotate`関数を使用します:

!!! Example "例"

    === "Python"
        ```python
        from ultralytics.data.annotator import auto_annotate

        auto_annotate(data="path/to/images", det_model="yolov8x.pt", sam_model='sam_b.pt')
        ```

| 引数         | タイプ              | 説明                                                           | デフォルト        |
|------------|------------------|--------------------------------------------------------------|--------------|
| data       | str              | 注釈を付ける画像が含まれるフォルダへのパス。                                       |              |
| det_model  | str, オプション       | 事前トレーニング済みのYOLO検出モデル。デフォルトは'yolov8x.pt'。                     | 'yolov8x.pt' |
| sam_model  | str, オプション       | 事前トレーニング済みのSAMセグメンテーションモデル。デフォルトは'sam_b.pt'。                 | 'sam_b.pt'   |
| device     | str, オプション       | モデルを実行するデバイス。デフォルトは空の文字列（CPUまたはGPUが利用可能な場合）。                 |              |
| output_dir | str, None, オプション | 注釈付け結果を保存するディレクトリ。デフォルトは、'data'と同じディレクトリ内の 'labels' フォルダーです。 | None         |

`auto_annotate`関数は、画像へのパス、任意の事前トレーニング済みの検出およびSAMセグメンテーションモデル、モデルを実行するデバイス、および注釈付け結果を保存する出力ディレクトリを指定するためのオプション引数を取ります。

事前トレーニング済みモデルを使用したオートアノテーションにより、高品質なセグメンテーションデータセットを作成するための時間と労力を大幅に節約することができます。この機能は、大量の画像コレクションに取り組んでいる研究者や開発者にとって特に有益であり、モデルの開発と評価に集中することができます。

## 引用と謝辞

SAMが研究や開発の場で役立つ場合は、引用にご協力いただけると幸いです。

!!! Quote ""

    === "BibTeX"

        ```bibtex
        @misc{kirillov2023segment,
              title={Segment Anything},
              author={Alexander Kirillov and Eric Mintun and Nikhila Ravi and Hanzi Mao and Chloe Rolland and Laura Gustafson and Tete Xiao and Spencer Whitehead and Alexander C. Berg and Wan-Yen Lo and Piotr Dollár and Ross Girshick},
              year={2023},
              eprint={2304.02643},
              archivePrefix={arXiv},
              primaryClass={cs.CV}
        }
        ```

この貴重なコンピュータビジョンコミュニティ向けのリソースを作成および維持してくれたMeta AIに感謝の意を表します。

*keywords: Segment Anything, Segment Anything Model, SAM, Meta SAM, 画像セグメンテーション, プロンプト可能なセグメンテーション, ゼロショットパフォーマンス, SA-1B データセット, 先進のアーキテクチャ, オートアノテーション, Ultralytics, 事前トレーニング済みモデル, SAM base, SAM large, インスタンスセグメンテーション, コンピュータビジョン, AI, 人工知能, 機械学習, データアノテーション, セグメンテーションマスク, ディテクションモデル, YOLOディテクションモデル, bibtex, Meta AI.*
