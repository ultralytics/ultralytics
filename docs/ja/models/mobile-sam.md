---
comments: true
description: Ultralyticsフレームワーク内でMobileSAMをダウンロードしてテストする方法、MobileSAMの実装、オリジナルのSAMとの比較について詳しく知ることができます。今日からモバイルアプリケーションを改善しましょう。
keywords: MobileSAM, Ultralytics, SAM, モバイルアプリケーション, Arxiv, GPU, API, 画像エンコーダ, マスクデコーダ, モデルのダウンロード, テスト方法
---

![MobileSAM ロゴ](https://github.com/ChaoningZhang/MobileSAM/blob/master/assets/logo2.png?raw=true)

# Mobile Segment Anything（MobileSAM）

MobileSAM論文が[arXiv](https://arxiv.org/pdf/2306.14289.pdf)で利用可能になりました。

CPU上で動作するMobileSAMのデモは、[こちらのデモリンク](https://huggingface.co/spaces/dhkim2810/MobileSAM)からアクセスできます。Mac i5 CPU上では、約3秒かかります。Hugging Faceのデモでは、インターフェースと低性能なCPUが遅い応答に寄与していますが、効果的に動作し続けます。

MobileSAMは、[Grounding-SAM](https://github.com/IDEA-Research/Grounded-Segment-Anything)、[AnyLabeling](https://github.com/vietanhdev/anylabeling)、および[Segment Anything in 3D](https://github.com/Jumpat/SegmentAnythingin3D)など、さまざまなプロジェクトで実装されています。

MobileSAMは、100kのデータセット（元の画像の1%）を単一のGPUで学習し、1日未満で訓練が完了します。このトレーニングのコードは将来公開される予定です。

## 利用可能なモデル、サポートされているタスク、および動作モード

この表は、利用可能なモデルとそれぞれの固有の事前学習重み、サポートされているタスク、および[予測](../modes/predict.md)、[検証](../modes/val.md)、[訓練](../modes/train.md)、および[エクスポート](../modes/export.md)のようなさまざまな動作モードに対する互換性を示しています。`✅`は対応しているモード、`❌`は対応していないモードを示しています。

| モデルタイプ    | 事前学習重み          | サポートされているタスク                           | 予測 | 検証 | 訓練 | エクスポート |
|-----------|-----------------|----------------------------------------|----|----|----|--------|
| MobileSAM | `mobile_sam.pt` | [インスタンスセグメンテーション](../tasks/segment.md) | ✅  | ❌  | ❌  | ✅      |

## SAMからMobileSAMへの移行

MobileSAMは、オリジナルのSAMと同じパイプラインを維持しているため、オリジナルの前処理、後処理、およびその他のインタフェースを組み込んでいます。そのため、現在オリジナルのSAMを使用している場合でも、MobileSAMへの移行は最小限の労力で行うことができます。

MobileSAMは、オリジナルのSAMと同等のパフォーマンスを発揮し、イメージエンコーダを変更することで同じパイプラインを保持しています。具体的には、元の重いViT-Hエンコーダ（632M）をより小さいTiny-ViT（5M）に置き換えています。単一のGPU上でMobileSAMは、おおよそ画像あたり12msで動作します：イメージエンコーダで8ms、マスクデコーダで4msです。

次の表は、ViTベースのイメージエンコーダの比較です：

| イメージエンコーダ | オリジナルのSAM | MobileSAM |
|-----------|-----------|-----------|
| パラメーター    | 611M      | 5M        |
| 速度        | 452ms     | 8ms       |

オリジナルのSAMとMobileSAMは、同じプロンプト誘導型マスクデコーダを使用しています：

| マスクデコーダ | オリジナルのSAM | MobileSAM |
|---------|-----------|-----------|
| パラメーター  | 3.876M    | 3.876M    |
| 速度      | 4ms       | 4ms       |

以下は、全体のパイプラインの比較です：

| パイプライン全体（エンコーダ+デコーダ） | オリジナルのSAM | MobileSAM |
|----------------------|-----------|-----------|
| パラメーター               | 615M      | 9.66M     |
| 速度                   | 456ms     | 12ms      |

MobileSAMとオリジナルのSAMのパフォーマンスは、ポイントとボックスをプロンプトとして使用した場合に示されます。

![ポイントをプロンプトにした画像](https://raw.githubusercontent.com/ChaoningZhang/MobileSAM/master/assets/mask_box.jpg?raw=true)

![ボックスをプロンプトにした画像](https://raw.githubusercontent.com/ChaoningZhang/MobileSAM/master/assets/mask_box.jpg?raw=true)

MobileSAMは、現在のFastSAMよりも約5倍小さく、約7倍高速です。詳細は[MobileSAMプロジェクトページ](https://github.com/ChaoningZhang/MobileSAM)でご覧いただけます。

## UltralyticsでのMobileSAMのテスト

オリジナルのSAMと同様に、ポイントとボックスのプロンプトの両方に対応したUltralyticsでの簡単なテスト方法を提供しています。

### モデルのダウンロード

モデルは[こちらからダウンロード](https://github.com/ChaoningZhang/MobileSAM/blob/master/weights/mobile_sam.pt)できます。

### ポイントプロンプト

!!! Example "例"

    === "Python"
        ```python
        from ultralytics import SAM

        # モデルをロード
        model = SAM('mobile_sam.pt')

        # ポイントプロンプトに基づいてセグメントを予測
        model.predict('ultralytics/assets/zidane.jpg', points=[900, 370], labels=[1])
        ```

### ボックスプロンプト

!!! Example "例"

    === "Python"
        ```python
        from ultralytics import SAM

        # モデルをロード
        model = SAM('mobile_sam.pt')

        # ボックスプロンプトに基づいてセグメントを予測
        model.predict('ultralytics/assets/zidane.jpg', bboxes=[439, 437, 524, 709])
        ```

`MobileSAM`と`SAM`は、同じAPIを使用して実装されています。詳細な使用方法については、[SAMページ](sam.md)をご覧ください。

## 引用と謝辞

MobileSAMが研究や開発のお役に立つ場合は、次の論文を引用していただけると幸いです：

!!! Quote文 ""

    === "BibTeX"

        ```bibtex
        @article{mobile_sam,
          title={Faster Segment Anything: Towards Lightweight SAM for Mobile Applications},
          author={Zhang, Chaoning and Han, Dongshen and Qiao, Yu and Kim, Jung Uk and Bae, Sung Ho and Lee, Seungkyu and Hong, Choong Seon},
          journal={arXiv preprint arXiv:2306.14289},
          year={2023}
        }
