---
comments: true
description: FastSAMは、画像内のオブジェクトをリアルタイムでセグメンテーションするためのCNNベースのソリューションです。利用者の対話、計算効率の向上、様々なビジョンタスクに対応可能です。
keywords: FastSAM, 機械学習, CNNベースのソリューション, オブジェクトセグメンテーション, リアルタイムソリューション, Ultralytics, ビジョンタスク, 画像処理, 工業用途, ユーザー対話
---

# Fast Segment Anything Model (FastSAM)

Fast Segment Anything Model（FastSAM）は、セグメントエニシングタスクのための新しいリアルタイムのCNNベースのソリューションです。このタスクは、さまざまなユーザー対話のプロンプトに基づいて画像内の任意のオブジェクトをセグメント化することを目的としています。FastSAMは、優れた性能を維持しながら、計算要件を大幅に削減し、様々なビジョンタスクに実用的な選択肢となります。

![Fast Segment Anything Model (FastSAM) architecture overview](https://user-images.githubusercontent.com/26833433/248551984-d98f0f6d-7535-45d0-b380-2e1440b52ad7.jpg)

## 概要

FastSAMは、[Segment Anything Model (SAM)](sam.md)の制約事項に対処するために設計されました。SAMは、大規模な計算リソースを要する重いTransformerモデルです。FastSAMは、セグメントエニシングタスクを2つの連続するステージに分割し、すべてのインスタンスセグメンテーションとプロンプトガイドの選択を行います。最初のステージでは、[YOLOv8-seg](../tasks/segment.md)を使用して、画像内のすべてのインスタンスのセグメンテーションマスクを生成します。2番目のステージでは、プロンプトに対応する領域を出力します。

## 主な特徴

1. **リアルタイムソリューション：** CNNの計算効率を活用することで、FastSAMはセグメントエニシングタスクのためのリアルタイムソリューションを提供し、迅速な結果を必要とする工業用途に価値をもたらします。

2. **効率と性能：** FastSAMは、計算およびリソースの要求を大幅に削減しながら、パフォーマンスの品質を損なうことなく、SAMと同等のパフォーマンスを達成します。これにより、リアルタイムアプリケーションが可能となります。

3. **プロンプトガイドのセグメンテーション：** FastSAMは、さまざまなユーザー対話のプロンプトに基づいて画像内の任意のオブジェクトをセグメント化することができます。これにより、様々なシナリオでの柔軟性と適応性が提供されます。

4. **YOLOv8-segに基づく：** FastSAMは、インスタンスセグメンテーションブランチを備えたオブジェクト検出器である[YOLOv8-seg](../tasks/segment.md)に基づいています。これにより、画像内のすべてのインスタンスのセグメンテーションマスクを効果的に生成することができます。

5. **ベンチマークでの競合力のある結果：** MS COCOのオブジェクトプロポーザルタスクにおいて、FastSAMは単一のNVIDIA RTX 3090上でのSAMよりもはるかに高速に高得点を獲得し、その効率性と能力を示しています。

6. **実用的な応用：** 提案されたアプローチは、現在の方法よりも数十倍または数百倍も高速な速度で、非常に高速なvisionタスクの新しい実用的なソリューションを提供します。

7. **モデルの圧縮の可能性：** FastSAMは、構造への人工的な事前条件を導入することにより、計算負荷を大幅に削減する可能な経路を示し、一般的なビジョンタスクの大規模モデルアーキテクチャの新たな可能性を開くことを示しています。

## 利用可能なモデル、サポートされるタスク、および動作モード

この表は、利用可能なモデルとそれぞれの特定の事前学習済みウェイト、サポートされるタスク、およびInference、Validation、Training、Exportなどの異なる操作モードとの互換性を示しています。サポートされているモードは✅、サポートされていないモードは❌の絵文字で示されます。

| モデルの種類    | 事前学習済みウェイト     | サポートされるタスク                             | Inference | Validation | Training | Export |
|-----------|----------------|----------------------------------------|-----------|------------|----------|--------|
| FastSAM-s | `FastSAM-s.pt` | [インスタンスセグメンテーション](../tasks/segment.md) | ✅         | ❌          | ❌        | ✅      |
| FastSAM-x | `FastSAM-x.pt` | [インスタンスセグメンテーション](../tasks/segment.md) | ✅         | ❌          | ❌        | ✅      |

## 使用例

FastSAMモデルは、Pythonアプリケーションに簡単に統合できます。Ultralyticsは、開発を効率化するためのユーザーフレンドリーなPython APIおよびCLIコマンドを提供しています。

### 予測の使用方法

画像のオブジェクト検出を実行するには、以下のように`predict`メソッドを使用します：

!!! Example "例"

    === "Python"
        ```python
        from ultralytics import FastSAM
        from ultralytics.models.fastsam import FastSAMPrompt

        # 推論元のソースを定義する
        source = 'path/to/bus.jpg'

        # FastSAMモデルを作成する
        model = FastSAM('FastSAM-s.pt')  # または FastSAM-x.pt

        # 画像への推論を実行する
        everything_results = model(source, device='cpu', retina_masks=True, imgsz=1024, conf=0.4, iou=0.9)

        # Prompt Processオブジェクトを準備する
        prompt_process = FastSAMPrompt(source, everything_results, device='cpu')

        # Everything prompt
        ann = prompt_process.everything_prompt()

        # バウンディングボックスのデフォルトの形状は [0,0,0,0] -> [x1,y1,x2,y2]
        ann = prompt_process.box_prompt(bbox=[200, 200, 300, 300])

        # テキストプロンプト
        ann = prompt_process.text_prompt(text='a photo of a dog')

        # ポイントプロンプト
        # pointsのデフォルトは [[0,0]] [[x1,y1],[x2,y2]]
        # point_labelのデフォルトは [0] [1,0] 0:background, 1:foreground
        ann = prompt_process.point_prompt(points=[[200, 200]], pointlabel=[1])
        prompt_process.plot(annotations=ann, output='./')
        ```

    === "CLI"
        ```bash
        # FastSAMモデルをロードし、それによってeverythingをセグメント化する
        yolo segment predict model=FastSAM-s.pt source=path/to/bus.jpg imgsz=640
        ```

このスニペットは、事前学習済みモデルをロードし、イメージに対する予測を実行するシンプルさを示しています。

### 検証の使用方法

データセット上でモデルの検証を行うには、以下のようにします：

!!! Example "例"

    === "Python"
        ```python
        from ultralytics import FastSAM

        # FastSAMモデルを作成する
        model = FastSAM('FastSAM-s.pt')  # または FastSAM-x.pt

        # モデルを検証する
        results = model.val(data='coco8-seg.yaml')
        ```

    === "CLI"
        ```bash
        # FastSAMモデルをロードし、COCO8の例のデータセットで検証する（イメージサイズ：640）
        yolo segment val model=FastSAM-s.pt data=coco8.yaml imgsz=640
        ```

FastSAMは、オブジェクトの検出とセグメンテーションを1つのクラスのオブジェクトに対してのみサポートしています。これは、すべてのオブジェクトを同じクラスとして認識し、セグメント化することを意味します。そのため、データセットを準備する際には、すべてのオブジェクトのカテゴリIDを0に変換する必要があります。

## FastSAM公式の使用方法

FastSAMは、[https://github.com/CASIA-IVA-Lab/FastSAM](https://github.com/CASIA-IVA-Lab/FastSAM)リポジトリから直接利用することもできます。以下は、FastSAMを使用するための一般的な手順の概要です。

### インストール

1. FastSAMリポジトリをクローンする：
   ```shell
   git clone https://github.com/CASIA-IVA-Lab/FastSAM.git
   ```

2. Python 3.9を使用したConda環境を作成してアクティベートする：
   ```shell
   conda create -n FastSAM python=3.9
   conda activate FastSAM
   ```

3. クローンされたリポジトリに移動し、必要なパッケージをインストールする：
   ```shell
   cd FastSAM
   pip install -r requirements.txt
   ```

4. CLIPモデルをインストールする：
   ```shell
   pip install git+https://github.com/openai/CLIP.git
   ```

### 使用例

1. [モデルのチェックポイント](https://drive.google.com/file/d/1m1sjY4ihXBU1fZXdQ-Xdj-mDltW-2Rqv/view?usp=sharing)をダウンロードします。

2. FastSAMを推論に使用します。以下は実行例です：

    - 画像内のすべてをセグメント化する：
      ```shell
      python Inference.py --model_path ./weights/FastSAM.pt --img_path ./images/dogs.jpg
      ```

    - テキストプロンプトを使用して特定のオブジェクトをセグメント化する：
      ```shell
      python Inference.py --model_path ./weights/FastSAM.pt --img_path ./images/dogs.jpg --text_prompt "the yellow dog"
      ```

    - バウンディングボックス内のオブジェクトをセグメント化する（xywh形式でボックス座標を指定します）：
      ```shell
      python Inference.py --model_path ./weights/FastSAM.pt --img_path ./images/dogs.jpg --box_prompt "[570,200,230,400]"
      ```

    - 特定のポイントの近くにあるオブジェクトをセグメント化する：
      ```shell
      python Inference.py --model_path ./weights/FastSAM.pt --img_path ./images/dogs.jpg --point_prompt "[[520,360],[620,300]]" --point_label "[1,0]"
      ```

さらに、FastSAMを[Colabデモ](https://colab.research.google.com/drive/1oX14f6IneGGw612WgVlAiy91UHwFAvr9?usp=sharing)や[HuggingFaceウェブデモ](https://huggingface.co/spaces/An-619/FastSAM)で試すこともできます。

## 引用と謝辞

FastSAMの著者には、リアルタイムインスタンスセグメンテーションの分野での重要な貢献を称えたいと思います。

!!! Quote ""

    === "BibTeX"

      ```bibtex
      @misc{zhao2023fast,
            title={Fast Segment Anything},
            author={Xu Zhao and Wenchao Ding and Yongqi An and Yinglong Du and Tao Yu and Min Li and Ming Tang and Jinqiao Wang},
            year={2023},
            eprint={2306.12156},
            archivePrefix={arXiv},
            primaryClass={cs.CV}
      }
      ```

FastSAMのオリジナルの論文は、[arXiv](https://arxiv.org/abs/2306.12156)で入手できます。著者は彼らの作品を広く公開し、コードベースは[GitHub](https://github.com/CASIA-IVA-Lab/FastSAM)でアクセスできるようにしています。私たちは、彼らがフィールドを進歩させ、その成果を広いコミュニティにアクセス可能にしてくれた彼らの努力に感謝しています。
