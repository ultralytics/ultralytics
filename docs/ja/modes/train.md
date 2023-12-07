---
comments: true
description: YOLOv8モデルをUltralytics YOLOを使用してトレーニングする手順についてのガイドで、シングルGPUとマルチGPUトレーニングの例を含む
keywords: Ultralytics, YOLOv8, YOLO, 物体検出, トレーニングモード, カスタムデータセット, GPUトレーニング, マルチGPU, ハイパーパラメータ, CLI例, Python例
---

# Ultralytics YOLOを使ったモデルトレーニング

<img width="1024" src="https://github.com/ultralytics/assets/raw/main/yolov8/banner-integrations.png" alt="Ultralytics YOLOエコシステムと統合">

## はじめに

ディープラーニングモデルのトレーニングは、データを与えてパラメーターを調整し、正確な予測を行えるようにするプロセスを含みます。UltralyticsのYOLOv8のトレーニングモードは、現代のハードウェアの能力をフルに活用して物体検出モデルを効果的かつ効率的にトレーニングするために設計されています。このガイドは、YOLOv8 の豊富な機能セットを使用して自身のモデルをトレーニングするために必要なすべての詳細をカバーすることを目的としています。

<p align="center">
  <br>
  <iframe width="720" height="405" src="https://www.youtube.com/embed/LNwODJXcvt4?si=7n1UvGRLSd9p5wKs"
    title="YouTube動画プレーヤー" frameborder="0"
    allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share"
    allowfullscreen>
  </iframe>
  <br>
  <strong>視聴:</strong> Google Colab でカスタムデータセットにYOLOv8モデルをトレーニングする方法。
</p>

## トレーニングにUltralyticsのYOLOを選ぶ理由

YOLOv8のトレーニングモードを選択するいくつかの魅力的な理由を以下に示します：

- **効率性：** シングルGPUセットアップであろうと複数のGPUにスケールする場合であろうと、あなたのハードウェアを最大限に活用します。
- **汎用性：** COCO、VOC、ImageNetのような既存のデータセットに加え、カスタムデータセットでのトレーニングが可能です。
- **ユーザーフレンドリー：** 直感的でありながら強力なCLIとPythonインターフェースを備え、簡単なトレーニング体験を提供します。
- **ハイパーパラメータの柔軟性：** モデルのパフォーマンスを微調整するための幅広いカスタマイズ可能なハイパーパラメータ。

### トレーニングモードの主な特徴

以下に、YOLOv8のトレーニングモードのいくつかの注目すべき特徴を挙げます：

- **自動データセットダウンロード：** COCO、VOC、ImageNetのような標準データセットは最初の使用時に自動的にダウンロードされます。
- **マルチGPUサポート：** 複数のGPUにわたってトレーニングをスケールし、プロセスを迅速に行います。
- **ハイパーパラメータの設定：** YAML設定ファイルやCLI引数を通じてハイパーパラメータを変更するオプション。
- **可視化とモニタリング：** トレーニング指標のリアルタイム追跡と学習プロセスの可視化により、より良い洞察を得ます。

!!! Tip "ヒント"

    * YOLOv8のデータセット、例えばCOCO、VOC、ImageNetなどは、最初の使用時に自動的にダウンロードされます。例：`yolo train data=coco.yaml`

## 使用例

COCO128データセットでYOLOv8nを100エポック、画像サイズ640でトレーニングする。トレーニングデバイスは、`device`引数を使って指定できます。引数が渡されない場合、利用可能であればGPU `device=0`が、そうでなければ`device=cpu`が利用されます。全てのトレーニング引数のリストは以下の引数セクションを参照してください。

!!! Example "シングルGPUとCPUトレーニング例"

    デバイスは自動的に決定されます。GPUが利用可能であればそれが使用され、そうでなければCPUでトレーニングが開始されます。

    === "Python"

        ```python
        from ultralytics import YOLO

        # モデルをロード
        model = YOLO('yolov8n.yaml')  # YAMLから新しいモデルを作成
        model = YOLO('yolov8n.pt')  # トレーニングにはおすすめの事前学習済みモデルをロード
        model = YOLO('yolov8n.yaml').load('yolov8n.pt')  # YAMLからモデルを作成し、重みを転送

        # モデルをトレーニング
        results = model.train(data='coco128.yaml', epochs=100, imgsz=640)
        ```

    === "CLI"

        ```bash
        # YAMLから新しいモデルを作成し、最初からトレーニングを開始
        yolo detect train data=coco128.yaml model=yolov8n.yaml epochs=100 imgsz=640

        # 事前学習済み*.ptモデルからトレーニングを開始
        yolo detect train data=coco128.yaml model=yolov8n.pt epochs=100 imgsz=640

        # YAMLから新しいモデルを作成し、事前学習済みの重みを転送してトレーニングを開始
        yolo detect train data=coco128.yaml model=yolov8n.yaml pretrained=yolov8n.pt epochs=100 imgsz=640
        ```

### マルチGPUトレーニング

マルチGPUトレーニングは、利用可能なハードウェアリソースをより効率的に活用するために、トレーニングの負荷を複数のGPUに分散させることを可能にします。この機能はPython APIとコマンドラインインターフェィスの両方を通じて利用できます。マルチGPUトレーニングを有効にするには、使用したいGPUデバイスIDを指定します。

!!! Example "マルチGPUトレーニング例"

    2つのGPUを使ってトレーニングするには、CUDAデバイス0と1を使い以下のコマンドを使用します。必要に応じて追加のGPUに拡張します。

    === "Python"

        ```python
        from ultralytics import YOLO

        # モデルをロード
        model = YOLO('yolov8n.pt')  # トレーニングにはおすすめの事前学習済みモデルをロード

        # 2つのGPUでモデルをトレーニング
        results = model.train(data='coco128.yaml', epochs=100, imgsz=640, device=[0, 1])
        ```

    === "CLI"

        ```bash
        # 事前学習済み*.ptモデルからGPU 0と1を使ってトレーニングを開始
        yolo detect train data=coco128.yaml model=yolov8n.pt epochs=100 imgsz=640 device=0,1
        ```

### Apple M1 および M2 MPSトレーニング

AppleのM1およびM2チップに対するサポートがUltralyticsのYOLOモデルに統合されたことで、Appleの強力なMetal Performance Shaders（MPS）フレームワークを使用してデバイスでモデルをトレーニングすることが可能になりました。 MPSは、Appleのカスタムシリコン上での計算や画像処理タスクの高性能な実行方法を提供します。

AppleのM1およびM2チップでのトレーニングを有効にするには、トレーニングプロセスを開始する際に`mps`をデバイスとして指定する必要があります。以下はPythonおよびコマンドラインでこれを行う例です：

!!! Example "MPSトレーニング例"

    === "Python"

        ```python
        from ultralytics import YOLO

        # モデルをロード
        model = YOLO('yolov8n.pt')  # トレーニングにはおすすめの事前学習済みモデルをロード

        # MPSを使ってモデルをトレーニング
        results = model.train(data='coco128.yaml', epochs=100, imgsz=640, device='mps')
        ```

    === "CLI"

        ```bash
        # MPSを使って、事前学習済み*.ptモデルからトレーニングを開始
        yolo detect train data=coco128.yaml model=yolov8n.pt epochs=100 imgsz=640 device=mps
        ```

M1/M2チップの計算能力を利用しながら、これによりトレーニングタスクのより効率的な処理が可能になります。より詳細なガイダンスや高度な設定オプションについては、[PyTorch MPSのドキュメント](https://pytorch.org/docs/stable/notes/mps.html)を参照してください。

## ロギング

YOLOv8モデルをトレーニングする際、モデルのパフォーマンスを時間とともに追跡することが価値あることであると考えられます。これがロギングの役割になります。UltralyticsのYOLOは、Comet、ClearML、TensorBoardの3種類のロガーをサポートしています。

ロガーを使用するには、上記のコードスニペットからドロップダウンメニューを選択し、実行します。選択したロガーがインストールされ、初期化されます。

### Comet

[Comet](https://www.comet.ml/site/)は、データサイエンティストや開発者が実験やモデルを追跡、比較、説明、最適化するためのプラットフォームです。リアルタイムメトリクスやコード差分、ハイパーパラメータの追跡などの機能を提供しています。

Cometを使用するには：

!!! Example "例"

    === "Python"
        ```python
        # pip install comet_ml
        import comet_ml

        comet_ml.init()
        ```

Cometアカウントにサインインし、APIキーを取得してください。このキーを環境変数またはスクリプトに追加して、実験をログに記録する必要があります。

### ClearML

[ClearML](https://www.clear.ml/)は、実験の追跡を自動化し、資源の効率的な共有を支援するオープンソースプラットフォームです。チームがML作業をより効率的に管理、実行、再現するのに役立ちます。

ClearMLを使用するには：

!!! Example "例"

    === "Python"
        ```python
        # pip install clearml
        import clearml

        clearml.browser_login()
        ```

このスクリプトを実行した後、ブラウザでClearMLアカウントにサインインし、セッションを認証する必要があります。

### TensorBoard

[TensorBoard](https://www.tensorflow.org/tensorboard)は、TensorFlowの視覚化ツールキットです。TensorFlowグラフを可視化し、グラフの実行に関する定量的メトリックをプロットし、それを通過する画像などの追加データを表示することができます。

[Google Colab](https://colab.research.google.com/github/ultralytics/ultralytics/blob/main/examples/tutorial.ipynb)でTensorBoardを使用するには：

!!! Example "例"

    === "CLI"
        ```bash
        load_ext tensorboard
        tensorboard --logdir ultralytics/runs  # 'runs'ディレクトリと置き換えてください
        ```

TensorBoardをローカルで使用する場合は、http://localhost:6006/ で結果を確認できます。

!!! Example "例"

    === "CLI"
        ```bash
        tensorboard --logdir ultralytics/runs  # 'runs'ディレクトリと置き換えてください
        ```

これでTensorBoardがロードされ、トレーニングログが保存されているディレクトリを指定します。

ログを設定した後、モデルのトレーニングを進めてください。すべてのトレーニングメトリクスが選択したプラットフォームに自動的に記録され、これらのログをアクセスして、時間とともにモデルのパフォーマンスを監視したり、さまざまなモデルを比較したり、改善の余地を特定したりすることができます。
