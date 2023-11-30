---
comments: true
description: Ultralyticsのpip、conda、git、Dockerを使用した様々なインストール方法を探索し、コマンドラインインターフェースまたはPythonプロジェクト内でのUltralyticsの使用方法を学びます。
keywords: Ultralyticsインストール, pipインストールUltralytics, DockerインストールUltralytics, Ultralyticsコマンドラインインターフェース, Ultralytics Pythonインターフェース
---

## Ultralyticsのインストール

Ultralyticsはpip、conda、Dockerを含むさまざまなインストール方法を提供しています。最新の安定版リリースである`ultralytics` pipパッケージを通じてYOLOv8をインストールするか、最新バージョンを取得するために[Ultralytics GitHubリポジトリ](https://github.com/ultralytics/ultralytics)をクローンします。Dockerは、ローカルインストールを回避し、孤立したコンテナ内でパッケージを実行するために使用できます。

!!! Example "インストール"

    === "Pipでのインストール（推奨）"
        pipを使用して`ultralytics`パッケージをインストールするか、`pip install -U ultralytics`を実行して既存のインストールをアップデートします。`ultralytics`パッケージの詳細については、Python Package Index（PyPI）を参照してください: [https://pypi.org/project/ultralytics/](https://pypi.org/project/ultralytics/)。

        [![PyPI version](https://badge.fury.io/py/ultralytics.svg)](https://badge.fury.io/py/ultralytics) [![Downloads](https://static.pepy.tech/badge/ultralytics)](https://pepy.tech/project/ultralytics)

        ```bash
        # PyPIからultralyticsパッケージをインストール
        pip install ultralytics
        ```

        GitHubの[リポジトリ](https://github.com/ultralytics/ultralytics)から直接`ultralytics`パッケージをインストールすることもできます。これは、最新の開発版が必要な場合に便利かもしれません。システムにGitコマンドラインツールがインストールされている必要があります。`@main`コマンドは`main`ブランチをインストールし、別のブランチ、例えば`@my-branch`に変更したり、`main`ブランチにデフォルトするために完全に削除することができます。

        ```bash
        # GitHubからultralyticsパッケージをインストール
        pip install git+https://github.com/ultralytics/ultralytics.git@main
        ```


    === "Condaでのインストール"
        Condaはpipの代わりのパッケージマネージャーで、インストールにも使用できます。より詳細はAnacondaを参照してください [https://anaconda.org/conda-forge/ultralytics](https://anaconda.org/conda-forge/ultralytics)。Condaパッケージを更新するためのUltralyticsフィードストックリポジトリはこちらです [https://github.com/conda-forge/ultralytics-feedstock/](https://github.com/conda-forge/ultralytics-feedstock/)。


        [![Conda Recipe](https://img.shields.io/badge/recipe-ultralytics-green.svg)](https://anaconda.org/conda-forge/ultralytics) [![Conda Downloads](https://img.shields.io/conda/dn/conda-forge/ultralytics.svg)](https://anaconda.org/conda-forge/ultralytics) [![Conda Version](https://img.shields.io/conda/vn/conda-forge/ultralytics.svg)](https://anaconda.org/conda-forge/ultralytics) [![Conda Platforms](https://img.shields.io/conda/pn/conda-forge/ultralytics.svg)](https://anaconda.org/conda-forge/ultralytics)

        ```bash
        # Condaを使用してultralyticsパッケージをインストール
        conda install -c conda-forge ultralytics
        ```

        !!! Note "ノート"

            CUDA環境でインストールする場合、パッケージマネージャーが競合を解決できるようにするため、`ultralytics`、`pytorch`、`pytorch-cuda`を同じコマンドで一緒にインストールするのがベストプラクティスです。または、CPU専用の`pytorch`パッケージに必要な場合は上書きするように`pytorch-cuda`を最後にインストールします。
            ```bash
            # Condaを使用して一緒にすべてのパッケージをインストール
            conda install -c pytorch -c nvidia -c conda-forge pytorch torchvision pytorch-cuda=11.8 ultralytics
            ```

        ### Conda Dockerイメージ

        UltralyticsのConda Dockerイメージも[DockerHub](https://hub.docker.com/r/ultralytics/ultralytics)から利用可能です。これらのイメージは[Miniconda3](https://docs.conda.io/projects/miniconda/en/latest/)に基づいており、Conda環境で`ultralytics`を使用する簡単な方法です。

        ```bash
        # イメージ名を変数として設定
        t=ultralytics/ultralytics:latest-conda

        # Docker Hubから最新のultralyticsイメージをプル
        sudo docker pull $t

        # すべてのGPUを持つコンテナでultralyticsイメージを実行
        sudo docker run -it --ipc=host --gpus all $t  # すべてのGPU
        sudo docker run -it --ipc=host --gpus '"device=2,3"' $t  # GPUを指定
        ```

    === "Gitクローン"
        開発への貢献に興味がある場合や、最新のソースコードで実験したい場合は、`ultralytics`リポジトリをクローンしてください。クローンした後、ディレクトリに移動し、pipを使って編集可能モード`-e`でパッケージをインストールします。
        ```bash
        # ultralyticsリポジトリをクローン
        git clone https://github.com/ultralytics/ultralytics

        # クローンしたディレクトリに移動
        cd ultralytics

        # 開発用に編集可能モードでパッケージをインストール
        pip install -e .
        ```

必要な依存関係のリストについては、`ultralytics`の[requirements.txt](https://github.com/ultralytics/ultralytics/blob/main/requirements.txt)ファイルを参照してください。上記の全ての例では、必要な依存関係を全てインストールします。

<p align="center">
  <br>
  <iframe width="720" height="405" src="https://www.youtube.com/embed/_a7cVL9hqnk"
    title="YouTube video player" frameborder="0"
    allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share"
    allowfullscreen>
  </iframe>
  <br>
  <strong>Watch:</strong> Ultralytics YOLO Quick Start Guide
</p>

!!! Tip "ヒント"

    PyTorchの要件はオペレーティングシステムとCUDAの要件によって異なるため、[https://pytorch.org/get-started/locally](https://pytorch.org/get-started/locally)に従って最初にPyTorchをインストールすることをお勧めします。

    <a href="https://pytorch.org/get-started/locally/">
        <img width="800" alt="PyTorch Installation Instructions" src="https://user-images.githubusercontent.com/26833433/228650108-ab0ec98a-b328-4f40-a40d-95355e8a84e3.png">
    </a>

## CLIでUltralyticsを使用

Ultralyticsコマンドラインインターフェース（CLI）を使用すると、Python環境がなくても単一の行のコマンドを簡単に実行できます。CLIはカスタマイズもPythonコードも必要ありません。単純にすべてのタスクを`yolo`コマンドでターミナルから実行することができます。コマンドラインからYOLOv8を使用する方法について詳しくは、[CLIガイド](/../usage/cli.md)を参照してください。

!!! Example "例"

    === "構文"

        Ultralyticsの`yolo`コマンドは以下の構文を使用します：
        ```bash
        yolo TASK MODE ARGS

        ここで  TASK（オプション）は[detect, segment, classify]のうちの1つ
                MODE（必須）は[train, val, predict, export, track]のうちの1つ
                ARGS（オプション）はデフォルトを上書きする任意の数のカスタム'arg=value'ペアです。
        ```
        full [Configuration Guide](/../usage/cfg.md)または`yolo cfg`で全てのARGSを確認してください

    === "トレーニング"

        10エポックにわたって初期学習率0.01で検出モデルをトレーニング
        ```bash
        yolo train data=coco128.yaml model=yolov8n.pt epochs=10 lr0=0.01
        ```

    === "予測"

        画像サイズ320で事前トレーニングされたセグメンテーションモデルを使用してYouTubeビデオを予測：
        ```bash
        yolo predict model=yolov8n-seg.pt source='https://youtu.be/LNwODJXcvt4' imgsz=320
        ```

    === "検証"

        バッチサイズ1および画像サイズ640で事前トレーニングされた検出モデルを検証する：
        ```bash
        yolo val model=yolov8n.pt data=coco128.yaml batch=1 imgsz=640
        ```

    === "エクスポート"

        画像サイズ224 x 128でYOLOv8n分類モデルをONNX形式にエクスポート（TASKは不要）
        ```bash
        yolo export model=yolov8n-cls.pt format=onnx imgsz=224,128
        ```

    === "特殊"

        バージョンを確認したり、設定を表示したり、チェックを行ったりするための特別なコマンドを実行します：
        ```bash
        yolo help
        yolo checks
        yolo version
        yolo settings
        yolo copy-cfg
        yolo cfg
        ```

!!! Warning "警告"

    引数は`arg=val`ペアとして渡され、`=`記号で分割され、ペア間にスペース` `が必要です。引数のプレフィックスに`--`や引数間にカンマ`,`を使用しないでください。

    - `yolo predict model=yolov8n.pt imgsz=640 conf=0.25` &nbsp; ✅
    - `yolo predict model yolov8n.pt imgsz 640 conf 0.25` &nbsp; ❌
    - `yolo predict --model yolov8n.pt --imgsz 640 --conf 0.25` &nbsp; ❌

[CLIガイド](/../usage/cli.md){ .md-button }

## PythonでUltralyticsを使用

YOLOv8のPythonインターフェースを使用すると、Pythonプロジェクトにシームレスに統合し、モデルをロード、実行、出力を処理することが可能です。簡単さと使いやすさを念頭に設計されたPythonインターフェースにより、ユーザーは素早くプロジェクトに物体検出、セグメンテーション、分類を実装することができます。このように、YOLOv8のPythonインターフェースは、これらの機能をPythonプロジェクトに取り入れたいと考えている方にとって貴重なツールです。

たとえば、ユーザーはモデルをロードして、トレーニングし、検証セットでのパフォーマンスを評価し、ONNX形式にエクスポートするまでの一連の処理を数行のコードで行うことができます。YOLOv8をPythonプロジェクトで使用する方法について詳しくは、[Pythonガイド](/../usage/python.md)を参照してください。

!!! Example "例"

    ```python
    from ultralytics import YOLO

    # スクラッチから新しいYOLOモデルを作成
    model = YOLO('yolov8n.yaml')

    # 事前トレーニドされたYOLOモデルをロード（トレーニングに推奨）
    model = YOLO('yolov8n.pt')

    # 'coco128.yaml'データセットを使用して3エポックでモデルをトレーニング
    results = model.train(data='coco128.yaml', epochs=3)

    # モデルのパフォーマンスを検証セットで評価
    results = model.val()

    # モデルを使用して画像で物体検出を実行
    results = model('https://ultralytics.com/images/bus.jpg')

    # モデルをONNX形式にエクスポート
    success = model.export(format='onnx')
    ```

[Pythonガイド](/../usage/python.md){.md-button .md-button--primary}
