---
comments: true
description: YOLO-NASは、優れた物体検出モデルです。その機能、事前学習モデル、Ultralytics Python APIの使用法などについて詳しく説明します。
keywords: YOLO-NAS, Deci AI, 物体検出, 深層学習, ニューラルアーキテクチャ検索, Ultralytics Python API, YOLOモデル, 事前学習モデル, 量子化, 最適化, COCO, Objects365, Roboflow 100
---

# YOLO-NAS

## 概要

Deci AIによって開発されたYOLO-NASは、画期的な物体検出ベースモデルです。従来のYOLOモデルの制約に対処するための高度なニューラルアーキテクチャ検索技術によって生み出されています。量子化のサポートと精度とレイテンシのトレードオフの改善により、YOLO-NASは物体検出において大きな進歩を遂げています。

![モデルの例の画像](https://learnopencv.com/wp-content/uploads/2023/05/yolo-nas_COCO_map_metrics.png)
**YOLO-NASの概要。** YOLO-NASは、量子化対応のブロックと選択的量子化を使用して最適なパフォーマンスを実現します。モデルをINT8で量子化すると、他のモデルよりも精度がほとんど低下せずに最適なパフォーマンスが得られます。これらの進歩により、前例のない物体検出能力と優れたパフォーマンスを備えた優れたアーキテクチャが実現されます。

### 主な特徴

- **量子化フレンドリーな基本ブロック:** YOLO-NASは、従来のYOLOモデルの制約の1つである量子化に対応した新しい基本ブロックを導入しています。
- **洗練されたトレーニングと量子化:** YOLO-NASは、高度なトレーニングスキームとポストトレーニング量子化を活用してパフォーマンスを向上させています。
- **AutoNAC最適化と事前学習:** YOLO-NASはAutoNAC最適化を利用し、COCO、Objects365、Roboflow 100などの注目されるデータセットで事前学習されています。この事前学習により、製品環境での下流物体検出タスクに非常に適しています。

## 事前学習モデル

Ultralyticsが提供する事前学習済みのYOLO-NASモデルを使用して、次世代の物体検出のパワーを体験してください。これらのモデルは、速度と精度の両方の面で優れたパフォーマンスを提供するように設計されています。特定のニーズに合わせてさまざまなオプションから選択できます。

| モデル              | mAP   | レイテンシ (ms) |
|------------------|-------|------------|
| YOLO-NAS S       | 47.5  | 3.21       |
| YOLO-NAS M       | 51.55 | 5.85       |
| YOLO-NAS L       | 52.22 | 7.87       |
| YOLO-NAS S INT-8 | 47.03 | 2.36       |
| YOLO-NAS M INT-8 | 51.0  | 3.78       |
| YOLO-NAS L INT-8 | 52.1  | 4.78       |

各モデルのバリエーションは、Mean Average Precision（mAP）とレイテンシのバランスを取り、パフォーマンスとスピードの両方に最適化されています。

## 使用例

Ultralyticsの`ultralytics` Pythonパッケージを使用して、YOLO-NASモデルをPythonアプリケーションに簡単に統合できるようにしました。このパッケージは、プロセスをスムーズにするユーザーフレンドリーなPython APIを提供します。

次の例では、推論と検証のために`ultralytics`パッケージを使用してYOLO-NASモデルをどのように使用するかを示しています。

### 推論と検証の例

この例では、COCO8データセットでYOLO-NAS-sを検証します。

!!! Example "例"

    この例では、YOLO-NASの推論と検証のためのシンプルなコードを提供しています。推論結果の処理については、[Predict](../modes/predict.md)モードを参照してください。他のモードでYOLO-NASを使用する方法については、[Val](../modes/val.md)および[Export](../modes/export.md)を参照してください。`ultralytics`パッケージのYOLO-NASはトレーニングをサポートしていません。

    === "Python"

        Pythonで、PyTorchの事前学習済みの`*.pt`モデルファイルを`NAS()`クラスに渡すことで、モデルのインスタンスを作成できます:

        ```python
        from ultralytics import NAS

        # COCO事前学習済みのYOLO-NAS-sモデルをロード
        model = NAS('yolo_nas_s.pt')

        # モデル情報の表示（オプション）
        model.info()

        # COCO8の例データセットでモデルを検証
        results = model.val(data='coco8.yaml')

        # 'bus.jpg'画像上でYOLO-NAS-sモデルを使用した推論
        results = model('path/to/bus.jpg')
        ```

    === "CLI"

        モデルを直接実行するためのCLIコマンドもあります:

        ```bash
        # COCO事前学習済みのYOLO-NAS-sモデルをロードし、COCO8の例データセットでパフォーマンスを検証
        yolo val model=yolo_nas_s.pt data=coco8.yaml

        # COCO事前学習済みのYOLO-NAS-sモデルをロードし、'bus.jpg'画像上で推論を実行
        yolo predict model=yolo_nas_s.pt source=path/to/bus.jpg
        ```

## サポートされているタスクとモード

YOLO-NASモデルは、Small（s）、Medium（m）、Large（l）の3つのバリエーションを提供しています。各バリエーションは、異なる計算リソースとパフォーマンスのニーズに対応するように設計されています:

- **YOLO-NAS-s:** 計算リソースが限られている環境で効率が重要な場合に最適化されています。
- **YOLO-NAS-m:** 幅広い一般的な物体検出のニーズに適したバランスの取れたアプローチです。
- **YOLO-NAS-l:** 計算リソースの制約が少ない最高の精度が求められるシナリオに対応しています。

以下は、各モデルの詳細な概要であり、それらの事前学習済み重みへのリンク、サポートされるタスク、さまざまな動作モードとの互換性が示されています。

| モデルの種類     | 事前学習済みの重み                                                                                     | サポートされるタスク                 | 推論 | 検証 | トレーニング | エクスポート |
|------------|-----------------------------------------------------------------------------------------------|----------------------------|----|----|--------|--------|
| YOLO-NAS-s | [yolo_nas_s.pt](https://github.com/ultralytics/assets/releases/download/v8.1.0/yolo_nas_s.pt) | [物体検出](../tasks/detect.md) | ✅  | ✅  | ❌      | ✅      |
| YOLO-NAS-m | [yolo_nas_m.pt](https://github.com/ultralytics/assets/releases/download/v8.1.0/yolo_nas_m.pt) | [物体検出](../tasks/detect.md) | ✅  | ✅  | ❌      | ✅      |
| YOLO-NAS-l | [yolo_nas_l.pt](https://github.com/ultralytics/assets/releases/download/v8.1.0/yolo_nas_l.pt) | [物体検出](../tasks/detect.md) | ✅  | ✅  | ❌      | ✅      |

## 引用と謝辞

研究や開発の中でYOLO-NASを使用する場合は、SuperGradientsを引用してください:

!!! Quote ""

    === "BibTeX"

        ```bibtex
        @misc{supergradients,
              doi = {10.5281/ZENODO.7789328},
              url = {https://zenodo.org/record/7789328},
              author = {Aharon,  Shay and {Louis-Dupont} and {Ofri Masad} and Yurkova,  Kate and {Lotem Fridman} and {Lkdci} and Khvedchenya,  Eugene and Rubin,  Ran and Bagrov,  Natan and Tymchenko,  Borys and Keren,  Tomer and Zhilko,  Alexander and {Eran-Deci}},
              title = {Super-Gradients},
              publisher = {GitHub},
              journal = {GitHub repository},
              year = {2021},
        }
        ```

このコンピュータビジョンコミュニティ向けの貴重なリソースを作成および維持するために、Deci AIの[SuperGradients](https://github.com/Deci-AI/super-gradients/)チームに感謝の意を表します。革新的なアーキテクチャと優れた物体検出能力を持つYOLO-NASが、開発者や研究者の重要なツールになると信じています。

*keywords: YOLO-NAS, Deci AI, 物体検出, 深層学習, ニューラルアーキテクチャ検索, Ultralytics Python API, YOLOモデル, SuperGradients, 事前学習モデル, 量子化フレンドリーな基本ブロック, 高度なトレーニングスキーム, ポストトレーニング量子化, AutoNAC最適化, COCO, Objects365, Roboflow 100*
