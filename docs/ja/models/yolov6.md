---
comments: true
description: 最先端の速度と精度のバランスを実現する、Meituan YOLOv6というオブジェクト検出モデルを紹介します。機能、事前学習済みモデル、Pythonの使用方法について深く掘り下げます。
keywords: Meituan YOLOv6、オブジェクト検出、Ultralytics、YOLOv6ドキュメント、Bi-directional Concatenation、Anchor-Aided Training、事前学習済みモデル、リアルタイムアプリケーション
---

# Meituan YOLOv6

## 概要

[Meituan](https://about.meituan.com/) YOLOv6は、速度と精度のバランスに優れた最先端のオブジェクト検出器であり、リアルタイムアプリケーションにおいては人気のある選択肢となっています。このモデルは、Bi-directional Concatenation（BiC）モジュール、アンカー支援トレーニング（AAT）戦略の実装、およびCOCOデータセットにおける最先端の精度を実現するための改良されたバックボーンとネックの設計など、アーキテクチャとトレーニング方法にいくつかの注目すべき技術的改善をもたらしました。

![Meituan YOLOv6](https://user-images.githubusercontent.com/26833433/240750495-4da954ce-8b3b-41c4-8afd-ddb74361d3c2.png)
![モデルの例](https://user-images.githubusercontent.com/26833433/240750557-3e9ec4f0-0598-49a8-83ea-f33c91eb6d68.png)
**YOLOv6の概要。** モデルのアーキテクチャ図は、重要な改善点として再設計されたネットワークコンポーネントとトレーニング戦略を示しており、著しいパフォーマンス向上につながっています。 (a) YOLOv6のネック（NおよびSが表示されています）。M/Lの場合、RepBlocksはCSPStackRepで置き換えられます。 (b) BiCモジュールの構造。 (c) SimCSPSPPFブロック。 ([出典](https://arxiv.org/pdf/2301.05586.pdf))。

### 主な特徴

- **Bi-directional Concatenation（BiC）モジュール：** YOLOv6は、検出器のネックにBiCモジュールを導入し、ローカリゼーション信号を強化して性能を向上させ、速度の低下をほとんど無視できる優れた結果を実現します。
- **アンカー支援トレーニング（AAT）戦略：** このモデルでは、AATを提案して、アンカーベースとアンカーフリーのパラダイムの両方の利点を享受することができます。これにより、推論効率を損なうことなく性能を向上させることができます。
- **改良されたバックボーンとネックの設計：** YOLOv6をバックボーンとネックに別のステージを追加することで、このモデルはCOCOデータセットでの最先端の性能を高解像度入力で実現しています。
- **自己蒸留戦略：** YOLOv6の小型モデルの性能を向上させるために、新しい自己蒸留戦略が実装されており、トレーニング中に補助回帰ブランチを強化し、推論時にはそれを除去して顕著な速度低下を回避します。

## パフォーマンスメトリクス

YOLOv6にはさまざまなスケールの事前学習済みモデルが提供されています。

- YOLOv6-N: NVIDIA Tesla T4 GPUで、COCO val2017において37.5%のAPを1187 FPSで達成。
- YOLOv6-S: 484 FPSで45.0%のAP。
- YOLOv6-M: 226 FPSで50.0%のAP。
- YOLOv6-L: 116 FPSで52.8%のAP。
- YOLOv6-L6: リアルタイムでの最先端の精度。

YOLOv6には、異なる精度に最適化されたクォンタイズ済みのモデルや、モバイルプラットフォーム向けに最適化されたモデルも提供されています。

## 使用例

この例では、YOLOv6のトレーニングおよび推論の簡単な使用例を示します。これらおよび他の[モード](../modes/index.md)の完全なドキュメントについては、[Predict](../modes/predict.md)、[Train](../modes/train.md)、[Val](../modes/val.md)、[Export](../modes/export.md)のドキュメントページを参照してください。

!!! Example "例"

    === "Python"

        PyTorchで事前学習済みの`*.pt`モデルと`*.yaml`設定ファイルを`YOLO()`クラスに渡すことで、Pythonでモデルインスタンスを作成することができます。

        ```python
        from ultralytics import YOLO

        # YOLOv6nモデルをゼロから構築する
        model = YOLO('yolov6n.yaml')

        # モデルの情報を表示する（オプション）
        model.info()

        # COCO8の例題データセットでモデルを100エポックトレーニングする
        results = model.train(data='coco8.yaml', epochs=100, imgsz=640)

        # YOLOv6nモデルで'bus.jpg'画像に対して推論を実行する
        results = model('path/to/bus.jpg')
        ```

    === "CLI"

        モデルを直接実行するためのCLIコマンドも利用できます。

        ```bash
        # ゼロからYOLOv6nモデルを構築し、COCO8の例題データセットで100エポックトレーニングする
        yolo train model=yolov6n.yaml data=coco8.yaml epochs=100 imgsz=640

        # ゼロからYOLOv6nモデルを構築し、'bus.jpg'画像に対して推論を実行する
        yolo predict model=yolov6n.yaml source=path/to/bus.jpg
        ```

## サポートされるタスクとモード

YOLOv6シリーズは、高性能の[オブジェクト検出](../tasks/detect.md)に最適化されたモデルを提供しています。これらのモデルは、さまざまな計算ニーズと精度要件に対応しており、幅広いアプリケーションに適応することができます。

| モデルタイプ    | 事前学習済みの重み      | サポートされるタスク                     | 推論 | 検証 | トレーニング | エクスポート |
|-----------|----------------|--------------------------------|----|----|--------|--------|
| YOLOv6-N  | `yolov6-n.pt`  | [オブジェクト検出](../tasks/detect.md) | ✅  | ✅  | ✅      | ✅      |
| YOLOv6-S  | `yolov6-s.pt`  | [オブジェクト検出](../tasks/detect.md) | ✅  | ✅  | ✅      | ✅      |
| YOLOv6-M  | `yolov6-m.pt`  | [オブジェクト検出](../tasks/detect.md) | ✅  | ✅  | ✅      | ✅      |
| YOLOv6-L  | `yolov6-l.pt`  | [オブジェクト検出](../tasks/detect.md) | ✅  | ✅  | ✅      | ✅      |
| YOLOv6-L6 | `yolov6-l6.pt` | [オブジェクト検出](../tasks/detect.md) | ✅  | ✅  | ✅      | ✅      |

この表は、YOLOv6モデルのバリアントについての詳細な概要を提供し、オブジェクト検出のタスクにおける機能と、[推論](../modes/predict.md)、[検証](../modes/val.md)、[トレーニング](../modes/train.md)、[エクスポート](../modes/export.md)などのさまざまな操作モードとの互換性を強調しています。この包括的なサポートにより、ユーザーはさまざまなオブジェクト検出シナリオでYOLOv6モデルの機能を十分に活用することができます。

## 引用と謝辞

リアルタイムオブジェクト検出の分野における重要な貢献をした著者に謝意を表します。

!!! Quote ""

    === "BibTeX"

        ```bibtex
        @misc{li2023yolov6,
              title={YOLOv6 v3.0: A Full-Scale Reloading},
              author={Chuyi Li and Lulu Li and Yifei Geng and Hongliang Jiang and Meng Cheng and Bo Zhang and Zaidan Ke and Xiaoming Xu and Xiangxiang Chu},
              year={2023},
              eprint={2301.05586},
              archivePrefix={arXiv},
              primaryClass={cs.CV}
        }
        ```

YOLOv6のオリジナル論文は[arXiv](https://arxiv.org/abs/2301.05586)で入手できます。著者は自身の研究を広く共有しており、コードベースは[GitHub](https://github.com/meituan/YOLOv6)でアクセスできます。私たちは彼らがこの分野の進歩に貢献し、その研究を広く公開していることに感謝しています。
