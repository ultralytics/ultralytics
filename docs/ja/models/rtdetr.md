---
comments: true
description: RT-DETRは、Baiduによって開発された、高速かつ高精度なリアルタイムオブジェクト検出器です。Vision Transformers（ViT）の力を借りて、マルチスケールの特徴を効率的に処理します。RT-DETRは非常に適応性があり、再学習せずに異なるデコーダーレイヤーを使用して推論速度を柔軟に調整できます。このモデルは、TensorRTを使用したCUDAなどの高速エンドバックエンドで優れた性能を発揮し、多くの他のリアルタイムオブジェクト検出器を凌駕します。
keywords: RT-DETR, Baidu, Vision Transformers, object detection, real-time performance, CUDA, TensorRT, IoU-aware query selection, Ultralytics, Python API, PaddlePaddle
---

# BaiduのRT-DETR: Vision Transformerベースのリアルタイムオブジェクト検出器

## 概要

Baiduが開発したリアルタイム検出Transformer（RT-DETR）は、高い精度を維持しながらリアルタイム性能を提供する最先端のエンドツーエンドのオブジェクト検出器です。Vision Transformers（ViT）の力を借りて、マルチスケールの特徴を効率的に処理することにより、RT-DETRは高い適応性を持ちます。再学習せずに異なるデコーダーレイヤーを使用して推論速度を柔軟に調整できるため、このモデルはTensorRTを使用したCUDAなどの高速バックエンドで多くの他のリアルタイムオブジェクト検出器を凌駕します。

![モデルの例](https://user-images.githubusercontent.com/26833433/238963168-90e8483f-90aa-4eb6-a5e1-0d408b23dd33.png)
**BaiduのRT-DETRの概要。** RT-DETRのモデルアーキテクチャダイアグラムでは、バックボーンの最後の3つのステージ{S3、S4、S5}がエンコーダーへの入力として表示されます。効率的なハイブリッドエンコーダーは、マルチスケールの特徴をイントラスケール特徴の相互作用（AIFI）とクロススケール特徴融合モジュール（CCFM）を介して画像特徴のシーケンスに変換します。IoU-awareクエリ選択は、デコーダーの初期オブジェクトクエリとして固定数の画像特徴を選択するために使用されます。最後に、デコーダーは補助予測ヘッドとともに、オブジェクトクエリを反復最適化してボックスと信頼スコアを生成します（[出典](https://arxiv.org/pdf/2304.08069.pdf)）。

### 主な特徴

- **効率的なハイブリッドエンコーダー：** BaiduのRT-DETRは、マルチスケールの特徴をイントラスケールの相互作用とクロススケールの融合を分離することで処理する効率的なハイブリッドエンコーダーを使用しています。このユニークなVision Transformersベースの設計により、計算コストを削減し、リアルタイムオブジェクト検出を実現しています。
- **IoU-awareクエリ選択：** BaiduのRT-DETRは、IoU-awareクエリ選択を活用してオブジェクトクエリの初期化を改善します。これにより、モデルはシーン内の関連性の高いオブジェクトに焦点を当てて検出の精度を向上させることができます。
- **適応可能な推論速度：** BaiduのRT-DETRは、再学習せずに異なるデコーダーレイヤーを使用して推論速度を柔軟に調整することができます。この適応性により、さまざまなリアルタイムオブジェクト検出シナリオでの実用的な応用が容易になります。

## 事前学習済みモデル

Ultralytics Python APIは、異なるスケールの事前学習済みPaddlePaddle RT-DETRモデルを提供しています。

- RT-DETR-L：COCO val2017で53.0%のAP、T4 GPUで114 FPS
- RT-DETR-X：COCO val2017で54.8%のAP、T4 GPUで74 FPS

## 使用例

この例では、RT-DETRの訓練と推論の簡単な例を提供します。これらと他の[モード](../modes/index.md)の詳しいドキュメントについては、[Predict](../modes/predict.md)、[Train](../modes/train.md)、[Val](../modes/val.md)、および[Export](../modes/export.md)ドキュメントページを参照してください。

!!! Example "例"

    === "Python"

        ```python
        from ultralytics import RTDETR

        # COCOで事前学習済みのRT-DETR-lモデルをロードします
        model = RTDETR('rtdetr-l.pt')

        # モデル情報を表示します（オプション）
        model.info()

        # COCO8の例のデータセットでモデルを100エポックトレーニングします
        results = model.train(data='coco8.yaml', epochs=100, imgsz=640)

        # 'bus.jpg'画像でRT-DETR-lモデルで推論を実行します
        results = model('path/to/bus.jpg')
        ```

    === "CLI"

        ```bash
        # COCOで事前学習済みのRT-DETR-lモデルをロードし、COCO8の例のデータセットで100エポックトレーニングします
        yolo train model=rtdetr-l.pt data=coco8.yaml epochs=100 imgsz=640

        # COCOで事前学習済みのRT-DETR-lモデルをロードし、'bus.jpg'画像で推論を実行します
        yolo predict model=rtdetr-l.pt source=path/to/bus.jpg
        ```

## サポートされているタスクとモード

この表には、各モデルがサポートするタスク、特定の事前学習済み重み、およびサポートされるさまざまなモード（[Train](../modes/train.md)、[Val](../modes/val.md)、[Predict](../modes/predict.md)、[Export](../modes/export.md)）が✅絵文字で示されている情報が示されています。

| モデルの種類              | 事前学習済み重み      | サポートされるタスク                     | 推論 | 検証 | 訓練 | エクスポート |
|---------------------|---------------|--------------------------------|----|----|----|--------|
| RT-DETR Large       | `rtdetr-l.pt` | [オブジェクト検出](../tasks/detect.md) | ✅  | ✅  | ✅  | ✅      |
| RT-DETR Extra-Large | `rtdetr-x.pt` | [オブジェクト検出](../tasks/detect.md) | ✅  | ✅  | ✅  | ✅      |

## 引用と謝辞

研究や開発の中でBaiduのRT-DETRを使用する場合は、[元の論文](https://arxiv.org/abs/2304.08069)を引用してください。

!!! Quote ""

    === "BibTeX"

        ```bibtex
        @misc{lv2023detrs,
              title={DETRs Beat YOLOs on Real-time Object Detection},
              author={Wenyu Lv and Shangliang Xu and Yian Zhao and Guanzhong Wang and Jinman Wei and Cheng Cui and Yuning Du and Qingqing Dang and Yi Liu},
              year={2023},
              eprint={2304.08069},
              archivePrefix={arXiv},
              primaryClass={cs.CV}
        }
        ```

私たちは、Baiduと[PaddlePaddle](https://github.com/PaddlePaddle/PaddleDetection)チームに、コンピュータビジョンコミュニティ向けのこの貴重なリソースを作成しメンテナンスしていただいたことに感謝いたします。Vision Transformersベースのリアルタイムオブジェクト検出器であるRT-DETRの開発による、彼らのフィールドへの貢献は非常に評価されています。

*Keywords: RT-DETR, Transformer, ViT, Vision Transformers, Baidu RT-DETR, PaddlePaddle, Paddle Paddle RT-DETR, real-time object detection, Vision Transformers-based object detection, pre-trained PaddlePaddle RT-DETR models, Baidu's RT-DETR usage, Ultralytics Python API*
