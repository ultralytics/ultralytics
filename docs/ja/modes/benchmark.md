---
comments: true
description: YOLOv8を様々なエクスポート形式でスピードと精度をプロファイリングする方法を学び、mAP50-95、accuracy_top5のメトリクスなどの洞察を得る。
keywords: Ultralytics, YOLOv8, ベンチマーク, スピードプロファイリング, 精度プロファイリング, mAP50-95, accuracy_top5, ONNX, OpenVINO, TensorRT, YOLOエクスポート形式
---

# Ultralytics YOLO でのモデルベンチマーク

<img width="1024" src="https://github.com/ultralytics/assets/raw/main/yolov8/banner-integrations.png" alt="Ultralytics YOLOエコシステムと統合">

## はじめに

モデルがトレーニングされ、検証された後、次の論理的なステップは、さまざまな現実世界のシナリオでのパフォーマンスを評価することです。Ultralytics YOLOv8 のベンチマークモードは、さまざまなエクスポート形式でモデルのスピードと精度を評価するための頑健なフレームワークを提供します。

## ベンチマークが重要な理由は？

- **情報に基づいた決定:** スピードと精度のトレードオフについての洞察を得る。
- **リソース割り当て:** 異なるハードウェアで異なるエクスポート形式がどのように動作するかを理解する。
- **最適化:** 特定のユースケースで最高のパフォーマンスを提供するエクスポート形式を学ぶ。
- **コスト効率:** ベンチマーク結果に基づいてハードウェアリソースをより効率的に使用する。

### ベンチマークモードでの主要なメトリクス

- **mAP50-95:** 物体検出、セグメンテーション、ポーズ推定に使用。
- **accuracy_top5:** 画像分類に使用。
- **推論時間:** 各画像に要する時間（ミリ秒）。

### サポートされるエクスポート形式

- **ONNX:** 最適なCPUパフォーマンスのために
- **TensorRT:** 最大限のGPU効率のために
- **OpenVINO:** Intelハードウェアの最適化のために
- **CoreML, TensorFlow SavedModel など:** 多様なデプロイメントニーズに。

!!! Tip "ヒント"

    * CPUスピードアップのためにONNXまたはOpenVINOにエクスポートする。
    * GPUスピードアップのためにTensorRTにエクスポートする。

## 使用例

ONNX、TensorRTなど、すべてのサポートされるエクスポート形式でYOLOv8nベンチマークを実行します。完全なエクスポート引数のリストについては、以下のArgumentsセクションを参照してください。

!!! Example "例"

    === "Python"

        ```python
        from ultralytics.utils.benchmarks import benchmark

        # GPUでベンチマーク
        benchmark(model='yolov8n.pt', data='coco8.yaml', imgsz=640, half=False, device=0)
        ```
    === "CLI"

        ```bash
        yolo benchmark model=yolov8n.pt data='coco8.yaml' imgsz=640 half=False device=0
        ```

## 引数

`model`、`data`、`imgsz`、`half`、`device`、`verbose` などの引数は、特定のニーズに合わせてベンチマークを微調整し、さまざまなエクスポート形式のパフォーマンスを容易に比較するためにユーザーに柔軟性を提供します。

| キー        | 値       | 説明                                                        |
|-----------|---------|-----------------------------------------------------------|
| `model`   | `None`  | モデルファイルへのパス、例: yolov8n.pt, yolov8n.yaml                   |
| `data`    | `None`  | ベンチマークデータセットを参照するYAMLへのパス（`val`ラベルの下）                     |
| `imgsz`   | `640`   | 画像サイズをスカラーまたは(h, w)リストで、例: (640, 480)                     |
| `half`    | `False` | FP16量子化                                                   |
| `int8`    | `False` | INT8量子化                                                   |
| `device`  | `None`  | 実行デバイス、例: cuda device=0 または device=0,1,2,3 または device=cpu |
| `verbose` | `False` | エラー時に続行しない（bool）、またはval床しきい値（float）                       |

## エクスポート形式

以下の可能なすべてのエクスポート形式で自動的にベンチマークを試みます。

| 形式                                                                 | `format` 引数   | モデル                       | メタデータ | 引数                                                  |
|--------------------------------------------------------------------|---------------|---------------------------|-------|-----------------------------------------------------|
| [PyTorch](https://pytorch.org/)                                    | -             | `yolov8n.pt`              | ✅     | -                                                   |
| [TorchScript](https://pytorch.org/docs/stable/jit.html)            | `torchscript` | `yolov8n.torchscript`     | ✅     | `imgsz`, `optimize`                                 |
| [ONNX](https://onnx.ai/)                                           | `onnx`        | `yolov8n.onnx`            | ✅     | `imgsz`, `half`, `dynamic`, `simplify`, `opset`     |
| [OpenVINO](https://docs.openvino.ai/latest/index.html)             | `openvino`    | `yolov8n_openvino_model/` | ✅     | `imgsz`, `half`                                     |
| [TensorRT](https://developer.nvidia.com/tensorrt)                  | `engine`      | `yolov8n.engine`          | ✅     | `imgsz`, `half`, `dynamic`, `simplify`, `workspace` |
| [CoreML](https://github.com/apple/coremltools)                     | `coreml`      | `yolov8n.mlpackage`       | ✅     | `imgsz`, `half`, `int8`, `nms`                      |
| [TF SavedModel](https://www.tensorflow.org/guide/saved_model)      | `saved_model` | `yolov8n_saved_model/`    | ✅     | `imgsz`, `keras`                                    |
| [TF GraphDef](https://www.tensorflow.org/api_docs/python/tf/Graph) | `pb`          | `yolov8n.pb`              | ❌     | `imgsz`                                             |
| [TF Lite](https://www.tensorflow.org/lite)                         | `tflite`      | `yolov8n.tflite`          | ✅     | `imgsz`, `half`, `int8`                             |
| [TF Edge TPU](https://coral.ai/docs/edgetpu/models-intro/)         | `edgetpu`     | `yolov8n_edgetpu.tflite`  | ✅     | `imgsz`                                             |
| [TF.js](https://www.tensorflow.org/js)                             | `tfjs`        | `yolov8n_web_model/`      | ✅     | `imgsz`                                             |
| [PaddlePaddle](https://github.com/PaddlePaddle)                    | `paddle`      | `yolov8n_paddle_model/`   | ✅     | `imgsz`                                             |
| [ncnn](https://github.com/Tencent/ncnn)                            | `ncnn`        | `yolov8n_ncnn_model/`     | ✅     | `imgsz`, `half`                                     |

[エクスポート](https://docs.ultralytics.com/modes/export/)ページでさらに詳しい`export`の詳細をご覧ください。
