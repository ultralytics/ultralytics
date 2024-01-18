---
comments: true
description: YOLOv8予測モードの使用方法について学び、画像、動画、データフォーマットなどさまざまな推論ソースについて解説します。
keywords: Ultralytics, YOLOv8, 予測モード, 推論ソース, 予測タスク, ストリーミングモード, 画像処理, 動画処理, 機械学習, AI
---

# Ultralytics YOLOによるモデル予測

<img width="1024" src="https://github.com/ultralytics/assets/raw/main/yolov8/banner-integrations.png" alt="Ultralytics YOLO ecosystem and integrations">

## イントロダクション

機械学習やコンピュータビジョンの世界では、視覚データから意味を引き出すプロセスを「推論」または「予測」と呼ばれています。UltralyticsのYOLOv8は、幅広いデータソースに対して高性能でリアルタイムな推論を行うために特化した、「予測モード」と呼ばれる強力な機能を提供しています。

<p align="center">
  <br>
  <iframe width="720" height="405" src="https://www.youtube.com/embed/QtsI0TnwDZs?si=ljesw75cMO2Eas14"
    title="YouTube video player" frameborder="0"
    allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share"
    allowfullscreen>
  </iframe>
  <br>
  <strong>視聴:</strong> Ultralytics YOLOv8モデルの出力をカスタムプロジェクトに取り込む方法を学ぶ。
</p>

## 実際の応用例

|                                                                 製造業                                                                 |                                                              スポーツ                                                               |                                                             安全                                                              |
|:-----------------------------------------------------------------------------------------------------------------------------------:|:-------------------------------------------------------------------------------------------------------------------------------:|:---------------------------------------------------------------------------------------------------------------------------:|
| ![Vehicle Spare Parts Detection](https://github.com/RizwanMunawar/ultralytics/assets/62513924/a0f802a8-0776-44cf-8f17-93974a4a28a1) | ![Football Player Detection](https://github.com/RizwanMunawar/ultralytics/assets/62513924/7d320e1f-fc57-4d7f-a691-78ee579c3442) | ![People Fall Detection](https://github.com/RizwanMunawar/ultralytics/assets/62513924/86437c4a-3227-4eee-90ef-9efb697bdb43) |
|                                                             車両のスペアパーツ検出                                                             |                                                           フットボール選手検出                                                            |                                                           人の転倒検出                                                            |

## 予測にUltralytics YOLOを使う理由

様々な推論ニーズにYOLOv8の予測モードを検討すべき理由です：

- **柔軟性:** 画像、動画、さらにはライブストリームにおいて推論を行う能力があります。
- **パフォーマンス:** 正確さを犠牲にすることなく、リアルタイムで高速な処理が行えるように設計されています。
- **使いやすさ:** 迅速な展開とテストのための直感的なPythonおよびCLIインターフェース。
- **高いカスタマイズ性:** 特定の要件に応じてモデルの推論動作を調整するためのさまざまな設定とパラメーター。

### 予測モードの主な特徴

YOLOv8の予測モードは、頑健で多様性があり、次の特徴を備えています：

- **複数のデータソースとの互換性:** データが個々の画像、画像の集合、動画ファイル、またはリアルタイムの動画ストリームのいずれの形式であっても、予測モードが対応しています。
- **ストリーミングモード:** `Results`オブジェクトのメモリ効率の良いジェネレータを生成するためにストリーミング機能を使用します。`stream=True`を予測器の呼び出しメソッドに設定することにより有効になります。
- **バッチ処理:** 単一のバッチで複数の画像や動画フレームを処理する能力は、さらに推論時間を短縮します。
- **統合が容易:** 柔軟なAPIのおかげで、既存のデータパイプラインや他のソフトウェアコンポーネントに簡単に統合できます。

UltralyticsのYOLOモデルは、`stream=True`が推論中にモデルに渡されると、Pythonの`Results`オブジェクトのリストまたは`Results`オブジェクトのメモリ効率の良いPythonジェネレータのいずれかを返します：

!!! Example "予測"

    === "`stream=False`でリストを返す"
        ```python
        from ultralytics import YOLO

        # モデルをロード
        model = YOLO('yolov8n.pt')  # 事前にトレーニングされたYOLOv8nモデル

        # 画像のリストに対してバッチ推論を実行
        results = model(['im1.jpg', 'im2.jpg'])  # Resultsオブジェクトのリストを返す

        # 結果リストを処理
        for result in results:
            boxes = result.boxes  # バウンディングボックス出力用のBoxesオブジェクト
            masks = result.masks  # セグメンテーションマスク出力用のMasksオブジェクト
            keypoints = result.keypoints  # 姿勢出力用のKeypointsオブジェクト
            probs = result.probs  # 分類出力用のProbsオブジェクト
        ```

    === "`stream=True`でジェネレータを返す"
        ```python
        from ultralytics import YOLO

        # モデルをロード
        model = YOLO('yolov8n.pt')  # 事前にトレーニングされたYOLOv8nモデル

        # 画像のリストに対してバッチ推論を実行
        results = model(['im1.jpg', 'im2.jpg'], stream=True)  # Resultsオブジェクトのジェネレータを返す

        # 結果ジェネレータを処理
        for result in results:
            boxes = result.boxes  # バウンディングボックス出力用のBoxesオブジェクト
            masks = result.masks  # セグメンテーションマスク出力用のMasksオブジェクト
            keypoints = result.keypoints  # 姿勢出力用のKeypointsオブジェクト
            probs = result.probs  # 分類出力用のProbsオブジェクト
        ```

## 推論ソース

YOLOv8は、以下の表に示されるように、異なるタイプの入力ソースを推論に処理できます。ソースには静止画像、動画ストリーム、およびさまざまなデータフォーマットが含まれます。表には、各ソースがストリーミングモードで使用できるかどうかも示されており、引数`stream=True`で✅が表示されています。ストリーミングモードは、動画やライブストリームを処理する場合に有利であり、すべてのフレームをメモリにロードする代わりに結果のジェネレータを作成します。

!!! Tip "ヒント"

    長い動画や大きなデータセットを効率的にメモリ管理するために`stream=True`を使用します。`stream=False`では、すべてのフレームまたはデータポイントの結果がメモリに格納されますが、大きな入力で迅速にメモリが積み上がり、メモリ不足のエラーを引き起こす可能性があります。対照的に、`stream=True`はジェネレータを利用し、現在のフレームまたはデータポイントの結果のみをメモリに保持し、メモリ消費を大幅に削減し、メモリ不足の問題を防ぎます。

| ソース        | 引数                                         | タイプ              | 備考                                                               |
|------------|--------------------------------------------|------------------|------------------------------------------------------------------|
| 画像         | `'image.jpg'`                              | `str` または `Path` | 単一の画像ファイル。                                                       |
| URL        | `'https://ultralytics.com/images/bus.jpg'` | `str`            | 画像へのURL。                                                         |
| スクリーンショット  | `'screen'`                                 | `str`            | スクリーンショットをキャプチャ。                                                 |
| PIL        | `Image.open('im.jpg')`                     | `PIL.Image`      | HWCフォーマットでRGBチャンネル。                                              |
| OpenCV     | `cv2.imread('im.jpg')`                     | `np.ndarray`     | HWCフォーマットでBGRチャンネル `uint8 (0-255)`。                              |
| numpy      | `np.zeros((640,1280,3))`                   | `np.ndarray`     | HWCフォーマットでBGRチャンネル `uint8 (0-255)`。                              |
| torch      | `torch.zeros(16,3,320,640)`                | `torch.Tensor`   | BCHWフォーマットでRGBチャンネル `float32 (0.0-1.0)`。                         |
| CSV        | `'sources.csv'`                            | `str` または `Path` | 画像、動画、ディレクトリへのパスを含むCSVファイル。                                      |
| 動画 ✅       | `'video.mp4'`                              | `str` または `Path` | MP4、AVIなどの形式の動画ファイル。                                             |
| ディレクトリ ✅   | `'path/'`                                  | `str` または `Path` | 画像または動画を含むディレクトリへのパス。                                            |
| グロブ ✅      | `'path/*.jpg'`                             | `str`            | 複数のファイルに一致するグロブパターン。ワイルドカードとして`*`文字を使用します。                       |
| YouTube ✅  | `'https://youtu.be/LNwODJXcvt4'`           | `str`            | YouTube動画のURL。                                                   |
| ストリーム ✅    | `'rtsp://example.com/media.mp4'`           | `str`            | RTSP、RTMP、TCP、IPアドレスなどのストリーミングプロトコルのためのURL。                      |
| マルチストリーム ✅ | `'list.streams'`                           | `str` または `Path` | ストリームURLを行ごとに1つ含む`*.streams`テキストファイル。つまり、8つのストリームをバッチサイズ8で実行します。 |

以下は、それぞれのソースタイプを使用するためのコード例です：

!!! Example "予測ソース"

    === "画像"
        画像ファイルに推論を実行します。
        ```python
        from ultralytics import YOLO

        # 事前にトレーニングされたYOLOv8nモデルをロード
        model = YOLO('yolov8n.pt')

        # 画像ファイルへのパスを定義
        source = 'path/to/image.jpg'

        # ソースに推論を実行
        results = model(source)  # Resultsオブジェクトのリスト
        ```

    === "スクリーンショット"
        現在の画面内容のスクリーンショットに推論を実行します。
        ```python
        from ultralytics import YOLO

        # 事前にトレーニングされたYOLOv8nモデルをロード
        model = YOLO('yolov8n.pt')

        # 現在のスクリーンショットをソースとして定義
        source = 'screen'

        # ソースに推論を実行
        results = model(source)  # Resultsオブジェクトのリスト
        ```

    === "URL"
        リモートのURL経由でホストされている画像や動画に推論を実行します。
        ```python
        from ultralytics import YOLO

        # 事前にトレーニングされたYOLOv8nモデルをロード
        model = YOLO('yolov8n.pt')

        # リモート画像や動画のURLを定義
        source = 'https://ultralytics.com/images/bus.jpg'

        # ソースに推論を実行
        results = model(source)  # Resultsオブジェクトのリスト
        ```

    === "PIL"
        Python Imaging Library (PIL)を使用して開いた画像に推論を実行します。
        ```python
        from PIL import Image
        from ultralytics import YOLO

        # 事前にトレーニングされたYOLOv8nモデルをロード
        model = YOLO('yolov8n.pt')

        # PILを使用して画像を開く
        source = Image.open('path/to/image.jpg')

        # ソースに推論を実行
        results = model(source)  # Resultsオブジェクトのリスト
        ```

    === "OpenCV"
        OpenCVを使用して読み込んだ画像に推論を実行します。
        ```python
        import cv2
        from ultralytics import YOLO

        # 事前にトレーニングされたYOLOv8nモデルをロード
        model = YOLO('yolov8n.pt')

        # OpenCVを使用して画像を読み込む
        source = cv2.imread('path/to/image.jpg')

        # ソースに推論を実行
        results = model(source)  # Resultsオブジェクトのリスト
        ```

    === "numpy"
        numpy配列として表される画像に推論を実行します。
        ```python
        import numpy as np
        from ultralytics import YOLO

        # 事前にトレーニングされたYOLOv8nモデルをロード
        model = YOLO('yolov8n.pt')

        # HWC形状（640, 640, 3）、範囲[0, 255]、型`uint8`のランダムなnumpy配列を作成
        source = np.random.randint(low=0, high=255, size=(640,640,3), dtype='uint8')

        # ソースに推論を実行
        results = model(source)  # Resultsオブジェクトのリスト
        ```
