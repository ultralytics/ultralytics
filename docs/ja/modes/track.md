---
comments: true
description: Ultralytics YOLOを使用したビデオストリームでのオブジェクトトラッキングの使用方法を学びます。異なるトラッカーの使用ガイドとトラッカー構成のカスタマイズについて。
keywords: Ultralytics, YOLO, オブジェクトトラッキング, ビデオストリーム, BoT-SORT, ByteTrack, Pythonガイド, CLIガイド
---

# Ultralytics YOLOによる複数オブジェクトのトラッキング

<img width="1024" src="https://user-images.githubusercontent.com/26833433/243418637-1d6250fd-1515-4c10-a844-a32818ae6d46.png" alt="複数オブジェクトのトラッキング例">

ビデオ分析の領域でのオブジェクトトラッキングは、フレーム内のオブジェクトの位置とクラスを特定するだけでなく、ビデオが進行するにつれてそれぞれの検出されたオブジェクトにユニークなIDを維持する重要なタスクです。その応用範囲は無限で、監視やセキュリティからリアルタイムスポーツ分析まで及びます。

## オブジェクトトラッキングにUltralytics YOLOを選ぶ理由は？

Ultralyticsのトラッカーからの出力は標準のオブジェクト検出と一致しており、さらにオブジェクトIDの付加価値があります。これにより、ビデオストリーム内のオブジェクトを追跡し、後続の分析を行うことが容易になります。Ultralytics YOLOをオブジェクトトラッキングのニーズに利用を検討する理由は以下の通りです：

- **効率性:** 精度を損なうことなくリアルタイムでビデオストリームを処理します。
- **柔軟性:** 複数のトラッキングアルゴリズムと構成をサポートしています。
- **使いやすさ:** 簡単なPython APIとCLIオプションで迅速な統合と展開が可能です。
- **カスタマイズ性:** カスタムトレーニング済みのYOLOモデルとの容易な使用により、ドメイン特有のアプリケーションへの統合が可能です。

<p align="center">
  <br>
  <iframe width="720" height="405" src="https://www.youtube.com/embed/hHyHmOtmEgs?si=VNZtXmm45Nb9s-N-"
    title="YouTube video player" frameborder="0"
    allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share"
    allowfullscreen>
  </iframe>
  <br>
  <strong>視聴：</strong> Ultralytics YOLOv8によるオブジェクト検出とトラッキング。
</p>

## 実世界での応用例

|                                                       交通                                                        |                                                       小売                                                        |                                                      水産業                                                       |
|:---------------------------------------------------------------------------------------------------------------:|:---------------------------------------------------------------------------------------------------------------:|:--------------------------------------------------------------------------------------------------------------:|
| ![車両のトラッキング](https://github.com/RizwanMunawar/ultralytics/assets/62513924/ee6e6038-383b-4f21-ac29-b2a1c7d386ab) | ![人々のトラッキング](https://github.com/RizwanMunawar/ultralytics/assets/62513924/93bb4ee2-77a0-4e4e-8eb6-eb8f527f0527) | ![魚のトラッキング](https://github.com/RizwanMunawar/ultralytics/assets/62513924/a5146d0f-bfa8-4e0a-b7df-3c1446cd8142) |
|                                                    車両トラッキング                                                     |                                                    人々のトラッキング                                                    |                                                    魚のトラッキング                                                    |

## 一目でわかる機能

Ultralytics YOLOは、オブジェクト検出機能を拡張して、堅牢で多機能なオブジェクトトラッキングを提供します：

- **リアルタイムトラッキング：** 高フレームレートのビデオでオブジェクトをシームレスに追跡します。
- **複数トラッカーサポート：** 確立されたトラッキングアルゴリズムから選択できます。
- **カスタマイズ可能なトラッカー構成：** 様々なパラメーターを調整することで特定の要件に合わせてトラッキングアルゴリズムを調整します。

## 利用可能なトラッカー

Ultralytics YOLOは、次のトラッキングアルゴリズムをサポートしています。それらは、関連するYAML構成ファイル（たとえば`tracker=tracker_type.yaml`）を渡すことで有効にすることができます：

* [BoT-SORT](https://github.com/NirAharon/BoT-SORT) - このトラッカーを有効にするには`botsort.yaml`を使用します。
* [ByteTrack](https://github.com/ifzhang/ByteTrack) - このトラッカーを有効にするには`bytetrack.yaml`を使用します。

デフォルトのトラッカーはBoT-SORTです。

## トラッキング

ビデオストリームでトラッカーを実行するには、YOLOv8n、YOLOv8n-seg、YOLOv8n-poseなどのトレーニング済みのDetect、Segment、またはPoseモデルを使用します。

!!! Example "例"

    === "Python"

        ```python
        from ultralytics import YOLO

        # 公式またはカスタムモデルをロード
        model = YOLO('yolov8n.pt')  # 公式のDetectモデルをロード
        model = YOLO('yolov8n-seg.pt')  # 公式のSegmentモデルをロード
        model = YOLO('yolov8n-pose.pt')  # 公式のPoseモデルをロード
        model = YOLO('path/to/best.pt')  # カスタムトレーニング済みモデルをロード

        # モデルでトラッキングを実行
        results = model.track(source="https://youtu.be/LNwODJXcvt4", show=True)  # デフォルトトラッカーでトラッキング
        results = model.track(source="https://youtu.be/LNwODJXcvt4", show=True, tracker="bytetrack.yaml")  # ByteTrackトラッカーでトラッキング
        ```

    === "CLI"

        ```bash
        # コマンドラインインターフェースを使用して、様々なモデルでトラッキングを実行
        yolo track model=yolov8n.pt source="https://youtu.be/LNwODJXcvt4"  # 公式のDetectモデル
        yolo track model=yolov8n-seg.pt source="https://youtu.be/LNwODJXcvt4"  # 公式のSegmentモデル
        yolo track model=yolov8n-pose.pt source="https://youtu.be/LNwODJXcvt4"  # 公式のPoseモデル
        yolo track model=path/to/best.pt source="https://youtu.be/LNwODJXcvt4"  # カスタムトレーニング済みモデル

        # ByteTrackトラッカーを使用してトラッキング
        yolo track model=path/to/best.pt tracker="bytetrack.yaml"
        ```

上記の使用法に示されているように、トラッキングはビデオやストリーミングソースで実行されるすべてのDetect、Segment、およびPoseモデルで利用可能です。

## 構成

### トラッキング引数

トラッキング構成は、`conf`、`iou`、および`show`などのPredictモードと同じプロパティを共有します。さらなる構成については、[Predict](https://docs.ultralytics.com/modes/predict/)モデルページを参照してください。

!!! Example "例"

    === "Python"

        ```python
        from ultralytics import YOLO

        # トラッキングパラメーターを構成し、トラッカーを実行
        model = YOLO('yolov8n.pt')
        results = model.track(source="https://youtu.be/LNwODJXcvt4", conf=0.3, iou=0.5, show=True)
        ```

    === "CLI"

        ```bash
        # コマンドラインインターフェースを使用してトラッキングパラメータを構成し、トラッカーを実行
        yolo track model=yolov8n.pt source="https://youtu.be/LNwODJXcvt4" conf=0.3, iou=0.5 show
        ```

### トラッカーの選択

Ultralyticsは、変更されたトラッカー構成ファイルの使用も可能にします。これを行うには、[ultralytics/cfg/trackers](https://github.com/ultralytics/ultralytics/tree/main/ultralytics/cfg/trackers)からトラッカー構成ファイル（たとえば`custom_tracker.yaml`）のコピーを作成し、必要に応じて任意の構成（`tracker_type`を除く）を変更します。

!!! Example "例"

    === "Python"

        ```python
        from ultralytics import YOLO

        # モデルをロードし、カスタム構成ファイルでトラッカーを実行
        model = YOLO('yolov8n.pt')
        results = model.track(source="https://youtu.be/LNwODJXcvt4", tracker='custom_tracker.yaml')
        ```

    === "CLI"

        ```bash
        # コマンドラインインターフェースを使用して、カスタム構成ファイルでモデルをロードし、トラッカーを実行
        yolo track model=yolov8n.pt source="https://youtu.be/LNwODJXcvt4" tracker='custom_tracker.yaml'
        ```

トラッキング引数の包括的なリストについては、[ultralytics/cfg/trackers](https://github.com/ultralytics/ultralytics/tree/main/ultralytics/cfg/trackers)ページを参照してください。

## Pythonの例

### トラックループの永続化

次は、OpenCV (`cv2`)とYOLOv8を使用してビデオフレームでオブジェクトトラッキングを実行するPythonスクリプトです。このスクリプトでは、必要なパッケージ（`opencv-python`および`ultralytics`）が既にインストールされていることが前提です。`persist=True`引数は、トラッカーに現在の画像またはフレームがシーケンスの次のものであり、現在の画像に前の画像からのトラックを期待することを伝えます。

!!! Example "トラッキングを伴うストリーミングforループ"

    ```python
    import cv2
    from ultralytics import YOLO

    # YOLOv8モデルをロード
    model = YOLO('yolov8n.pt')

    # ビデオファイルを開く
    video_path = "path/to/video.mp4"
    cap = cv2.VideoCapture(video_path)

    # ビデオフレームをループする
    while cap.isOpened():
        # ビデオからフレームを読み込む
        success, frame = cap.read()

        if success:
            # フレームでYOLOv8トラッキングを実行し、フレーム間でトラックを永続化
            results = model.track(frame, persist=True)

            # フレームに結果を可視化
            annotated_frame = results[0].plot()

            # 注釈付きのフレームを表示
            cv2.imshow("YOLOv8トラッキング", annotated_frame)

            # 'q'が押されたらループから抜ける
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        else:
            # ビデオの終わりに到達したらループから抜ける
            break

    # ビデオキャプチャオブジェクトを解放し、表示ウィンドウを閉じる
    cap.release()
    cv2.destroyAllWindows()
    ```

ここでの変更は、単純な検出ではなくオブジェクトトラッキングを有効にする`model(frame)`から`model.track(frame)`への変更です。この変更されたスクリプトは、ビデオの各フレームでトラッカーを実行し、結果を視覚化し、ウィンドウに表示します。ループは'q'を押すことで終了できます。

## 新しいトラッカーの貢献

あなたはマルチオブジェクトトラッキングに精通しており、Ultralytics YOLOでトラッキングアルゴリズムをうまく実装または適応させたことがありますか？[ultralytics/cfg/trackers](https://github.com/ultralytics/ultralytics/tree/main/ultralytics/cfg/trackers)セクションへの貢献を私たちは歓迎します！あなたの実世界での応用例とソリューションは、トラッキングタスクに取り組むユーザーにとって非常に有益かもしれません。

このセクションへの貢献により、Ultralytics YOLOフレームワーク内で利用可能なトラッキングソリューションの範囲が広がり、コミュニティにとっての機能性とユーティリティーに新たな層が加わります。

ご自身の貢献を開始するには、プルリクエスト（PR）の送信に関する総合的な指示について我々の[貢献ガイド](https://docs.ultralytics.com/help/contributing)をご参照ください 🛠️。あなたが何をもたらすか私たちは期待しています！

一緒に、Ultralytics YOLOエコシステムのトラッキング機能を高めましょう 🙏！
