---
comments: true
description: Offizielle Dokumentation für YOLOv8 von Ultralytics. Erfahren Sie, wie Sie Modelle trainieren, validieren, vorhersagen und in verschiedenen Formaten exportieren. Einschließlich detaillierter Leistungsstatistiken.
keywords: YOLOv8, Ultralytics, Objekterkennung, vortrainierte Modelle, Training, Validierung, Vorhersage, Modell-Export, COCO, ImageNet, PyTorch, ONNX, CoreML
---

# Objekterkennung

<img width="1024" src="https://user-images.githubusercontent.com/26833433/243418624-5785cb93-74c9-4541-9179-d5c6782d491a.png" alt="Beispiele für Objekterkennung">

Objekterkennung ist eine Aufgabe, die das Identifizieren der Position und Klasse von Objekten in einem Bild oder Videostream umfasst.

Die Ausgabe eines Objekterkenners ist eine Menge von Begrenzungsrahmen, die die Objekte im Bild umschließen, zusammen mit Klassenbezeichnungen und Vertrauenswerten für jedes Feld. Objekterkennung ist eine gute Wahl, wenn Sie Objekte von Interesse in einer Szene identifizieren müssen, aber nicht genau wissen müssen, wo das Objekt ist oder wie seine genaue Form ist.

<p align="center">
  <br>
  <iframe width="720" height="405" src="https://www.youtube.com/embed/5ku7npMrW40?si=6HQO1dDXunV8gekh"
    title="YouTube-Video-Player" frameborder="0"
    allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share"
    allowfullscreen>
  </iframe>
  <br>
  <strong>Sehen Sie:</strong> Objekterkennung mit vortrainiertem Ultralytics YOLOv8 Modell.
</p>

!!! Tip "Tipp"

    YOLOv8 Detect Modelle sind die Standard YOLOv8 Modelle, zum Beispiel `yolov8n.pt`, und sind vortrainiert auf dem [COCO](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/datasets/coco.yaml)-Datensatz.

## [Modelle](https://github.com/ultralytics/ultralytics/tree/main/ultralytics/cfg/models/v8)

Hier werden die vortrainierten YOLOv8 Detect Modelle gezeigt. Detect, Segment und Pose Modelle sind vortrainiert auf dem [COCO](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/datasets/coco.yaml)-Datensatz, während die Classify Modelle vortrainiert sind auf dem [ImageNet](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/datasets/ImageNet.yaml)-Datensatz.

[Modelle](https://github.com/ultralytics/ultralytics/tree/main/ultralytics/cfg/models) werden automatisch von der neuesten Ultralytics [Veröffentlichung](https://github.com/ultralytics/assets/releases) bei Erstbenutzung heruntergeladen.

| Modell                                                                               | Größe<br><sup>(Pixel) | mAP<sup>val<br>50-95 | Geschwindigkeit<br><sup>CPU ONNX<br>(ms) | Geschwindigkeit<br><sup>A100 TensorRT<br>(ms) | params<br><sup>(M) | FLOPs<br><sup>(B) |
|--------------------------------------------------------------------------------------|-----------------------|----------------------|------------------------------------------|-----------------------------------------------|--------------------|-------------------|
| [YOLOv8n](https://github.com/ultralytics/assets/releases/download/v8.1.0/yolov8n.pt) | 640                   | 37.3                 | 80.4                                     | 0.99                                          | 3.2                | 8.7               |
| [YOLOv8s](https://github.com/ultralytics/assets/releases/download/v8.1.0/yolov8s.pt) | 640                   | 44.9                 | 128.4                                    | 1.20                                          | 11.2               | 28.6              |
| [YOLOv8m](https://github.com/ultralytics/assets/releases/download/v8.1.0/yolov8m.pt) | 640                   | 50.2                 | 234.7                                    | 1.83                                          | 25.9               | 78.9              |
| [YOLOv8l](https://github.com/ultralytics/assets/releases/download/v8.1.0/yolov8l.pt) | 640                   | 52.9                 | 375.2                                    | 2.39                                          | 43.7               | 165.2             |
| [YOLOv8x](https://github.com/ultralytics/assets/releases/download/v8.1.0/yolov8x.pt) | 640                   | 53.9                 | 479.1                                    | 3.53                                          | 68.2               | 257.8             |

- **mAP<sup>val</sup>** Werte sind für Single-Modell Single-Scale auf dem [COCO val2017](http://cocodataset.org) Datensatz.
  <br>Reproduzieren mit `yolo val detect data=coco.yaml device=0`
- **Geschwindigkeit** gemittelt über COCO Val Bilder mit einer [Amazon EC2 P4d](https://aws.amazon.com/ec2/instance-types/p4/)-Instanz.
  <br>Reproduzieren mit `yolo val detect data=coco128.yaml batch=1 device=0|cpu`

## Training

YOLOv8n auf dem COCO128-Datensatz für 100 Epochen bei Bildgröße 640 trainieren. Für eine vollständige Liste verfügbarer Argumente siehe die [Konfigurationsseite](/../usage/cfg.md).

!!! Example "Beispiel"

    === "Python"

        ```python
        from ultralytics import YOLO

        # Modell laden
        model = YOLO('yolov8n.yaml')  # ein neues Modell aus YAML aufbauen
        model = YOLO('yolov8n.pt')  # ein vortrainiertes Modell laden (empfohlen für Training)
        model = YOLO('yolov8n.yaml').load('yolov8n.pt')  # aus YAML aufbauen und Gewichte übertragen

        # Das Modell trainieren
        results = model.train(data='coco128.yaml', epochs=100, imgsz=640)
        ```
    === "CLI"

        ```bash
        # Ein neues Modell aus YAML aufbauen und Training von Grund auf starten
        yolo detect train data=coco128.yaml model=yolov8n.yaml epochs=100 imgsz=640

        # Training von einem vortrainierten *.pt Modell starten
        yolo detect train data=coco128.yaml model=yolov8n.pt epochs=100 imgsz=640

        # Ein neues Modell aus YAML aufbauen, vortrainierte Gewichte übertragen und Training starten
        yolo detect train data=coco128.yaml model=yolov8n.yaml pretrained=yolov8n.pt epochs=100 imgsz=640
        ```

### Datenformat

Das Datenformat für YOLO-Erkennungsdatensätze finden Sie detailliert im [Dataset Guide](../../../datasets/detect/index.md). Um Ihren vorhandenen Datensatz von anderen Formaten (wie COCO etc.) in das YOLO-Format zu konvertieren, verwenden Sie bitte das [JSON2YOLO](https://github.com/ultralytics/JSON2YOLO)-Tool von Ultralytics.

## Validierung

Genauigkeit des trainierten YOLOv8n-Modells auf dem COCO128-Datensatz validieren. Es müssen keine Argumente übergeben werden, da das `modell` seine Trainingsdaten und Argumente als Modellattribute beibehält.

!!! Example "Beispiel"

    === "Python"

        ```python
        from ultralytics import YOLO

        # Modell laden
        model = YOLO('yolov8n.pt')  # ein offizielles Modell laden
        model = YOLO('pfad/zum/besten.pt')  # ein benutzerdefiniertes Modell laden

        # Das Modell validieren
        metrics = model.val()  # keine Argumente nötig, Datensatz und Einstellungen erinnert
        metrics.box.map    # map50-95
        metrics.box.map50  # map50
        metrics.box.map75  # map75
        metrics.box.maps   # eine Liste enthält map50-95 jeder Kategorie
        ```
    === "CLI"

        ```bash
        yolo detect val model=yolov8n.pt  # offizielles Modell validieren
        yolo detect val model=pfad/zum/besten.pt  # benutzerdefiniertes Modell validieren
        ```

## Vorhersage

Ein trainiertes YOLOv8n-Modell verwenden, um Vorhersagen auf Bildern durchzuführen.

!!! Example "Beispiel"

    === "Python"

        ```python
        from ultralytics import YOLO

        # Modell laden
        model = YOLO('yolov8n.pt')  # ein offizielles Modell laden
        model = YOLO('pfad/zum/besten.pt')  # ein benutzerdefiniertes Modell laden

        # Mit dem Modell vorhersagen
        results = model('https://ultralytics.com/images/bus.jpg')  # Vorhersage auf einem Bild
        ```
    === "CLI"

        ```bash
        yolo detect predict model=yolov8n.pt source='https://ultralytics.com/images/bus.jpg'  # Vorhersage mit offiziellem Modell
        yolo detect predict model=pfad/zum/besten.pt source='https://ultralytics.com/images/bus.jpg'  # Vorhersage mit benutzerdefiniertem Modell
        ```

Volle Details über den `predict`-Modus finden Sie auf der [Predict-Seite](https://docs.ultralytics.com/modes/predict/).

## Export

Ein YOLOv8n-Modell in ein anderes Format wie ONNX, CoreML usw. exportieren.

!!! Example "Beispiel"

    === "Python"

        ```python
        from ultralytics import YOLO

        # Modell laden
        model = YOLO('yolov8n.pt')  # ein offizielles Modell laden
        model = YOLO('pfad/zum/besten.pt')  # ein benutzerdefiniert trainiertes Modell laden

        # Das Modell exportieren
        model.export(format='onnx')
        ```
    === "CLI"

        ```bash
        yolo export model=yolov8n.pt format=onnx  # offizielles Modell exportieren
        yolo export model=pfad/zum/besten.pt format=onnx  # benutzerdefiniert trainiertes Modell exportieren
        ```

Verfügbare YOLOv8 Exportformate sind in der untenstehenden Tabelle aufgeführt. Sie können direkt auf den exportierten Modellen Vorhersagen treffen oder diese validieren, zum Beispiel `yolo predict model=yolov8n.onnx`. Verwendungsbeispiele werden für Ihr Modell nach Abschluss des Exports angezeigt.

| Format                                                             | `format`-Argument | Modell                    | Metadaten | Argumente                                           |
|--------------------------------------------------------------------|-------------------|---------------------------|-----------|-----------------------------------------------------|
| [PyTorch](https://pytorch.org/)                                    | -                 | `yolov8n.pt`              | ✅         | -                                                   |
| [TorchScript](https://pytorch.org/docs/stable/jit.html)            | `torchscript`     | `yolov8n.torchscript`     | ✅         | `imgsz`, `optimize`                                 |
| [ONNX](https://onnx.ai/)                                           | `onnx`            | `yolov8n.onnx`            | ✅         | `imgsz`, `half`, `dynamic`, `simplify`, `opset`     |
| [OpenVINO](https://docs.openvino.ai/latest/index.html)             | `openvino`        | `yolov8n_openvino_model/` | ✅         | `imgsz`, `half`                                     |
| [TensorRT](https://developer.nvidia.com/tensorrt)                  | `engine`          | `yolov8n.engine`          | ✅         | `imgsz`, `half`, `dynamic`, `simplify`, `workspace` |
| [CoreML](https://github.com/apple/coremltools)                     | `coreml`          | `yolov8n.mlpackage`       | ✅         | `imgsz`, `half`, `int8`, `nms`                      |
| [TF SavedModel](https://www.tensorflow.org/guide/saved_model)      | `saved_model`     | `yolov8n_saved_model/`    | ✅         | `imgsz`, `keras`                                    |
| [TF GraphDef](https://www.tensorflow.org/api_docs/python/tf/Graph) | `pb`              | `yolov8n.pb`              | ❌         | `imgsz`                                             |
| [TF Lite](https://www.tensorflow.org/lite)                         | `tflite`          | `yolov8n.tflite`          | ✅         | `imgsz`, `half`, `int8`                             |
| [TF Edge TPU](https://coral.ai/docs/edgetpu/models-intro/)         | `edgetpu`         | `yolov8n_edgetpu.tflite`  | ✅         | `imgsz`                                             |
| [TF.js](https://www.tensorflow.org/js)                             | `tfjs`            | `yolov8n_web_model/`      | ✅         | `imgsz`                                             |
| [PaddlePaddle](https://github.com/PaddlePaddle)                    | `paddle`          | `yolov8n_paddle_model/`   | ✅         | `imgsz`                                             |
| [ncnn](https://github.com/Tencent/ncnn)                            | `ncnn`            | `yolov8n_ncnn_model/`     | ✅         | `imgsz`, `half`                                     |

Volle Details zum `export` finden Sie auf der [Export-Seite](https://docs.ultralytics.com/modes/export/).
