---
comments: true
description: Erfahren Sie, wie Sie Ultralytics YOLOv8 für Aufgaben der Pose-Schätzung verwenden können. Finden Sie vortrainierte Modelle, lernen Sie, wie man eigene trainiert, validiert, vorhersagt und exportiert.
keywords: Ultralytics, YOLO, YOLOv8, Pose-Schätzung, Erkennung von Schlüsselpunkten, Objekterkennung, vortrainierte Modelle, maschinelles Lernen, künstliche Intelligenz
---

# Pose-Schätzung

![Beispiele für die Pose-Schätzung](https://user-images.githubusercontent.com/26833433/243418616-9811ac0b-a4a7-452a-8aba-484ba32bb4a8.png)

Die Pose-Schätzung ist eine Aufgabe, die das Identifizieren der Lage spezifischer Punkte in einem Bild beinhaltet, die normalerweise als Schlüsselpunkte bezeichnet werden. Die Schlüsselpunkte können verschiedene Teile des Objekts wie Gelenke, Landmarken oder andere charakteristische Merkmale repräsentieren. Die Positionen der Schlüsselpunkte sind üblicherweise als eine Gruppe von 2D `[x, y]` oder 3D `[x, y, sichtbar]` Koordinaten dargestellt.

Das Ergebnis eines Pose-Schätzungsmodells ist eine Gruppe von Punkten, die die Schlüsselpunkte auf einem Objekt im Bild darstellen, normalerweise zusammen mit den Konfidenzwerten für jeden Punkt. Die Pose-Schätzung eignet sich gut, wenn Sie spezifische Teile eines Objekts in einer Szene identifizieren müssen und deren Lage zueinander.

<p align="center">
  <br>
  <iframe width="720" height="405" src="https://www.youtube.com/embed/Y28xXQmju64?si=pCY4ZwejZFu6Z4kZ"
    title="YouTube-Video-Player" frameborder="0"
    allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share"
    allowfullscreen>
  </iframe>
  <br>
  <strong>Ansehen:</strong> Pose-Schätzung mit Ultralytics YOLOv8.
</p>

!!! Tip "Tipp"

    YOLOv8 _pose_-Modelle verwenden den Suffix `-pose`, z. B. `yolov8n-pose.pt`. Diese Modelle sind auf dem [COCO-Schlüsselpunkte](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/datasets/coco-pose.yaml)-Datensatz trainiert und für eine Vielzahl von Pose-Schätzungsaufgaben geeignet.

## [Modelle](https://github.com/ultralytics/ultralytics/tree/main/ultralytics/cfg/models/v8)

Hier werden vortrainierte YOLOv8 Pose-Modelle gezeigt. Erkennungs-, Segmentierungs- und Pose-Modelle sind auf dem [COCO](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/datasets/coco.yaml)-Datensatz vortrainiert, während Klassifizierungsmodelle auf dem [ImageNet](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/datasets/ImageNet.yaml)-Datensatz vortrainiert sind.

[Modelle](https://github.com/ultralytics/ultralytics/tree/main/ultralytics/cfg/models) werden automatisch aus der neuesten Ultralytics-[Veröffentlichung](https://github.com/ultralytics/assets/releases) bei erstmaliger Verwendung heruntergeladen.

| Modell                                                                                               | Größe<br/><sup>(Pixel) | mAP<sup>pose<br/>50-95 | mAP<sup>pose<br/>50 | Geschwindigkeit<br/><sup>CPU ONNX<br/>(ms) | Geschwindigkeit<br/><sup>A100 TensorRT<br/>(ms) | Parameter<br/><sup>(M) | FLOPs<br/><sup>(B) |
|------------------------------------------------------------------------------------------------------|------------------------|------------------------|---------------------|--------------------------------------------|-------------------------------------------------|------------------------|--------------------|
| [YOLOv8n-pose](https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n-pose.pt)       | 640                    | 50,4                   | 80,1                | 131,8                                      | 1,18                                            | 3,3                    | 9,2                |
| [YOLOv8s-pose](https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8s-pose.pt)       | 640                    | 60,0                   | 86,2                | 233,2                                      | 1,42                                            | 11,6                   | 30,2               |
| [YOLOv8m-pose](https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8m-pose.pt)       | 640                    | 65,0                   | 88,8                | 456,3                                      | 2,00                                            | 26,4                   | 81,0               |
| [YOLOv8l-pose](https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8l-pose.pt)       | 640                    | 67,6                   | 90,0                | 784,5                                      | 2,59                                            | 44,4                   | 168,6              |
| [YOLOv8x-pose](https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8x-pose.pt)       | 640                    | 69,2                   | 90,2                | 1607,1                                     | 3,73                                            | 69,4                   | 263,2              |
| [YOLOv8x-pose-p6](https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8x-pose-p6.pt) | 1280                   | 71,6                   | 91,2                | 4088,7                                     | 10,04                                           | 99,1                   | 1066,4             |

- **mAP<sup>val</sup>** Werte gelten für ein einzelnes Modell mit einfacher Skala auf dem [COCO Keypoints val2017](http://cocodataset.org)-Datensatz.
  <br>Zu reproduzieren mit `yolo val pose data=coco-pose.yaml device=0`.
- **Geschwindigkeit** gemittelt über COCO-Validierungsbilder mit einer [Amazon EC2 P4d](https://aws.amazon.com/de/ec2/instance-types/p4/)-Instanz.
  <br>Zu reproduzieren mit `yolo val pose data=coco8-pose.yaml batch=1 device=0|cpu`.

## Trainieren

Trainieren Sie ein YOLOv8-Pose-Modell auf dem COCO128-Pose-Datensatz.

!!! Example "Beispiel"

    === "Python"

        ```python
        from ultralytics import YOLO

        # Modell laden
        model = YOLO('yolov8n-pose.yaml')  # ein neues Modell aus YAML bauen
        model = YOLO('yolov8n-pose.pt')  # ein vortrainiertes Modell laden (empfohlen für das Training)
        model = YOLO('yolov8n-pose.yaml').load('yolov8n-pose.pt')  # aus YAML bauen und Gewichte übertragen

        # Modell trainieren
        results = model.train(data='coco8-pose.yaml', epochs=100, imgsz=640)
        ```
    === "CLI"

        ```bash
        # Ein neues Modell aus YAML bauen und das Training von Grund auf starten
        yolo pose train data=coco8-pose.yaml model=yolov8n-pose.yaml epochs=100 imgsz=640

        # Training von einem vortrainierten *.pt Modell starten
        yolo pose train data=coco8-pose.yaml model=yolov8n-pose.pt epochs=100 imgsz=640

        # Ein neues Modell aus YAML bauen, vortrainierte Gewichte übertragen und das Training starten
        yolo pose train data=coco8-pose.yaml model=yolov8n-pose.yaml pretrained=yolov8n-pose.pt epochs=100 imgsz=640
        ```

### Datensatzformat

Das YOLO-Pose-Datensatzformat finden Sie detailliert im [Datensatz-Leitfaden](../../../datasets/pose/index.md). Um Ihren bestehenden Datensatz aus anderen Formaten (wie COCO usw.) in das YOLO-Format zu konvertieren, verwenden Sie bitte das [JSON2YOLO](https://github.com/ultralytics/JSON2YOLO)-Tool von Ultralytics.

## Validieren

Die Genauigkeit des trainierten YOLOv8n-Pose-Modells auf dem COCO128-Pose-Datensatz validieren. Es müssen keine Argumente übergeben werden, da das `Modell` seine Trainings`daten` und Argumente als Modellattribute beibehält.

!!! Example "Beispiel"

    === "Python"

        ```python
        from ultralytics import YOLO

        # Modell laden
        model = YOLO('yolov8n-pose.pt')  # ein offizielles Modell laden
        model = YOLO('pfad/zu/best.pt')  # ein benutzerdefiniertes Modell laden

        # Modell validieren
        metrics = model.val()  # keine Argumente nötig, Datensatz und Einstellungen sind gespeichert
        metrics.box.map    # map50-95
        metrics.box.map50  # map50
        metrics.box.map75  # map75
        metrics.box.maps   # Liste enthält map50-95 jeder Kategorie
        ```
    === "CLI"

        ```bash
        yolo pose val model=yolov8n-pose.pt  # offizielles Modell validieren
        yolo pose val model=pfad/zu/best.pt  # benutzerdefiniertes Modell validieren
        ```

## Vorhersagen

Ein trainiertes YOLOv8n-Pose-Modell verwenden, um Vorhersagen auf Bildern zu machen.

!!! Example "Beispiel"

    === "Python"

        ```python
        from ultralytics import YOLO

        # Modell laden
        model = YOLO('yolov8n-pose.pt')  # ein offizielles Modell laden
        model = YOLO('pfad/zu/best.pt')  # ein benutzerdefiniertes Modell laden

        # Mit dem Modell Vorhersagen machen
        results = model('https://ultralytics.com/images/bus.jpg')  # Vorhersage auf einem Bild machen
        ```
    === "CLI"

        ```bash
        yolo pose predict model=yolov8n-pose.pt source='https://ultralytics.com/images/bus.jpg'  # Vorhersage mit dem offiziellen Modell machen
        yolo pose predict model=pfad/zu/best.pt source='https://ultralytics.com/images/bus.jpg'  # Vorhersage mit dem benutzerdefinierten Modell machen
        ```

Vollständige `predict`-Modusdetails finden Sie auf der [Vorhersage](https://docs.ultralytics.com/modes/predict/)-Seite.

## Exportieren

Ein YOLOv8n-Pose-Modell in ein anderes Format wie ONNX, CoreML usw. exportieren.

!!! Example "Beispiel"

    === "Python"

        ```python
        from ultralytics import YOLO

        # Modell laden
        model = YOLO('yolov8n-pose.pt')  # ein offizielles Modell laden
        model = YOLO('pfad/zu/best.pt')  # ein benutzerdefiniertes Modell laden

        # Modell exportieren
        model.export(format='onnx')
        ```
    === "CLI"

        ```bash
        yolo export model=yolov8n-pose.pt format=onnx  # offizielles Modell exportieren
        yolo export model=pfad/zu/best.pt format=onnx  # benutzerdefiniertes Modell exportieren
        ```

Verfügbare YOLOv8-Pose-Exportformate sind in der folgenden Tabelle aufgeführt. Sie können direkt auf exportierten Modellen vorhersagen oder validieren, z. B. `yolo predict model=yolov8n-pose.onnx`. Verwendungsbeispiele werden für Ihr Modell nach Abschluss des Exports angezeigt.

| Format                                                             | `format` Argument | Modell                         | Metadaten | Argumente                                                 |
|--------------------------------------------------------------------|-------------------|--------------------------------|-----------|-----------------------------------------------------------|
| [PyTorch](https://pytorch.org/)                                    | -                 | `yolov8n-pose.pt`              | ✅         | -                                                         |
| [TorchScript](https://pytorch.org/docs/stable/jit.html)            | `torchscript`     | `yolov8n-pose.torchscript`     | ✅         | `imgsz`, `optimieren`                                     |
| [ONNX](https://onnx.ai/)                                           | `onnx`            | `yolov8n-pose.onnx`            | ✅         | `imgsz`, `halb`, `dynamisch`, `vereinfachen`, `opset`     |
| [OpenVINO](https://docs.openvino.ai/latest/index.html)             | `openvino`        | `yolov8n-pose_openvino_model/` | ✅         | `imgsz`, `halb`                                           |
| [TensorRT](https://developer.nvidia.com/tensorrt)                  | `engine`          | `yolov8n-pose.engine`          | ✅         | `imgsz`, `halb`, `dynamisch`, `vereinfachen`, `workspace` |
| [CoreML](https://github.com/apple/coremltools)                     | `coreml`          | `yolov8n-pose.mlpackage`       | ✅         | `imgsz`, `halb`, `int8`, `nms`                            |
| [TF SavedModel](https://www.tensorflow.org/guide/saved_model)      | `saved_model`     | `yolov8n-pose_saved_model/`    | ✅         | `imgsz`, `keras`                                          |
| [TF GraphDef](https://www.tensorflow.org/api_docs/python/tf/Graph) | `pb`              | `yolov8n-pose.pb`              | ❌         | `imgsz`                                                   |
| [TF Lite](https://www.tensorflow.org/lite)                         | `tflite`          | `yolov8n-pose.tflite`          | ✅         | `imgsz`, `halb`, `int8`                                   |
| [TF Edge TPU](https://coral.ai/docs/edgetpu/models-intro/)         | `edgetpu`         | `yolov8n-pose_edgetpu.tflite`  | ✅         | `imgsz`                                                   |
| [TF.js](https://www.tensorflow.org/js)                             | `tfjs`            | `yolov8n-pose_web_model/`      | ✅         | `imgsz`                                                   |
| [PaddlePaddle](https://github.com/PaddlePaddle)                    | `paddle`          | `yolov8n-pose_paddle_model/`   | ✅         | `imgsz`                                                   |
| [ncnn](https://github.com/Tencent/ncnn)                            | `ncnn`            | `yolov8n-pose_ncnn_model/`     | ✅         | `imgsz`, `halb`                                           |

Vollständige `export`-Details finden Sie auf der [Export](https://docs.ultralytics.com/modes/export/)-Seite.
