---
comments: true
description: Erfahren Sie, wie Sie Instanzsegmentierungsmodelle mit Ultralytics YOLO verwenden. Anleitungen zum Training, zur Validierung, zur Bildvorhersage und zum Export von Modellen.
Schlagworte: yolov8, Instanzsegmentierung, Ultralytics, COCO-Datensatz, Bildsegmentierung, Objekterkennung, Modelltraining, Modellvalidierung, Bildvorhersage, Modellexport
---

# Instanzsegmentierung

![Beispiele für Instanzsegmentierung](https://user-images.githubusercontent.com/26833433/243418644-7df320b8-098d-47f1-85c5-26604d761286.png)

Instanzsegmentierung geht einen Schritt weiter als die Objekterkennung und beinhaltet die Identifizierung einzelner Objekte in einem Bild und deren Abtrennung vom Rest des Bildes.

Das Ergebnis eines Instanzsegmentierungsmodells ist eine Reihe von Masken oder Konturen, die jedes Objekt im Bild umreißen, zusammen mit Klassenbezeichnungen und Vertrauensscores für jedes Objekt. Instanzsegmentierung ist nützlich, wenn man nicht nur wissen muss, wo sich Objekte in einem Bild befinden, sondern auch, welche genaue Form sie haben.

<p align="center">
  <br>
  <iframe width="720" height="405" src="https://www.youtube.com/embed/o4Zd-IeMlSY?si=37nusCzDTd74Obsp"
    title="YouTube-Video-Player" frameborder="0"
    allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share"
    allowfullscreen>
  </iframe>
  <br>
  <strong>Schauen Sie:</strong> Führen Sie Segmentierung mit dem vortrainierten Ultralytics YOLOv8 Modell in Python aus.
</p>

!!! Tip "Tipp"

    YOLOv8 Segment-Modelle verwenden das Suffix `-seg`, d.h. `yolov8n-seg.pt` und sind auf dem [COCO](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/datasets/coco.yaml)-Datensatz vortrainiert.

## [Modelle](https://github.com/ultralytics/ultralytics/tree/main/ultralytics/cfg/models/v8)

Hier werden vortrainierte YOLOv8 Segment-Modelle gezeigt. Detect-, Segment- und Pose-Modelle sind auf dem [COCO](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/datasets/coco.yaml)-Datensatz vortrainiert, während Classify-Modelle auf dem [ImageNet](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/datasets/ImageNet.yaml)-Datensatz vortrainiert sind.

[Modelle](https://github.com/ultralytics/ultralytics/tree/main/ultralytics/cfg/models) laden sich automatisch von der neuesten Ultralytics [Veröffentlichung](https://github.com/ultralytics/assets/releases) beim ersten Gebrauch herunter.

| Modell                                                                                       | Größe<br><sup>(Pixel) | mAP<sup>Kasten<br>50-95 | mAP<sup>Masken<br>50-95 | Geschwindigkeit<br><sup>CPU ONNX<br>(ms) | Geschwindigkeit<br><sup>A100 TensorRT<br>(ms) | Parameter<br><sup>(M) | FLOPs<br><sup>(B) |
|----------------------------------------------------------------------------------------------|-----------------------|-------------------------|-------------------------|------------------------------------------|-----------------------------------------------|-----------------------|-------------------|
| [YOLOv8n-seg](https://github.com/ultralytics/assets/releases/download/v8.1.0/yolov8n-seg.pt) | 640                   | 36.7                    | 30.5                    | 96.1                                     | 1.21                                          | 3.4                   | 12.6              |
| [YOLOv8s-seg](https://github.com/ultralytics/assets/releases/download/v8.1.0/yolov8s-seg.pt) | 640                   | 44.6                    | 36.8                    | 155.7                                    | 1.47                                          | 11.8                  | 42.6              |
| [YOLOv8m-seg](https://github.com/ultralytics/assets/releases/download/v8.1.0/yolov8m-seg.pt) | 640                   | 49.9                    | 40.8                    | 317.0                                    | 2.18                                          | 27.3                  | 110.2             |
| [YOLOv8l-seg](https://github.com/ultralytics/assets/releases/download/v8.1.0/yolov8l-seg.pt) | 640                   | 52.3                    | 42.6                    | 572.4                                    | 2.79                                          | 46.0                  | 220.5             |
| [YOLOv8x-seg](https://github.com/ultralytics/assets/releases/download/v8.1.0/yolov8x-seg.pt) | 640                   | 53.4                    | 43.4                    | 712.1                                    | 4.02                                          | 71.8                  | 344.1             |

- Die **mAP<sup>val</sup>**-Werte sind für ein einzelnes Modell, einzelne Skala auf dem [COCO val2017](http://cocodataset.org)-Datensatz.
  <br>Zum Reproduzieren nutzen Sie `yolo val segment data=coco.yaml device=0`
- Die **Geschwindigkeit** ist über die COCO-Validierungsbilder gemittelt und verwendet eine [Amazon EC2 P4d](https://aws.amazon.com/ec2/instance-types/p4/)-Instanz.
  <br>Zum Reproduzieren `yolo val segment data=coco128-seg.yaml batch=1 device=0|cpu`

## Training

Trainieren Sie YOLOv8n-seg auf dem COCO128-seg-Datensatz für 100 Epochen mit einer Bildgröße von 640. Eine vollständige Liste der verfügbaren Argumente finden Sie auf der Seite [Konfiguration](/../usage/cfg.md).

!!! Example "Beispiel"

    === "Python"

        ```python
        from ultralytics import YOLO

        # Modell laden
        model = YOLO('yolov8n-seg.yaml')  # ein neues Modell aus YAML erstellen
        model = YOLO('yolov8n-seg.pt')  # ein vortrainiertes Modell laden (empfohlen für das Training)
        model = YOLO('yolov8n-seg.yaml').load('yolov8n.pt')  # aus YAML erstellen und Gewichte übertragen

        # Das Modell trainieren
        results = model.train(data='coco128-seg.yaml', epochs=100, imgsz=640)
        ```
    === "CLI"

        ```bash
        # Ein neues Modell aus YAML erstellen und das Training von vorne beginnen
        yolo segment train data=coco128-seg.yaml model=yolov8n-seg.yaml epochs=100 imgsz=640

        # Das Training von einem vortrainierten *.pt Modell aus starten
        yolo segment train data=coco128-seg.yaml model=yolov8n-seg.pt epochs=100 imgsz=640

        # Ein neues Modell aus YAML erstellen, vortrainierte Gewichte darauf übertragen und das Training beginnen
        yolo segment train data=coco128-seg.yaml model=yolov8n-seg.yaml pretrained=yolov8n-seg.pt epochs=100 imgsz=640
        ```

### Datenformat

Das YOLO Segmentierungsdatenformat finden Sie detailliert im [Dataset Guide](../../../datasets/segment/index.md). Um Ihre vorhandenen Daten aus anderen Formaten (wie COCO usw.) in das YOLO-Format umzuwandeln, verwenden Sie bitte das [JSON2YOLO](https://github.com/ultralytics/JSON2YOLO)-Tool von Ultralytics.

## Val

Validieren Sie die Genauigkeit des trainierten YOLOv8n-seg-Modells auf dem COCO128-seg-Datensatz. Es müssen keine Argumente übergeben werden, da das `Modell` seine Trainingsdaten und -argumente als Modellattribute behält.

!!! Example "Beispiel"

    === "Python"

        ```python
        from ultralytics import YOLO

        # Modell laden
        model = YOLO('yolov8n-seg.pt')  # offizielles Modell laden
        model = YOLO('pfad/zu/best.pt')  # benutzerdefiniertes Modell laden

        # Das Modell validieren
        metrics = model.val()  # Keine Argumente erforderlich, Datensatz und Einstellungen werden behalten
        metrics.box.map    # mAP50-95(B)
        metrics.box.map50  # mAP50(B)
        metrics.box.map75  # mAP75(B)
        metrics.box.maps   # eine Liste enthält mAP50-95(B) für jede Kategorie
        metrics.seg.map    # mAP50-95(M)
        metrics.seg.map50  # mAP50(M)
        metrics.seg.map75  # mAP75(M)
        metrics.seg.maps   # eine Liste enthält mAP50-95(M) für jede Kategorie
        ```
    === "CLI"

        ```bash
        yolo segment val model=yolov8n-seg.pt  # offizielles Modell validieren
        yolo segment val model=pfad/zu/best.pt  # benutzerdefiniertes Modell validieren
        ```

## Predict

Verwenden Sie ein trainiertes YOLOv8n-seg-Modell für Vorhersagen auf Bildern.

!!! Example "Beispiel"

    === "Python"

        ```python
        from ultralytics import YOLO

        # Modell laden
        model = YOLO('yolov8n-seg.pt')  # offizielles Modell laden
        model = YOLO('pfad/zu/best.pt')  # benutzerdefiniertes Modell laden

        # Mit dem Modell Vorhersagen treffen
        results = model('https://ultralytics.com/images/bus.jpg')  # Vorhersage auf einem Bild
        ```
    === "CLI"

        ```bash
        yolo segment predict model=yolov8n-seg.pt source='https://ultralytics.com/images/bus.jpg'  # Vorhersage mit offiziellem Modell treffen
        yolo segment predict model=pfad/zu/best.pt source='https://ultralytics.com/images/bus.jpg'  # Vorhersage mit benutzerdefiniertem Modell treffen
        ```

Die vollständigen Details zum `predict`-Modus finden Sie auf der Seite [Predict](https://docs.ultralytics.com/modes/predict/).

## Export

Exportieren Sie ein YOLOv8n-seg-Modell in ein anderes Format wie ONNX, CoreML usw.

!!! Example "Beispiel"

    === "Python"

        ```python
        from ultralytics import YOLO

        # Modell laden
        model = YOLO('yolov8n-seg.pt')  # offizielles Modell laden
        model = YOLO('pfad/zu/best.pt')  # benutzerdefiniertes trainiertes Modell laden

        # Das Modell exportieren
        model.export(format='onnx')
        ```
    === "CLI"

        ```bash
        yolo export model=yolov8n-seg.pt format=onnx  # offizielles Modell exportieren
        yolo export model=pfad/zu/best.pt format=onnx  # benutzerdefiniertes trainiertes Modell exportieren
        ```

Die verfügbaren YOLOv8-seg-Exportformate sind in der folgenden Tabelle aufgeführt. Sie können direkt auf exportierten Modellen Vorhersagen treffen oder sie validieren, z.B. `yolo predict model=yolov8n-seg.onnx`. Verwendungsbeispiele werden für Ihr Modell nach dem Export angezeigt.

| Format                                                             | `format`-Argument | Modell                        | Metadaten | Argumente                                                       |
|--------------------------------------------------------------------|-------------------|-------------------------------|-----------|-----------------------------------------------------------------|
| [PyTorch](https://pytorch.org/)                                    | -                 | `yolov8n-seg.pt`              | ✅         | -                                                               |
| [TorchScript](https://pytorch.org/docs/stable/jit.html)            | `torchscript`     | `yolov8n-seg.torchscript`     | ✅         | `imgsz`, `optimieren`                                           |
| [ONNX](https://onnx.ai/)                                           | `onnx`            | `yolov8n-seg.onnx`            | ✅         | `imgsz`, `halb`, `dynamisch`, `vereinfachen`, `opset`           |
| [OpenVINO](https://docs.openvino.ai/latest/index.html)             | `openvino`        | `yolov8n-seg_openvino_model/` | ✅         | `imgsz`, `halb`                                                 |
| [TensorRT](https://developer.nvidia.com/tensorrt)                  | `engine`          | `yolov8n-seg.engine`          | ✅         | `imgsz`, `halb`, `dynamisch`, `vereinfachen`, `Arbeitsspeicher` |
| [CoreML](https://github.com/apple/coremltools)                     | `coreml`          | `yolov8n-seg.mlpackage`       | ✅         | `imgsz`, `halb`, `int8`, `nms`                                  |
| [TF SavedModel](https://www.tensorflow.org/guide/saved_model)      | `saved_model`     | `yolov8n-seg_saved_model/`    | ✅         | `imgsz`, `keras`                                                |
| [TF GraphDef](https://www.tensorflow.org/api_docs/python/tf/Graph) | `pb`              | `yolov8n-seg.pb`              | ❌         | `imgsz`                                                         |
| [TF Lite](https://www.tensorflow.org/lite)                         | `tflite`          | `yolov8n-seg.tflite`          | ✅         | `imgsz`, `halb`, `int8`                                         |
| [TF Edge TPU](https://coral.ai/docs/edgetpu/models-intro/)         | `edgetpu`         | `yolov8n-seg_edgetpu.tflite`  | ✅         | `imgsz`                                                         |
| [TF.js](https://www.tensorflow.org/js)                             | `tfjs`            | `yolov8n-seg_web_model/`      | ✅         | `imgsz`                                                         |
| [PaddlePaddle](https://github.com/PaddlePaddle)                    | `paddle`          | `yolov8n-seg_paddle_model/`   | ✅         | `imgsz`                                                         |
| [ncnn](https://github.com/Tencent/ncnn)                            | `ncnn`            | `yolov8n-seg_ncnn_model/`     | ✅         | `imgsz`, `halb`                                                 |

Die vollständigen Details zum `export` finden Sie auf der Seite [Export](https://docs.ultralytics.com/modes/export/).
