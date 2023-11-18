---
comments: true
description: Erfahren Sie mehr über YOLOv8 Classify-Modelle zur Bildklassifizierung. Erhalten Sie detaillierte Informationen über die Liste vortrainierter Modelle und wie man Modelle trainiert, validiert, vorhersagt und exportiert.
keywords: Ultralytics, YOLOv8, Bildklassifizierung, Vortrainierte Modelle, YOLOv8n-cls, Training, Validierung, Vorhersage, Modellexport
---

# Bildklassifizierung

<img width="1024" src="https://user-images.githubusercontent.com/26833433/243418606-adf35c62-2e11-405d-84c6-b84e7d013804.png" alt="Beispiele für Bildklassifizierung">

Bildklassifizierung ist die einfachste der drei Aufgaben und besteht darin, ein ganzes Bild in eine von einem Satz vordefinierter Klassen zu klassifizieren.

Die Ausgabe eines Bildklassifizierers ist ein einzelnes Klassenlabel und eine Vertrauenspunktzahl. Bildklassifizierung ist nützlich, wenn Sie nur wissen müssen, zu welcher Klasse ein Bild gehört, und nicht wissen müssen, wo sich Objekte dieser Klasse befinden oder wie ihre genaue Form ist.

!!! Tip "Tipp"

    YOLOv8 Classify-Modelle verwenden den Suffix `-cls`, z.B. `yolov8n-cls.pt` und sind auf [ImageNet](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/datasets/ImageNet.yaml) vortrainiert.

## [Modelle](https://github.com/ultralytics/ultralytics/tree/main/ultralytics/cfg/models/v8)

Hier werden vortrainierte YOLOv8 Classify-Modelle gezeigt. Detect-, Segment- und Pose-Modelle sind auf dem [COCO](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/datasets/coco.yaml)-Datensatz vortrainiert, während Classify-Modelle auf dem [ImageNet](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/datasets/ImageNet.yaml)-Datensatz vortrainiert sind.

[Modelle](https://github.com/ultralytics/ultralytics/tree/main/ultralytics/cfg/models) werden automatisch vom neuesten Ultralytics-[Release](https://github.com/ultralytics/assets/releases) beim ersten Gebrauch heruntergeladen.

| Modell                                                                                       | Größe<br><sup>(Pixel) | Genauigkeit<br><sup>top1 | Genauigkeit<br><sup>top5 | Geschwindigkeit<br><sup>CPU ONNX<br>(ms) | Geschwindigkeit<br><sup>A100 TensorRT<br>(ms) | Parameter<br><sup>(M) | FLOPs<br><sup>(B) bei 640 |
|----------------------------------------------------------------------------------------------|-----------------------|--------------------------|--------------------------|------------------------------------------|-----------------------------------------------|-----------------------|---------------------------|
| [YOLOv8n-cls](https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n-cls.pt) | 224                   | 66.6                     | 87.0                     | 12.9                                     | 0.31                                          | 2.7                   | 4.3                       |
| [YOLOv8s-cls](https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8s-cls.pt) | 224                   | 72.3                     | 91.1                     | 23.4                                     | 0.35                                          | 6.4                   | 13.5                      |
| [YOLOv8m-cls](https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8m-cls.pt) | 224                   | 76.4                     | 93.2                     | 85.4                                     | 0.62                                          | 17.0                  | 42.7                      |
| [YOLOv8l-cls](https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8l-cls.pt) | 224                   | 78.0                     | 94.1                     | 163.0                                    | 0.87                                          | 37.5                  | 99.7                      |
| [YOLOv8x-cls](https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8x-cls.pt) | 224                   | 78.4                     | 94.3                     | 232.0                                    | 1.01                                          | 57.4                  | 154.8                     |

- **Genauigkeit**-Werte sind Modellgenauigkeiten auf dem [ImageNet](https://www.image-net.org/)-Datensatz Validierungsset.
  <br>Zur Reproduktion `yolo val classify data=pfad/zu/ImageNet device=0 verwenden`
- **Geschwindigkeit** Durchschnitt über ImageNet-Validierungsbilder mit einer [Amazon EC2 P4d](https://aws.amazon.com/de/ec2/instance-types/p4/)-Instanz.
  <br>Zur Reproduktion `yolo val classify data=pfad/zu/ImageNet batch=1 device=0|cpu verwenden`

## Trainieren

Trainieren Sie das YOLOv8n-cls-Modell auf dem MNIST160-Datensatz für 100 Epochen bei Bildgröße 64. Eine vollständige Liste der verfügbaren Argumente finden Sie auf der Seite [Konfiguration](/../usage/cfg.md).

!!! Example "Beispiel"

    === "Python"

        ```python
        from ultralytics import YOLO

        # Ein Modell laden
        model = YOLO('yolov8n-cls.yaml')  # ein neues Modell aus YAML erstellen
        model = YOLO('yolov8n-cls.pt')  # ein vortrainiertes Modell laden (empfohlen für das Training)
        model = YOLO('yolov8n-cls.yaml').load('yolov8n-cls.pt')  # aus YAML erstellen und Gewichte übertragen

        # Das Modell trainieren
        results = model.train(data='mnist160', epochs=100, imgsz=64)
        ```

    === "CLI"

        ```bash
        # Ein neues Modell aus YAML erstellen und das Training von Grund auf starten
        yolo classify train data=mnist160 model=yolov8n-cls.yaml epochs=100 imgsz=64

        # Das Training von einem vortrainierten *.pt Modell starten
        yolo classify train data=mnist160 model=yolov8n-cls.pt epochs=100 imgsz=64

        # Ein neues Modell aus YAML erstellen, vortrainierte Gewichte übertragen und das Training starten
        yolo classify train data=mnist160 model=yolov8n-cls.yaml pretrained=yolov8n-cls.pt epochs=100 imgsz=64
        ```

### Datenformat

Das Datenformat für YOLO-Klassifizierungsdatensätze finden Sie im Detail im [Datenleitfaden](../../../datasets/classify/index.md).

## Validieren

Validieren Sie die Genauigkeit des trainierten YOLOv8n-cls-Modells auf dem MNIST160-Datensatz. Kein Argument muss übergeben werden, da das `modell` seine Trainings`daten` und Argumente als Modellattribute behält.

!!! Example "Beispiel"

    === "Python"

        ```python
        from ultralytics import YOLO

        # Ein Modell laden
        model = YOLO('yolov8n-cls.pt')  # ein offizielles Modell laden
        model = YOLO('pfad/zu/best.pt')  # ein benutzerdefiniertes Modell laden

        # Das Modell validieren
        metrics = model.val()  # keine Argumente benötigt, Datensatz und Einstellungen gespeichert
        metrics.top1   # top1 Genauigkeit
        metrics.top5   # top5 Genauigkeit
        ```
    === "CLI"

        ```bash
        yolo classify val model=yolov8n-cls.pt  # ein offizielles Modell validieren
        yolo classify val model=pfad/zu/best.pt  # ein benutzerdefiniertes Modell validieren
        ```

## Vorhersagen

Verwenden Sie ein trainiertes YOLOv8n-cls-Modell, um Vorhersagen auf Bildern durchzuführen.

!!! Example "Beispiel"

    === "Python"

        ```python
        from ultralytics import YOLO

        # Ein Modell laden
        model = YOLO('yolov8n-cls.pt')  # ein offizielles Modell laden
        model = YOLO('pfad/zu/best.pt')  # ein benutzerdefiniertes Modell laden

        # Mit dem Modell vorhersagen
        results = model('https://ultralytics.com/images/bus.jpg')  # Vorhersage auf einem Bild
        ```
    === "CLI"

        ```bash
        yolo classify predict model=yolov8n-cls.pt source='https://ultralytics.com/images/bus.jpg'  # mit offiziellem Modell vorhersagen
        yolo classify predict model=pfad/zu/best.pt source='https://ultralytics.com/images/bus.jpg'  # mit benutzerdefiniertem Modell vorhersagen
        ```

Vollständige Details zum `predict`-Modus finden Sie auf der Seite [Vorhersage](https://docs.ultralytics.com/modes/predict/).

## Exportieren

Exportieren Sie ein YOLOv8n-cls-Modell in ein anderes Format wie ONNX, CoreML usw.

!!! Example "Beispiel"

    === "Python"

        ```python
        from ultralytics import YOLO

        # Ein Modell laden
        model = YOLO('yolov8n-cls.pt')  # ein offizielles Modell laden
        model = YOLO('pfad/zu/best.pt')  # ein benutzerdefiniertes trainiertes Modell laden

        # Das Modell exportieren
        model.export(format='onnx')
        ```
    === "CLI"

        ```bash
        yolo export model=yolov8n-cls.pt format=onnx  # offizielles Modell exportieren
        yolo export model=pfad/zu/best.pt format=onnx  # benutzerdefiniertes trainiertes Modell exportieren
        ```

Verfügbare YOLOv8-cls Exportformate stehen in der folgenden Tabelle. Sie können direkt auf exportierten Modellen vorhersagen oder validieren, d.h. `yolo predict model=yolov8n-cls.onnx`. Nutzungsexempel werden für Ihr Modell nach Abschluss des Exports angezeigt.

| Format                                                             | `format`-Argument | Modell                        | Metadaten | Argumente                                           |
|--------------------------------------------------------------------|-------------------|-------------------------------|-----------|-----------------------------------------------------|
| [PyTorch](https://pytorch.org/)                                    | -                 | `yolov8n-cls.pt`              | ✅         | -                                                   |
| [TorchScript](https://pytorch.org/docs/stable/jit.html)            | `torchscript`     | `yolov8n-cls.torchscript`     | ✅         | `imgsz`, `optimize`                                 |
| [ONNX](https://onnx.ai/)                                           | `onnx`            | `yolov8n-cls.onnx`            | ✅         | `imgsz`, `half`, `dynamic`, `simplify`, `opset`     |
| [OpenVINO](https://docs.openvino.ai/latest/index.html)             | `openvino`        | `yolov8n-cls_openvino_model/` | ✅         | `imgsz`, `half`                                     |
| [TensorRT](https://developer.nvidia.com/tensorrt)                  | `engine`          | `yolov8n-cls.engine`          | ✅         | `imgsz`, `half`, `dynamic`, `simplify`, `workspace` |
| [CoreML](https://github.com/apple/coremltools)                     | `coreml`          | `yolov8n-cls.mlpackage`       | ✅         | `imgsz`, `half`, `int8`, `nms`                      |
| [TF SavedModel](https://www.tensorflow.org/guide/saved_model)      | `saved_model`     | `yolov8n-cls_saved_model/`    | ✅         | `imgsz`, `keras`                                    |
| [TF GraphDef](https://www.tensorflow.org/api_docs/python/tf/Graph) | `pb`              | `yolov8n-cls.pb`              | ❌         | `imgsz`                                             |
| [TF Lite](https://www.tensorflow.org/lite)                         | `tflite`          | `yolov8n-cls.tflite`          | ✅         | `imgsz`, `half`, `int8`                             |
| [TF Edge TPU](https://coral.ai/docs/edgetpu/models-intro/)         | `edgetpu`         | `yolov8n-cls_edgetpu.tflite`  | ✅         | `imgsz`                                             |
| [TF.js](https://www.tensorflow.org/js)                             | `tfjs`            | `yolov8n-cls_web_model/`      | ✅         | `imgsz`                                             |
| [PaddlePaddle](https://github.com/PaddlePaddle)                    | `paddle`          | `yolov8n-cls_paddle_model/`   | ✅         | `imgsz`                                             |
| [ncnn](https://github.com/Tencent/ncnn)                            | `ncnn`            | `yolov8n-cls_ncnn_model/`     | ✅         | `imgsz`, `half`                                     |

Vollständige Details zum `export` finden Sie auf der Seite [Export](https://docs.ultralytics.com/modes/export/).
