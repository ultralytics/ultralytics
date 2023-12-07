---
comments: true
description: Schritt-für-Schritt-Anleitung zum Exportieren Ihrer YOLOv8-Modelle in verschiedene Formate wie ONNX, TensorRT, CoreML und mehr für den Einsatz.
keywords: YOLO, YOLOv8, Ultralytics, Modell-Export, ONNX, TensorRT, CoreML, TensorFlow SavedModel, OpenVINO, PyTorch, Modell exportieren
---

# Modell-Export mit Ultralytics YOLO

<img width="1024" src="https://github.com/ultralytics/assets/raw/main/yolov8/banner-integrations.png" alt="Ultralytics YOLO Ökosystem und Integrationen">

## Einführung

Das ultimative Ziel des Trainierens eines Modells besteht darin, es für reale Anwendungen einzusetzen. Der Exportmodus in Ultralytics YOLOv8 bietet eine vielseitige Palette von Optionen für den Export Ihres trainierten Modells in verschiedene Formate, sodass es auf verschiedenen Plattformen und Geräten eingesetzt werden kann. Dieser umfassende Leitfaden soll Sie durch die Nuancen des Modell-Exports führen und zeigen, wie Sie maximale Kompatibilität und Leistung erzielen können.

<p align="center">
  <br>
  <iframe width="720" height="405" src="https://www.youtube.com/embed/WbomGeoOT_k?si=aGmuyooWftA0ue9X"
    title="YouTube-Video-Player" frameborder="0"
    allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share"
    allowfullscreen>
  </iframe>
  <br>
  <strong>Ansehen:</strong> Wie man ein benutzerdefiniertes trainiertes Ultralytics YOLOv8-Modell exportiert und Live-Inferenz auf der Webcam ausführt.
</p>

## Warum den Exportmodus von YOLOv8 wählen?

- **Vielseitigkeit:** Export in verschiedene Formate einschließlich ONNX, TensorRT, CoreML und mehr.
- **Leistung:** Bis zu 5-fache GPU-Beschleunigung mit TensorRT und 3-fache CPU-Beschleunigung mit ONNX oder OpenVINO.
- **Kompatibilität:** Machen Sie Ihr Modell universell einsetzbar in zahlreichen Hardware- und Softwareumgebungen.
- **Benutzerfreundlichkeit:** Einfache CLI- und Python-API für schnellen und unkomplizierten Modell-Export.

### Schlüsselfunktionen des Exportmodus

Hier sind einige der herausragenden Funktionen:

- **Ein-Klick-Export:** Einfache Befehle für den Export in verschiedene Formate.
- **Batch-Export:** Export von Modellen, die Batch-Inferenz unterstützen.
- **Optimiertes Inferenzverhalten:** Exportierte Modelle sind für schnellere Inferenzzeiten optimiert.
- **Tutorial-Videos:** Ausführliche Anleitungen und Tutorials für ein reibungsloses Exporterlebnis.

!!! Tip "Tipp"

    * Exportieren Sie nach ONNX oder OpenVINO für bis zu 3-fache CPU-Beschleunigung.
    * Exportieren Sie nach TensorRT für bis zu 5-fache GPU-Beschleunigung.

## Nutzungsbeispiele

Exportieren Sie ein YOLOv8n-Modell in ein anderes Format wie ONNX oder TensorRT. Weitere Informationen zu den Exportargumenten finden Sie im Abschnitt „Argumente“ unten.

!!! Example "Beispiel"

    === "Python"

        ```python
        from ultralytics import YOLO

        # Laden eines Modells
        model = YOLO('yolov8n.pt')  # offizielles Modell laden
        model = YOLO('path/to/best.pt')  # benutzerdefiniertes trainiertes Modell laden

        # Exportieren des Modells
        model.export(format='onnx')
        ```
    === "CLI"

        ```bash
        yolo export model=yolov8n.pt format=onnx  # offizielles Modell exportieren
        yolo export model=path/to/best.pt format=onnx  # benutzerdefiniertes trainiertes Modell exportieren
        ```

## Argumente

Exporteinstellungen für YOLO-Modelle beziehen sich auf verschiedene Konfigurationen und Optionen, die verwendet werden, um das Modell zu speichern oder für den Einsatz in anderen Umgebungen oder Plattformen zu exportieren. Diese Einstellungen können die Leistung, Größe und Kompatibilität des Modells mit verschiedenen Systemen beeinflussen. Zu den gängigen Exporteinstellungen von YOLO gehören das Format der exportierten Modelldatei (z. B. ONNX, TensorFlow SavedModel), das Gerät, auf dem das Modell ausgeführt wird (z. B. CPU, GPU) und das Vorhandensein zusätzlicher Funktionen wie Masken oder mehrere Labels pro Box. Andere Faktoren, die den Exportprozess beeinflussen können, sind die spezifische Aufgabe, für die das Modell verwendet wird, und die Anforderungen oder Einschränkungen der Zielumgebung oder -plattform. Es ist wichtig, diese Einstellungen sorgfältig zu berücksichtigen und zu konfigurieren, um sicherzustellen, dass das exportierte Modell für den beabsichtigten Einsatzzweck optimiert ist und in der Zielumgebung effektiv eingesetzt werden kann.

| Schlüssel   | Wert            | Beschreibung                                             |
|-------------|-----------------|----------------------------------------------------------|
| `format`    | `'torchscript'` | Format für den Export                                    |
| `imgsz`     | `640`           | Bildgröße als Skalar oder (h, w)-Liste, z.B. (640, 480)  |
| `keras`     | `False`         | Verwendung von Keras für TensorFlow SavedModel-Export    |
| `optimize`  | `False`         | TorchScript: Optimierung für mobile Geräte               |
| `half`      | `False`         | FP16-Quantisierung                                       |
| `int8`      | `False`         | INT8-Quantisierung                                       |
| `dynamic`   | `False`         | ONNX/TensorRT: dynamische Achsen                         |
| `simplify`  | `False`         | ONNX/TensorRT: Vereinfachung des Modells                 |
| `opset`     | `None`          | ONNX: Opset-Version (optional, Standardwert ist neueste) |
| `workspace` | `4`             | TensorRT: Arbeitsbereichgröße (GB)                       |
| `nms`       | `False`         | CoreML: Hinzufügen von NMS                               |

## Exportformate

Verfügbare YOLOv8-Exportformate finden Sie in der Tabelle unten. Sie können in jedes Format exportieren, indem Sie das `format`-Argument verwenden, z. B. `format='onnx'` oder `format='engine'`.

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
