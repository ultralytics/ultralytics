---
comments: true
description: Lernen Sie, wie Sie die Geschwindigkeit und Genauigkeit von YOLOv8 über verschiedene Exportformate hinweg profilieren können; erhalten Sie Einblicke in mAP50-95, Genauigkeit_top5 Kennzahlen und mehr.
keywords: Ultralytics, YOLOv8, Benchmarking, Geschwindigkeitsprofilierung, Genauigkeitsprofilierung, mAP50-95, accuracy_top5, ONNX, OpenVINO, TensorRT, YOLO-Exportformate
---

# Modell-Benchmarking mit Ultralytics YOLO

<img width="1024" src="https://github.com/ultralytics/assets/raw/main/yolov8/banner-integrations.png" alt="Ultralytics YOLO-Ökosystem und Integrationen">

## Einführung

Nachdem Ihr Modell trainiert und validiert wurde, ist der nächste logische Schritt, seine Leistung in verschiedenen realen Szenarien zu bewerten. Der Benchmark-Modus in Ultralytics YOLOv8 dient diesem Zweck, indem er einen robusten Rahmen für die Beurteilung von Geschwindigkeit und Genauigkeit Ihres Modells über eine Reihe von Exportformaten hinweg bietet.

## Warum ist Benchmarking entscheidend?

- **Informierte Entscheidungen:** Erhalten Sie Einblicke in die Kompromisse zwischen Geschwindigkeit und Genauigkeit.
- **Ressourcenzuweisung:** Verstehen Sie, wie sich verschiedene Exportformate auf unterschiedlicher Hardware verhalten.
- **Optimierung:** Erfahren Sie, welches Exportformat die beste Leistung für Ihren spezifischen Anwendungsfall bietet.
- **Kosteneffizienz:** Nutzen Sie Hardware-Ressourcen basierend auf den Benchmark-Ergebnissen effizienter.

### Schlüsselmetriken im Benchmark-Modus

- **mAP50-95:** Für Objekterkennung, Segmentierung und Posenschätzung.
- **accuracy_top5:** Für die Bildklassifizierung.
- **Inferenzzeit:** Zeit, die für jedes Bild in Millisekunden benötigt wird.

### Unterstützte Exportformate

- **ONNX:** Für optimale CPU-Leistung
- **TensorRT:** Für maximale GPU-Effizienz
- **OpenVINO:** Für die Optimierung von Intel-Hardware
- **CoreML, TensorFlow SavedModel, und mehr:** Für vielfältige Deployment-Anforderungen.

!!! Tip "Tipp"

    * Exportieren Sie in ONNX oder OpenVINO für bis zu 3x CPU-Beschleunigung.
    * Exportieren Sie in TensorRT für bis zu 5x GPU-Beschleunigung.

## Anwendungsbeispiele

Führen Sie YOLOv8n-Benchmarks auf allen unterstützten Exportformaten einschließlich ONNX, TensorRT usw. durch. Siehe den Abschnitt Argumente unten für eine vollständige Liste der Exportargumente.

!!! Example "Beispiel"

    === "Python"

        ```python
        from ultralytics.utils.benchmarks import benchmark

        # Benchmark auf GPU
        benchmark(model='yolov8n.pt', data='coco8.yaml', imgsz=640, half=False, device=0)
        ```
    === "CLI"

        ```bash
        yolo benchmark model=yolov8n.pt data='coco8.yaml' imgsz=640 half=False device=0
        ```

## Argumente

Argumente wie `model`, `data`, `imgsz`, `half`, `device` und `verbose` bieten Benutzern die Flexibilität, die Benchmarks auf ihre spezifischen Bedürfnisse abzustimmen und die Leistung verschiedener Exportformate mühelos zu vergleichen.

| Schlüssel | Wert    | Beschreibung                                                                         |
|-----------|---------|--------------------------------------------------------------------------------------|
| `model`   | `None`  | Pfad zur Modelldatei, z. B. yolov8n.pt, yolov8n.yaml                                 |
| `data`    | `None`  | Pfad zur YAML, die das Benchmarking-Dataset referenziert (unter `val`-Kennzeichnung) |
| `imgsz`   | `640`   | Bildgröße als Skalar oder Liste (h, w), z. B. (640, 480)                             |
| `half`    | `False` | FP16-Quantisierung                                                                   |
| `int8`    | `False` | INT8-Quantisierung                                                                   |
| `device`  | `None`  | Gerät zum Ausführen, z. B. CUDA device=0 oder device=0,1,2,3 oder device=cpu         |
| `verbose` | `False` | bei Fehlern nicht fortsetzen (bool), oder Wertebereichsschwelle (float)              |

## Exportformate

Benchmarks werden automatisch auf allen möglichen Exportformaten unten ausgeführt.

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

Vollständige Details zum `export` finden Sie auf der [Export](https://docs.ultralytics.com/modes/export/)-Seite.
