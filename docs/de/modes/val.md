---
comments: true
description: Anleitung zur Validierung von YOLOv8-Modellen. Erfahren Sie, wie Sie die Leistung Ihrer YOLO-Modelle mit Validierungseinstellungen und Metriken in Python und CLI-Beispielen bewerten können.
keywords: Ultralytics, YOLO-Dokumente, YOLOv8, Validierung, Modellbewertung, Hyperparameter, Genauigkeit, Metriken, Python, CLI
---

# Modellvalidierung mit Ultralytics YOLO

<img width="1024" src="https://github.com/ultralytics/assets/raw/main/yolov8/banner-integrations.png" alt="Ultralytics YOLO-Ökosystem und Integrationen">

## Einführung

Die Validierung ist ein kritischer Schritt im Machine-Learning-Prozess, der es Ihnen ermöglicht, die Qualität Ihrer trainierten Modelle zu bewerten. Der Val-Modus in Ultralytics YOLOv8 bietet eine robuste Suite von Tools und Metriken zur Bewertung der Leistung Ihrer Objekterkennungsmodelle. Dieser Leitfaden dient als umfassende Ressource, um zu verstehen, wie Sie den Val-Modus effektiv nutzen können, um sicherzustellen, dass Ihre Modelle sowohl genau als auch zuverlässig sind.

## Warum mit Ultralytics YOLO validieren?

Hier sind die Vorteile der Verwendung des Val-Modus von YOLOv8:

- **Präzision:** Erhalten Sie genaue Metriken wie mAP50, mAP75 und mAP50-95, um Ihr Modell umfassend zu bewerten.
- **Bequemlichkeit:** Nutzen Sie integrierte Funktionen, die Trainingseinstellungen speichern und so den Validierungsprozess vereinfachen.
- **Flexibilität:** Validieren Sie Ihr Modell mit den gleichen oder verschiedenen Datensätzen und Bildgrößen.
- **Hyperparameter-Tuning:** Verwenden Sie Validierungsmetriken, um Ihr Modell für eine bessere Leistung zu optimieren.

### Schlüsselfunktionen des Val-Modus

Dies sind die bemerkenswerten Funktionen, die der Val-Modus von YOLOv8 bietet:

- **Automatisierte Einstellungen:** Modelle erinnern sich an ihre Trainingskonfigurationen für eine unkomplizierte Validierung.
- **Unterstützung mehrerer Metriken:** Bewerten Sie Ihr Modell anhand einer Reihe von Genauigkeitsmetriken.
- **CLI- und Python-API:** Wählen Sie zwischen Befehlszeilenschnittstelle oder Python-API basierend auf Ihrer Präferenz für die Validierung.
- **Datenkompatibilität:** Funktioniert nahtlos mit Datensätzen, die während der Trainingsphase sowie mit benutzerdefinierten Datensätzen verwendet wurden.

!!! Tip "Tipp"

    * YOLOv8-Modelle speichern automatisch ihre Trainingseinstellungen, sodass Sie ein Modell mit der gleichen Bildgröße und dem ursprünglichen Datensatz leicht validieren können, indem Sie einfach `yolo val model=yolov8n.pt` oder `model('yolov8n.pt').val()` ausführen

## Beispielverwendung

Validieren Sie die Genauigkeit des trainierten YOLOv8n-Modells auf dem COCO128-Datensatz. Es muss kein Argument übergeben werden, da das `model` seine Trainings-`data` und Argumente als Modellattribute speichert. Siehe Abschnitt „Argumente“ unten für eine vollständige Liste der Exportargumente.

!!! Example "Beispiel"

    === "Python"

        ```python
        from ultralytics import YOLO

        # Modell laden
        model = YOLO('yolov8n.pt')  # ein offizielles Modell laden
        model = YOLO('path/to/best.pt')  # ein benutzerdefiniertes Modell laden

        # Modell validieren
        metrics = model.val()  # keine Argumente benötigt, Datensatz und Einstellungen gespeichert
        metrics.box.map    # map50-95
        metrics.box.map50  # map50
        metrics.box.map75  # map75
        metrics.box.maps   # eine Liste enthält map50-95 jeder Kategorie
        ```
    === "CLI"

        ```bash
        yolo detect val model=yolov8n.pt  # offizielles Modell validieren
        yolo detect val model=path/to/best.pt  # benutzerdefiniertes Modell validieren
        ```

## Argumente

Validierungseinstellungen für YOLO-Modelle beziehen sich auf verschiedene Hyperparameter und Konfigurationen, die verwendet werden, um die Leistung des Modells an einem Validierungsdatensatz zu bewerten. Diese Einstellungen können die Leistung, Geschwindigkeit und Genauigkeit des Modells beeinflussen. Einige gängige YOLO-Validierungseinstellungen umfassen die Batch-Größe, die Häufigkeit der Validierung während des Trainings und die Metriken zur Bewertung der Modellleistung. Andere Faktoren, die den Validierungsprozess beeinflussen können, sind die Größe und Zusammensetzung des Validierungsdatensatzes und die spezifische Aufgabe, für die das Modell verwendet wird. Es ist wichtig, diese Einstellungen sorgfältig abzustimmen und zu experimentieren, um sicherzustellen, dass das Modell auf dem Validierungsdatensatz gut funktioniert sowie Überanpassung zu erkennen und zu verhindern.

| Key           | Value   | Beschreibung                                                                    |
|---------------|---------|---------------------------------------------------------------------------------|
| `data`        | `None`  | Pfad zur Datendatei, z.B. coco128.yaml                                          |
| `imgsz`       | `640`   | Größe der Eingabebilder als ganzzahlige Zahl                                    |
| `batch`       | `16`    | Anzahl der Bilder pro Batch (-1 für AutoBatch)                                  |
| `save_json`   | `False` | Ergebnisse in JSON-Datei speichern                                              |
| `save_hybrid` | `False` | hybride Version der Labels speichern (Labels + zusätzliche Vorhersagen)         |
| `conf`        | `0.001` | Objekterkennungsschwelle für Zuversichtlichkeit                                 |
| `iou`         | `0.6`   | Schwellenwert für IoU (Intersection over Union) für NMS                         |
| `max_det`     | `300`   | maximale Anzahl an Vorhersagen pro Bild                                         |
| `half`        | `True`  | Halbpräzision verwenden (FP16)                                                  |
| `device`      | `None`  | Gerät zur Ausführung, z.B. CUDA device=0/1/2/3 oder device=cpu                  |
| `dnn`         | `False` | OpenCV DNN für ONNX-Inf erenz nutzen                                            |
| `plots`       | `False` | Diagramme während des Trainings anzeigen                                        |
| `rect`        | `False` | rechteckige Validierung mit jeder Batch-Charge für minimale Polsterung          |
| `split`       | `val`   | Zu verwendende Daten-Teilmenge für Validierung, z.B. 'val', 'test' oder 'train' |
|
