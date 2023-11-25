---
comments: true
description: Vom Training bis zum Tracking - Nutzen Sie YOLOv8 von Ultralytics optimal. Erhalten Sie Einblicke und Beispiele für jeden unterstützten Modus, einschließlich Validierung, Export und Benchmarking.
keywords: Ultralytics, YOLOv8, Maschinelles Lernen, Objekterkennung, Training, Validierung, Vorhersage, Export, Tracking, Benchmarking
---

# Ultralytics YOLOv8 Modi

<img width="1024" src="https://github.com/ultralytics/assets/raw/main/yolov8/banner-integrations.png" alt="Ultralytics YOLO-Ökosystem und Integrationen">

## Einführung

Ultralytics YOLOv8 ist nicht nur ein weiteres Objekterkennungsmodell; es ist ein vielseitiges Framework, das den gesamten Lebenszyklus von Machine-Learning-Modellen abdeckt - von der Dateneingabe und dem Modelltraining über die Validierung und Bereitstellung bis hin zum Tracking in der realen Welt. Jeder Modus dient einem bestimmten Zweck und ist darauf ausgelegt, Ihnen die Flexibilität und Effizienz zu bieten, die für verschiedene Aufgaben und Anwendungsfälle erforderlich ist.

<p align="center">
  <br>
  <iframe width="720" height="405" src="https://www.youtube.com/embed/j8uQc0qB91s?si=dhnGKgqvs7nPgeaM"
    title="YouTube-Videoplayer" frameborder="0"
    allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share"
    allowfullscreen>
  </iframe>
  <br>
  <strong>Anschauen:</strong> Ultralytics Modi Tutorial: Trainieren, Validieren, Vorhersagen, Exportieren & Benchmarking.
</p>

### Modi im Überblick

Das Verständnis der verschiedenen **Modi**, die Ultralytics YOLOv8 unterstützt, ist entscheidend, um das Beste aus Ihren Modellen herauszuholen:

- **Train**-Modus: Verfeinern Sie Ihr Modell mit angepassten oder vorgeladenen Datensätzen.
- **Val**-Modus: Eine Nachtrainingsprüfung zur Validierung der Modellleistung.
- **Predict**-Modus: Entfesseln Sie die Vorhersagekraft Ihres Modells mit realen Daten.
- **Export**-Modus: Machen Sie Ihr Modell in verschiedenen Formaten einsatzbereit.
- **Track**-Modus: Erweitern Sie Ihr Objekterkennungsmodell um Echtzeit-Tracking-Anwendungen.
- **Benchmark**-Modus: Analysieren Sie die Geschwindigkeit und Genauigkeit Ihres Modells in verschiedenen Einsatzumgebungen.

Dieser umfassende Leitfaden soll Ihnen einen Überblick und praktische Einblicke in jeden Modus geben, um Ihnen zu helfen, das volle Potenzial von YOLOv8 zu nutzen.

## [Trainieren](train.md)

Der Trainingsmodus wird verwendet, um ein YOLOv8-Modell mit einem angepassten Datensatz zu trainieren. In diesem Modus wird das Modell mit dem angegebenen Datensatz und den Hyperparametern trainiert. Der Trainingsprozess beinhaltet die Optimierung der Modellparameter, damit es die Klassen und Standorte von Objekten in einem Bild genau vorhersagen kann.

[Trainingsbeispiele](train.md){ .md-button }

## [Validieren](val.md)

Der Validierungsmodus wird genutzt, um ein YOLOv8-Modell nach dem Training zu bewerten. In diesem Modus wird das Modell auf einem Validierungsset getestet, um seine Genauigkeit und Generalisierungsleistung zu messen. Dieser Modus kann verwendet werden, um die Hyperparameter des Modells für eine bessere Leistung zu optimieren.

[Validierungsbeispiele](val.md){ .md-button }

## [Vorhersagen](predict.md)

Der Vorhersagemodus wird verwendet, um mit einem trainierten YOLOv8-Modell Vorhersagen für neue Bilder oder Videos zu treffen. In diesem Modus wird das Modell aus einer Checkpoint-Datei geladen, und der Benutzer kann Bilder oder Videos zur Inferenz bereitstellen. Das Modell sagt die Klassen und Standorte von Objekten in den Eingabebildern oder -videos voraus.

[Vorhersagebeispiele](predict.md){ .md-button }

## [Exportieren](export.md)

Der Exportmodus wird verwendet, um ein YOLOv8-Modell in ein Format zu exportieren, das für die Bereitstellung verwendet werden kann. In diesem Modus wird das Modell in ein Format konvertiert, das von anderen Softwareanwendungen oder Hardwaregeräten verwendet werden kann. Dieser Modus ist nützlich, wenn das Modell in Produktionsumgebungen eingesetzt wird.

[Exportbeispiele](export.md){ .md-button }

## [Verfolgen](track.md)

Der Trackingmodus wird zur Echtzeitverfolgung von Objekten mit einem YOLOv8-Modell verwendet. In diesem Modus wird das Modell aus einer Checkpoint-Datei geladen, und der Benutzer kann einen Live-Videostream für das Echtzeitobjekttracking bereitstellen. Dieser Modus ist nützlich für Anwendungen wie Überwachungssysteme oder selbstfahrende Autos.

[Trackingbeispiele](track.md){ .md-button }

## [Benchmarking](benchmark.md)

Der Benchmark-Modus wird verwendet, um die Geschwindigkeit und Genauigkeit verschiedener Exportformate für YOLOv8 zu profilieren. Die Benchmarks liefern Informationen über die Größe des exportierten Formats, seine `mAP50-95`-Metriken (für Objekterkennung, Segmentierung und Pose)
oder `accuracy_top5`-Metriken (für Klassifizierung) und die Inferenzzeit in Millisekunden pro Bild für verschiedene Exportformate wie ONNX, OpenVINO, TensorRT und andere. Diese Informationen können den Benutzern dabei helfen, das optimale Exportformat für ihren spezifischen Anwendungsfall basierend auf ihren Anforderungen an Geschwindigkeit und Genauigkeit auszuwählen.

[Benchmarkbeispiele](benchmark.md){ .md-button }
