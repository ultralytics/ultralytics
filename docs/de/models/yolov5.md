---
comments: true
description: Entdecken Sie YOLOv5u, eine verbesserte Version des YOLOv5-Modells mit einem optimierten Verhältnis von Genauigkeit und Geschwindigkeit sowie zahlreiche vorab trainierte Modelle für verschiedene Objekterkennungsaufgaben.
keywords: YOLOv5u, Objekterkennung, vorab trainierte Modelle, Ultralytics, Inferenz, Validierung, YOLOv5, YOLOv8, Ankerfrei, Objektlos, Echtzeitanwendungen, Maschinelles Lernen
---

# YOLOv5

## Übersicht

YOLOv5u steht für eine Weiterentwicklung der Methoden zur Objekterkennung. Basierend auf der grundlegenden Architektur des von Ultralytics entwickelten YOLOv5-Modells integriert YOLOv5u den ankerfreien, objektlosen Split-Kopf, ein Feature, das zuvor in den YOLOv8-Modellen eingeführt wurde. Diese Anpassung verfeinert die Architektur des Modells und führt zu einem optimierten Verhältnis von Genauigkeit und Geschwindigkeit bei der Objekterkennung. Basierend auf den empirischen Ergebnissen und den abgeleiteten Features bietet YOLOv5u eine effiziente Alternative für diejenigen, die robuste Lösungen sowohl in der Forschung als auch in praktischen Anwendungen suchen.

![Ultralytics YOLOv5](https://raw.githubusercontent.com/ultralytics/assets/main/yolov5/v70/splash.png)

## Hauptmerkmale

- **Ankerfreier Split-Ultralytics-Kopf:** Herkömmliche Objekterkennungsmodelle verwenden vordefinierte Ankerboxen, um die Position von Objekten vorherzusagen. YOLOv5u modernisiert diesen Ansatz. Durch die Verwendung eines ankerfreien Split-Ultralytics-Kopfes wird ein flexiblerer und anpassungsfähigerer Detektionsmechanismus gewährleistet, der die Leistung in verschiedenen Szenarien verbessert.

- **Optimiertes Verhältnis von Genauigkeit und Geschwindigkeit:** Geschwindigkeit und Genauigkeit ziehen oft in entgegengesetzte Richtungen. Aber YOLOv5u stellt diese Abwägung in Frage. Es bietet eine ausgewogene Balance, die Echtzeitdetektionen ohne Einbußen bei der Genauigkeit ermöglicht. Diese Funktion ist besonders wertvoll für Anwendungen, die schnelle Reaktionen erfordern, wie autonome Fahrzeuge, Robotik und Echtzeitanalyse von Videos.

- **Vielfalt an vorab trainierten Modellen:** YOLOv5u bietet eine Vielzahl von vorab trainierten Modellen, da verschiedene Aufgaben unterschiedliche Werkzeuge erfordern. Ob Sie sich auf Inferenz, Validierung oder Training konzentrieren, es wartet ein maßgeschneidertes Modell auf Sie. Diese Vielfalt gewährleistet, dass Sie nicht nur eine Einheitslösung verwenden, sondern ein speziell für Ihre einzigartige Herausforderung feinabgestimmtes Modell.

## Unterstützte Aufgaben und Modi

Die YOLOv5u-Modelle mit verschiedenen vorab trainierten Gewichten eignen sich hervorragend für Aufgaben zur [Objekterkennung](../tasks/detect.md). Sie unterstützen eine umfassende Palette von Modi, die sie für verschiedene Anwendungen von der Entwicklung bis zur Bereitstellung geeignet machen.

| Modelltyp | Vorab trainierte Gewichte                                                                                                   | Aufgabe                               | Inferenz | Validierung | Training | Export |
|-----------|-----------------------------------------------------------------------------------------------------------------------------|---------------------------------------|----------|-------------|----------|--------|
| YOLOv5u   | `yolov5nu`, `yolov5su`, `yolov5mu`, `yolov5lu`, `yolov5xu`, `yolov5n6u`, `yolov5s6u`, `yolov5m6u`, `yolov5l6u`, `yolov5x6u` | [Objekterkennung](../tasks/detect.md) | ✅        | ✅           | ✅        | ✅      |

Diese Tabelle bietet eine detaillierte Übersicht über die verschiedenen Varianten des YOLOv5u-Modells und hebt ihre Anwendbarkeit in der Objekterkennung sowie die Unterstützung unterschiedlicher Betriebsmodi wie [Inferenz](../modes/predict.md), [Validierung](../modes/val.md), [Training](../modes/train.md) und [Export](../modes/export.md) hervor. Diese umfassende Unterstützung ermöglicht es Benutzern, die Fähigkeiten der YOLOv5u-Modelle in einer Vielzahl von Objekterkennungsszenarien voll auszuschöpfen.

## Leistungskennzahlen

!!! Leistung

    === "Erkennung"

    Siehe [Erkennungsdokumentation](https://docs.ultralytics.com/tasks/detect/) für Beispiele zur Verwendung dieser Modelle, die auf [COCO](https://docs.ultralytics.com/datasets/detect/coco/) trainiert wurden und 80 vorab trainierte Klassen enthalten.

    | Modell                                                                                       | YAML                                                                                                           | Größe<br><sup>(Pixel) | mAP<sup>val<br>50-95 | Geschwindigkeit<br><sup>CPU ONNX<br>(ms) | Geschwindigkeit<br><sup>A100 TensorRT<br>(ms) | Parameter<br><sup>(M) | FLOPs<br><sup>(B) |
    |---------------------------------------------------------------------------------------------|----------------------------------------------------------------------------------------------------------------|-----------------------|----------------------|------------------------------------------|-----------------------------------------------|--------------------|-------------------|
    | [yolov5nu.pt](https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov5nu.pt)   | [yolov5n.yaml](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/models/v5/yolov5.yaml)     | 640                   | 34,3                 | 73,6                                     | 1,06                                          | 2,6                  | 7,7               |
    | [yolov5su.pt](https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov5su.pt)   | [yolov5s.yaml](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/models/v5/yolov5.yaml)     | 640                   | 43,0                 | 120,7                                    | 1,27                                          | 9,1                  | 24,0              |
    | [yolov5mu.pt](https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov5mu.pt)   | [yolov5m.yaml](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/models/v5/yolov5.yaml)     | 640                   | 49,0                 | 233,9                                    | 1,86                                          | 25,1                 | 64,2              |
    | [yolov5lu.pt](https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov5lu.pt)   | [yolov5l.yaml](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/models/v5/yolov5.yaml)     | 640                   | 52,2                 | 408,4                                    | 2,50                                          | 53,2                 | 135,0             |
    | [yolov5xu.pt](https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov5xu.pt)   | [yolov5x.yaml](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/models/v5/yolov5.yaml)     | 640                   | 53,2                 | 763,2                                    | 3,81                                          | 97,2                 | 246,4             |
    |                                                                                             |                                                                                                                |                       |                      |                                          |                                               |                    |                   |
    | [yolov5n6u.pt](https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov5n6u.pt) | [yolov5n6.yaml](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/models/v5/yolov5-p6.yaml) | 1.280                 | 42,1                 | 211,0                                    | 1,83                                          | 4,3                  | 7,8               |
    | [yolov5s6u.pt](https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov5s6u.pt) | [yolov5s6.yaml](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/models/v5/yolov5-p6.yaml) | 1.280                 | 48,6                 | 422,6                                    | 2,34                                          | 15,3                 | 24,6              |
    | [yolov5m6u.pt](https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov5m6u.pt) | [yolov5m6.yaml](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/models/v5/yolov5-p6.yaml) | 1.280                 | 53,6                 | 810,9                                    | 4,36                                          | 41,2                 | 65,7              |
    | [yolov5l6u.pt](https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov5l6u.pt) | [yolov5l6.yaml](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/models/v5/yolov5-p6.yaml) | 1.280                 | 55,7                 | 1.470,9                                  | 5,47                                          | 86,1                 | 137,4             |
    | [yolov5x6u.pt](https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov5x6u.pt) | [yolov5x6.yaml](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/models/v5/yolov5-p6.yaml) | 1.280                 | 56,8                 | 2.436,5                                  | 8,98                                          | 155,4                | 250,7             |

## Beispiele zur Verwendung

Dieses Beispiel enthält einfache Beispiele zur Schulung und Inferenz mit YOLOv5. Die vollständige Dokumentation zu diesen und anderen [Modi](../modes/index.md) finden Sie in den Seiten [Predict](../modes/predict.md), [Train](../modes/train.md), [Val](../modes/val.md) und [Export](../modes/export.md).

!!! Example "Beispiel"

    === "Python"

        PyTorch-vortrainierte `*.pt`-Modelle sowie Konfigurationsdateien `*.yaml` können an die `YOLO()`-Klasse übergeben werden, um eine Modellinstanz in Python zu erstellen:

        ```python
        from ultralytics import YOLO

        # Laden Sie ein vortrainiertes YOLOv5n-Modell für COCO-Daten
        modell = YOLO('yolov5n.pt')

        # Informationen zum Modell anzeigen (optional)
        model.info()

        # Trainieren Sie das Modell anhand des COCO8-Beispieldatensatzes für 100 Epochen
        ergebnisse = model.train(data='coco8.yaml', epochs=100, imgsz=640)

        # Führen Sie die Inferenz mit dem YOLOv5n-Modell auf dem Bild 'bus.jpg' durch
        ergebnisse = model('path/to/bus.jpg')
        ```

    === "CLI"

        CLI-Befehle sind verfügbar, um die Modelle direkt auszuführen:

        ```bash
        # Laden Sie ein vortrainiertes YOLOv5n-Modell und trainieren Sie es anhand des COCO8-Beispieldatensatzes für 100 Epochen
        yolo train model=yolov5n.pt data=coco8.yaml epochs=100 imgsz=640

        # Laden Sie ein vortrainiertes YOLOv5n-Modell und führen Sie die Inferenz auf dem Bild 'bus.jpg' durch
        yolo predict model=yolov5n.pt source=path/to/bus.jpg
        ```

## Zitate und Danksagungen

Wenn Sie YOLOv5 oder YOLOv5u in Ihrer Forschung verwenden, zitieren Sie bitte das Ultralytics YOLOv5-Repository wie folgt:

!!! Quote ""

    === "BibTeX"
        ```bibtex
        @software{yolov5,
          title = {Ultralytics YOLOv5},
          author = {Glenn Jocher},
          year = {2020},
          version = {7.0},
          license = {AGPL-3.0},
          url = {https://github.com/ultralytics/yolov5},
          doi = {10.5281/zenodo.3908559},
          orcid = {0000-0001-5950-6979}
        }
        ```

Bitte beachten Sie, dass die YOLOv5-Modelle unter den Lizenzen [AGPL-3.0](https://github.com/ultralytics/ultralytics/blob/main/LICENSE) und [Enterprise](https://ultralytics.com/license) bereitgestellt werden.
