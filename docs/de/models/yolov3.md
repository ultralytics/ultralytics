---
comments: true
description: Erhalten Sie eine Übersicht über YOLOv3, YOLOv3-Ultralytics und YOLOv3u. Erfahren Sie mehr über ihre wichtigsten Funktionen, Verwendung und unterstützte Aufgaben für die Objekterkennung.
keywords: YOLOv3, YOLOv3-Ultralytics, YOLOv3u, Objekterkennung, Inferenz, Training, Ultralytics
---

# YOLOv3, YOLOv3-Ultralytics und YOLOv3u

## Übersicht

Dieses Dokument bietet eine Übersicht über drei eng verwandte Modelle zur Objekterkennung, nämlich [YOLOv3](https://pjreddie.com/darknet/yolo/), [YOLOv3-Ultralytics](https://github.com/ultralytics/yolov3) und [YOLOv3u](https://github.com/ultralytics/ultralytics).

1. **YOLOv3:** Dies ist die dritte Version des You Only Look Once (YOLO) Objekterkennungsalgorithmus. Ursprünglich entwickelt von Joseph Redmon, verbesserte YOLOv3 seine Vorgängermodelle durch die Einführung von Funktionen wie mehrskaligen Vorhersagen und drei verschiedenen Größen von Erkennungskernen.

2. **YOLOv3-Ultralytics:** Dies ist die Implementierung des YOLOv3-Modells von Ultralytics. Es reproduziert die ursprüngliche YOLOv3-Architektur und bietet zusätzliche Funktionalitäten, wie die Unterstützung für weitere vortrainierte Modelle und einfachere Anpassungsoptionen.

3. **YOLOv3u:** Dies ist eine aktualisierte Version von YOLOv3-Ultralytics, die den anchor-freien, objektfreien Split Head aus den YOLOv8-Modellen einbezieht. YOLOv3u verwendet die gleiche Backbone- und Neck-Architektur wie YOLOv3, aber mit dem aktualisierten Erkennungskopf von YOLOv8.

![Ultralytics YOLOv3](https://raw.githubusercontent.com/ultralytics/assets/main/yolov3/banner-yolov3.png)

## Wichtigste Funktionen

- **YOLOv3:** Einführung der Verwendung von drei unterschiedlichen Skalen für die Erkennung unter Verwendung von drei verschiedenen Größen von Erkennungskernen: 13x13, 26x26 und 52x52. Dadurch wurde die Erkennungsgenauigkeit für Objekte unterschiedlicher Größe erheblich verbessert. Darüber hinaus fügte YOLOv3 Funktionen wie Mehrfachkennzeichnungen für jeden Begrenzungsrahmen und ein besseres Feature-Extraktionsnetzwerk hinzu.

- **YOLOv3-Ultralytics:** Ultralytics' Implementierung von YOLOv3 bietet die gleiche Leistung wie das ursprüngliche Modell, bietet jedoch zusätzliche Unterstützung für weitere vortrainierte Modelle, zusätzliche Trainingsmethoden und einfachere Anpassungsoptionen. Dadurch wird es vielseitiger und benutzerfreundlicher für praktische Anwendungen.

- **YOLOv3u:** Dieses aktualisierte Modell enthält den anchor-freien, objektfreien Split Head aus YOLOv8. Durch die Beseitigung der Notwendigkeit vordefinierter Ankerfelder und Objektheitsscores kann dieses Entwurfsmerkmal für den Erkennungskopf die Fähigkeit des Modells verbessern, Objekte unterschiedlicher Größe und Form zu erkennen. Dadurch wird YOLOv3u robuster und genauer für Aufgaben der Objekterkennung.

## Unterstützte Aufgaben und Modi

Die YOLOv3-Serie, einschließlich YOLOv3, YOLOv3-Ultralytics und YOLOv3u, ist speziell für Aufgaben der Objekterkennung konzipiert. Diese Modelle sind bekannt für ihre Effektivität in verschiedenen realen Szenarien und kombinieren Genauigkeit und Geschwindigkeit. Jede Variante bietet einzigartige Funktionen und Optimierungen, die sie für eine Vielzahl von Anwendungen geeignet machen.

Alle drei Modelle unterstützen einen umfangreichen Satz von Modi, um Vielseitigkeit in verschiedenen Phasen der Modellbereitstellung und -entwicklung zu gewährleisten. Zu diesen Modi gehören [Inferenz](../modes/predict.md), [Validierung](../modes/val.md), [Training](../modes/train.md) und [Export](../modes/export.md), was den Benutzern ein vollständiges Toolkit für eine effektive Objekterkennung bietet.

| Modelltyp          | Unterstützte Aufgaben                 | Inferenz | Validierung | Training | Export |
|--------------------|---------------------------------------|----------|-------------|----------|--------|
| YOLOv3             | [Objekterkennung](../tasks/detect.md) | ✅        | ✅           | ✅        | ✅      |
| YOLOv3-Ultralytics | [Objekterkennung](../tasks/detect.md) | ✅        | ✅           | ✅        | ✅      |
| YOLOv3u            | [Objekterkennung](../tasks/detect.md) | ✅        | ✅           | ✅        | ✅      |

Diese Tabelle bietet einen schnellen Überblick über die Fähigkeiten jeder YOLOv3-Variante und hebt ihre Vielseitigkeit und Eignung für verschiedene Aufgaben und Betriebsmodi in Workflows zur Objekterkennung hervor.

## Beispiele zur Verwendung

Dieses Beispiel enthält einfache Trainings- und Inferenzbeispiele für YOLOv3. Für die vollständige Dokumentation zu diesen und anderen [Modi](../modes/index.md) siehe die Seiten zur [Predict](../modes/predict.md), [Train](../modes/train.md), [Val](../modes/val.md) und [Export](../modes/export.md).

!!! Example "Beispiel"

    === "Python"

        Vorgefertigte PyTorch-Modelle im `*.pt`-Format sowie Konfigurationsdateien im `*.yaml`-Format können an die `YOLO()`-Klasse übergeben werden, um eine Modellinstanz in Python zu erstellen:

        ```python
        from ultralytics import YOLO

        # Lade ein vortrainiertes YOLOv3n-Modell für COCO
        model = YOLO('yolov3n.pt')

        # Zeige Informationen zum Modell an (optional)
        model.info()

        # Trainiere das Modell mit dem COCO8-Beispieldatensatz für 100 Epochen
        results = model.train(data='coco8.yaml', epochs=100, imgsz=640)

        # Führe Inferenz mit dem YOLOv3n-Modell auf dem Bild "bus.jpg" durch
        results = model('path/to/bus.jpg')
        ```

    === "CLI"

        CLI-Befehle stehen zur Verfügung, um die Modelle direkt auszuführen:

        ```bash
        # Lade ein vortrainiertes YOLOv3n-Modell und trainiere es mit dem COCO8-Beispieldatensatz für 100 Epochen
        yolo train model=yolov3n.pt data=coco8.yaml epochs=100 imgsz=640

        # Lade ein vortrainiertes YOLOv3n-Modell und führe Inferenz auf dem Bild "bus.jpg" aus
        yolo predict model=yolov3n.pt source=path/to/bus.jpg
        ```

## Zitate und Anerkennungen

Wenn Sie YOLOv3 in Ihrer Forschung verwenden, zitieren Sie bitte die ursprünglichen YOLO-Papiere und das Ultralytics YOLOv3-Repository:

!!! Quote ""

    === "BibTeX"

        ```bibtex
        @article{redmon2018yolov3,
          title={YOLOv3: An Incremental Improvement},
          author={Redmon, Joseph and Farhadi, Ali},
          journal={arXiv preprint arXiv:1804.02767},
          year={2018}
        }
        ```

Vielen Dank an Joseph Redmon und Ali Farhadi für die Entwicklung des originalen YOLOv3.
