---
comments: true
description: Erfahren Sie mehr über YOLO-NAS, ein herausragendes Modell für die Objekterkennung. Erfahren Sie mehr über seine Funktionen, vortrainierte Modelle, Nutzung mit der Ultralytics Python API und vieles mehr.
keywords: YOLO-NAS, Deci AI, Objekterkennung, Deep Learning, Neural Architecture Search, Ultralytics Python API, YOLO-Modell, vortrainierte Modelle, Quantisierung, Optimierung, COCO, Objects365, Roboflow 100
---

# YOLO-NAS

## Übersicht

Entwickelt von Deci AI, ist YOLO-NAS ein bahnbrechendes Modell für die Objekterkennung. Es ist das Ergebnis fortschrittlicher Technologien zur Neural Architecture Search und wurde sorgfältig entworfen, um die Einschränkungen früherer YOLO-Modelle zu überwinden. Mit signifikanten Verbesserungen in der Quantisierungsunterstützung und Abwägung von Genauigkeit und Latenz stellt YOLO-NAS einen großen Fortschritt in der Objekterkennung dar.

![Modellbeispielbild](https://learnopencv.com/wp-content/uploads/2023/05/yolo-nas_COCO_map_metrics.png)
**Übersicht über YOLO-NAS.** YOLO-NAS verwendet Quantisierungsblöcke und selektive Quantisierung für optimale Leistung. Das Modell weist bei der Konvertierung in seine quantisierte Version mit INT8 einen minimalen Präzisionsverlust auf, was im Vergleich zu anderen Modellen eine signifikante Verbesserung darstellt. Diese Entwicklungen führen zu einer überlegenen Architektur mit beispiellosen Fähigkeiten zur Objekterkennung und herausragender Leistung.

### Schlüsselfunktionen

- **Quantisierungsfreundlicher Basiselement:** YOLO-NAS führt ein neues Basiselement ein, das für Quantisierung geeignet ist und eine der wesentlichen Einschränkungen früherer YOLO-Modelle angeht.
- **Raffiniertes Training und Quantisierung:** YOLO-NAS nutzt fortschrittliche Trainingsschemata und post-training Quantisierung zur Leistungsverbesserung.
- **AutoNAC-Optimierung und Vortraining:** YOLO-NAS verwendet die AutoNAC-Optimierung und wird auf prominenten Datensätzen wie COCO, Objects365 und Roboflow 100 vortrainiert. Dieses Vortraining macht es äußerst geeignet für die Objekterkennung in Produktionsumgebungen.

## Vortrainierte Modelle

Erleben Sie die Leistungsfähigkeit der Objekterkennung der nächsten Generation mit den vortrainierten YOLO-NAS-Modellen von Ultralytics. Diese Modelle sind darauf ausgelegt, sowohl bei Geschwindigkeit als auch bei Genauigkeit hervorragende Leistung zu liefern. Wählen Sie aus einer Vielzahl von Optionen, die auf Ihre spezifischen Anforderungen zugeschnitten sind:

| Modell           | mAP   | Latenz (ms) |
|------------------|-------|-------------|
| YOLO-NAS S       | 47,5  | 3,21        |
| YOLO-NAS M       | 51,55 | 5,85        |
| YOLO-NAS L       | 52,22 | 7,87        |
| YOLO-NAS S INT-8 | 47,03 | 2,36        |
| YOLO-NAS M INT-8 | 51,0  | 3,78        |
| YOLO-NAS L INT-8 | 52,1  | 4,78        |

Jede Modellvariante ist darauf ausgelegt, eine Balance zwischen Mean Average Precision (mAP) und Latenz zu bieten und Ihre Objekterkennungsaufgaben für Performance und Geschwindigkeit zu optimieren.

## Beispiele zur Verwendung

Ultralytics hat es einfach gemacht, YOLO-NAS-Modelle in Ihre Python-Anwendungen über unser `ultralytics` Python-Paket zu integrieren. Das Paket bietet eine benutzerfreundliche Python-API, um den Prozess zu optimieren.

Die folgenden Beispiele zeigen, wie Sie YOLO-NAS-Modelle mit dem `ultralytics`-Paket für Inferenz und Validierung verwenden:

### Beispiele für Inferenz und Validierung

In diesem Beispiel validieren wir YOLO-NAS-s auf dem COCO8-Datensatz.

!!! Example "Beispiel"

    Dieses Beispiel bietet einfachen Code für Inferenz und Validierung für YOLO-NAS. Für die Verarbeitung von Inferenzergebnissen siehe den [Predict](../modes/predict.md)-Modus. Für die Verwendung von YOLO-NAS mit zusätzlichen Modi siehe [Val](../modes/val.md) und [Export](../modes/export.md). Das YOLO-NAS-Modell im `ultralytics`-Paket unterstützt kein Training.

    === "Python"

        Vorab trainierte `*.pt`-Modelldateien von PyTorch können der Klasse `NAS()` übergeben werden, um eine Modellinstanz in Python zu erstellen:

        ```python
        from ultralytics import NAS

        # Laden Sie ein auf COCO vortrainiertes YOLO-NAS-s-Modell
        model = NAS('yolo_nas_s.pt')

        # Modelinformationen anzeigen (optional)
        model.info()

        # Validieren Sie das Modell am Beispiel des COCO8-Datensatzes
        results = model.val(data='coco8.yaml')

        # Führen Sie Inferenz mit dem YOLO-NAS-s-Modell auf dem Bild 'bus.jpg' aus
        results = model('path/to/bus.jpg')
        ```

    === "CLI"

        CLI-Befehle sind verfügbar, um die Modelle direkt auszuführen:

        ```bash
        # Laden Sie ein auf COCO vortrainiertes YOLO-NAS-s-Modell und validieren Sie die Leistung am Beispiel des COCO8-Datensatzes
        yolo val model=yolo_nas_s.pt data=coco8.yaml

        # Laden Sie ein auf COCO vortrainiertes YOLO-NAS-s-Modell und führen Sie Inferenz auf dem Bild 'bus.jpg' aus
        yolo predict model=yolo_nas_s.pt source=path/to/bus.jpg
        ```

## Unterstützte Aufgaben und Modi

Wir bieten drei Varianten der YOLO-NAS-Modelle an: Small (s), Medium (m) und Large (l). Jede Variante ist dazu gedacht, unterschiedliche Berechnungs- und Leistungsanforderungen zu erfüllen:

- **YOLO-NAS-s**: Optimiert für Umgebungen mit begrenzten Rechenressourcen, bei denen Effizienz entscheidend ist.
- **YOLO-NAS-m**: Bietet einen ausgewogenen Ansatz und ist für die Objekterkennung im Allgemeinen mit höherer Genauigkeit geeignet.
- **YOLO-NAS-l**: Maßgeschneidert für Szenarien, bei denen höchste Genauigkeit gefordert ist und Rechenressourcen weniger einschränkend sind.

Im Folgenden finden Sie eine detaillierte Übersicht über jedes Modell, einschließlich Links zu den vortrainierten Gewichten, den unterstützten Aufgaben und deren Kompatibilität mit verschiedenen Betriebsmodi.

| Modelltyp  | Vortrainierte Gewichte                                                                        | Unterstützte Aufgaben                 | Inferenz | Validierung | Training | Export |
|------------|-----------------------------------------------------------------------------------------------|---------------------------------------|----------|-------------|----------|--------|
| YOLO-NAS-s | [yolo_nas_s.pt](https://github.com/ultralytics/assets/releases/download/v0.0.0/yolo_nas_s.pt) | [Objekterkennung](../tasks/detect.md) | ✅        | ✅           | ❌        | ✅      |
| YOLO-NAS-m | [yolo_nas_m.pt](https://github.com/ultralytics/assets/releases/download/v0.0.0/yolo_nas_m.pt) | [Objekterkennung](../tasks/detect.md) | ✅        | ✅           | ❌        | ✅      |
| YOLO-NAS-l | [yolo_nas_l.pt](https://github.com/ultralytics/assets/releases/download/v0.0.0/yolo_nas_l.pt) | [Objekterkennung](../tasks/detect.md) | ✅        | ✅           | ❌        | ✅      |

## Zitierungen und Danksagungen

Wenn Sie YOLO-NAS in Ihrer Forschungs- oder Entwicklungsarbeit verwenden, zitieren Sie bitte SuperGradients:

!!! Quote ""

    === "BibTeX"

        ```bibtex
        @misc{supergradients,
              doi = {10.5281/ZENODO.7789328},
              url = {https://zenodo.org/record/7789328},
              author = {Aharon,  Shay and {Louis-Dupont} and {Ofri Masad} and Yurkova,  Kate and {Lotem Fridman} and {Lkdci} and Khvedchenya,  Eugene and Rubin,  Ran and Bagrov,  Natan and Tymchenko,  Borys and Keren,  Tomer and Zhilko,  Alexander and {Eran-Deci}},
              title = {Super-Gradients},
              publisher = {GitHub},
              journal = {GitHub repository},
              year = {2021},
        }
        ```

Wir möchten dem [SuperGradients](https://github.com/Deci-AI/super-gradients/)-Team von Deci AI für ihre Bemühungen bei der Erstellung und Pflege dieser wertvollen Ressource für die Computer Vision Community danken. Wir sind der Meinung, dass YOLO-NAS mit seiner innovativen Architektur und seinen herausragenden Fähigkeiten zur Objekterkennung ein wichtiges Werkzeug für Entwickler und Forscher gleichermaßen wird.

*Keywords: YOLO-NAS, Deci AI, Objekterkennung, Deep Learning, Neural Architecture Search, Ultralytics Python API, YOLO-Modell, SuperGradients, vortrainierte Modelle, quantisierungsfreundliches Basiselement, fortschrittliche Trainingsschemata, post-training Quantisierung, AutoNAC-Optimierung, COCO, Objects365, Roboflow 100*
