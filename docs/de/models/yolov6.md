---
comments: true
description: Erforschen Sie Meituan YOLOv6, ein modernes Objekterkennungsmodell, das eine ausgewogene Kombination aus Geschwindigkeit und Genauigkeit bietet. Tauchen Sie ein in Funktionen, vorab trainierte Modelle und die Verwendung von Python.
keywords: Meituan YOLOv6, Objekterkennung, Ultralytics, YOLOv6 Dokumentation, Bi-direktionale Konkatenation, Anchor-Aided Training, vorab trainierte Modelle, Echtzeitanwendungen
---

# Meituan YOLOv6

## Überblick

[Meituan](https://about.meituan.com/) YOLOv6 ist ein moderner Objekterkenner, der eine bemerkenswerte Balance zwischen Geschwindigkeit und Genauigkeit bietet und somit eine beliebte Wahl für Echtzeitanwendungen darstellt. Dieses Modell bietet mehrere bemerkenswerte Verbesserungen in seiner Architektur und seinem Trainingsschema, einschließlich der Implementierung eines Bi-direktionalen Konkatenationsmoduls (BiC), einer anchor-aided training (AAT)-Strategie und einem verbesserten Backpropagation- und Neck-Design für Spitzenleistungen auf dem COCO-Datensatz.

![Meituan YOLOv6](https://user-images.githubusercontent.com/26833433/240750495-4da954ce-8b3b-41c4-8afd-ddb74361d3c2.png)
![Modellbeispielbild](https://user-images.githubusercontent.com/26833433/240750557-3e9ec4f0-0598-49a8-83ea-f33c91eb6d68.png)
**Übersicht über YOLOv6.** Diagramm der Modellarchitektur, das die neu gestalteten Netzwerkkomponenten und Trainingstrategien zeigt, die zu signifikanten Leistungsverbesserungen geführt haben. (a) Der Nacken von YOLOv6 (N und S sind dargestellt). Beachten Sie, dass bei M/L RepBlocks durch CSPStackRep ersetzt wird. (b) Die Struktur eines BiC-Moduls. (c) Ein SimCSPSPPF-Block. ([Quelle](https://arxiv.org/pdf/2301.05586.pdf)).

### Hauptmerkmale

- **Bi-direktionales Konkatenations (BiC) Modul:** YOLOv6 führt ein BiC-Modul im Nacken des Erkenners ein, das die Lokalisierungssignale verbessert und eine Leistungssteigerung bei vernachlässigbarem Geschwindigkeitsabfall liefert.
- **Anchor-aided Training (AAT) Strategie:** Dieses Modell schlägt AAT vor, um die Vorteile sowohl von ankerbasierten als auch von ankerfreien Paradigmen zu nutzen, ohne die Inferenzeffizienz zu beeinträchtigen.
- **Verbessertes Backpropagation- und Neck-Design:** Durch Vertiefung von YOLOv6 um eine weitere Stufe im Backpropagation und Nacken erreicht dieses Modell Spitzenleistungen auf dem COCO-Datensatz bei hochauflösenden Eingaben.
- **Self-Distillation Strategie:** Eine neue Self-Distillation-Strategie wird implementiert, um die Leistung von kleineren Modellen von YOLOv6 zu steigern, indem der Hilfsregressionszweig während des Trainings verstärkt und bei der Inferenz entfernt wird, um einen deutlichen Geschwindigkeitsabfall zu vermeiden.

## Leistungsmetriken

YOLOv6 bietet verschiedene vorab trainierte Modelle mit unterschiedlichen Maßstäben:

- YOLOv6-N: 37,5% AP auf COCO val2017 bei 1187 FPS mit NVIDIA Tesla T4 GPU.
- YOLOv6-S: 45,0% AP bei 484 FPS.
- YOLOv6-M: 50,0% AP bei 226 FPS.
- YOLOv6-L: 52,8% AP bei 116 FPS.
- YOLOv6-L6: Spitzenleistung in Echtzeit.

YOLOv6 bietet auch quantisierte Modelle für verschiedene Genauigkeiten sowie Modelle, die für mobile Plattformen optimiert sind.

## Beispiele zur Verwendung

In diesem Beispiel werden einfache Schulungs- und Inferenzbeispiele für YOLOv6 bereitgestellt. Weitere Dokumentation zu diesen und anderen [Modi](../modes/index.md) finden Sie auf den Seiten [Predict](../modes/predict.md), [Train](../modes/train.md), [Val](../modes/val.md) und [Export](../modes/export.md).

!!! Example "Beispiel"

    === "Python"

        In Python kann PyTorch-vorab trainierte `*.pt`-Modelle sowie Konfigurations-`*.yaml`-Dateien an die `YOLO()`-Klasse übergeben werden, um eine Modellinstanz zu erstellen:

        ```python
        from ultralytics import YOLO

        # Erstellen Sie ein YOLOv6n-Modell von Grund auf
        model = YOLO('yolov6n.yaml')

        # Zeigen Sie Informationen zum Modell an (optional)
        model.info()

        # Trainieren Sie das Modell am Beispiel des COCO8-Datensatzes für 100 Epochen
        results = model.train(data='coco8.yaml', epochs=100, imgsz=640)

        # Führen Sie Inferenz mit dem YOLOv6n-Modell auf dem Bild 'bus.jpg' durch
        results = model('path/to/bus.jpg')
        ```

    === "CLI"

        CLI-Befehle stehen zur Verfügung, um die Modelle direkt auszuführen:

        ```bash
        # Erstellen Sie ein YOLOv6n-Modell von Grund auf und trainieren Sie es am Beispiel des COCO8-Datensatzes für 100 Epochen
        yolo train model=yolov6n.yaml data=coco8.yaml epochs=100 imgsz=640

        # Erstellen Sie ein YOLOv6n-Modell von Grund auf und führen Sie Inferenz auf dem Bild 'bus.jpg' durch
        yolo predict model=yolov6n.yaml source=path/to/bus.jpg
        ```

## Unterstützte Aufgaben und Modi

Die YOLOv6-Serie bietet eine Reihe von Modellen, die jeweils für die Hochleistungs-[Objekterkennung](../tasks/detect.md) optimiert sind. Diese Modelle erfüllen unterschiedliche Rechenanforderungen und Genauigkeitsanforderungen und sind daher vielseitig für eine Vielzahl von Anwendungen einsetzbar.

| Modelltyp | Vorab trainierte Gewichte | Unterstützte Aufgaben                 | Inferenz | Validierung | Training | Exportieren |
|-----------|---------------------------|---------------------------------------|----------|-------------|----------|-------------|
| YOLOv6-N  | `yolov6-n.pt`             | [Objekterkennung](../tasks/detect.md) | ✅        | ✅           | ✅        | ✅           |
| YOLOv6-S  | `yolov6-s.pt`             | [Objekterkennung](../tasks/detect.md) | ✅        | ✅           | ✅        | ✅           |
| YOLOv6-M  | `yolov6-m.pt`             | [Objekterkennung](../tasks/detect.md) | ✅        | ✅           | ✅        | ✅           |
| YOLOv6-L  | `yolov6-l.pt`             | [Objekterkennung](../tasks/detect.md) | ✅        | ✅           | ✅        | ✅           |
| YOLOv6-L6 | `yolov6-l6.pt`            | [Objekterkennung](../tasks/detect.md) | ✅        | ✅           | ✅        | ✅           |

Diese Tabelle bietet einen detaillierten Überblick über die YOLOv6-Modellvarianten und hebt ihre Fähigkeiten bei der Objekterkennung sowie ihre Kompatibilität mit verschiedenen Betriebsmodi wie [Inferenz](../modes/predict.md), [Validierung](../modes/val.md), [Training](../modes/train.md) und [Exportieren](../modes/export.md) hervor. Diese umfassende Unterstützung ermöglicht es den Benutzern, die Fähigkeiten von YOLOv6-Modellen in einer Vielzahl von Objekterkennungsszenarien vollständig zu nutzen.

## Zitate und Anerkennungen

Wir möchten den Autoren für ihre bedeutenden Beiträge auf dem Gebiet der Echtzeit-Objekterkennung danken:

!!! Quote ""

    === "BibTeX"

        ```bibtex
        @misc{li2023yolov6,
              title={YOLOv6 v3.0: A Full-Scale Reloading},
              author={Chuyi Li and Lulu Li and Yifei Geng and Hongliang Jiang and Meng Cheng and Bo Zhang and Zaidan Ke and Xiaoming Xu and Xiangxiang Chu},
              year={2023},
              eprint={2301.05586},
              archivePrefix={arXiv},
              primaryClass={cs.CV}
        }
        ```

Das ursprüngliche YOLOv6-Papier finden Sie auf [arXiv](https://arxiv.org/abs/2301.05586). Die Autoren haben ihre Arbeit öffentlich zugänglich gemacht, und der Code kann auf [GitHub](https://github.com/meituan/YOLOv6) abgerufen werden. Wir schätzen ihre Bemühungen zur Weiterentwicklung des Fachgebiets und zur Zugänglichmachung ihrer Arbeit für die breitere Gemeinschaft.
