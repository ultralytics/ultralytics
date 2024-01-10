---
comments: true
description: Erfahren Sie mehr über MobileSAM, dessen Implementierung, den Vergleich mit dem Original-SAM und wie Sie es im Ultralytics-Framework herunterladen und testen können. Verbessern Sie Ihre mobilen Anwendungen heute.
keywords: MobileSAM, Ultralytics, SAM, mobile Anwendungen, Arxiv, GPU, API, Bildencoder, Maskendekoder, Modell-Download, Testmethode
---

![MobileSAM Logo](https://github.com/ChaoningZhang/MobileSAM/blob/master/assets/logo2.png?raw=true)

# Mobile Segment Anything (MobileSAM)

Das MobileSAM-Paper ist jetzt auf [arXiv](https://arxiv.org/pdf/2306.14289.pdf) verfügbar.

Eine Demonstration von MobileSAM, das auf einer CPU ausgeführt wird, finden Sie unter diesem [Demo-Link](https://huggingface.co/spaces/dhkim2810/MobileSAM). Die Leistung auf einer Mac i5 CPU beträgt etwa 3 Sekunden. Auf der Hugging Face-Demo führt die Benutzeroberfläche und CPUs mit niedrigerer Leistung zu einer langsameren Reaktion, aber die Funktion bleibt effektiv.

MobileSAM ist in verschiedenen Projekten implementiert, darunter [Grounding-SAM](https://github.com/IDEA-Research/Grounded-Segment-Anything), [AnyLabeling](https://github.com/vietanhdev/anylabeling) und [Segment Anything in 3D](https://github.com/Jumpat/SegmentAnythingin3D).

MobileSAM wird mit einem einzigen GPU und einem 100K-Datensatz (1% der Originalbilder) in weniger als einem Tag trainiert. Der Code für dieses Training wird in Zukunft verfügbar gemacht.

## Verfügbarkeit von Modellen, unterstützte Aufgaben und Betriebsarten

Die folgende Tabelle zeigt die verfügbaren Modelle mit ihren spezifischen vortrainierten Gewichten, die unterstützten Aufgaben und ihre Kompatibilität mit unterschiedlichen Betriebsarten wie [Inferenz](../modes/predict.md), [Validierung](../modes/val.md), [Training](../modes/train.md) und [Export](../modes/export.md). Unterstützte Betriebsarten werden mit ✅-Emojis und nicht unterstützte Betriebsarten mit ❌-Emojis angezeigt.

| Modelltyp | Vortrainierte Gewichte | Unterstützte Aufgaben                       | Inferenz | Validierung | Training | Export |
|-----------|------------------------|---------------------------------------------|----------|-------------|----------|--------|
| MobileSAM | `mobile_sam.pt`        | [Instanzsegmentierung](../tasks/segment.md) | ✅        | ❌           | ❌        | ✅      |

## Anpassung von SAM zu MobileSAM

Da MobileSAM die gleiche Pipeline wie das Original-SAM beibehält, haben wir das ursprüngliche Preprocessing, Postprocessing und alle anderen Schnittstellen eingebunden. Personen, die derzeit das ursprüngliche SAM verwenden, können daher mit minimalem Aufwand zu MobileSAM wechseln.

MobileSAM bietet vergleichbare Leistungen wie das ursprüngliche SAM und behält dieselbe Pipeline, mit Ausnahme eines Wechsels des Bildencoders. Konkret ersetzen wir den ursprünglichen, leistungsstarken ViT-H-Encoder (632M) durch einen kleineren Tiny-ViT-Encoder (5M). Auf einem einzelnen GPU arbeitet MobileSAM in etwa 12 ms pro Bild: 8 ms auf dem Bildencoder und 4 ms auf dem Maskendekoder.

Die folgende Tabelle bietet einen Vergleich der Bildencoder, die auf ViT basieren:

| Bildencoder     | Original-SAM | MobileSAM |
|-----------------|--------------|-----------|
| Parameter       | 611M         | 5M        |
| Geschwindigkeit | 452ms        | 8ms       |

Sowohl das ursprüngliche SAM als auch MobileSAM verwenden denselben promptgeführten Maskendekoder:

| Maskendekoder   | Original-SAM | MobileSAM |
|-----------------|--------------|-----------|
| Parameter       | 3.876M       | 3.876M    |
| Geschwindigkeit | 4ms          | 4ms       |

Hier ist ein Vergleich der gesamten Pipeline:

| Gesamte Pipeline (Enc+Dec) | Original-SAM | MobileSAM |
|----------------------------|--------------|-----------|
| Parameter                  | 615M         | 9.66M     |
| Geschwindigkeit            | 456ms        | 12ms      |

Die Leistung von MobileSAM und des ursprünglichen SAM werden sowohl mit einem Punkt als auch mit einem Kasten als Prompt demonstriert.

![Bild mit Punkt als Prompt](https://raw.githubusercontent.com/ChaoningZhang/MobileSAM/master/assets/mask_box.jpg?raw=true)

![Bild mit Kasten als Prompt](https://raw.githubusercontent.com/ChaoningZhang/MobileSAM/master/assets/mask_box.jpg?raw=true)

Mit seiner überlegenen Leistung ist MobileSAM etwa 5-mal kleiner und 7-mal schneller als das aktuelle FastSAM. Weitere Details finden Sie auf der [MobileSAM-Projektseite](https://github.com/ChaoningZhang/MobileSAM).

## Testen von MobileSAM in Ultralytics

Wie beim ursprünglichen SAM bieten wir eine unkomplizierte Testmethode in Ultralytics an, einschließlich Modi für Punkt- und Kasten-Prompts.

### Modell-Download

Sie können das Modell [hier](https://github.com/ChaoningZhang/MobileSAM/blob/master/weights/mobile_sam.pt) herunterladen.

### Punkt-Prompt

!!! Example "Beispiel"

    === "Python"
        ```python
        from ultralytics import SAM

        # Laden Sie das Modell
        model = SAM('mobile_sam.pt')

        # Vorhersage einer Segmentierung basierend auf einem Punkt-Prompt
        model.predict('ultralytics/assets/zidane.jpg', points=[900, 370], labels=[1])
        ```

### Kasten-Prompt

!!! Example "Beispiel"

    === "Python"
        ```python
        from ultralytics import SAM

        # Laden Sie das Modell
        model = SAM('mobile_sam.pt')

        # Vorhersage einer Segmentierung basierend auf einem Kasten-Prompt
        model.predict('ultralytics/assets/zidane.jpg', bboxes=[439, 437, 524, 709])
        ```

Wir haben `MobileSAM` und `SAM` mit derselben API implementiert. Für weitere Verwendungsinformationen sehen Sie bitte die [SAM-Seite](sam.md).

## Zitate und Danksagungen

Wenn Sie MobileSAM in Ihrer Forschungs- oder Entwicklungsarbeit nützlich finden, zitieren Sie bitte unser Paper:

!!! Quote ""

    === "BibTeX"

        ```bibtex
        @article{mobile_sam,
          title={Faster Segment Anything: Towards Lightweight SAM for Mobile Applications},
          author={Zhang, Chaoning and Han, Dongshen and Qiao, Yu and Kim, Jung Uk and Bae, Sung Ho and Lee, Seungkyu and Hong, Choong Seon},
          journal={arXiv preprint arXiv:2306.14289},
          year={2023}
        }
