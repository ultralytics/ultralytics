---
comments: true
description: Erkunden Sie FastSAM, eine CNN-basierte Lösung zur Echtzeit-Segmentierung von Objekten in Bildern. Verbesserte Benutzerinteraktion, Recheneffizienz und anpassbar für verschiedene Vision-Aufgaben.
keywords: FastSAM, maschinelles Lernen, CNN-basierte Lösung, Objektsegmentierung, Echtzeillösung, Ultralytics, Vision-Aufgaben, Bildverarbeitung, industrielle Anwendungen, Benutzerinteraktion
---

# Fast Segment Anything Model (FastSAM)

Das Fast Segment Anything Model (FastSAM) ist eine neuartige, Echtzeit-CNN-basierte Lösung für die Segment Anything Aufgabe. Diese Aufgabe zielt darauf ab, jedes Objekt in einem Bild auf Basis verschiedener möglicher Benutzerinteraktionen zu segmentieren. FastSAM reduziert signifikant den Rechenbedarf, während es eine wettbewerbsfähige Leistung beibehält und somit für eine Vielzahl von Vision-Aufgaben praktisch einsetzbar ist.

![Übersicht über die Architektur des Fast Segment Anything Model (FastSAM)](https://user-images.githubusercontent.com/26833433/248551984-d98f0f6d-7535-45d0-b380-2e1440b52ad7.jpg)

## Überblick

FastSAM wurde entwickelt, um die Einschränkungen des [Segment Anything Model (SAM)](sam.md) zu beheben, einem schweren Transformer-Modell mit erheblichem Rechenressourcenbedarf. Das FastSAM teilt die Segment Anything Aufgabe in zwei aufeinanderfolgende Stufen auf: die Instanzsegmentierung und die promptgesteuerte Auswahl. In der ersten Stufe wird [YOLOv8-seg](../tasks/segment.md) verwendet, um die Segmentierungsmasken aller Instanzen im Bild zu erzeugen. In der zweiten Stufe gibt es den Bereich von Interesse aus, der dem Prompt entspricht.

## Hauptmerkmale

1. **Echtzeitlösung:** Durch die Nutzung der Recheneffizienz von CNNs bietet FastSAM eine Echtzeitlösung für die Segment Anything Aufgabe und eignet sich somit für industrielle Anwendungen, die schnelle Ergebnisse erfordern.

2. **Effizienz und Leistung:** FastSAM bietet eine signifikante Reduzierung des Rechen- und Ressourcenbedarfs, ohne die Leistungsqualität zu beeinträchtigen. Es erzielt eine vergleichbare Leistung wie SAM, verwendet jedoch drastisch reduzierte Rechenressourcen und ermöglicht so eine Echtzeitanwendung.

3. **Promptgesteuerte Segmentierung:** FastSAM kann jedes Objekt in einem Bild anhand verschiedener möglicher Benutzerinteraktionsaufforderungen segmentieren. Dies ermöglicht Flexibilität und Anpassungsfähigkeit in verschiedenen Szenarien.

4. **Basierend auf YOLOv8-seg:** FastSAM basiert auf [YOLOv8-seg](../tasks/segment.md), einem Objektdetektor mit einem Instanzsegmentierungsmodul. Dadurch ist es in der Lage, die Segmentierungsmasken aller Instanzen in einem Bild effektiv zu erzeugen.

5. **Wettbewerbsfähige Ergebnisse auf Benchmarks:** Bei der Objektvorschlagsaufgabe auf MS COCO erzielt FastSAM hohe Punktzahlen bei deutlich schnellerem Tempo als [SAM](sam.md) auf einer einzelnen NVIDIA RTX 3090. Dies demonstriert seine Effizienz und Leistungsfähigkeit.

6. **Praktische Anwendungen:** Der vorgeschlagene Ansatz bietet eine neue, praktische Lösung für eine Vielzahl von Vision-Aufgaben mit sehr hoher Geschwindigkeit, die zehn- oder hundertmal schneller ist als vorhandene Methoden.

7. **Möglichkeit zur Modellkompression:** FastSAM zeigt, dass der Rechenaufwand erheblich reduziert werden kann, indem ein künstlicher Prior in die Struktur eingeführt wird. Dadurch eröffnen sich neue Möglichkeiten für große Modellarchitekturen für allgemeine Vision-Aufgaben.

## Verfügbare Modelle, unterstützte Aufgaben und Betriebsmodi

In dieser Tabelle werden die verfügbaren Modelle mit ihren spezifischen vorab trainierten Gewichten, den unterstützten Aufgaben und ihrer Kompatibilität mit verschiedenen Betriebsmodi wie [Inferenz](../modes/predict.md), [Validierung](../modes/val.md), [Training](../modes/train.md) und [Export](../modes/export.md) angezeigt. Dabei stehen ✅ Emojis für unterstützte Modi und ❌ Emojis für nicht unterstützte Modi.

| Modelltyp | Vorab trainierte Gewichte | Unterstützte Aufgaben                       | Inferenz | Validierung | Training | Export |
|-----------|---------------------------|---------------------------------------------|----------|-------------|----------|--------|
| FastSAM-s | `FastSAM-s.pt`            | [Instanzsegmentierung](../tasks/segment.md) | ✅        | ❌           | ❌        | ✅      |
| FastSAM-x | `FastSAM-x.pt`            | [Instanzsegmentierung](../tasks/segment.md) | ✅        | ❌           | ❌        | ✅      |

## Beispiele für die Verwendung

Die FastSAM-Modelle lassen sich problemlos in Ihre Python-Anwendungen integrieren. Ultralytics bietet eine benutzerfreundliche Python-API und CLI-Befehle zur Vereinfachung der Entwicklung.

### Verwendung der Methode `predict`

Um eine Objekterkennung auf einem Bild durchzuführen, verwenden Sie die Methode `predict` wie folgt:

!!! Example "Beispiel"

    === "Python"
        ```python
        from ultralytics import FastSAM
        from ultralytics.models.fastsam import FastSAMPrompt

        # Definieren Sie die Quelle für die Inferenz
        source = 'Pfad/zum/bus.jpg'

        # Erstellen Sie ein FastSAM-Modell
        model = FastSAM('FastSAM-s.pt')  # oder FastSAM-x.pt

        # Führen Sie die Inferenz auf einem Bild durch
        everything_results = model(source, device='cpu', retina_masks=True, imgsz=1024, conf=0.4, iou=0.9)

        # Bereiten Sie ein Prompt-Process-Objekt vor
        prompt_process = FastSAMPrompt(source, everything_results, device='cpu')

        # Alles-Prompt
        ann = prompt_process.everything_prompt()

        # Bbox Standardform [0,0,0,0] -> [x1,y1,x2,y2]
        ann = prompt_process.box_prompt(bbox=[200, 200, 300, 300])

        # Text-Prompt
        ann = prompt_process.text_prompt(text='ein Foto von einem Hund')

        # Punkt-Prompt
        # Punkte Standard [[0,0]] [[x1,y1],[x2,y2]]
        # Punktbezeichnung Standard [0] [1,0] 0:Hintergrund, 1:Vordergrund
        ann = prompt_process.point_prompt(points=[[200, 200]], pointlabel=[1])
        prompt_process.plot(annotations=ann, output='./')
        ```

    === "CLI"
        ```bash
        # Laden Sie ein FastSAM-Modell und segmentieren Sie alles damit
        yolo segment predict model=FastSAM-s.pt source=Pfad/zum/bus.jpg imgsz=640
        ```

Dieser Code-Ausschnitt zeigt die Einfachheit des Ladens eines vorab trainierten Modells und das Durchführen einer Vorhersage auf einem Bild.

### Verwendung von `val`

Die Validierung des Modells auf einem Datensatz kann wie folgt durchgeführt werden:

!!! Example "Beispiel"

    === "Python"
        ```python
        from ultralytics import FastSAM

        # Erstellen Sie ein FastSAM-Modell
        model = FastSAM('FastSAM-s.pt')  # oder FastSAM-x.pt

        # Validieren Sie das Modell
        results = model.val(data='coco8-seg.yaml')
        ```

    === "CLI"
        ```bash
        # Laden Sie ein FastSAM-Modell und validieren Sie es auf dem COCO8-Beispieldatensatz mit Bildgröße 640
        yolo segment val model=FastSAM-s.pt data=coco8.yaml imgsz=640
        ```

Bitte beachten Sie, dass FastSAM nur die Erkennung und Segmentierung einer einzigen Objektklasse unterstützt. Das bedeutet, dass es alle Objekte als dieselbe Klasse erkennt und segmentiert. Daher müssen Sie beim Vorbereiten des Datensatzes alle Objektkategorie-IDs in 0 umwandeln.

## Offizielle Verwendung von FastSAM

FastSAM ist auch direkt aus dem [https://github.com/CASIA-IVA-Lab/FastSAM](https://github.com/CASIA-IVA-Lab/FastSAM) Repository erhältlich. Hier ist ein kurzer Überblick über die typischen Schritte, die Sie unternehmen könnten, um FastSAM zu verwenden:

### Installation

1. Klonen Sie das FastSAM-Repository:
   ```shell
   git clone https://github.com/CASIA-IVA-Lab/FastSAM.git
   ```

2. Erstellen und aktivieren Sie eine Conda-Umgebung mit Python 3.9:
   ```shell
   conda create -n FastSAM python=3.9
   conda activate FastSAM
   ```

3. Navigieren Sie zum geklonten Repository und installieren Sie die erforderlichen Pakete:
   ```shell
   cd FastSAM
   pip install -r requirements.txt
   ```

4. Installieren Sie das CLIP-Modell:
   ```shell
   pip install git+https://github.com/openai/CLIP.git
   ```

### Beispielverwendung

1. Laden Sie eine [Modell-Sicherung](https://drive.google.com/file/d/1m1sjY4ihXBU1fZXdQ-Xdj-mDltW-2Rqv/view?usp=sharing) herunter.

2. Verwenden Sie FastSAM für Inferenz. Beispielbefehle:

    - Segmentieren Sie alles in einem Bild:
      ```shell
      python Inference.py --model_path ./weights/FastSAM.pt --img_path ./images/dogs.jpg
      ```

    - Segmentieren Sie bestimmte Objekte anhand eines Textprompts:
      ```shell
      python Inference.py --model_path ./weights/FastSAM.pt --img_path ./images/dogs.jpg --text_prompt "der gelbe Hund"
      ```

    - Segmentieren Sie Objekte innerhalb eines Begrenzungsrahmens (geben Sie die Boxkoordinaten im xywh-Format an):
      ```shell
      python Inference.py --model_path ./weights/FastSAM.pt --img_path ./images/dogs.jpg --box_prompt "[570,200,230,400]"
      ```

    - Segmentieren Sie Objekte in der Nähe bestimmter Punkte:
      ```shell
      python Inference.py --model_path ./weights/FastSAM.pt --img_path ./images/dogs.jpg --point_prompt "[[520,360],[620,300]]" --point_label "[1,0]"
      ```

Sie können FastSAM auch über eine [Colab-Demo](https://colab.research.google.com/drive/1oX14f6IneGGw612WgVlAiy91UHwFAvr9?usp=sharing) oder die [HuggingFace-Web-Demo](https://huggingface.co/spaces/An-619/FastSAM) testen, um eine visuelle Erfahrung zu machen.

## Zitate und Danksagungen

Wir möchten den Autoren von FastSAM für ihre bedeutenden Beiträge auf dem Gebiet der Echtzeit-Instanzsegmentierung danken:

!!! Quote ""

    === "BibTeX"

      ```bibtex
      @misc{zhao2023fast,
            title={Fast Segment Anything},
            author={Xu Zhao and Wenchao Ding and Yongqi An and Yinglong Du and Tao Yu and Min Li and Ming Tang and Jinqiao Wang},
            year={2023},
            eprint={2306.12156},
            archivePrefix={arXiv},
            primaryClass={cs.CV}
      }
      ```

Die ursprüngliche FastSAM-Arbeit ist auf [arXiv](https://arxiv.org/abs/2306.12156) zu finden. Die Autoren haben ihre Arbeit öffentlich zugänglich gemacht, und der Code ist auf [GitHub](https://github.com/CASIA-IVA-Lab/FastSAM) verfügbar. Wir schätzen ihre Bemühungen, das Fachgebiet voranzutreiben und ihre Arbeit der breiteren Gemeinschaft zugänglich zu machen.
