---
comments: true
description: Erkunden Sie das innovative Segment Anything Model (SAM) von Ultralytics, das Echtzeit-Bildsegmentierung ermöglicht. Erfahren Sie mehr über die promptable Segmentierung, die Zero-Shot-Performance und die Anwendung.
keywords: Ultralytics, Bildsegmentierung, Segment Anything Model, SAM, SA-1B-Datensatz, Echtzeit-Performance, Zero-Shot-Transfer, Objekterkennung, Bildanalyse, maschinelles Lernen
---

# Segment Anything Model (SAM)

Willkommen an der Spitze der Bildsegmentierung mit dem Segment Anything Model (SAM). Dieses revolutionäre Modell hat mit promptabler Bildsegmentierung und Echtzeit-Performance neue Standards in diesem Bereich gesetzt.

## Einführung in SAM: Das Segment Anything Model

Das Segment Anything Model (SAM) ist ein innovatives Bildsegmentierungsmodell, das promptable Segmentierung ermöglicht und so eine beispiellose Vielseitigkeit bei der Bildanalyse bietet. SAM bildet das Herzstück der Segment Anything Initiative, einem bahnbrechenden Projekt, das ein neuartiges Modell, eine neue Aufgabe und einen neuen Datensatz für die Bildsegmentierung einführt.

Dank seiner fortschrittlichen Konstruktion kann SAM sich an neue Bildverteilungen und Aufgaben anpassen, auch ohne Vorwissen. Das wird als Zero-Shot-Transfer bezeichnet. Trainiert wurde SAM auf dem umfangreichen [SA-1B-Datensatz](https://ai.facebook.com/datasets/segment-anything/), der über 1 Milliarde Masken auf 11 Millionen sorgfältig kuratierten Bildern enthält. SAM hat beeindruckende Zero-Shot-Performance gezeigt und in vielen Fällen frühere vollständig überwachte Ergebnisse übertroffen.

![Beispielbild aus dem Datensatz](https://user-images.githubusercontent.com/26833433/238056229-0e8ffbeb-f81a-477e-a490-aff3d82fd8ce.jpg)
Beispielimagen mit überlagernden Masken aus unserem neu eingeführten Datensatz SA-1B. SA-1B enthält 11 Millionen diverse, hochauflösende, lizenzierte und die Privatsphäre schützende Bilder und 1,1 Milliarden qualitativ hochwertige Segmentierungsmasken. Diese wurden vollautomatisch von SAM annotiert und sind nach menschlichen Bewertungen und zahlreichen Experimenten von hoher Qualität und Vielfalt. Die Bilder sind nach der Anzahl der Masken pro Bild gruppiert (im Durchschnitt sind es etwa 100 Masken pro Bild).

## Hauptmerkmale des Segment Anything Model (SAM)

- **Promptable Segmentierungsaufgabe:** SAM wurde mit der Ausführung einer promptable Segmentierungsaufgabe entwickelt, wodurch es valide Segmentierungsmasken aus beliebigen Prompts generieren kann, z. B. räumlichen oder textuellen Hinweisen zur Identifizierung eines Objekts.
- **Fortgeschrittene Architektur:** Das Segment Anything Model verwendet einen leistungsfähigen Bild-Encoder, einen Prompt-Encoder und einen leichten Masken-Decoder. Diese einzigartige Architektur ermöglicht flexibles Prompting, Echtzeitmaskenberechnung und Berücksichtigung von Mehrdeutigkeiten in Segmentierungsaufgaben.
- **Der SA-1B-Datensatz:** Eingeführt durch das Segment Anything Projekt, enthält der SA-1B-Datensatz über 1 Milliarde Masken auf 11 Millionen Bildern. Als bisher größter Segmentierungsdatensatz liefert er SAM eine vielfältige und umfangreiche Datenquelle für das Training.
- **Zero-Shot-Performance:** SAM zeigt herausragende Zero-Shot-Performance in verschiedenen Segmentierungsaufgaben und ist damit ein einsatzbereites Werkzeug für vielfältige Anwendungen mit minimalem Bedarf an prompt engineering.

Für eine detaillierte Betrachtung des Segment Anything Models und des SA-1B-Datensatzes besuchen Sie bitte die [Segment Anything Website](https://segment-anything.com) und lesen Sie das Forschungspapier [Segment Anything](https://arxiv.org/abs/2304.02643).

## Verfügbare Modelle, unterstützte Aufgaben und Betriebsmodi

Diese Tabelle zeigt die verfügbaren Modelle mit ihren spezifischen vortrainierten Gewichten, die unterstützten Aufgaben und ihre Kompatibilität mit verschiedenen Betriebsmodi wie [Inference](../modes/predict.md), [Validierung](../modes/val.md), [Training](../modes/train.md) und [Export](../modes/export.md), wobei ✅ Emojis für unterstützte Modi und ❌ Emojis für nicht unterstützte Modi verwendet werden.

| Modelltyp | Vortrainierte Gewichte | Unterstützte Aufgaben                       | Inference | Validierung | Training | Export |
|-----------|------------------------|---------------------------------------------|-----------|-------------|----------|--------|
| SAM base  | `sam_b.pt`             | [Instanzsegmentierung](../tasks/segment.md) | ✅         | ❌           | ❌        | ✅      |
| SAM large | `sam_l.pt`             | [Instanzsegmentierung](../tasks/segment.md) | ✅         | ❌           | ❌        | ✅      |

## Wie man SAM verwendet: Vielseitigkeit und Power in der Bildsegmentierung

Das Segment Anything Model kann für eine Vielzahl von Aufgaben verwendet werden, die über die Trainingsdaten hinausgehen. Dazu gehören Kantenerkennung, Generierung von Objektvorschlägen, Instanzsegmentierung und vorläufige Text-to-Mask-Vorhersage. Mit prompt engineering kann SAM sich schnell an neue Aufgaben und Datenverteilungen anpassen und sich so als vielseitiges und leistungsstarkes Werkzeug für alle Anforderungen der Bildsegmentierung etablieren.

### Beispiel für SAM-Vorhersage

!!! Example "Segmentierung mit Prompts"

    Bildsegmentierung mit gegebenen Prompts.

    === "Python"

        ```python
        from ultralytics import SAM

        # Modell laden
        model = SAM('sam_b.pt')

        # Modellinformationen anzeigen (optional)
        model.info()

        # Inferenz mit Bounding Box Prompt
        model('ultralytics/assets/zidane.jpg', bboxes=[439, 437, 524, 709])

        # Inferenz mit Point Prompt
        model('ultralytics/assets/zidane.jpg', points=[900, 370], labels=[1])
        ```

!!! Example "Alles segmentieren"

    Das ganze Bild segmentieren.

    === "Python"

        ```python
        from ultralytics import SAM

        # Modell laden
        model = SAM('sam_b.pt')

        # Modellinformationen anzeigen (optional)
        model.info()

        # Inferenz
        model('Pfad/zum/Bild.jpg')
        ```

    === "CLI"

        ```bash
        # Inferenz mit einem SAM-Modell
        yolo predict model=sam_b.pt source=Pfad/zum/Bild.jpg
        ```

- Die Logik hier besteht darin, das gesamte Bild zu segmentieren, wenn keine Prompts (Bounding Box/Point/Maske) übergeben werden.

!!! Example "Beispiel SAMPredictor"

    Dadurch können Sie das Bild einmal festlegen und mehrmals Inferenz mit Prompts ausführen, ohne den Bild-Encoder mehrfach auszuführen.

    === "Prompt-Inferenz"

        ```python
        from ultralytics.models.sam import Predictor as SAMPredictor

        # SAMPredictor erstellen
        overrides = dict(conf=0.25, task='segment', mode='predict', imgsz=1024, model="mobile_sam.pt")
        predictor = SAMPredictor(overrides=overrides)

        # Bild festlegen
        predictor.set_image("ultralytics/assets/zidane.jpg")  # Festlegung mit Bild-Datei
        predictor.set_image(cv2.imread("ultralytics/assets/zidane.jpg"))  # Festlegung mit np.ndarray
        results = predictor(bboxes=[439, 437, 524, 709])
        results = predictor(points=[900, 370], labels=[1])

        # Bild zurücksetzen
        predictor.reset_image()
        ```

    Alles segmentieren mit zusätzlichen Argumenten.

    === "Alles segmentieren"

        ```python
        from ultralytics.models.sam import Predictor as SAMPredictor

        # SAMPredictor erstellen
        overrides = dict(conf=0.25, task='segment', mode='predict', imgsz=1024, model="mobile_sam.pt")
        predictor = SAMPredictor(overrides=overrides)

        # Mit zusätzlichen Argumenten segmentieren
        results = predictor(source="ultralytics/assets/zidane.jpg", crop_n_layers=1, points_stride=64)
        ```

- Weitere zusätzliche Argumente für `Alles segmentieren` finden Sie in der [`Predictor/generate` Referenz](../../../reference/models/sam/predict.md).

## Vergleich von SAM und YOLOv8

Hier vergleichen wir Meta's kleinstes SAM-Modell, SAM-b, mit Ultralytics kleinstem Segmentierungsmodell, [YOLOv8n-seg](../tasks/segment.md):

| Modell                                         | Größe                         | Parameter                    | Geschwindigkeit (CPU)                  |
|------------------------------------------------|-------------------------------|------------------------------|----------------------------------------|
| Meta's SAM-b                                   | 358 MB                        | 94,7 M                       | 51096 ms/pro Bild                      |
| [MobileSAM](mobile-sam.md)                     | 40,7 MB                       | 10,1 M                       | 46122 ms/pro Bild                      |
| [FastSAM-s](fast-sam.md) mit YOLOv8-Backbone   | 23,7 MB                       | 11,8 M                       | 115 ms/pro Bild                        |
| Ultralytics [YOLOv8n-seg](../tasks/segment.md) | **6,7 MB** (53,4-mal kleiner) | **3,4 M** (27,9-mal kleiner) | **59 ms/pro Bild** (866-mal schneller) |

Dieser Vergleich zeigt die Größen- und Geschwindigkeitsunterschiede zwischen den Modellen. Während SAM einzigartige Fähigkeiten für die automatische Segmentierung bietet, konkurriert es nicht direkt mit YOLOv8-Segmentierungsmodellen, die kleiner, schneller und effizienter sind.

Die Tests wurden auf einem Apple M2 MacBook aus dem Jahr 2023 mit 16 GB RAM durchgeführt. Um diesen Test zu reproduzieren:

!!! Example "Beispiel"

    === "Python"
        ```python
        from ultralytics import FastSAM, SAM, YOLO

        # SAM-b profilieren
        model = SAM('sam_b.pt')
        model.info()
        model('ultralytics/assets')

        # MobileSAM profilieren
        model = SAM('mobile_sam.pt')
        model.info()
        model('ultralytics/assets')

        # FastSAM-s profilieren
        model = FastSAM('FastSAM-s.pt')
        model.info()
        model('ultralytics/assets')

        # YOLOv8n-seg profilieren
        model = YOLO('yolov8n-seg.pt')
        model.info()
        model('ultralytics/assets')
        ```

## Auto-Annotierung: Der schnelle Weg zu Segmentierungsdatensätzen

Die Auto-Annotierung ist eine wichtige Funktion von SAM, mit der Benutzer mithilfe eines vortrainierten Detektionsmodells einen [Segmentierungsdatensatz](https://docs.ultralytics.com/datasets/segment) generieren können. Diese Funktion ermöglicht eine schnelle und genaue Annotation einer großen Anzahl von Bildern, ohne dass zeitaufwändiges manuelles Labeling erforderlich ist.

### Generieren Sie Ihren Segmentierungsdatensatz mit einem Detektionsmodell

Um Ihren Datensatz mit dem Ultralytics-Framework automatisch zu annotieren, verwenden Sie die `auto_annotate` Funktion wie folgt:

!!! Example "Beispiel"

    === "Python"
        ```python
        from ultralytics.data.annotator import auto_annotate

        auto_annotate(data="Pfad/zum/Bilderordner", det_model="yolov8x.pt", sam_model='sam_b.pt')
        ```

| Argument   | Typ                 | Beschreibung                                                                                                              | Standard     |
|------------|---------------------|---------------------------------------------------------------------------------------------------------------------------|--------------|
| data       | str                 | Pfad zu einem Ordner, der die zu annotierenden Bilder enthält.                                                            |              |
| det_model  | str, optional       | Vortrainiertes YOLO-Detektionsmodell. Standardmäßig 'yolov8x.pt'.                                                         | 'yolov8x.pt' |
| sam_model  | str, optional       | Vortrainiertes SAM-Segmentierungsmodell. Standardmäßig 'sam_b.pt'.                                                        | 'sam_b.pt'   |
| device     | str, optional       | Gerät, auf dem die Modelle ausgeführt werden. Standardmäßig ein leerer String (CPU oder GPU, falls verfügbar).            |              |
| output_dir | str, None, optional | Verzeichnis zum Speichern der annotierten Ergebnisse. Standardmäßig ein 'labels'-Ordner im selben Verzeichnis wie 'data'. | None         |

Die `auto_annotate` Funktion nimmt den Pfad zu Ihren Bildern entgegen, mit optionalen Argumenten für das vortrainierte Detektions- und SAM-Segmentierungsmodell, das Gerät, auf dem die Modelle ausgeführt werden sollen, und das Ausgabeverzeichnis, in dem die annotierten Ergebnisse gespeichert werden sollen.

Die Auto-Annotierung mit vortrainierten Modellen kann die Zeit und den Aufwand für die Erstellung hochwertiger Segmentierungsdatensätze erheblich reduzieren. Diese Funktion ist besonders vorteilhaft für Forscher und Entwickler, die mit großen Bildersammlungen arbeiten. Sie ermöglicht es ihnen, sich auf die Modellentwicklung und -bewertung zu konzentrieren, anstatt auf die manuelle Annotation.

## Zitate und Danksagungen

Wenn Sie SAM in Ihrer Forschungs- oder Entwicklungsarbeit nützlich finden, erwägen Sie bitte, unser Paper zu zitieren:

!!! Quote ""

    === "BibTeX"

        ```bibtex
        @misc{kirillov2023segment,
              title={Segment Anything},
              author={Alexander Kirillov and Eric Mintun and Nikhila Ravi and Hanzi Mao and Chloe Rolland and Laura Gustafson and Tete Xiao and Spencer Whitehead and Alexander C. Berg and Wan-Yen Lo and Piotr Dollár and Ross Girshick},
              year={2023},
              eprint={2304.02643},
              archivePrefix={arXiv},
              primaryClass={cs.CV}
        }
        ```

Wir möchten Meta AI für die Erstellung und Pflege dieser wertvollen Ressource für die Computer Vision Community danken.

*Stichworte: Segment Anything, Segment Anything Model, SAM, Meta SAM, Bildsegmentierung, Promptable Segmentierung, Zero-Shot-Performance, SA-1B-Datensatz, fortschrittliche Architektur, Auto-Annotierung, Ultralytics, vortrainierte Modelle, SAM Base, SAM Large, Instanzsegmentierung, Computer Vision, Künstliche Intelligenz, maschinelles Lernen, Datenannotation, Segmentierungsmasken, Detektionsmodell, YOLO Detektionsmodell, Bibtex, Meta AI.*
