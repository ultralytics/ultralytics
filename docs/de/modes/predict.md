---
comments: true
description: Erkunden Sie, wie der YOLOv8-Prognosemodus für verschiedene Aufgaben verwendet werden kann. Erfahren Sie mehr über verschiedene Inferenzquellen wie Bilder, Videos und Datenformate.
keywords: Ultralytics, YOLOv8, Vorhersagemodus, Inferenzquellen, Vorhersageaufgaben, Streaming-Modus, Bildverarbeitung, Videoverarbeitung, maschinelles Lernen, KI
---

# Modellvorhersage mit Ultralytics YOLO

<img width="1024" src="https://github.com/ultralytics/assets/raw/main/yolov8/banner-integrations.png" alt="Ultralytics YOLO Ökosystem und Integrationen">

## Einführung

Im Bereich des maschinellen Lernens und der Computer Vision wird der Prozess des Verstehens visueller Daten als 'Inferenz' oder 'Vorhersage' bezeichnet. Ultralytics YOLOv8 bietet eine leistungsstarke Funktion, die als **Prognosemodus** bekannt ist und für eine hochleistungsfähige, echtzeitfähige Inferenz auf einer breiten Palette von Datenquellen zugeschnitten ist.

<p align="center">
  <br>
  <iframe width="720" height="405" src="https://www.youtube.com/embed/QtsI0TnwDZs?si=ljesw75cMO2Eas14"
    title="YouTube Video Player" frameborder="0"
    allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share"
    allowfullscreen>
  </iframe>
  <br>
  <strong>Anschauen:</strong> Wie man die Ausgaben vom Ultralytics YOLOv8 Modell für individuelle Projekte extrahiert.
</p>

## Anwendungen in der realen Welt

|                                                               Herstellung                                                               |                                                                Sport                                                                |                                                               Sicherheit                                                                |
|:---------------------------------------------------------------------------------------------------------------------------------------:|:-----------------------------------------------------------------------------------------------------------------------------------:|:---------------------------------------------------------------------------------------------------------------------------------------:|
| ![Ersatzteilerkennung für Fahrzeuge](https://github.com/RizwanMunawar/ultralytics/assets/62513924/a0f802a8-0776-44cf-8f17-93974a4a28a1) | ![Erkennung von Fußballspielern](https://github.com/RizwanMunawar/ultralytics/assets/62513924/7d320e1f-fc57-4d7f-a691-78ee579c3442) | ![Erkennung von stürzenden Personen](https://github.com/RizwanMunawar/ultralytics/assets/62513924/86437c4a-3227-4eee-90ef-9efb697bdb43) |
|                                                   Erkennung von Fahrzeugersatzteilen                                                    |                                                    Erkennung von Fußballspielern                                                    |                                                    Erkennung von stürzenden Personen                                                    |

## Warum Ultralytics YOLO für Inferenz nutzen?

Hier sind Gründe, warum Sie den Prognosemodus von YOLOv8 für Ihre verschiedenen Inferenzanforderungen in Betracht ziehen sollten:

- **Vielseitigkeit:** Fähig, Inferenzen auf Bilder, Videos und sogar Live-Streams zu machen.
- **Leistung:** Entwickelt für Echtzeit-Hochgeschwindigkeitsverarbeitung ohne Genauigkeitsverlust.
- **Einfache Bedienung:** Intuitive Python- und CLI-Schnittstellen für schnelle Einsatzbereitschaft und Tests.
- **Hohe Anpassbarkeit:** Verschiedene Einstellungen und Parameter, um das Verhalten der Modellinferenz entsprechend Ihren spezifischen Anforderungen zu optimieren.

### Schlüsselfunktionen des Prognosemodus

Der Prognosemodus von YOLOv8 ist robust und vielseitig konzipiert und verfügt über:

- **Kompatibilität mit mehreren Datenquellen:** Ganz gleich, ob Ihre Daten in Form von Einzelbildern, einer Bildersammlung, Videodateien oder Echtzeit-Videostreams vorliegen, der Prognosemodus deckt alles ab.
- **Streaming-Modus:** Nutzen Sie die Streaming-Funktion, um einen speichereffizienten Generator von `Results`-Objekten zu erzeugen. Aktivieren Sie dies, indem Sie `stream=True` in der Aufrufmethode des Predictors einstellen.
- **Batchverarbeitung:** Die Möglichkeit, mehrere Bilder oder Videoframes in einem einzigen Batch zu verarbeiten, wodurch die Inferenzzeit weiter verkürzt wird.
- **Integrationsfreundlich:** Dank der flexiblen API leicht in bestehende Datenpipelines und andere Softwarekomponenten zu integrieren.

Ultralytics YOLO-Modelle geben entweder eine Python-Liste von `Results`-Objekten zurück, oder einen speichereffizienten Python-Generator von `Results`-Objekten, wenn `stream=True` beim Inferenzvorgang an das Modell übergeben wird:

!!! Example "Predict"

    === "Gibt eine Liste mit `stream=False` zurück"
        ```python
        from ultralytics import YOLO

        # Ein Modell laden
        model = YOLO('yolov8n.pt')  # vortrainiertes YOLOv8n Modell

        # Batch-Inferenz auf einer Liste von Bildern ausführen
        results = model(['im1.jpg', 'im2.jpg'])  # gibt eine Liste von Results-Objekten zurück

        # Ergebnisliste verarbeiten
        for result in results:
            boxes = result.boxes  # Boxes-Objekt für Bbox-Ausgaben
            masks = result.masks  # Masks-Objekt für Segmentierungsmasken-Ausgaben
            keypoints = result.keypoints  # Keypoints-Objekt für Pose-Ausgaben
            probs = result.probs  # Probs-Objekt für Klassifizierungs-Ausgaben
        ```

    === "Gibt einen Generator mit `stream=True` zurück"
        ```python
        from ultralytics import YOLO

        # Ein Modell laden
        model = YOLO('yolov8n.pt')  # vortrainiertes YOLOv8n Modell

        # Batch-Inferenz auf einer Liste von Bildern ausführen
        results = model(['im1.jpg', 'im2.jpg'], stream=True)  # gibt einen Generator von Results-Objekten zurück

        # Generator von Ergebnissen verarbeiten
        for result in results:
            boxes = result.boxes  # Boxes-Objekt für Bbox-Ausgaben
            masks = result.masks  # Masks-Objekt für Segmentierungsmasken-Ausgaben
            keypoints = result.keypoints  # Keypoints-Objekt für Pose-Ausgaben
            probs = result.probs  # Probs-Objekt für Klassifizierungs-Ausgaben
        ```

## Inferenzquellen

YOLOv8 kann verschiedene Arten von Eingabequellen für die Inferenz verarbeiten, wie in der folgenden Tabelle gezeigt. Die Quellen umfassen statische Bilder, Videostreams und verschiedene Datenformate. Die Tabelle gibt ebenfalls an, ob jede Quelle im Streaming-Modus mit dem Argument `stream=True` ✅ verwendet werden kann. Der Streaming-Modus ist vorteilhaft für die Verarbeitung von Videos oder Live-Streams, da er einen Generator von Ergebnissen statt das Laden aller Frames in den Speicher erzeugt.

!!! Tip "Tipp"

    Verwenden Sie `stream=True` für die Verarbeitung langer Videos oder großer Datensätze, um den Speicher effizient zu verwalten. Bei `stream=False` werden die Ergebnisse für alle Frames oder Datenpunkte im Speicher gehalten, was bei großen Eingaben schnell zu Speicherüberläufen führen kann. Im Gegensatz dazu verwendet `stream=True` einen Generator, der nur die Ergebnisse des aktuellen Frames oder Datenpunkts im Speicher behält, was den Speicherverbrauch erheblich reduziert und Speicherüberlaufprobleme verhindert.

| Quelle             | Argument                                   | Typ               | Hinweise                                                                                       |
|--------------------|--------------------------------------------|-------------------|------------------------------------------------------------------------------------------------|
| Bild               | `'image.jpg'`                              | `str` oder `Path` | Einzelbilddatei.                                                                               |
| URL                | `'https://ultralytics.com/images/bus.jpg'` | `str`             | URL zu einem Bild.                                                                             |
| Bildschirmaufnahme | `'screen'`                                 | `str`             | Eine Bildschirmaufnahme erstellen.                                                             |
| PIL                | `Image.open('im.jpg')`                     | `PIL.Image`       | HWC-Format mit RGB-Kanälen.                                                                    |
| OpenCV             | `cv2.imread('im.jpg')`                     | `np.ndarray`      | HWC-Format mit BGR-Kanälen `uint8 (0-255)`.                                                    |
| numpy              | `np.zeros((640,1280,3))`                   | `np.ndarray`      | HWC-Format mit BGR-Kanälen `uint8 (0-255)`.                                                    |
| torch              | `torch.zeros(16,3,320,640)`                | `torch.Tensor`    | BCHW-Format mit RGB-Kanälen `float32 (0.0-1.0)`.                                               |
| CSV                | `'sources.csv'`                            | `str` oder `Path` | CSV-Datei mit Pfaden zu Bildern, Videos oder Verzeichnissen.                                   |
| video ✅            | `'video.mp4'`                              | `str` oder `Path` | Videodatei in Formaten wie MP4, AVI, usw.                                                      |
| Verzeichnis ✅      | `'path/'`                                  | `str` oder `Path` | Pfad zu einem Verzeichnis mit Bildern oder Videos.                                             |
| glob ✅             | `'path/*.jpg'`                             | `str`             | Glob-Muster, um mehrere Dateien zu finden. Verwenden Sie das `*` Zeichen als Platzhalter.      |
| YouTube ✅          | `'https://youtu.be/LNwODJXcvt4'`           | `str`             | URL zu einem YouTube-Video.                                                                    |
| stream ✅           | `'rtsp://example.com/media.mp4'`           | `str`             | URL für Streaming-Protokolle wie RTSP, RTMP, TCP oder eine IP-Adresse.                         |
| Multi-Stream ✅     | `'list.streams'`                           | `str` oder `Path` | `*.streams` Textdatei mit einer Stream-URL pro Zeile, z.B. 8 Streams laufen bei Batch-Größe 8. |

Untenstehend finden Sie Codebeispiele für die Verwendung jedes Quelltyps:

!!! Example "Vorhersagequellen"

    === "Bild"
        Führen Sie die Inferenz auf einer Bilddatei aus.
        ```python
        from ultralytics import YOLO

        # Ein vortrainiertes YOLOv8n Modell laden
        model = YOLO('yolov8n.pt')

        # Pfad zur Bilddatei definieren
        quell = 'Pfad/zum/Bild.jpg'

        # Inferenz auf der Quelle ausführen
        ergebnisse = model(quell)  # Liste von Results-Objekten
        ```

    === "Bildschirmaufnahme"
        Führen Sie die Inferenz auf dem aktuellen Bildschirminhalt als Screenshot aus.
        ```python
        from ultralytics import YOLO

        # Ein vortrainiertes YOLOv8n Modell laden
        model = YOLO('yolov8n.pt')

        # Aktuellen Screenshot als Quelle definieren
        quell = 'Bildschirm'

        # Inferenz auf der Quelle ausführen
        ergebnisse = model(quell)  # Liste von Results-Objekten
        ```

    === "URL"
        Führen Sie die Inferenz auf einem Bild oder Video aus, das über eine URL remote gehostet wird.
        ```python
        from ultralytics import YOLO

        # Ein vortrainiertes YOLOv8n Modell laden
        model = YOLO('yolov8n.pt')

        # Remote-Bild- oder Video-URL definieren
        quell = 'https://ultralytics.com/images/bus.jpg'

        # Inferenz auf der Quelle ausführen
        ergebnisse = model(quell)  # Liste von Results-Objekten
        ```

    === "PIL"
        Führen Sie die Inferenz auf einem Bild aus, das mit der Python Imaging Library (PIL) geöffnet wurde.
        ```python
        from PIL import Image
        from ultralytics import YOLO

        # Ein vortrainiertes YOLOv8n Modell laden
        model = YOLO('yolov8n.pt')

        # Ein Bild mit PIL öffnen
        quell = Image.open('Pfad/zum/Bild.jpg')

        # Inferenz auf der Quelle ausführen
        ergebnisse = model(quell)  # Liste von Results-Objekten
        ```

    === "OpenCV"
        Führen Sie die Inferenz auf einem Bild aus, das mit OpenCV gelesen wurde.
        ```python
        import cv2
        from ultralytics import YOLO

        # Ein vortrainiertes YOLOv8n Modell laden
        model = YOLO('yolov8n.pt')

        # Ein Bild mit OpenCV lesen
        quell = cv2.imread('Pfad/zum/Bild.jpg')

        # Inferenz auf der Quelle ausführen
        ergebnisse = model(quell)  # Liste von Results-Objekten
        ```

    === "numpy"
        Führen Sie die Inferenz auf einem Bild aus, das als numpy-Array dargestellt wird.
        ```python
        import numpy as np
        from ultralytics import YOLO

        # Ein vortrainiertes YOLOv8n Modell laden
        model = YOLO('yolov8n.pt')

        # Ein zufälliges numpy-Array der HWC-Form (640, 640, 3) mit Werten im Bereich [0, 255] und Typ uint8 erstellen
        quell = np.random.randint(low=0, high=255, size=(640, 640, 3), dtype='uint8')

        # Inferenz auf der Quelle ausführen
        ergebnisse = model(quell)  # Liste von Results-Objekten
        ```

    === "torch"
        Führen Sie die Inferenz auf einem Bild aus, das als PyTorch-Tensor dargestellt wird.
        ```python
        import torch
        from ultralytics import YOLO

        # Ein vortrainiertes YOLOv8n Modell laden
        model = YOLO('yolov8n.pt')

        # Ein zufälliger torch-Tensor der BCHW-Form (1, 3, 640, 640) mit Werten im Bereich [0, 1] und Typ float32 erstellen
        quell = torch.rand(1, 3, 640, 640, dtype=torch.float32)

        # Inferenz auf der Quelle ausführen
        ergebnisse = model(quell)  # Liste von Results-Objekten
