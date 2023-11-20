---
comments: true
description: Entdecken Sie die Vielfalt der von Ultralytics unterstützten Modelle der YOLO-Familie, SAM, MobileSAM, FastSAM, YOLO-NAS und RT-DETR Modelle. Beginnen Sie mit Beispielen für die Verwendung in CLI und Python.
keywords: Ultralytics, Dokumentation, YOLO, SAM, MobileSAM, FastSAM, YOLO-NAS, RT-DETR, Modelle, Architekturen, Python, CLI
---

# Von Ultralytics unterstützte Modelle

Willkommen in der Modell-Dokumentation von Ultralytics! Wir bieten Unterstützung für eine breite Palette von Modellen, die für spezifische Aufgaben wie [Objekterkennung](../tasks/detect.md), [Instanzsegmentierung](../tasks/segment.md), [Bildklassifizierung](../tasks/classify.md), [Poseerkennung](../tasks/pose.md) und [Multi-Objekt-Tracking](../modes/track.md) zugeschnitten sind. Wenn Sie daran interessiert sind, Ihre Modellarchitektur an Ultralytics beizutragen, werfen Sie einen Blick auf unseren [Beitragenden-Leitfaden](../../help/contributing.md).

!!! Note "Hinweis"

    🚧 Unsere mehrsprachige Dokumentation befindet sich derzeit im Aufbau, und wir arbeiten hart daran, sie zu verbessern. Vielen Dank für Ihre Geduld! 🙏

## Vorgestellte Modelle

Hier sind einige der wesentlichen unterstützten Modelle:

1. **[YOLOv3](../../models/yolov3.md)**: Die dritte Iteration der YOLO-Modellfamilie, ursprünglich von Joseph Redmon entwickelt und bekannt für ihre effiziente Echtzeit-Objekterkennung.
2. **[YOLOv4](../../models/yolov4.md)**: Eine darknet-native Aktualisierung von YOLOv3, die 2020 von Alexey Bochkovskiy veröffentlicht wurde.
3. **[YOLOv5](../../models/yolov5.md)**: Eine verbesserte Version der YOLO-Architektur von Ultralytics, die im Vergleich zu früheren Versionen bessere Leistungs- und Geschwindigkeitstrade-offs bietet.
4. **[YOLOv6](../../models/yolov6.md)**: Im Jahr 2022 von [Meituan](https://about.meituan.com/) veröffentlicht und in vielen autonomen Zustellrobotern des Unternehmens verwendet.
5. **[YOLOv7](../../models/yolov7.md)**: Im Jahr 2022 von den Autoren von YOLOv4 aktualisierte YOLO-Modelle.
6. **[YOLOv8](../../models/yolov8.md)**: Die neueste Version der YOLO-Familie mit erweiterten Fähigkeiten wie Instanzsegmentierung, Pose-/Schlüsselpunktschätzung und Klassifizierung.
7. **[Segment Anything Model (SAM)](../../models/sam.md)**: Metas Segment Anything Model (SAM).
8. **[Mobile Segment Anything Model (MobileSAM)](../../models/mobile-sam.md)**: MobileSAM für mobile Anwendungen von der Kyung Hee Universität.
9. **[Fast Segment Anything Model (FastSAM)](../../models/fast-sam.md)**: FastSAM von der Bild- und Videoanalysegruppe des Instituts für Automatisierung, Chinesische Akademie der Wissenschaften.
10. **[YOLO-NAS](../../models/yolo-nas.md)**: YOLO Neural Architecture Search (NAS) Modelle.
11. **[Realtime Detection Transformers (RT-DETR)](../../models/rtdetr.md)**: Baidus PaddlePaddle Realtime Detection Transformer (RT-DETR) Modelle.

<p align="center">
  <br>
  <iframe width="720" height="405" src="https://www.youtube.com/embed/MWq1UxqTClU?si=nHAW-lYDzrz68jR0"
    title="YouTube-Video-Player" frameborder="0"
    allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share"
    allowfullscreen>
  </iframe>
  <br>
  <strong>Sehen Sie:</strong> Ultralytics YOLO-Modelle in nur wenigen Zeilen Code ausführen.
</p>

## Erste Schritte: Anwendungsbeispiele

!!! Example "Beispiel"

    === "Python"

        PyTorch vortrainierte `*.pt` Modelle sowie Konfigurations-`*.yaml` Dateien können den Klassen `YOLO()`, `SAM()`, `NAS()` und `RTDETR()` übergeben werden, um in Python eine Modellinstanz zu erstellen:

        ```python
        from ultralytics import YOLO

        # Laden eines auf COCO vortrainierten YOLOv8n-Modells
        model = YOLO('yolov8n.pt')

        # Modellinformationen anzeigen (optional)
        model.info()

        # Das Modell mit dem COCO8-Beispieldatensatz für 100 Epochen trainieren
        results = model.train(data='coco8.yaml', epochs=100, imgsz=640)

        # Inferenz mit dem YOLOv8n-Modell am Bild 'bus.jpg' durchführen
        results = model('path/to/bus.jpg')
        ```

    === "CLI"

        CLI-Befehle sind verfügbar, um die Modelle direkt auszuführen:

        ```bash
        # Laden eines auf COCO vortrainierten YOLOv8n-Modells und Trainieren auf dem COCO8-Beispieldatensatz für 100 Epochen
        yolo train model=yolov8n.pt data=coco8.yaml epochs=100 imgsz=640

        # Laden eines auf COCO vortrainierten YOLOv8n-Modells und Durchführung der Inferenz am Bild 'bus.jpg'
        yolo predict model=yolov8n.pt source=path/to/bus.jpg
        ```

## Neue Modelle beitragen

Interessiert, Ihr Modell bei Ultralytics beizutragen? Großartig! Wir sind immer offen, unser Modellportfolio zu erweitern.

1. **Das Repository forken**: Beginnen Sie damit, das [GitHub-Repository von Ultralytics](https://github.com/ultralytics/ultralytics) zu forken.

2. **Ihren Fork klonen**: Klonen Sie Ihren Fork auf Ihre lokale Maschine und erstellen Sie einen neuen Branch, um daran zu arbeiten.

3. **Ihr Modell implementieren**: Fügen Sie Ihr Modell gemäß den in unserem [Beitragenden-Leitfaden](../../help/contributing.md) bereitgestellten Codierstandards und Richtlinien hinzu.

4. **Gründlich testen**: Stellen Sie sicher, dass Sie Ihr Modell sowohl isoliert als auch als Teil der Pipeline rigoros testen.

5. **Einen Pull Request erstellen**: Wenn Sie mit Ihrem Modell zufrieden sind, erstellen Sie einen Pull Request zum Hauptrepository zur Überprüfung.

6. **Code-Überprüfung und Merging**: Nach der Überprüfung wird Ihr Modell, wenn es unseren Kriterien entspricht, in das Hauptrepository übernommen.

Für detaillierte Schritte konsultieren Sie unseren [Beitragenden-Leitfaden](../../help/contributing.md).
