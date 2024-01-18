---
comments: true
description: Entdecken Sie die vielfältige Palette an Modellen der YOLO-Familie, SAM, MobileSAM, FastSAM, YOLO-NAS und RT-DETR, die von Ultralytics unterstützt werden. Beginnen Sie mit Beispielen für die CLI- und Python-Nutzung.
keywords: Ultralytics, Dokumentation, YOLO, SAM, MobileSAM, FastSAM, YOLO-NAS, RT-DETR, Modelle, Architekturen, Python, CLI
---

# Von Ultralytics unterstützte Modelle

Willkommen bei der Modell-Dokumentation von Ultralytics! Wir bieten Unterstützung für eine breite Palette von Modellen, die jeweils für spezifische Aufgaben wie [Objekterkennung](../tasks/detect.md), [Instanzsegmentierung](../tasks/segment.md), [Bildklassifizierung](../tasks/classify.md), [Posenschätzung](../tasks/pose.md) und [Multi-Objekt-Tracking](../modes/track.md) maßgeschneidert sind. Wenn Sie daran interessiert sind, Ihre Modellarchitektur bei Ultralytics beizutragen, sehen Sie sich unseren [Beitragenden-Leitfaden](../../help/contributing.md) an.

!!! Note "Hinweis"

    🚧 Unsere Dokumentation in verschiedenen Sprachen ist derzeit im Aufbau und wir arbeiten hart daran, sie zu verbessern. Vielen Dank für Ihre Geduld! 🙏

## Vorgestellte Modelle

Hier sind einige der wichtigsten unterstützten Modelle:

1. **[YOLOv3](yolov3.md)**: Die dritte Iteration der YOLO-Modellfamilie, ursprünglich von Joseph Redmon, bekannt für ihre effiziente Echtzeit-Objekterkennungsfähigkeiten.
2. **[YOLOv4](yolov4.md)**: Ein dunkelnetz-natives Update von YOLOv3, veröffentlicht von Alexey Bochkovskiy im Jahr 2020.
3. **[YOLOv5](yolov5.md)**: Eine verbesserte Version der YOLO-Architektur von Ultralytics, die bessere Leistungs- und Geschwindigkeitskompromisse im Vergleich zu früheren Versionen bietet.
4. **[YOLOv6](yolov6.md)**: Veröffentlicht von [Meituan](https://about.meituan.com/) im Jahr 2022 und in vielen autonomen Lieferrobotern des Unternehmens im Einsatz.
5. **[YOLOv7](yolov7.md)**: Aktualisierte YOLO-Modelle, die 2022 von den Autoren von YOLOv4 veröffentlicht wurden.
6. **[YOLOv8](yolov8.md) NEU 🚀**: Die neueste Version der YOLO-Familie, mit erweiterten Fähigkeiten wie Instanzsegmentierung, Pose/Schlüsselpunktschätzung und Klassifizierung.
7. **[Segment Anything Model (SAM)](sam.md)**: Metas Segment Anything Model (SAM).
8. **[Mobile Segment Anything Model (MobileSAM)](mobile-sam.md)**: MobileSAM für mobile Anwendungen, von der Kyung Hee University.
9. **[Fast Segment Anything Model (FastSAM)](fast-sam.md)**: FastSAM von der Image & Video Analysis Group, Institute of Automation, Chinesische Akademie der Wissenschaften.
10. **[YOLO-NAS](yolo-nas.md)**: YOLO Neural Architecture Search (NAS) Modelle.
11. **[Realtime Detection Transformers (RT-DETR)](rtdetr.md)**: Baidus PaddlePaddle Realtime Detection Transformer (RT-DETR) Modelle.

<p align="center">
  <br>
  <iframe width="720" height="405" src="https://www.youtube.com/embed/MWq1UxqTClU?si=nHAW-lYDzrz68jR0"
    title="YouTube-Video-Player" frameborder="0"
    allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share"
    allowfullscreen>
  </iframe>
  <br>
  <strong>Anschauen:</strong> Führen Sie Ultralytics YOLO-Modelle in nur wenigen Codezeilen aus.
</p>

## Einstieg: Nutzungbeispiele

Dieses Beispiel bietet einfache YOLO-Trainings- und Inferenzbeispiele. Für vollständige Dokumentationen über diese und andere [Modi](../modes/index.md) siehe die Dokumentationsseiten [Predict](../modes/predict.md),  [Train](../modes/train.md), [Val](../modes/val.md) und [Export](../modes/export.md).

Beachten Sie, dass das folgende Beispiel für YOLOv8 [Detect](../tasks/detect.md) Modelle zur Objekterkennung ist. Für zusätzliche unterstützte Aufgaben siehe die Dokumentation zu [Segment](../tasks/segment.md), [Classify](../tasks/classify.md) und [Pose](../tasks/pose.md).

!!! Example "Beispiel"

    === "Python"

        Vorgefertigte PyTorch `*.pt` Modelle sowie Konfigurationsdateien `*.yaml` können den Klassen `YOLO()`, `SAM()`, `NAS()` und `RTDETR()` übergeben werden, um eine Modellinstanz in Python zu erstellen:

        ```python
        from ultralytics import YOLO

        # Laden eines COCO-vortrainierten YOLOv8n Modells
        model = YOLO('yolov8n.pt')

        # Modellinformationen anzeigen (optional)
        model.info()

        # Model auf dem COCO8-Beispieldatensatz für 100 Epochen trainieren
        results = model.train(data='coco8.yaml', epochs=100, imgsz=640)

        # Inferenz mit dem YOLOv8n Modell auf das Bild 'bus.jpg' ausführen
        results = model('path/to/bus.jpg')
        ```

    === "CLI"

        CLI-Befehle sind verfügbar, um die Modelle direkt auszuführen:

        ```bash
        # Ein COCO-vortrainiertes YOLOv8n Modell laden und auf dem COCO8-Beispieldatensatz für 100 Epochen trainieren
        yolo train model=yolov8n.pt data=coco8.yaml epochs=100 imgsz=640

        # Ein COCO-vortrainiertes YOLOv8n Modell laden und Inferenz auf das Bild 'bus.jpg' ausführen
        yolo predict model=yolov8n.pt source=path/to/bus.jpg
        ```

## Neue Modelle beitragen

Sind Sie daran interessiert, Ihr Modell bei Ultralytics beizutragen? Großartig! Wir sind immer offen dafür, unser Modellportfolio zu erweitern.

1. **Repository forken**: Beginnen Sie mit dem Forken des [Ultralytics GitHub-Repositorys](https://github.com/ultralytics/ultralytics).

2. **Ihren Fork klonen**: Klonen Sie Ihren Fork auf Ihre lokale Maschine und erstellen Sie einen neuen Branch, um daran zu arbeiten.

3. **Ihr Modell implementieren**: Fügen Sie Ihr Modell entsprechend den in unserem [Beitragenden-Leitfaden](../../help/contributing.md) bereitgestellten Kodierungsstandards und Richtlinien hinzu.

4. **Gründlich testen**: Stellen Sie sicher, dass Sie Ihr Modell sowohl isoliert als auch als Teil des Pipelines gründlich testen.

5. **Eine Pull-Anfrage erstellen**: Sobald Sie mit Ihrem Modell zufrieden sind, erstellen Sie eine Pull-Anfrage zum Hauptrepository zur Überprüfung.

6. **Code-Review & Zusammenführen**: Nach der Überprüfung, wenn Ihr Modell unseren Kriterien entspricht, wird es in das Hauptrepository zusammengeführt.

Für detaillierte Schritte konsultieren Sie unseren [Beitragenden-Leitfaden](../../help/contributing.md).
