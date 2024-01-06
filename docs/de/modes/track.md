---
comments: true
description: Erfahren Sie, wie Sie Ultralytics YOLO für Objektverfolgung in Videostreams verwenden. Anleitungen zum Einsatz verschiedener Tracker und zur Anpassung von Tracker-Konfigurationen.
keywords: Ultralytics, YOLO, Objektverfolgung, Videostreams, BoT-SORT, ByteTrack, Python-Anleitung, CLI-Anleitung
---

# Multi-Objektverfolgung mit Ultralytics YOLO

<img width="1024" src="https://user-images.githubusercontent.com/26833433/243418637-1d6250fd-1515-4c10-a844-a32818ae6d46.png" alt="Beispiele für Multi-Objektverfolgung">

Objektverfolgung im Bereich der Videoanalytik ist eine essentielle Aufgabe, die nicht nur den Standort und die Klasse von Objekten innerhalb des Frames identifiziert, sondern auch eine eindeutige ID für jedes erkannte Objekt, während das Video fortschreitet, erhält. Die Anwendungsmöglichkeiten sind grenzenlos – von Überwachung und Sicherheit bis hin zur Echtzeitsportanalytik.

## Warum Ultralytics YOLO für Objektverfolgung wählen?

Die Ausgabe von Ultralytics Trackern ist konsistent mit der standardmäßigen Objekterkennung, bietet aber zusätzlich Objekt-IDs. Dies erleichtert das Verfolgen von Objekten in Videostreams und das Durchführen nachfolgender Analysen. Hier sind einige Gründe, warum Sie Ultralytics YOLO für Ihre Objektverfolgungsaufgaben in Betracht ziehen sollten:

- **Effizienz:** Verarbeitung von Videostreams in Echtzeit ohne Einbußen bei der Genauigkeit.
- **Flexibilität:** Unterstützt mehrere Tracking-Algorithmen und -Konfigurationen.
- **Benutzerfreundlichkeit:** Einfache Python-API und CLI-Optionen für schnelle Integration und Bereitstellung.
- **Anpassbarkeit:** Einfache Verwendung mit individuell trainierten YOLO-Modellen, ermöglicht Integration in branchenspezifische Anwendungen.

<p align="center">
  <br>
  <iframe width="720" height="405" src="https://www.youtube.com/embed/hHyHmOtmEgs?si=VNZtXmm45Nb9s-N-"
    title="YouTube-Videoplayer" frameborder="0"
    allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share"
    allowfullscreen>
  </iframe>
  <br>
  <strong>Ansehen:</strong> Objekterkennung und -verfolgung mit Ultralytics YOLOv8.
</p>

## Anwendungen in der realen Welt

|                                                      Transportwesen                                                      |                                                       Einzelhandel                                                       |                                                      Aquakultur                                                       |
|:------------------------------------------------------------------------------------------------------------------------:|:------------------------------------------------------------------------------------------------------------------------:|:---------------------------------------------------------------------------------------------------------------------:|
| ![Fahrzeugverfolgung](https://github.com/RizwanMunawar/ultralytics/assets/62513924/ee6e6038-383b-4f21-ac29-b2a1c7d386ab) | ![Personenverfolgung](https://github.com/RizwanMunawar/ultralytics/assets/62513924/93bb4ee2-77a0-4e4e-8eb6-eb8f527f0527) | ![Fischverfolgung](https://github.com/RizwanMunawar/ultralytics/assets/62513924/a5146d0f-bfa8-4e0a-b7df-3c1446cd8142) |
|                                                    Fahrzeugverfolgung                                                    |                                                    Personenverfolgung                                                    |                                                    Fischverfolgung                                                    |

## Eigenschaften auf einen Blick

Ultralytics YOLO erweitert seine Objekterkennungsfunktionen, um eine robuste und vielseitige Objektverfolgung bereitzustellen:

- **Echtzeitverfolgung:** Nahtloses Verfolgen von Objekten in Videos mit hoher Bildfrequenz.
- **Unterstützung mehrerer Tracker:** Auswahl aus einer Vielzahl etablierter Tracking-Algorithmen.
- **Anpassbare Tracker-Konfigurationen:** Anpassen des Tracking-Algorithmus an spezifische Anforderungen durch Einstellung verschiedener Parameter.

## Verfügbare Tracker

Ultralytics YOLO unterstützt die folgenden Tracking-Algorithmen. Sie können aktiviert werden, indem Sie die entsprechende YAML-Konfigurationsdatei wie `tracker=tracker_type.yaml` übergeben:

* [BoT-SORT](https://github.com/NirAharon/BoT-SORT) - Verwenden Sie `botsort.yaml`, um diesen Tracker zu aktivieren.
* [ByteTrack](https://github.com/ifzhang/ByteTrack) - Verwenden Sie `bytetrack.yaml`, um diesen Tracker zu aktivieren.

Der Standardtracker ist BoT-SORT.

## Verfolgung

Um den Tracker auf Videostreams auszuführen, verwenden Sie ein trainiertes Erkennungs-, Segmentierungs- oder Posierungsmodell wie YOLOv8n, YOLOv8n-seg und YOLOv8n-pose.

!!! Example "Beispiel"

    === "Python"

        ```python
        from ultralytics import YOLO

        # Laden Sie ein offizielles oder individuelles Modell
        model = YOLO('yolov8n.pt')  # Laden Sie ein offizielles Erkennungsmodell
        model = YOLO('yolov8n-seg.pt')  # Laden Sie ein offizielles Segmentierungsmodell
        model = YOLO('yolov8n-pose.pt')  # Laden Sie ein offizielles Posierungsmodell
        model = YOLO('path/to/best.pt')  # Laden Sie ein individuell trainiertes Modell

        # Führen Sie die Verfolgung mit dem Modell durch
        results = model.track(source="https://youtu.be/LNwODJXcvt4", show=True)  # Verfolgung mit Standardtracker
        results = model.track(source="https://youtu.be/LNwODJXcvt4", show=True, tracker="bytetrack.yaml")  # Verfolgung mit ByteTrack-Tracker
        ```

    === "CLI"

        ```bash
        # Führen Sie die Verfolgung mit verschiedenen Modellen über die Befehlszeilenschnittstelle durch
        yolo track model=yolov8n.pt source="https://youtu.be/LNwODJXcvt4"  # Offizielles Erkennungsmodell
        yolo track model=yolov8n-seg.pt source="https://youtu.be/LNwODJXcvt4"  # Offizielles Segmentierungsmodell
        yolo track model=yolov8n-pose.pt source="https://youtu.be/LNwODJXcvt4"  # Offizielles Posierungsmodell
        yolo track model=path/to/best.pt source="https://youtu.be/LNwODJXcvt4"  # Individuell trainiertes Modell

        # Verfolgung mit ByteTrack-Tracker
        yolo track model=path/to/best.pt tracker="bytetrack.yaml"
        ```

Wie in der obigen Nutzung zu sehen ist, ist die Verfolgung für alle Detect-, Segment- und Pose-Modelle verfügbar, die auf Videos oder Streaming-Quellen ausgeführt werden.

## Konfiguration

### Tracking-Argumente

Die Tracking-Konfiguration teilt Eigenschaften mit dem Predict-Modus, wie `conf`, `iou` und `show`. Für weitere Konfigurationen siehe die Seite des [Predict](https://docs.ultralytics.com/modes/predict/)-Modells.

!!! Example "Beispiel"

    === "Python"

        ```python
        from ultralytics import YOLO

        # Konfigurieren Sie die Tracking-Parameter und führen Sie den Tracker aus
        model = YOLO('yolov8n.pt')
        results = model.track(source="https://youtu.be/LNwODJXcvt4", conf=0.3, iou=0.5, show=True)
        ```

    === "CLI"

        ```bash
        # Konfigurieren Sie die Tracking-Parameter und führen Sie den Tracker über die Befehlszeilenschnittstelle aus
        yolo track model=yolov8n.pt source="https://youtu.be/LNwODJXcvt4" conf=0.3, iou=0.5 show
        ```

### Tracker-Auswahl

Ultralytics ermöglicht es Ihnen auch, eine modifizierte Tracker-Konfigurationsdatei zu verwenden. Hierfür kopieren Sie einfach eine Tracker-Konfigurationsdatei (zum Beispiel `custom_tracker.yaml`) von [ultralytics/cfg/trackers](https://github.com/ultralytics/ultralytics/tree/main/ultralytics/cfg/trackers) und ändern jede Konfiguration (außer dem `tracker_type`), wie es Ihren Bedürfnissen entspricht.

!!! Example "Beispiel"

    === "Python"

        ```python
        from ultralytics import YOLO

        # Laden Sie das Modell und führen Sie den Tracker mit einer individuellen Konfigurationsdatei aus
        model = YOLO('yolov8n.pt')
        results = model.track(source="https://youtu.be/LNwODJXcvt4", tracker='custom_tracker.yaml')
        ```

    === "CLI"

        ```bash
        # Laden Sie das Modell und führen Sie den Tracker mit einer individuellen Konfigurationsdatei über die Befehlszeilenschnittstelle aus
        yolo track model=yolov8n.pt source="https://youtu.be/LNwODJXcvt4" tracker='custom_tracker.yaml'
        ```

Für eine umfassende Liste der Tracking-Argumente siehe die Seite [ultralytics/cfg/trackers](https://github.com/ultralytics/ultralytics/tree/main/ultralytics/cfg/trackers).

## Python-Beispiele

### Persistierende Tracks-Schleife

Hier ist ein Python-Skript, das OpenCV (`cv2`) und YOLOv8 verwendet, um Objektverfolgung in Videoframes durchzuführen. Dieses Skript setzt voraus, dass Sie die notwendigen Pakete (`opencv-python` und `ultralytics`) bereits installiert haben. Das Argument `persist=True` teilt dem Tracker mit, dass das aktuelle Bild oder Frame das nächste in einer Sequenz ist und Tracks aus dem vorherigen Bild im aktuellen Bild erwartet werden.

!!! Example "Streaming-For-Schleife mit Tracking"

    ```python
    import cv2
    from ultralytics import YOLO

    # Laden Sie das YOLOv8-Modell
    model = YOLO('yolov8n.pt')

    # Öffnen Sie die Videodatei
    video_path = "path/to/video.mp4"
    cap = cv2.VideoCapture(video_path)

    # Schleife durch die Videoframes
    while cap.isOpened():
        # Einen Frame aus dem Video lesen
        success, frame = cap.read()

        if success:
            # Führen Sie YOLOv8-Tracking im Frame aus, wobei Tracks zwischen Frames beibehalten werden
            results = model.track(frame, persist=True)

            # Visualisieren Sie die Ergebnisse im Frame
            annotated_frame = results[0].plot()

            # Zeigen Sie den kommentierten Frame an
            cv2.imshow("YOLOv8-Tracking", annotated_frame)

            # Beenden Sie die Schleife, wenn 'q' gedrückt wird
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        else:
            # Beenden Sie die Schleife, wenn das Ende des Videos erreicht ist
            break

    # Geben Sie das Videoaufnahmeobjekt frei und schließen Sie das Anzeigefenster
    cap.release()
    cv2.destroyAllWindows()
    ```

Bitte beachten Sie die Änderung von `model(frame)` zu `model.track(frame)`, welche die Objektverfolgung anstelle der einfachen Erkennung aktiviert. Dieses modifizierte Skript führt den Tracker auf jedem Frame des Videos aus, visualisiert die Ergebnisse und zeigt sie in einem Fenster an. Die Schleife kann durch Drücken von 'q' beendet werden.

## Neue Tracker beisteuern

Sind Sie versiert in der Multi-Objektverfolgung und haben erfolgreich einen Tracking-Algorithmus mit Ultralytics YOLO implementiert oder angepasst? Wir laden Sie ein, zu unserem Trackers-Bereich in [ultralytics/cfg/trackers](https://github.com/ultralytics/ultralytics/tree/main/ultralytics/cfg/trackers) beizutragen! Ihre realen Anwendungen und Lösungen könnten für Benutzer, die an Tracking-Aufgaben arbeiten, von unschätzbarem Wert sein.

Indem Sie zu diesem Bereich beitragen, helfen Sie, das Spektrum verfügbarer Tracking-Lösungen innerhalb des Ultralytics YOLO-Frameworks zu erweitern und fügen eine weitere Funktionsschicht für die Gemeinschaft hinzu.

Um Ihren Beitrag einzuleiten, sehen Sie bitte in unserem [Contributing Guide](https://docs.ultralytics.com/help/contributing) für umfassende Anweisungen zur Einreichung eines Pull Requests (PR) 🛠️. Wir sind gespannt darauf, was Sie beitragen!

Gemeinsam verbessern wir die Tracking-Fähigkeiten des Ultralytics YOLO-Ökosystems 🙏!
