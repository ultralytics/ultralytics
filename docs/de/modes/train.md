---
comments: true
description: Schritt-für-Schritt-Leitfaden zum Trainieren von YOLOv8-Modellen mit Ultralytics YOLO, einschließlich Beispielen für Single-GPU- und Multi-GPU-Training
keywords: Ultralytics, YOLOv8, YOLO, Objekterkennung, Trainingsmodus, benutzerdefinierter Datensatz, GPU-Training, Multi-GPU, Hyperparameter, CLI-Beispiele, Python-Beispiele
---

# Modelltraining mit Ultralytics YOLO

<img width="1024" src="https://github.com/ultralytics/assets/raw/main/yolov8/banner-integrations.png" alt="Ultralytics YOLO Ökosystem und Integrationen">

## Einleitung

Das Training eines Deep-Learning-Modells beinhaltet das Einspeisen von Daten und die Anpassung seiner Parameter, so dass es genaue Vorhersagen treffen kann. Der Trainingsmodus in Ultralytics YOLOv8 ist für das effektive und effiziente Training von Objekterkennungsmodellen konzipiert und nutzt dabei die Fähigkeiten moderner Hardware voll aus. Dieser Leitfaden zielt darauf ab, alle Details zu vermitteln, die Sie benötigen, um mit dem Training Ihrer eigenen Modelle unter Verwendung des robusten Funktionssatzes von YOLOv8 zu beginnen.

<p align="center">
  <br>
  <iframe width="720" height="405" src="https://www.youtube.com/embed/LNwODJXcvt4?si=7n1UvGRLSd9p5wKs"
    title="YouTube-Videoplayer" frameborder="0"
    allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share"
    allowfullscreen>
  </iframe>
  <br>
  <strong>Video anschauen:</strong> Wie man ein YOLOv8-Modell auf Ihrem benutzerdefinierten Datensatz in Google Colab trainiert.
</p>

## Warum Ultralytics YOLO für das Training wählen?

Hier einige überzeugende Gründe, sich für den Trainingsmodus von YOLOv8 zu entscheiden:

- **Effizienz:** Machen Sie das Beste aus Ihrer Hardware, egal ob Sie auf einem Single-GPU-Setup sind oder über mehrere GPUs skalieren.
- **Vielseitigkeit:** Training auf benutzerdefinierten Datensätzen zusätzlich zu den bereits verfügbaren Datensätzen wie COCO, VOC und ImageNet.
- **Benutzerfreundlich:** Einfache, aber leistungsstarke CLI- und Python-Schnittstellen für ein unkompliziertes Trainingserlebnis.
- **Flexibilität der Hyperparameter:** Eine breite Palette von anpassbaren Hyperparametern, um die Modellleistung zu optimieren.

### Schlüsselfunktionen des Trainingsmodus

Die folgenden sind einige bemerkenswerte Funktionen von YOLOv8s Trainingsmodus:

- **Automatischer Datensatz-Download:** Standarddatensätze wie COCO, VOC und ImageNet werden bei der ersten Verwendung automatisch heruntergeladen.
- **Multi-GPU-Unterstützung:** Skalieren Sie Ihr Training nahtlos über mehrere GPUs, um den Prozess zu beschleunigen.
- **Konfiguration der Hyperparameter:** Die Möglichkeit zur Modifikation der Hyperparameter über YAML-Konfigurationsdateien oder CLI-Argumente.
- **Visualisierung und Überwachung:** Echtzeit-Tracking von Trainingsmetriken und Visualisierung des Lernprozesses für bessere Einsichten.

!!! Tip "Tipp"

    * YOLOv8-Datensätze wie COCO, VOC, ImageNet und viele andere werden automatisch bei der ersten Verwendung heruntergeladen, d.h. `yolo train data=coco.yaml`

## Nutzungsbeispiele

Trainieren Sie YOLOv8n auf dem COCO128-Datensatz für 100 Epochen bei einer Bildgröße von 640. Das Trainingsgerät kann mit dem Argument `device` spezifiziert werden. Wenn kein Argument übergeben wird, wird GPU `device=0` verwendet, wenn verfügbar, sonst wird `device=cpu` verwendet. Siehe den Abschnitt Argumente unten für eine vollständige Liste der Trainingsargumente.

!!! Example "Beispiel für Single-GPU- und CPU-Training"

    Das Gerät wird automatisch ermittelt. Wenn eine GPU verfügbar ist, dann wird diese verwendet, sonst beginnt das Training auf der CPU.

    === "Python"

        ```python
        from ultralytics import YOLO

        # Laden Sie ein Modell
        model = YOLO('yolov8n.yaml')  # bauen Sie ein neues Modell aus YAML
        model = YOLO('yolov8n.pt')  # laden Sie ein vortrainiertes Modell (empfohlen für das Training)
        model = YOLO('yolov8n.yaml').load('yolov8n.pt')  # bauen Sie aus YAML und übertragen Sie Gewichte

        # Trainieren Sie das Modell
        results = model.train(data='coco128.yaml', epochs=100, imgsz=640)
        ```

    === "CLI"

        ```bash
        # Bauen Sie ein neues Modell aus YAML und beginnen Sie das Training von Grund auf
        yolo detect train data=coco128.yaml model=yolov8n.yaml epochs=100 imgsz=640

        # Beginnen Sie das Training von einem vortrainierten *.pt Modell
        yolo detect train data=coco128.yaml model=yolov8n.pt epochs=100 imgsz=640

        # Bauen Sie ein neues Modell aus YAML, übertragen Sie vortrainierte Gewichte darauf und beginnen Sie das Training
        yolo detect train data=coco128.yaml model=yolov8n.yaml pretrained=yolov8n.pt epochs=100 imgsz=640
        ```

### Multi-GPU-Training

Multi-GPU-Training ermöglicht eine effizientere Nutzung von verfügbaren Hardware-Ressourcen, indem die Trainingslast über mehrere GPUs verteilt wird. Diese Funktion ist über sowohl die Python-API als auch die Befehlszeilenschnittstelle verfügbar. Um das Multi-GPU-Training zu aktivieren, geben Sie die GPU-Geräte-IDs an, die Sie verwenden möchten.

!!! Example "Beispiel für Multi-GPU-Training"

    Um mit 2 GPUs zu trainieren, verwenden Sie die folgenden Befehle für CUDA-Geräte 0 und 1. Erweitern Sie dies bei Bedarf auf zusätzliche GPUs.

    === "Python"

        ```python
        from ultralytics import YOLO

        # Laden Sie ein Modell
        model = YOLO('yolov8n.pt')  # laden Sie ein vortrainiertes Modell (empfohlen für das Training)

        # Trainieren Sie das Modell mit 2 GPUs
        results = model.train(data='coco128.yaml', epochs=100, imgsz=640, device=[0, 1])
        ```

    === "CLI"

        ```bash
        # Beginnen Sie das Training von einem vortrainierten *.pt Modell unter Verwendung der GPUs 0 und 1
        yolo detect train data=coco128.yaml model=yolov8n.pt epochs=100 imgsz=640 device=0,1
        ```

### Apple M1- und M2-MPS-Training

Mit der Unterstützung für Apple M1- und M2-Chips, die in den Ultralytics YOLO-Modellen integriert ist, ist es jetzt möglich, Ihre Modelle auf Geräten zu trainieren, die das leistungsstarke Metal Performance Shaders (MPS)-Framework nutzen. MPS bietet eine leistungsstarke Methode zur Ausführung von Berechnungs- und Bildverarbeitungsaufgaben auf Apples benutzerdefinierten Siliziumchips.

Um das Training auf Apple M1- und M2-Chips zu ermöglichen, sollten Sie 'mps' als Ihr Gerät angeben, wenn Sie den Trainingsprozess starten. Unten ist ein Beispiel, wie Sie dies in Python und über die Befehlszeile tun könnten:

!!! Example "MPS-Training Beispiel"

    === "Python"

        ```python
        from ultralytics import YOLO

        # Laden Sie ein Modell
        model = YOLO('yolov8n.pt')  # laden Sie ein vortrainiertes Modell (empfohlen für das Training)

        # Trainieren Sie das Modell mit 2 GPUs
        results = model.train(data='coco128.yaml', epochs=100, imgsz=640, device='mps')
        ```

    === "CLI"

        ```bash
        # Beginnen Sie das Training von einem vortrainierten *.pt Modell unter Verwendung der GPUs 0 und 1
        yolo detect train data=coco128.yaml model=yolov8n.pt epochs=100 imgsz=640 device=mps
        ```

Indem sie die Rechenleistung der M1/M2-Chips nutzen, ermöglicht dies eine effizientere Verarbeitung der Trainingsaufgaben. Für detailliertere Anleitungen und fortgeschrittene Konfigurationsoptionen beziehen Sie sich bitte auf die [PyTorch MPS-Dokumentation](https://pytorch.org/docs/stable/notes/mps.html).

## Protokollierung

Beim Training eines YOLOv8-Modells kann es wertvoll sein, die Leistung des Modells im Laufe der Zeit zu verfolgen. Hier kommt die Protokollierung ins Spiel. Ultralytics' YOLO unterstützt drei Typen von Loggern - Comet, ClearML und TensorBoard.

Um einen Logger zu verwenden, wählen Sie ihn aus dem Dropdown-Menü im obigen Codeausschnitt aus und führen ihn aus. Der ausgewählte Logger wird installiert und initialisiert.

### Comet

[Comet](https://www.comet.ml/site/) ist eine Plattform, die Datenwissenschaftlern und Entwicklern erlaubt, Experimente und Modelle zu verfolgen, zu vergleichen, zu erklären und zu optimieren. Es bietet Funktionen wie Echtzeitmetriken, Code-Diffs und das Verfolgen von Hyperparametern.

Um Comet zu verwenden:

!!! Example "Beispiel"

    === "Python"
        ```python
        # pip installieren comet_ml
        import comet_ml

        comet_ml.init()
        ```

Vergessen Sie nicht, sich auf der Comet-Website anzumelden und Ihren API-Schlüssel zu erhalten. Sie müssen diesen zu Ihren Umgebungsvariablen oder Ihrem Skript hinzufügen, um Ihre Experimente zu protokollieren.

### ClearML

[ClearML](https://www.clear.ml/) ist eine Open-Source-Plattform, die das Verfolgen von Experimenten automatisiert und hilft, Ressourcen effizient zu teilen. Sie ist darauf ausgelegt, Teams bei der Verwaltung, Ausführung und Reproduktion ihrer ML-Arbeiten effizienter zu unterstützen.

Um ClearML zu verwenden:

!!! Example "Beispiel"

    === "Python"
        ```python
        # pip installieren clearml
        import clearml

        clearml.browser_login()
        ```

Nach dem Ausführen dieses Skripts müssen Sie sich auf dem Browser bei Ihrem ClearML-Konto anmelden und Ihre Sitzung authentifizieren.

### TensorBoard

[TensorBoard](https://www.tensorflow.org/tensorboard) ist ein Visualisierungstoolset für TensorFlow. Es ermöglicht Ihnen, Ihren TensorFlow-Graphen zu visualisieren, quantitative Metriken über die Ausführung Ihres Graphen zu plotten und zusätzliche Daten wie Bilder zu zeigen, die durch ihn hindurchgehen.

Um TensorBoard in [Google Colab](https://colab.research.google.com/github/ultralytics/ultralytics/blob/main/examples/tutorial.ipynb) zu verwenden:

!!! Example "Beispiel"

    === "CLI"
        ```bash
        load_ext tensorboard
        tensorboard --logdir ultralytics/runs  # ersetzen Sie mit Ihrem 'runs' Verzeichnis
        ```

Um TensorBoard lokal auszuführen, führen Sie den folgenden Befehl aus und betrachten Sie die Ergebnisse unter http://localhost:6006/.

!!! Example "Beispiel"

    === "CLI"
        ```bash
        tensorboard --logdir ultralytics/runs  # ersetzen Sie mit Ihrem 'runs' Verzeichnis
        ```

Dies lädt TensorBoard und weist es an, das Verzeichnis zu verwenden, in dem Ihre Trainingsprotokolle gespeichert sind.

Nachdem Sie Ihren Logger eingerichtet haben, können Sie mit Ihrem Modelltraining fortfahren. Alle Trainingsmetriken werden automatisch in Ihrer gewählten Plattform protokolliert, und Sie können auf diese Protokolle zugreifen, um die Leistung Ihres Modells im Laufe der Zeit zu überwachen, verschiedene Modelle zu vergleichen und Bereiche für Verbesserungen zu identifizieren.
