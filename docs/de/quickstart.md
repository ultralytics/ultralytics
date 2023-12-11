---
comments: true
description: Entdecken Sie verschiedene Methoden zur Installation von Ultralytics mit Pip, Conda, Git und Docker. Erfahren Sie, wie Sie Ultralytics über die Befehlszeilenschnittstelle oder innerhalb Ihrer Python-Projekte verwenden können.
keywords: Ultralytics-Installation, pip installieren Ultralytics, Docker installieren Ultralytics, Ultralytics-Befehlszeilenschnittstelle, Ultralytics Python-Schnittstelle
---

## Ultralytics installieren

Ultralytics bietet verschiedene Installationsmethoden, darunter Pip, Conda und Docker. Installiere YOLOv8 über das `ultralytics` Pip-Paket für die neueste stabile Veröffentlichung oder indem du das [Ultralytics GitHub-Repository](https://github.com/ultralytics/ultralytics) klonst für die aktuellste Version. Docker kann verwendet werden, um das Paket in einem isolierten Container auszuführen, ohne eine lokale Installation vornehmen zu müssen.

!!! Example "Installieren"

    === "Pip-Installation (empfohlen)"
        Installieren Sie das `ultralytics` Paket mit Pip oder aktualisieren Sie eine bestehende Installation, indem Sie `pip install -U ultralytics` ausführen. Besuchen Sie den Python Package Index (PyPI) für weitere Details zum `ultralytics` Paket: [https://pypi.org/project/ultralytics/](https://pypi.org/project/ultralytics/).

        [![PyPI-Version](https://badge.fury.io/py/ultralytics.svg)](https://badge.fury.io/py/ultralytics) [![Downloads](https://static.pepy.tech/badge/ultralytics)](https://pepy.tech/project/ultralytics)

        ```bash
        # Installiere das ultralytics Paket von PyPI
        pip install ultralytics
        ```

        Sie können auch das `ultralytics` Paket direkt vom GitHub [Repository](https://github.com/ultralytics/ultralytics) installieren. Dies könnte nützlich sein, wenn Sie die neueste Entwicklerversion möchten. Stellen Sie sicher, dass das Git-Kommandozeilen-Tool auf Ihrem System installiert ist. Der Befehl `@main` installiert den `main` Branch und kann zu einem anderen Branch geändert werden, z. B. `@my-branch`, oder ganz entfernt werden, um auf den `main` Branch standardmäßig zurückzugreifen.

        ```bash
        # Installiere das ultralytics Paket von GitHub
        pip install git+https://github.com/ultralytics/ultralytics.git@main
        ```


    === "Conda-Installation"
        Conda ist ein alternativer Paketmanager zu Pip, der ebenfalls für die Installation verwendet werden kann. Besuche Anaconda für weitere Details unter [https://anaconda.org/conda-forge/ultralytics](https://anaconda.org/conda-forge/ultralytics). Ultralytics Feedstock Repository für die Aktualisierung des Conda-Pakets befindet sich unter [https://github.com/conda-forge/ultralytics-feedstock/](https://github.com/conda-forge/ultralytics-feedstock/).


        [![Conda Rezept](https://img.shields.io/badge/recipe-ultralytics-green.svg)](https://anaconda.org/conda-forge/ultralytics) [![Conda Downloads](https://img.shields.io/conda/dn/conda-forge/ultralytics.svg)](https://anaconda.org/conda-forge/ultralytics) [![Conda Version](https://img.shields.io/conda/vn/conda-forge/ultralytics.svg)](https://anaconda.org/conda-forge/ultralytics) [![Conda Plattformen](https://img.shields.io/conda/pn/conda-forge/ultralytics.svg)](https://anaconda.org/conda-forge/ultralytics)

        ```bash
        # Installiere das ultralytics Paket mit Conda
        conda install -c conda-forge ultralytics
        ```

        !!! Note "Hinweis"

            Wenn Sie in einer CUDA-Umgebung installieren, ist es am besten, `ultralytics`, `pytorch` und `pytorch-cuda` im selben Befehl zu installieren, um dem Conda-Paketmanager zu ermöglichen, Konflikte zu lösen, oder `pytorch-cuda` als letztes zu installieren, damit es das CPU-spezifische `pytorch` Paket bei Bedarf überschreiben kann.
            ```bash
            # Installiere alle Pakete zusammen mit Conda
            conda install -c pytorch -c nvidia -c conda-forge pytorch torchvision pytorch-cuda=11.8 ultralytics
            ```

        ### Conda Docker-Image

        Ultralytics Conda Docker-Images sind ebenfalls von [DockerHub](https://hub.docker.com/r/ultralytics/ultralytics) verfügbar. Diese Bilder basieren auf [Miniconda3](https://docs.conda.io/projects/miniconda/en/latest/) und bieten eine einfache Möglichkeit, `ultralytics` in einer Conda-Umgebung zu nutzen.

        ```bash
        # Setze Image-Name als Variable
        t=ultralytics/ultralytics:latest-conda

        # Ziehe das neuste ultralytics Image von Docker Hub
        sudo docker pull $t

        # Führe das ultralytics Image in einem Container mit GPU-Unterstützung aus
        sudo docker run -it --ipc=host --gpus all $t  # alle GPUs
        sudo docker run -it --ipc=host --gpus '"device=2,3"' $t  # spezifische GPUs angeben
        ```

    === "Git klonen"
        Klonen Sie das `ultralytics` Repository, wenn Sie einen Beitrag zur Entwicklung leisten möchten oder mit dem neuesten Quellcode experimentieren wollen. Nach dem Klonen navigieren Sie in das Verzeichnis und installieren das Paket im editierbaren Modus `-e` mit Pip.
        ```bash
        # Klonen Sie das ultralytics Repository
        git clone https://github.com/ultralytics/ultralytics

        # Navigiere zum geklonten Verzeichnis
        cd ultralytics

        # Installiere das Paket im editierbaren Modus für die Entwicklung
        pip install -e .
        ```

Siehe die `ultralytics` [requirements.txt](https://github.com/ultralytics/ultralytics/blob/main/requirements.txt) Datei für eine Liste der Abhängigkeiten. Beachten Sie, dass alle oben genannten Beispiele alle erforderlichen Abhängigkeiten installieren.

<p align="center">
  <br>
  <iframe width="720" height="405" src="https://www.youtube.com/embed/_a7cVL9hqnk"
    title="YouTube video player" frameborder="0"
    allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share"
    allowfullscreen>
  </iframe>
  <br>
  <strong>Watch:</strong> Ultralytics YOLO Quick Start Guide
</p>

!!! Tip "Tipp"

    PyTorch-Anforderungen variieren je nach Betriebssystem und CUDA-Anforderungen, daher wird empfohlen, PyTorch zuerst gemäß den Anweisungen unter [https://pytorch.org/get-started/locally](https://pytorch.org/get-started/locally) zu installieren.

    <a href="https://pytorch.org/get-started/locally/">
        <img width="800" alt="PyTorch Installationsanweisungen" src="https://user-images.githubusercontent.com/26833433/228650108-ab0ec98a-b328-4f40-a40d-95355e8a84e3.png">
    </a>

## Ultralytics mit CLI verwenden

Die Befehlszeilenschnittstelle (CLI) von Ultralytics ermöglicht einfache Einzeilige Befehle ohne die Notwendigkeit einer Python-Umgebung. CLI erfordert keine Anpassung oder Python-Code. Sie können alle Aufgaben einfach vom Terminal aus mit dem `yolo` Befehl ausführen. Schauen Sie sich den [CLI-Leitfaden](/../usage/cli.md) an, um mehr über die Verwendung von YOLOv8 über die Befehlszeile zu erfahren.

!!! Example "Beispiel"

    === "Syntax"

        Ultralytics `yolo` Befehle verwenden die folgende Syntax:
        ```bash
        yolo TASK MODE ARGS

        Wo   TASK (optional) einer von [detect, segment, classify] ist
                MODE (erforderlich) einer von [train, val, predict, export, track] ist
                ARGS (optional) eine beliebige Anzahl von benutzerdefinierten 'arg=value' Paaren wie 'imgsz=320', die Vorgaben überschreiben.
        ```
        Sehen Sie alle ARGS im vollständigen [Konfigurationsleitfaden](/../usage/cfg.md) oder mit `yolo cfg`

    === "Trainieren"

        Trainieren Sie ein Erkennungsmodell für 10 Epochen mit einer Anfangslernerate von 0.01
        ```bash
        yolo train data=coco128.yaml model=yolov8n.pt epochs=10 lr0=0.01
        ```

    === "Vorhersagen"

        Vorhersagen eines YouTube-Videos mit einem vortrainierten Segmentierungsmodell bei einer Bildgröße von 320:
        ```bash
        yolo predict model=yolov8n-seg.pt source='https://youtu.be/LNwODJXcvt4' imgsz=320
        ```

    === "Val"

        Val ein vortrainiertes Erkennungsmodell bei Batch-Größe 1 und Bildgröße 640:
        ```bash
        yolo val model=yolov8n.pt data=coco128.yaml batch=1 imgsz=640
        ```

    === "Exportieren"

        Exportieren Sie ein YOLOv8n-Klassifikationsmodell im ONNX-Format bei einer Bildgröße von 224 mal 128 (kein TASK erforderlich)
        ```bash
        yolo export model=yolov8n-cls.pt format=onnx imgsz=224,128
        ```

    === "Speziell"

        Führen Sie spezielle Befehle aus, um Version, Einstellungen zu sehen, Checks auszuführen und mehr:
        ```bash
        yolo help
        yolo checks
        yolo version
        yolo settings
        yolo copy-cfg
        yolo cfg
        ```

!!! Warning "Warnung"

    Argumente müssen als `arg=val` Paare übergeben werden, getrennt durch ein Gleichheitszeichen `=` und durch Leerzeichen ` ` zwischen den Paaren. Verwenden Sie keine `--` Argumentpräfixe oder Kommata `,` zwischen den Argumenten.

    - `yolo predict model=yolov8n.pt imgsz=640 conf=0.25` &nbsp; ✅
    - `yolo predict model yolov8n.pt imgsz 640 conf 0.25` &nbsp; ❌
    - `yolo predict --model yolov8n.pt --imgsz 640 --conf 0.25` &nbsp; ❌

[CLI-Leitfaden](/../usage/cli.md){ .md-button }

## Ultralytics mit Python verwenden

Die Python-Schnittstelle von YOLOv8 ermöglicht eine nahtlose Integration in Ihre Python-Projekte und erleichtert das Laden, Ausführen und Verarbeiten der Modellausgabe. Konzipiert für Einfachheit und Benutzerfreundlichkeit, ermöglicht die Python-Schnittstelle Benutzern, Objekterkennung, Segmentierung und Klassifizierung schnell in ihren Projekten zu implementieren. Dies macht die Python-Schnittstelle von YOLOv8 zu einem unschätzbaren Werkzeug für jeden, der diese Funktionalitäten in seine Python-Projekte integrieren möchte.

Benutzer können beispielsweise ein Modell laden, es trainieren, seine Leistung an einem Validierungsset auswerten und sogar in das ONNX-Format exportieren, und das alles mit nur wenigen Codezeilen. Schauen Sie sich den [Python-Leitfaden](/../usage/python.md) an, um mehr über die Verwendung von YOLOv8 in Ihren_python_pro_jek_ten zu erfahren.

!!! Example "Beispiel"

    ```python
    from ultralytics import YOLO

    # Erstellen Sie ein neues YOLO Modell von Grund auf
    model = YOLO('yolov8n.yaml')

    # Laden Sie ein vortrainiertes YOLO Modell (empfohlen für das Training)
    model = YOLO('yolov8n.pt')

    # Trainieren Sie das Modell mit dem Datensatz 'coco128.yaml' für 3 Epochen
    results = model.train(data='coco128.yaml', epochs=3)

    # Bewerten Sie die Leistung des Modells am Validierungssatz
    results = model.val()

    # Führen Sie eine Objekterkennung an einem Bild mit dem Modell durch
    results = model('https://ultralytics.com/images/bus.jpg')

    # Exportieren Sie das Modell ins ONNX-Format
    success = model.export(format='onnx')
    ```

[Python-Leitfaden](/../usage/python.md){.md-button .md-button--primary}
