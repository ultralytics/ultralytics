---
comments: true
description: Erforschen Sie die aufregenden Funktionen von YOLOv8, der neuesten Version unseres Echtzeit-Objekterkenners! Erfahren Sie, wie fortschrittliche Architekturen, vortrainierte Modelle und eine optimale Balance zwischen Genauigkeit und Geschwindigkeit YOLOv8 zur perfekten Wahl für Ihre Aufgaben zur Objekterkennung machen.
keywords: YOLOv8, Ultralytics, Echtzeit-Objekterkenner, vortrainierte Modelle, Dokumentation, Objekterkennung, YOLO-Serie, fortschrittliche Architekturen, Genauigkeit, Geschwindigkeit
---

# YOLOv8

## Überblick

YOLOv8 ist die neueste Iteration der YOLO-Serie von Echtzeit-Objekterkennern und bietet eine herausragende Leistung in Bezug auf Genauigkeit und Geschwindigkeit. YOLOv8 baut auf den Fortschritten früherer YOLO-Versionen auf und bietet neue Funktionen und Optimierungen, die es zu einer idealen Wahl für verschiedene Aufgaben zur Objekterkennung in einer Vielzahl von Anwendungen machen.

![Ultralytics YOLOv8](https://raw.githubusercontent.com/ultralytics/assets/main/yolov8/yolo-comparison-plots.png)

## Hauptmerkmale

- **Fortgeschrittene Backbone- und Neck-Architekturen:** YOLOv8 verwendet modernste Backbone- und Neck-Architekturen, die zu einer verbesserten Merkmalsextraktion und Objekterkennungsleistung führen.
- **Ankerfreier Split Ultralytics Head:** YOLOv8 verwendet einen ankerfreien Split Ultralytics Head, der im Vergleich zu ankerbasierten Ansätzen eine bessere Genauigkeit und einen effizienteren Erkennungsprozess bietet.
- **Optimierte Genauigkeits-Geschwindigkeitsabwägung:** Mit dem Fokus auf eine optimale Balance zwischen Genauigkeit und Geschwindigkeit eignet sich YOLOv8 für Echtzeit-Objekterkennungsaufgaben in verschiedenen Anwendungsbereichen.
- **Verschiedene vortrainierte Modelle:** YOLOv8 bietet eine Auswahl an vortrainierten Modellen, um unterschiedlichen Aufgaben und Leistungsanforderungen gerecht zu werden und das richtige Modell für Ihren spezifischen Anwendungsfall zu finden.

## Unterstützte Aufgaben

| Modeltyp    | Vortrainierte Gewichte                                                                                              | Aufgabe              |
|-------------|---------------------------------------------------------------------------------------------------------------------|----------------------|
| YOLOv8      | `yolov8n.pt`, `yolov8s.pt`, `yolov8m.pt`, `yolov8l.pt`, `yolov8x.pt`                                                | Detektion            |
| YOLOv8-seg  | `yolov8n-seg.pt`, `yolov8s-seg.pt`, `yolov8m-seg.pt`, `yolov8l-seg.pt`, `yolov8x-seg.pt`                            | Instanzsegmentierung |
| YOLOv8-pose | `yolov8n-pose.pt`, `yolov8s-pose.pt`, `yolov8m-pose.pt`, `yolov8l-pose.pt`, `yolov8x-pose.pt`, `yolov8x-pose-p6.pt` | Pose/Schlüsselpunkte |
| YOLOv8-cls  | `yolov8n-cls.pt`, `yolov8s-cls.pt`, `yolov8m-cls.pt`, `yolov8l-cls.pt`, `yolov8x-cls.pt`                            | Klassifikation       |

## Unterstützte Modi

| Modus       | Unterstützt |
|-------------|-------------|
| Inferenz    | ✅           |
| Validierung | ✅           |
| Training    | ✅           |

!!! Leistung

    === "Detektion (COCO)"

        | Modell                                                                                | Größe<br><sup>(Pixel) | mAP<sup>val<br>50-95 | Geschwindigkeit<br><sup>CPU ONNX<br>(ms) | Geschwindigkeit<br><sup>A100 TensorRT<br>(ms) | Parameter<br><sup>(M) | FLOPs<br><sup>(B) |
        |---------------------------------------------------------------------------------------|-----------------------|----------------------|-----------------------------------------|----------------------------------------------|------------------------|------------------|
        | [YOLOv8n](https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n.pt) | 640                   | 37,3                 | 80,4                                    | 0,99                                         | 3,2                    | 8,7              |
        | [YOLOv8s](https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8s.pt) | 640                   | 44,9                 | 128,4                                   | 1,20                                         | 11,2                   | 28,6             |
        | [YOLOv8m](https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8m.pt) | 640                   | 50,2                 | 234,7                                   | 1,83                                         | 25,9                   | 78,9             |
        | [YOLOv8l](https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8l.pt) | 640                   | 52,9                 | 375,2                                   | 2,39                                         | 43,7                   | 165,2            |
        | [YOLOv8x](https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8x.pt) | 640                   | 53,9                 | 479,1                                   | 3,53                                         | 68,2                   | 257,8            |

    === "Detektion (Open Images V7)"

        Informationen zu Verwendungsbeispielen mit diesen Modellen, die auf [Open Image V7](https://docs.ultralytics.com/datasets/detect/open-images-v7/) trainiert wurden, finden Sie in der [Detektionsdokumentation](https://docs.ultralytics.com/tasks/detect/). Die Modelle umfassen 600 vortrainierte Klassen.

        | Modell                                                                                        | Größe<br><sup>(Pixel) | mAP<sup>val<br>50-95 | Geschwindigkeit<br><sup>CPU ONNX<br>(ms) | Geschwindigkeit<br><sup>A100 TensorRT<br>(ms) | Parameter<br><sup>(M) | FLOPs<br><sup>(B) |
        | -------------------------------------------------------------------------------------------- | ----------------------- | ---------------------- | ----------------------------------------- | ---------------------------------------------- | ------------------------ | ------------------ |
        | [YOLOv8n](https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n-oiv7.pt) | 640                     | 18,4                     | 142,4                                    | 1,21                                           | 3,5                     | 10,5             |
        | [YOLOv8s](https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8s-oiv7.pt) | 640                     | 27,7                     | 183,1                                    | 1,40                                           | 11,4                   | 29,7             |
        | [YOLOv8m](https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8m-oiv7.pt) | 640                     | 33,6                     | 408,5                                    | 2,26                                           | 26,2                   | 80,6             |
        | [YOLOv8l](https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8l-oiv7.pt) | 640                     | 34,9                     | 596,9                                    | 2,43                                           | 44,1                   | 167,4            |
        | [YOLOv8x](https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8x-oiv7.pt) | 640                     | 36,3                     | 860,6                                    | 3,56                                           | 68,7                   | 260,6            |

    === "Segmentierung (COCO)"

        | Modell                                                                                            | Größe<br><sup>(Pixel) | mAP<sup>Box<br>50-95 | mAP<sup>Mask<br>50-95 | Geschwindigkeit<br><sup>CPU ONNX<br>(ms) | Geschwindigkeit<br><sup>A100 TensorRT<br>(ms) | Parameter<br><sup>(M) | FLOPs<br><sup>(B) |
        | ------------------------------------------------------------------------------------------------ | ----------------------- | ----------------------- | ---------------------- | ----------------------------------------- | ---------------------------------------------- | --------------------- | ------------------ |
        | [YOLOv8n-seg](https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n-seg.pt) | 640                     | 36,7                     | 30,5                   | 96,1                                      | 1,21                                           | 3,4                     | 12,6             |
        | [YOLOv8s-seg](https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8s-seg.pt) | 640                     | 44,6                     | 36,8                   | 155,7                                    | 1,47                                           | 11,8                   | 42,6             |
        | [YOLOv8m-seg](https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8m-seg.pt) | 640                     | 49,9                     | 40,8                   | 317,0                                    | 2,18                                           | 27,3                   | 110,2            |
        | [YOLOv8l-seg](https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8l-seg.pt) | 640                     | 52,3                     | 42,6                   | 572,4                                    | 2,79                                           | 46,0                   | 220,5            |
        | [YOLOv8x-seg](https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8x-seg.pt) | 640                     | 53,4                     | 43,4                   | 712,1                                    | 4,02                                           | 71,8                   | 344,1            |

    === "Klassifikation (ImageNet)"

        | Modell                                                                                            | Größe<br><sup>(Pixel) | Genauigkeit<br><sup>Top 1 | Genauigkeit<br><sup>Top 5 | Geschwindigkeit<br><sup>CPU ONNX<br>(ms) | Geschwindigkeit<br><sup>A100 TensorRT<br>(ms) | Parameter<br><sup>(M) | FLOPs<br><sup>(B) bei 640 |
        | ------------------------------------------------------------------------------------------------ | ----------------------- | ------------------------ | ------------------------ | ----------------------------------------- | ---------------------------------------------- | --------------------- | ------------------------ |
        | [YOLOv8n-cls](https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n-cls.pt) | 224                     | 66,6                     | 87,0                     | 12,9                                      | 0,31                                           | 2,7                     | 4,3                      |
        | [YOLOv8s-cls](https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8s-cls.pt) | 224                     | 72,3                     | 91,1                     | 23,4                                      | 0,35                                           | 6,4                     | 13,5                     |
        | [YOLOv8m-cls](https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8m-cls.pt) | 224                     | 76,4                     | 93,2                     | 85,4                                      | 0,62                                           | 17,0                   | 42,7                     |
        | [YOLOv8l-cls](https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8l-cls.pt) | 224                     | 78,0                     | 94,1                     | 163,0                                     | 0,87                                           | 37,5                   | 99,7                     |
        | [YOLOv8x-cls](https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8x-cls.pt) | 224                     | 78,4                     | 94,3                     | 232,0                                     | 1,01                                           | 57,4                   | 154,8                    |

    === "Pose (COCO)"

        | Modell                                                                                          | Größe<br><sup>(Pixel) | mAP<sup>Pose<br>50-95 | mAP<sup>Pose<br>50 | Geschwindigkeit<br><sup>CPU ONNX<br>(ms) | Geschwindigkeit<br><sup>A100 TensorRT<br>(ms) | Parameter<br><sup>(M) | FLOPs<br><sup>(B) |
        | ---------------------------------------------------------------------------------------------- | ----------------------- | ---------------------- | ------------------ | ----------------------------------------- | ---------------------------------------------- | --------------------- | ------------------ |
        | [YOLOv8n-pose](https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n-pose.pt) | 640                     | 50,4                   | 80,1               | 131,8                                     | 1,18                                           | 3,3                     | 9,2               |
        | [YOLOv8s-pose](https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8s-pose.pt) | 640                     | 60,0                   | 86,2               | 233,2                                     | 1,42                                           | 11,6                   | 30,2              |
        | [YOLOv8m-pose](https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8m-pose.pt) | 640                     | 65,0                   | 88,8               | 456,3                                     | 2,00                                           | 26,4                   | 81,0              |
        | [YOLOv8l-pose](https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8l-pose.pt) | 640                     | 67,6                   | 90,0               | 784,5                                     | 2,59                                           | 44,4                   | 168,6             |
        | [YOLOv8x-pose](https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8x-pose.pt) | 640                     | 69,2                   | 90,2               | 1607,1                                    | 3,73                                           | 69,4                   | 263,2             |
        | [YOLOv8x-pose-p6](https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8x-pose-p6.pt) | 1280                   | 71,6                   | 91,2               | 4088,7                                    | 10,04                                          | 99,1                   | 1066,4            |

## Verwendung

Sie können YOLOv8 für Aufgaben zur Objekterkennung mit dem Ultralytics-Paket verwenden. Der folgende Code-Ausschnitt zeigt, wie YOLOv8-Modelle für die Inferenz verwendet werden:

!!! Beispiel ""

    In diesem Beispiel finden Sie einen einfachen Inferenz-Code für YOLOv8. Weitere Optionen, einschließlich der Verarbeitung von Inferenzergebnissen, finden Sie im [Predict](../modes/predict.md)-Modus. Um YOLOv8 mit zusätzlichen Modi zu verwenden, siehe [Train](../modes/train.md), [Val](../modes/val.md) und [Export](../modes/export.md).

    === "Python"

        Vorkonfigurierte `*.pt` Modelle von PyTorch sowie Konfigurationsdateien `*.yaml` können an die `YOLO()`-Klasse übergeben werden, um eine Modellinstanz in Python zu erstellen:

        ```python
        from ultralytics import YOLO

        # Lade ein mit COCO vortrainiertes YOLOv8n Modell
        model = YOLO('yolov8n.pt')

        # Zeige Infos zum Modell an (optional)
        model.info()

        # Trainiere das Modell für 100 Epochen anhand des COCO8-Beispieldatensatzes
        results = model.train(data='coco8.yaml', epochs=100, imgsz=640)

        # Führe Inferenz mit dem YOLOv8n Modell auf dem Bild 'bus.jpg' aus
        results = model('path/to/bus.jpg')
        ```

    === "CLI"

        Es stehen CLI-Befehle zur direkten Ausführung der Modelle zur Verfügung:

        ```bash
        # Lade ein mit COCO vortrainiertes YOLOv8n Modell und trainiere es für 100 Epochen anhand des COCO8-Beispieldatensatzes
        yolo train model=yolov8n.pt data=coco8.yaml epochs=100 imgsz=640

        # Lade ein mit COCO vortrainiertes YOLOv8n Modell und führe Inferenz auf dem Bild 'bus.jpg' aus
        yolo predict model=yolov8n.pt source=path/to/bus.jpg
        ```

## Zitierungen und Danksagungen

Wenn Sie das YOLOv8-Modell oder eine andere Software aus diesem Repository in Ihrer Arbeit verwenden, zitieren Sie es bitte in folgendem Format:

!!! Hinweis ""

    === "BibTeX"

        ```bibtex
        @software{yolov8_ultralytics,
          author = {Glenn Jocher and Ayush Chaurasia and Jing Qiu},
          title = {Ultralytics YOLOv8},
          version = {8.0.0},
          year = {2023},
          url = {https://github.com/ultralytics/ultralytics},
          orcid = {0000-0001-5950-6979, 0000-0002-7603-6750, 0000-0003-3783-7069},
          license = {AGPL-3.0}
        }
        ```

Bitte beachten Sie, dass die DOI aussteht und der Zitation hinzugefügt wird, sobald sie verfügbar ist. Die Verwendung der Software erfolgt gemäß der AGPL-3.0-Lizenz.
