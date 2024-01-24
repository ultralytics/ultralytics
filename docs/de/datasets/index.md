---
comments: true
description: Erkunden Sie verschiedene von Ultralytics unterstützte Computer Vision Datensätze für Objekterkennung, Segmentierung, Posenschätzung, Bildklassifizierung und Multi-Objekt-Verfolgung.
keywords: Computer Vision, Datensätze, Ultralytics, YOLO, Objekterkennung, Instanzsegmentierung, Posenschätzung, Bildklassifizierung, Multi-Objekt-Verfolgung
---

# Übersicht über Datensätze

Ultralytics bietet Unterstützung für verschiedene Datensätze an, um Computervisionsaufgaben wie Erkennung, Instanzsegmentierung, Posenschätzung, Klassifizierung und Verfolgung mehrerer Objekte zu erleichtern. Unten finden Sie eine Liste der wichtigsten Ultralytics-Datensätze, gefolgt von einer Zusammenfassung jeder Computervisionsaufgabe und den jeweiligen Datensätzen.

!!! Note "Hinweis"

    🚧 Unsere mehrsprachige Dokumentation befindet sich derzeit im Aufbau und wir arbeiten intensiv an deren Verbesserung. Vielen Dank für Ihre Geduld! 🙏

## [Erkennungsdatensätze](../../datasets/detect/index.md)

Die Objekterkennung mittels Bounding Box ist eine Computervisionstechnik, die das Erkennen und Lokalisieren von Objekten in einem Bild anhand des Zeichnens einer Bounding Box um jedes Objekt beinhaltet.

- [Argoverse](../../datasets/detect/argoverse.md): Ein Datensatz mit 3D-Tracking- und Bewegungsvorhersagedaten aus städtischen Umgebungen mit umfassenden Annotationen.
- [COCO](../../datasets/detect/coco.md): Ein umfangreicher Datensatz für Objekterkennung, Segmentierung und Beschreibung mit über 200.000 beschrifteten Bildern.
- [COCO8](../../datasets/detect/coco8.md): Enthält die ersten 4 Bilder aus COCO Train und COCO Val, geeignet für schnelle Tests.
- [Global Wheat 2020](../../datasets/detect/globalwheat2020.md): Ein Datensatz mit Bildern von Weizenköpfen aus aller Welt für Objekterkennungs- und Lokalisierungsaufgaben.
- [Objects365](../../datasets/detect/objects365.md): Ein hochwertiger, großer Datensatz für Objekterkennung mit 365 Objektkategorien und über 600.000 annotierten Bildern.
- [OpenImagesV7](../../datasets/detect/open-images-v7.md): Ein umfassender Datensatz von Google mit 1,7 Millionen Trainingsbildern und 42.000 Validierungsbildern.
- [SKU-110K](../../datasets/detect/sku-110k.md): Ein Datensatz mit dichter Objekterkennung in Einzelhandelsumgebungen mit über 11.000 Bildern und 1,7 Millionen Bounding Boxen.
- [VisDrone](../../datasets/detect/visdrone.md): Ein Datensatz mit Objekterkennungs- und Multi-Objekt-Tracking-Daten aus Drohnenaufnahmen mit über 10.000 Bildern und Videosequenzen.
- [VOC](../../datasets/detect/voc.md): Der Pascal Visual Object Classes (VOC) Datensatz für Objekterkennung und Segmentierung mit 20 Objektklassen und über 11.000 Bildern.
- [xView](../../datasets/detect/xview.md): Ein Datensatz für Objekterkennung in Überwachungsbildern mit 60 Objektkategorien und über 1 Million annotierten Objekten.

## [Datensätze für Instanzsegmentierung](../../datasets/segment/index.md)

Die Instanzsegmentierung ist eine Computervisionstechnik, die das Identifizieren und Lokalisieren von Objekten in einem Bild auf Pixelebene beinhaltet.

- [COCO](../../datasets/segment/coco.md): Ein großer Datensatz für Objekterkennung, Segmentierung und Beschreibungsaufgaben mit über 200.000 beschrifteten Bildern.
- [COCO8-seg](../../datasets/segment/coco8-seg.md): Ein kleinerer Datensatz für Instanzsegmentierungsaufgaben, der eine Teilmenge von 8 COCO-Bildern mit Segmentierungsannotationen enthält.

## [Posenschätzung](../../datasets/pose/index.md)

Die Posenschätzung ist eine Technik, die verwendet wird, um die Position des Objekts relativ zur Kamera oder zum Weltkoordinatensystem zu bestimmen.

- [COCO](../../datasets/pose/coco.md): Ein großer Datensatz mit menschlichen Pose-Annotationen für Posenschätzungsaufgaben.
- [COCO8-pose](../../datasets/pose/coco8-pose.md): Ein kleinerer Datensatz für Posenschätzungsaufgaben, der eine Teilmenge von 8 COCO-Bildern mit menschlichen Pose-Annotationen enthält.
- [Tiger-pose](../../datasets/pose/tiger-pose.md): Ein kompakter Datensatz bestehend aus 263 Bildern, die auf Tiger fokussiert sind, mit Annotationen von 12 Schlüsselpunkten pro Tiger für Posenschätzungsaufgaben.

## [Bildklassifizierung](../../datasets/classify/index.md)

Die Bildklassifizierung ist eine Computervisionsaufgabe, bei der ein Bild basierend auf seinem visuellen Inhalt in eine oder mehrere vordefinierte Klassen oder Kategorien eingeteilt wird.

- [Caltech 101](../../datasets/classify/caltech101.md): Enthält Bilder von 101 Objektkategorien für Bildklassifizierungsaufgaben.
- [Caltech 256](../../datasets/classify/caltech256.md): Eine erweiterte Version von Caltech 101 mit 256 Objektkategorien und herausfordernderen Bildern.
- [CIFAR-10](../../datasets/classify/cifar10.md): Ein Datensatz mit 60.000 32x32 Farbbildern in 10 Klassen, mit 6.000 Bildern pro Klasse.
- [CIFAR-100](../../datasets/classify/cifar100.md): Eine erweiterte Version von CIFAR-10 mit 100 Objektkategorien und 600 Bildern pro Klasse.
- [Fashion-MNIST](../../datasets/classify/fashion-mnist.md): Ein Datensatz mit 70.000 Graustufenbildern von 10 Modekategorien für Bildklassifizierungsaufgaben.
- [ImageNet](../../datasets/classify/imagenet.md): Ein großer Datensatz für Objekterkennung und Bildklassifizierung mit über 14 Millionen Bildern und 20.000 Kategorien.
- [ImageNet-10](../../datasets/classify/imagenet10.md): Ein kleinerer Teildatensatz von ImageNet mit 10 Kategorien für schnelleres Experimentieren und Testen.
- [Imagenette](../../datasets/classify/imagenette.md): Ein kleinerer Teildatensatz von ImageNet, der 10 leicht unterscheidbare Klassen für ein schnelleres Training und Testen enthält.
- [Imagewoof](../../datasets/classify/imagewoof.md): Ein herausfordernderer Teildatensatz von ImageNet mit 10 Hundezuchtkategorien für Bildklassifizierungsaufgaben.
- [MNIST](../../datasets/classify/mnist.md): Ein Datensatz mit 70.000 Graustufenbildern von handgeschriebenen Ziffern für Bildklassifizierungsaufgaben.

## [Orientierte Bounding Boxes (OBB)](../../datasets/obb/index.md)

Orientierte Bounding Boxes (OBB) ist eine Methode in der Computervision für die Erkennung von geneigten Objekten in Bildern mithilfe von rotierten Bounding Boxen, die oft auf Luft- und Satellitenbilder angewendet wird.

- [DOTAv2](../../datasets/obb/dota-v2.md): Ein beliebter OBB-Datensatz für Luftbildaufnahmen mit 1,7 Millionen Instanzen und 11.268 Bildern.

## [Multi-Objekt-Verfolgung](../../datasets/track/index.md)

Die Verfolgung mehrerer Objekte ist eine Computervisionstechnik, die das Erkennen und Verfolgen mehrerer Objekte über die Zeit in einer Videosequenz beinhaltet.

- [Argoverse](../../datasets/detect/argoverse.md): Ein Datensatz mit 3D-Tracking- und Bewegungsvorhersagedaten aus städtischen Umgebungen mit umfassenden Annotationen für Multi-Objekt-Verfolgungsaufgaben.
- [VisDrone](../../datasets/detect/visdrone.md): Ein Datensatz mit Daten zur Objekterkennung und Multi-Objekt-Verfolgung aus Drohnenaufnahmen mit über 10.000 Bildern und Videosequenzen.

## Neue Datensätze beitragen

Das Bereitstellen eines neuen Datensatzes umfasst mehrere Schritte, um sicherzustellen, dass er gut in die bestehende Infrastruktur integriert werden kann. Unten finden Sie die notwendigen Schritte:

### Schritte um einen neuen Datensatz beizutragen

1. **Bilder sammeln**: Sammeln Sie die Bilder, die zum Datensatz gehören. Diese können von verschiedenen Quellen gesammelt werden, wie öffentlichen Datenbanken oder Ihrer eigenen Sammlung.

2. **Bilder annotieren**: Annotieren Sie diese Bilder mit Bounding Boxen, Segmenten oder Schlüsselpunkten, je nach Aufgabe.

3. **Annotationen exportieren**: Konvertieren Sie diese Annotationen in das von Ultralytics unterstützte YOLO `*.txt`-Dateiformat.

4. **Datensatz organisieren**: Ordnen Sie Ihren Datensatz in die richtige Ordnerstruktur an. Sie sollten übergeordnete Verzeichnisse `train/` und `val/` haben, und innerhalb dieser je ein Unterverzeichnis `images/` und `labels/`.

    ```
    dataset/
    ├── train/
    │   ├── images/
    │   └── labels/
    └── val/
        ├── images/
        └── labels/
    ```

5. **Eine `data.yaml`-Datei erstellen**: Erstellen Sie in Ihrem Stammverzeichnis des Datensatzes eine Datei `data.yaml`, die den Datensatz, die Klassen und andere notwendige Informationen beschreibt.

6. **Bilder optimieren (Optional)**: Wenn Sie die Größe des Datensatzes für eine effizientere Verarbeitung reduzieren möchten, können Sie die Bilder mit dem untenstehenden Code optimieren. Dies ist nicht erforderlich, wird aber für kleinere Datensatzgrößen und schnellere Download-Geschwindigkeiten empfohlen.

7. **Datensatz zippen**: Komprimieren Sie das gesamte Datensatzverzeichnis in eine Zip-Datei.

8. **Dokumentation und PR**: Erstellen Sie eine Dokumentationsseite, die Ihren Datensatz beschreibt und wie er in das bestehende Framework passt. Danach reichen Sie einen Pull Request (PR) ein. Weitere Details zur Einreichung eines PR finden Sie in den [Ultralytics Beitragshinweisen](https://docs.ultralytics.com/help/contributing).

### Beispielcode zum Optimieren und Zippen eines Datensatzes

!!! Example "Optimieren und Zippen eines Datensatzes"

    === "Python"

    ```python
    from pathlib import Path
    from ultralytics.data.utils import compress_one_image
    from ultralytics.utils.downloads import zip_directory

    # Definieren des Verzeichnisses des Datensatzes
    path = Path('Pfad/zum/Datensatz')

    # Bilder im Datensatz optimieren (optional)
    for f in path.rglob('*.jpg'):
        compress_one_image(f)

    # Datensatz in 'Pfad/zum/Datensatz.zip' zippen
    zip_directory(path)
    ```

Indem Sie diesen Schritten folgen, können Sie einen neuen Datensatz beitragen, der gut in die bestehende Struktur von Ultralytics integriert wird.
