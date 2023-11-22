---
comments: true
description: Erforschen Sie unseren detaillierten Leitfaden zu YOLOv4, einem hochmodernen Echtzeit-Objektdetektor. Erfahren Sie mehr über seine architektonischen Highlights, innovativen Funktionen und Anwendungsbeispiele.
keywords: ultralytics, YOLOv4, Objekterkennung, neuronales Netzwerk, Echtzeit-Erkennung, Objektdetektor, maschinelles Lernen
---

# YOLOv4: Schnelle und präzise Objekterkennung

Willkommen auf der Ultralytics-Dokumentationsseite für YOLOv4, einem hochmodernen, Echtzeit-Objektdetektor, der 2020 von Alexey Bochkovskiy unter [https://github.com/AlexeyAB/darknet](https://github.com/AlexeyAB/darknet) veröffentlicht wurde. YOLOv4 wurde entwickelt, um das optimale Gleichgewicht zwischen Geschwindigkeit und Genauigkeit zu bieten und ist somit eine ausgezeichnete Wahl für viele Anwendungen.

![YOLOv4 Architekturdiagramm](https://user-images.githubusercontent.com/26833433/246185689-530b7fe8-737b-4bb0-b5dd-de10ef5aface.png)
**YOLOv4 Architekturdiagramm**. Zeigt das komplexe Netzwerkdesign von YOLOv4, einschließlich der Backbone-, Neck- und Head-Komponenten sowie ihrer verbundenen Schichten für eine optimale Echtzeit-Objekterkennung.

## Einleitung

YOLOv4 steht für You Only Look Once Version 4. Es handelt sich um ein Echtzeit-Objekterkennungsmodell, das entwickelt wurde, um die Grenzen früherer YOLO-Versionen wie [YOLOv3](yolov3.md) und anderer Objekterkennungsmodelle zu überwinden. Im Gegensatz zu anderen konvolutionellen neuronalen Netzwerken (CNN), die auf Objekterkennung basieren, ist YOLOv4 nicht nur für Empfehlungssysteme geeignet, sondern auch für eigenständiges Prozessmanagement und Reduzierung der Benutzereingabe. Durch den Einsatz von herkömmlichen Grafikprozessoreinheiten (GPUs) ermöglicht es YOLOv4 eine Massennutzung zu einem erschwinglichen Preis und ist so konzipiert, dass es in Echtzeit auf einer herkömmlichen GPU funktioniert, wobei nur eine solche GPU für das Training erforderlich ist.

## Architektur

YOLOv4 nutzt mehrere innovative Funktionen, die zusammenarbeiten, um seine Leistung zu optimieren. Dazu gehören Weighted-Residual-Connections (WRC), Cross-Stage-Partial-connections (CSP), Cross mini-Batch Normalization (CmBN), Self-adversarial-training (SAT), Mish-Aktivierung, Mosaic-Datenaugmentation, DropBlock-Regularisierung und CIoU-Verlust. Diese Funktionen werden kombiniert, um erstklassige Ergebnisse zu erzielen.

Ein typischer Objektdetektor besteht aus mehreren Teilen, darunter der Eingabe, dem Backbone, dem Neck und dem Head. Das Backbone von YOLOv4 ist auf ImageNet vorgeschult und wird zur Vorhersage von Klassen und Begrenzungsrahmen von Objekten verwendet. Das Backbone kann aus verschiedenen Modellen wie VGG, ResNet, ResNeXt oder DenseNet stammen. Der Neck-Teil des Detektors wird verwendet, um Merkmalskarten von verschiedenen Stufen zu sammeln und umfasst normalerweise mehrere Bottom-up-Pfade und mehrere Top-down-Pfade. Der Head-Teil wird schließlich zur Durchführung der endgültigen Objekterkennung und Klassifizierung verwendet.

## Bag of Freebies

YOLOv4 verwendet auch Methoden, die als "Bag of Freebies" bekannt sind. Dabei handelt es sich um Techniken, die die Genauigkeit des Modells während des Trainings verbessern, ohne die Kosten der Inferenz zu erhöhen. Datenaugmentation ist eine häufige Bag of Freebies-Technik, die in der Objekterkennung verwendet wird, um die Variabilität der Eingabebilder zu erhöhen und die Robustheit des Modells zu verbessern. Beispiele für Datenaugmentation sind photometrische Verzerrungen (Anpassung von Helligkeit, Kontrast, Farbton, Sättigung und Rauschen eines Bildes) und geometrische Verzerrungen (Hinzufügen von zufälliger Skalierung, Ausschnitt, Spiegelung und Rotation). Diese Techniken helfen dem Modell, sich besser an verschiedene Arten von Bildern anzupassen.

## Funktionen und Leistung

YOLOv4 ist für optimale Geschwindigkeit und Genauigkeit in der Objekterkennung konzipiert. Die Architektur von YOLOv4 umfasst CSPDarknet53 als Backbone, PANet als Neck und YOLOv3 als Detektionskopf. Diese Konstruktion ermöglicht es YOLOv4, beeindruckend schnelle Objekterkennungen durchzuführen und ist somit für Echtzeitanwendungen geeignet. YOLOv4 zeichnet sich auch durch Genauigkeit aus und erzielt erstklassige Ergebnisse in Objekterkennungs-Benchmarks.

## Beispiele für die Verwendung

Zum Zeitpunkt der Erstellung dieser Dokumentation unterstützt Ultralytics derzeit keine YOLOv4-Modelle. Daher müssen sich Benutzer, die YOLOv4 verwenden möchten, direkt an das YOLOv4 GitHub-Repository für Installations- und Verwendungshinweise wenden.

Hier ist ein kurzer Überblick über die typischen Schritte, die Sie unternehmen könnten, um YOLOv4 zu verwenden:

1. Besuchen Sie das YOLOv4 GitHub-Repository: [https://github.com/AlexeyAB/darknet](https://github.com/AlexeyAB/darknet).

2. Befolgen Sie die in der README-Datei bereitgestellten Anweisungen zur Installation. Dies beinhaltet in der Regel das Klonen des Repositories, die Installation der erforderlichen Abhängigkeiten und das Einrichten der erforderlichen Umgebungsvariablen.

3. Sobald die Installation abgeschlossen ist, können Sie das Modell gemäß den in dem Repository bereitgestellten Verwendungshinweisen trainieren und verwenden. Dies beinhaltet in der Regel die Vorbereitung des Datensatzes, die Konfiguration der Modellparameter, das Training des Modells und die anschließende Verwendung des trainierten Modells zur Durchführung der Objekterkennung.

Bitte beachten Sie, dass die spezifischen Schritte je nach Ihrer spezifischen Anwendung und dem aktuellen Stand des YOLOv4-Repositories variieren können. Es wird daher dringend empfohlen, sich direkt an die Anweisungen im YOLOv4-GitHub-Repository zu halten.

Wir bedauern etwaige Unannehmlichkeiten und werden uns bemühen, dieses Dokument mit Verwendungsbeispielen für Ultralytics zu aktualisieren, sobald die Unterstützung für YOLOv4 implementiert ist.

## Fazit

YOLOv4 ist ein leistungsstarkes und effizientes Modell zur Objekterkennung, das eine Balance zwischen Geschwindigkeit und Genauigkeit bietet. Durch den Einsatz einzigartiger Funktionen und Bag of Freebies-Techniken während des Trainings erzielt es hervorragende Ergebnisse in Echtzeit-Objekterkennungsaufgaben. YOLOv4 kann von jedem mit einer herkömmlichen GPU trainiert und verwendet werden, was es für eine Vielzahl von Anwendungen zugänglich und praktisch macht.

## Zitate und Anerkennungen

Wir möchten den Autoren von YOLOv4 für ihren bedeutenden Beitrag auf dem Gebiet der Echtzeit-Objekterkennung danken:

!!! Quote ""

    === "BibTeX"

        ```bibtex
        @misc{bochkovskiy2020yolov4,
              title={YOLOv4: Optimal Speed and Accuracy of Object Detection},
              author={Alexey Bochkovskiy and Chien-Yao Wang and Hong-Yuan Mark Liao},
              year={2020},
              eprint={2004.10934},
              archivePrefix={arXiv},
              primaryClass={cs.CV}
        }
        ```

Die originale YOLOv4-Publikation finden Sie auf [arXiv](https://arxiv.org/pdf/2004.10934.pdf). Die Autoren haben ihre Arbeit öffentlich zugänglich gemacht und der Code kann auf [GitHub](https://github.com/AlexeyAB/darknet) abgerufen werden. Wir schätzen ihre Bemühungen, das Fachgebiet voranzubringen und ihre Arbeit der breiteren Community zugänglich zu machen.
