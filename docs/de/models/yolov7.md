---
comments: true
description: Erforsche den YOLOv7, einen echtzeitfähigen Objektdetektor. Verstehe seine überlegene Geschwindigkeit, beeindruckende Genauigkeit und seinen einzigartigen Fokus auf die optimierte Ausbildung mit "trainable bag-of-freebies".
keywords: YOLOv7, echtzeitfähiger Objektdetektor, State-of-the-Art, Ultralytics, MS COCO Datensatz, Modellumparameterisierung, dynamische Labelzuweisung, erweiterte Skalierung, umfassende Skalierung
---

# YOLOv7: Trainable Bag-of-Freebies

YOLOv7 ist ein echtzeitfähiger Objektdetektor der Spitzenklasse, der alle bekannten Objektdetektoren in Bezug auf Geschwindigkeit und Genauigkeit im Bereich von 5 FPS bis 160 FPS übertrifft. Mit einer Genauigkeit von 56,8% AP ist er der präziseste Echtzeit-Objektdetektor unter allen bekannten Modellen mit einer FPS von 30 oder höher auf der GPU V100. Darüber hinaus übertrifft YOLOv7 andere Objektdetektoren wie YOLOR, YOLOX, Scaled-YOLOv4, YOLOv5 und viele andere in Bezug auf Geschwindigkeit und Genauigkeit. Das Modell wird ausschließlich auf dem MS COCO-Datensatz trainiert, ohne andere Datensätze oder vortrainierte Gewichte zu verwenden. Sourcecode für YOLOv7 ist auf GitHub verfügbar.

![Vergleich von YOLOv7 mit SOTA-Objektdetektoren](https://github.com/ultralytics/ultralytics/assets/26833433/5e1e0420-8122-4c79-b8d0-2860aa79af92)
**Vergleich von Spitzen-Objektdetektoren.
** Aus den Ergebnissen in Tabelle 2 wissen wir, dass die vorgeschlagene Methode das beste Verhältnis von Geschwindigkeit und Genauigkeit umfassend aufweist. Vergleichen wir YOLOv7-tiny-SiLU mit YOLOv5-N (r6.1), so ist unsere Methode 127 FPS schneller und um 10,7% genauer beim AP. Darüber hinaus erreicht YOLOv7 bei einer Bildrate von 161 FPS einen AP von 51,4%, während PPYOLOE-L mit demselben AP nur eine Bildrate von 78 FPS aufweist. In Bezug auf die Parameterverwendung ist YOLOv7 um 41% geringer als PPYOLOE-L. Vergleicht man YOLOv7-X mit 114 FPS Inferenzgeschwindigkeit mit YOLOv5-L (r6.1) mit 99 FPS Inferenzgeschwindigkeit, kann YOLOv7-X den AP um 3,9% verbessern. Wenn YOLOv7-X mit YOLOv5-X (r6.1) in ähnlichem Maßstab verglichen wird, ist die Inferenzgeschwindigkeit von YOLOv7-X 31 FPS schneller. Darüber hinaus reduziert YOLOv7-X in Bezug auf die Anzahl der Parameter und Berechnungen 22% der Parameter und 8% der Berechnungen im Vergleich zu YOLOv5-X (r6.1), verbessert jedoch den AP um 2,2% ([Source](https://arxiv.org/pdf/2207.02696.pdf)).

## Übersicht

Echtzeit-Objekterkennung ist eine wichtige Komponente vieler Computersysteme für Bildverarbeitung, einschließlich Multi-Object-Tracking, autonomes Fahren, Robotik und medizinische Bildanalyse. In den letzten Jahren konzentrierte sich die Entwicklung der Echtzeit-Objekterkennung auf die Gestaltung effizienter Architekturen und die Verbesserung der Inferenzgeschwindigkeit verschiedener CPUs, GPUs und Neural Processing Units (NPUs). YOLOv7 unterstützt sowohl mobile GPUs als auch GPU-Geräte, von der Edge bis zur Cloud.

Im Gegensatz zu herkömmlichen, echtzeitfähigen Objektdetektoren, die sich auf die Architekturoptimierung konzentrieren, führt YOLOv7 eine Fokussierung auf die Optimierung des Schulungsprozesses ein. Dazu gehören Module und Optimierungsmethoden, die darauf abzielen, die Genauigkeit der Objekterkennung zu verbessern, ohne die Inferenzkosten zu erhöhen - ein Konzept, das als "trainable bag-of-freebies" bekannt ist.

## Hauptmerkmale

YOLOv7 führt mehrere Schlüsselfunktionen ein:

1. **Modellumparameterisierung**: YOLOv7 schlägt ein geplantes umparameterisiertes Modell vor, das eine in verschiedenen Netzwerken anwendbare Strategie darstellt und auf dem Konzept des Gradientenpropagationspfades basiert.

2. **Dynamische Labelzuweisung**: Das Training des Modells mit mehreren Ausgabeschichten stellt ein neues Problem dar: "Wie weist man dynamische Ziele für die Ausgaben der verschiedenen Zweige zu?" Zur Lösung dieses Problems führt YOLOv7 eine neue Methode zur Labelzuweisung ein, die als coarse-to-fine lead guided label assignment bekannt ist.

3. **Erweiterte und umfassende Skalierung**: YOLOv7 schlägt Methoden zur "erweiterten" und "umfassenden Skalierung" des echtzeitfähigen Objektdetektors vor, die Parameter und Berechnungen effektiv nutzen können.

4. **Effizienz**: Die von YOLOv7 vorgeschlagene Methode kann etwa 40 % der Parameter und 50 % der Berechnungen des state-of-the-art echtzeitfähigen Objektdetektors wirksam reduzieren und weist eine schnellere Inferenzgeschwindigkeit und eine höhere Detektionsgenauigkeit auf.

## Beispiele zur Nutzung

Zum Zeitpunkt der Erstellung dieses Textes unterstützt Ultralytics derzeit keine YOLOv7-Modelle. Daher müssen sich alle Benutzer, die YOLOv7 verwenden möchten, direkt an das YOLOv7 GitHub-Repository für Installations- und Nutzungshinweise wenden.

Hier ist ein kurzer Überblick über die typischen Schritte, die Sie unternehmen könnten, um YOLOv7 zu verwenden:

1. Besuchen Sie das YOLOv7 GitHub-Repository: [https://github.com/WongKinYiu/yolov7](https://github.com/WongKinYiu/yolov7).

2. Befolgen Sie die in der README-Datei bereitgestellten Anweisungen zur Installation. Dies beinhaltet in der Regel das Klonen des Repositories, die Installation der erforderlichen Abhängigkeiten und das Einrichten eventuell notwendiger Umgebungsvariablen.

3. Sobald die Installation abgeschlossen ist, können Sie das Modell entsprechend den im Repository bereitgestellten Anleitungen trainieren und verwenden. Dies umfasst in der Regel die Vorbereitung des Datensatzes, das Konfigurieren der Modellparameter, das Training des Modells und anschließend die Verwendung des trainierten Modells zur Durchführung der Objekterkennung.

Bitte beachten Sie, dass die spezifischen Schritte je nach Ihrem spezifischen Anwendungsfall und dem aktuellen Stand des YOLOv7-Repositories variieren können. Es wird daher dringend empfohlen, sich direkt an die im YOLOv7 GitHub-Repository bereitgestellten Anweisungen zu halten.

Wir bedauern etwaige Unannehmlichkeiten und werden uns bemühen, dieses Dokument mit Anwendungsbeispielen für Ultralytics zu aktualisieren, sobald die Unterstützung für YOLOv7 implementiert ist.

## Zitationen und Danksagungen

Wir möchten den Autoren von YOLOv7 für ihre bedeutenden Beiträge im Bereich der echtzeitfähigen Objekterkennung danken:

!!! Quote ""

    === "BibTeX"

        ```bibtex
        @article{wang2022yolov7,
          title={{YOLOv7}: Trainable bag-of-freebies sets new state-of-the-art for real-time object detectors},
          author={Wang, Chien-Yao and Bochkovskiy, Alexey and Liao, Hong-Yuan Mark},
          journal={arXiv preprint arXiv:2207.02696},
          year={2022}
        }
        ```

Die ursprüngliche YOLOv7-Studie kann auf [arXiv](https://arxiv.org/pdf/2207.02696.pdf) gefunden werden. Die Autoren haben ihre Arbeit öffentlich zugänglich gemacht, und der Code kann auf [GitHub](https://github.com/WongKinYiu/yolov7) abgerufen werden. Wir schätzen ihre Bemühungen, das Feld voranzubringen und ihre Arbeit der breiteren Gemeinschaft zugänglich zu machen.
