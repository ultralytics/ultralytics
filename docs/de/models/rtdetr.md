---
comments: true
description: Entdecken Sie die Funktionen und Vorteile von RT-DETR, dem effizienten und anpassungsfähigen Echtzeitobjektdetektor von Baidu, der von Vision Transformers unterstützt wird, einschließlich vortrainierter Modelle.
keywords: RT-DETR, Baidu, Vision Transformers, Objekterkennung, Echtzeitleistung, CUDA, TensorRT, IoU-bewusste Query-Auswahl, Ultralytics, Python API, PaddlePaddle
---

# Baidus RT-DETR: Ein Echtzeit-Objektdetektor auf Basis von Vision Transformers

## Überblick

Der Real-Time Detection Transformer (RT-DETR), entwickelt von Baidu, ist ein moderner End-to-End-Objektdetektor, der Echtzeitleistung mit hoher Genauigkeit bietet. Er nutzt die Leistung von Vision Transformers (ViT), um Multiskalen-Funktionen effizient zu verarbeiten, indem intra-skaliere Interaktion und eine skalenübergreifende Fusion entkoppelt werden. RT-DETR ist hoch anpassungsfähig und unterstützt flexible Anpassung der Inferenzgeschwindigkeit durch Verwendung verschiedener Decoder-Schichten ohne erneutes Training. Das Modell übertrifft viele andere Echtzeit-Objektdetektoren auf beschleunigten Backends wie CUDA mit TensorRT.

![Beispielbild des Modells](https://user-images.githubusercontent.com/26833433/238963168-90e8483f-90aa-4eb6-a5e1-0d408b23dd33.png)
**Übersicht von Baidus RT-DETR.** Die Modellarchitekturdiagramm des RT-DETR zeigt die letzten drei Stufen des Backbone {S3, S4, S5} als Eingabe für den Encoder. Der effiziente Hybrid-Encoder verwandelt Multiskalen-Funktionen durch intraskalare Feature-Interaktion (AIFI) und das skalenübergreifende Feature-Fusion-Modul (CCFM) in eine Sequenz von Bildmerkmalen. Die IoU-bewusste Query-Auswahl wird verwendet, um eine feste Anzahl von Bildmerkmalen als anfängliche Objekt-Queries für den Decoder auszuwählen. Der Decoder optimiert iterativ Objekt-Queries, um Boxen und Vertrauenswerte zu generieren ([Quelle](https://arxiv.org/pdf/2304.08069.pdf)).

### Hauptmerkmale

- **Effizienter Hybrid-Encoder:** Baidus RT-DETR verwendet einen effizienten Hybrid-Encoder, der Multiskalen-Funktionen verarbeitet, indem intra-skaliere Interaktion und eine skalenübergreifende Fusion entkoppelt werden. Dieses einzigartige Design auf Basis von Vision Transformers reduziert die Rechenkosten und ermöglicht die Echtzeit-Objekterkennung.
- **IoU-bewusste Query-Auswahl:** Baidus RT-DETR verbessert die Initialisierung von Objekt-Queries, indem IoU-bewusste Query-Auswahl verwendet wird. Dadurch kann das Modell sich auf die relevantesten Objekte in der Szene konzentrieren und die Erkennungsgenauigkeit verbessern.
- **Anpassbare Inferenzgeschwindigkeit:** Baidus RT-DETR ermöglicht flexible Anpassungen der Inferenzgeschwindigkeit durch Verwendung unterschiedlicher Decoder-Schichten ohne erneutes Training. Diese Anpassungsfähigkeit erleichtert den praktischen Einsatz in verschiedenen Echtzeit-Objekterkennungsszenarien.

## Vortrainierte Modelle

Die Ultralytics Python API bietet vortrainierte PaddlePaddle RT-DETR-Modelle in verschiedenen Skalierungen:

- RT-DETR-L: 53,0% AP auf COCO val2017, 114 FPS auf T4 GPU
- RT-DETR-X: 54,8% AP auf COCO val2017, 74 FPS auf T4 GPU

## Beispiele für die Verwendung

Das folgende Beispiel enthält einfache Trainings- und Inferenzbeispiele für RT-DETRR. Für die vollständige Dokumentation zu diesen und anderen [Modi](../modes/index.md) siehe die Dokumentationsseiten für [Predict](../modes/predict.md), [Train](../modes/train.md), [Val](../modes/val.md) und [Export](../modes/export.md).

!!! Example "Beispiel"

    === "Python"

        ```python
        from ultralytics import RTDETR

        # Laden Sie ein vortrainiertes RT-DETR-l Modell auf COCO
        model = RTDETR('rtdetr-l.pt')

        # Zeigen Sie Informationen über das Modell an (optional)
        model.info()

        # Trainieren Sie das Modell auf dem COCO8-Beispiel-Datensatz für 100 Epochen
        results = model.train(data='coco8.yaml', epochs=100, imgsz=640)

        # Führen Sie die Inferenz mit dem RT-DETR-l Modell auf dem Bild 'bus.jpg' aus
        results = model('path/to/bus.jpg')
        ```

    === "CLI"

        ```bash
        # Laden Sie ein vortrainiertes RT-DETR-l Modell auf COCO und trainieren Sie es auf dem COCO8-Beispiel-Datensatz für 100 Epochen
        yolo train model=rtdetr-l.pt data=coco8.yaml epochs=100 imgsz=640

        # Laden Sie ein vortrainiertes RT-DETR-l Modell auf COCO und führen Sie die Inferenz auf dem Bild 'bus.jpg' aus
        yolo predict model=rtdetr-l.pt source=path/to/bus.jpg
        ```

## Unterstützte Aufgaben und Modi

In dieser Tabelle werden die Modelltypen, die spezifischen vortrainierten Gewichte, die von jedem Modell unterstützten Aufgaben und die verschiedenen Modi ([Train](../modes/train.md), [Val](../modes/val.md), [Predict](../modes/predict.md), [Export](../modes/export.md)), die unterstützt werden, mit ✅-Emoji angezeigt.

| Modelltyp          | Vortrainierte Gewichte | Unterstützte Aufgaben                 | Inferenz | Validierung | Training | Exportieren |
|--------------------|------------------------|---------------------------------------|----------|-------------|----------|-------------|
| RT-DETR Groß       | `rtdetr-l.pt`          | [Objekterkennung](../tasks/detect.md) | ✅        | ✅           | ✅        | ✅           |
| RT-DETR Extra-Groß | `rtdetr-x.pt`          | [Objekterkennung](../tasks/detect.md) | ✅        | ✅           | ✅        | ✅           |

## Zitate und Danksagungen

Wenn Sie Baidus RT-DETR in Ihrer Forschungs- oder Entwicklungsarbeit verwenden, zitieren Sie bitte das [ursprüngliche Papier](https://arxiv.org/abs/2304.08069):

!!! Quote ""

    === "BibTeX"

        ```bibtex
        @misc{lv2023detrs,
              title={DETRs Beat YOLOs on Real-time Object Detection},
              author={Wenyu Lv and Shangliang Xu and Yian Zhao and Guanzhong Wang and Jinman Wei and Cheng Cui and Yuning Du and Qingqing Dang and Yi Liu},
              year={2023},
              eprint={2304.08069},
              archivePrefix={arXiv},
              primaryClass={cs.CV}
        }
        ```

Wir möchten Baidu und dem [PaddlePaddle](https://github.com/PaddlePaddle/PaddleDetection)-Team für die Erstellung und Pflege dieser wertvollen Ressource für die Computer-Vision-Community danken. Ihre Beitrag zum Gebiet der Entwicklung des Echtzeit-Objekterkenners auf Basis von Vision Transformers, RT-DETR, wird sehr geschätzt.

*Keywords: RT-DETR, Transformer, ViT, Vision Transformers, Baidu RT-DETR, PaddlePaddle, Paddle Paddle RT-DETR, Objekterkennung in Echtzeit, objekterkennung basierend auf Vision Transformers, vortrainierte PaddlePaddle RT-DETR Modelle, Verwendung von Baidus RT-DETR, Ultralytics Python API*
