---
comments: true
description: Découvrez les fonctionnalités passionnantes de YOLOv8, la dernière version de notre détecteur d'objets en temps réel ! Apprenez comment les architectures avancées, les modèles pré-entraînés et l'équilibre optimal entre précision et vitesse font de YOLOv8 le choix parfait pour vos tâches de détection d'objets.
keywords: YOLOv8, Ultralytics, détecteur d'objets en temps réel, modèles pré-entraînés, documentation, détection d'objets, série YOLO, architectures avancées, précision, vitesse
---

# YOLOv8

## Aperçu

YOLOv8 est la dernière itération de la série YOLO de détecteurs d'objets en temps réel, offrant des performances de pointe en termes de précision et de vitesse. En s'appuyant sur les avancées des précédentes versions de YOLO, YOLOv8 introduit de nouvelles fonctionnalités et optimisations qui en font un choix idéal pour diverses tâches de détection d'objets dans un large éventail d'applications.

![Ultralytics YOLOv8](https://raw.githubusercontent.com/ultralytics/assets/main/yolov8/yolo-comparison-plots.png)

## Principales fonctionnalités

- **Architectures avancées pour le réseau de base et le cou intermédiaire :** YOLOv8 utilise des architectures de réseau de base et de cou intermédiaire de pointe, ce qui améliore l'extraction des caractéristiques et les performances de détection des objets.
- **Tête Ultralytics sans ancre :** YOLOv8 adopte une tête Ultralytics sans ancre, ce qui contribue à une meilleure précision et à un processus de détection plus efficace par rapport aux approches par ancrage.
- **Équilibre optimal entre précision et vitesse :** Avec l'objectif de maintenir un équilibre optimal entre précision et vitesse, YOLOv8 convient aux tâches de détection d'objets en temps réel dans divers domaines d'application.
- **Variété de modèles pré-entraînés :** YOLOv8 propose une gamme de modèles pré-entraînés pour répondre à diverses tâches et exigences de performance, ce qui facilite la recherche du modèle adapté à votre cas d'utilisation spécifique.

## Tâches prises en charge

| Type de modèle | Poids pré-entraînés                                                                                                 | Tâche                   |
|----------------|---------------------------------------------------------------------------------------------------------------------|-------------------------|
| YOLOv8         | `yolov8n.pt`, `yolov8s.pt`, `yolov8m.pt`, `yolov8l.pt`, `yolov8x.pt`                                                | Détection               |
| YOLOv8-seg     | `yolov8n-seg.pt`, `yolov8s-seg.pt`, `yolov8m-seg.pt`, `yolov8l-seg.pt`, `yolov8x-seg.pt`                            | Segmentation d'instance |
| YOLOv8-pose    | `yolov8n-pose.pt`, `yolov8s-pose.pt`, `yolov8m-pose.pt`, `yolov8l-pose.pt`, `yolov8x-pose.pt`, `yolov8x-pose-p6.pt` | Pose/Points clés        |
| YOLOv8-cls     | `yolov8n-cls.pt`, `yolov8s-cls.pt`, `yolov8m-cls.pt`, `yolov8l-cls.pt`, `yolov8x-cls.pt`                            | Classification          |

## Modes pris en charge

| Mode         | Pris en charge |
|--------------|----------------|
| Inférence    | ✅              |
| Validation   | ✅              |
| Entraînement | ✅              |

!!! Performance

    === "Détection (COCO)"

        | Modèle                                                                                | taille<br><sup>(pixels) | mAP<sup>val<br>50-95 | Vitesse<br><sup>CPU ONNX<br>(ms) | Vitesse<br><sup>A100 TensorRT<br>(ms) | paramètres<br><sup>(M) | FLOPs<br><sup>(B) |
        | ------------------------------------------------------------------------------------ | --------------------- | -------------------- | ------------------------------ | ----------------------------------- | ------------------ | ----------------- |
        | [YOLOv8n](https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n.pt) | 640                   | 37,3                 | 80,4                           | 0,99                                | 3,2                | 8,7               |
        | [YOLOv8s](https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8s.pt) | 640                   | 44,9                 | 128,4                          | 1,20                                | 11,2               | 28,6              |
        | [YOLOv8m](https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8m.pt) | 640                   | 50,2                 | 234,7                          | 1,83                                | 25,9               | 78,9              |
        | [YOLOv8l](https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8l.pt) | 640                   | 52,9                 | 375,2                          | 2,39                                | 43,7               | 165,2             |
        | [YOLOv8x](https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8x.pt) | 640                   | 53,9                 | 479,1                          | 3,53                                | 68,2               | 257,8             |

    === "Détection (Open Images V7)"

        Voir la [documentation sur la détection](https://docs.ultralytics.com/tasks/detect/) pour des exemples d'utilisation avec ces modèles entraînés sur [Open Image V7](https://docs.ultralytics.com/datasets/detect/open-images-v7/), qui incluent 600 classes pré-entraînées.

        | Modèle                                                                                      | taille<br><sup>(pixels) | mAP<sup>val<br>50-95 | Vitesse<br><sup>CPU ONNX<br>(ms) | Vitesse<br><sup>A100 TensorRT<br>(ms) | paramètres<br><sup>(M) | FLOPs<br><sup>(B) |
        | ------------------------------------------------------------------------------------------ | --------------------- | -------------------- | ------------------------------ | ----------------------------------- | ------------------ | ----------------- |
        | [YOLOv8n](https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n-oiv7.pt)   | 640                   | 18,4                 | 142,4                          | 1,21                                | 3,5                | 10,5              |
        | [YOLOv8s](https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8s-oiv7.pt)   | 640                   | 27,7                 | 183,1                          | 1,40                                | 11,4               | 29,7              |
        | [YOLOv8m](https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8m-oiv7.pt)   | 640                   | 33,6                 | 408,5                          | 2,26                                | 26,2               | 80,6              |
        | [YOLOv8l](https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8l-oiv7.pt)   | 640                   | 34,9                 | 596,9                          | 2,43                                | 44,1               | 167,4             |
        | [YOLOv8x](https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8x-oiv7.pt)   | 640                   | 36,3                 | 860,6                          | 3,56                                | 68,7               | 260,6             |

    === "Segmentation (COCO)"

        | Modèle                                                                                          | taille<br><sup>(pixels) | mAP<sup>boîte<br>50-95 | mAP<sup>masque<br>50-95 | Vitesse<br><sup>CPU ONNX<br>(ms) | Vitesse<br><sup>A100 TensorRT<br>(ms) | paramètres<br><sup>(M) | FLOPs<br><sup>(B) |
        | ------------------------------------------------------------------------------------------------ | --------------------- | --------------------- | ---------------------- | ------------------------------ | ----------------------------------- | ------------------ | ----------------- |
        | [YOLOv8n-seg](https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n-seg.pt)   | 640                   | 36,7                  | 30,5                   | 96,1                           | 1,21                                | 3,4                | 12,6              |
        | [YOLOv8s-seg](https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8s-seg.pt)   | 640                   | 44,6                  | 36,8                   | 155,7                          | 1,47                                | 11,8               | 42,6              |
        | [YOLOv8m-seg](https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8m-seg.pt)   | 640                   | 49,9                  | 40,8                   | 317,0                          | 2,18                                | 27,3               | 110,2             |
        | [YOLOv8l-seg](https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8l-seg.pt)   | 640                   | 52,3                  | 42,6                   | 572,4                          | 2,79                                | 46,0               | 220,5             |
        | [YOLOv8x-seg](https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8x-seg.pt)   | 640                   | 53,4                  | 43,4                   | 712,1                          | 4,02                                | 71,8               | 344,1             |

    === "Classification (ImageNet)"

        | Modèle                                                                                          | taille<br><sup>(pixels) | acc<br><sup>top1 | acc<br><sup>top5 | Vitesse<br><sup>CPU ONNX<br>(ms) | Vitesse<br><sup>A100 TensorRT<br>(ms) | paramètres<br><sup>(M) | FLOPs<br><sup>(B) pour 640 |
        | ------------------------------------------------------------------------------------------------ | --------------------- | ---------------- | ---------------- | ------------------------------ | ----------------------------------- | ------------------ | ------------------------ |
        | [YOLOv8n-cls](https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n-cls.pt)   | 224                   | 66,6             | 87,0             | 12,9                           | 0,31                                | 2,7                | 4,3                      |
        | [YOLOv8s-cls](https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8s-cls.pt)   | 224                   | 72,3             | 91,1             | 23,4                           | 0,35                                | 6,4                | 13,5                     |
        | [YOLOv8m-cls](https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8m-cls.pt)   | 224                   | 76,4             | 93,2             | 85,4                           | 0,62                                | 17,0               | 42,7                     |
        | [YOLOv8l-cls](https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8l-cls.pt)   | 224                   | 78,0             | 94,1             | 163,0                          | 0,87                                | 37,5               | 99,7                     |
        | [YOLOv8x-cls](https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8x-cls.pt)   | 224                   | 78,4             | 94,3             | 232,0                          | 1,01                                | 57,4               | 154,8                    |

    === "Pose (COCO)"

        | Modèle                                                                                          | taille<br><sup>(pixels) | mAP<sup>pose<br>50-95 | mAP<sup>pose<br>50 | Vitesse<br><sup>CPU ONNX<br>(ms) | Vitesse<br><sup>A100 TensorRT<br>(ms) | paramètres<br><sup>(M) | FLOPs<br><sup>(B) |
        | ------------------------------------------------------------------------------------------------ | --------------------- | --------------------- | ------------------ | ------------------------------ | ----------------------------------- | ------------------ | ----------------- |
        | [YOLOv8n-pose](https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n-pose.pt) | 640                   | 50,4                  | 80,1               | 131,8                          | 1,18                                | 3,3                | 9,2               |
        | [YOLOv8s-pose](https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8s-pose.pt) | 640                   | 60,0                  | 86,2               | 233,2                          | 1,42                                | 11,6               | 30,2              |
        | [YOLOv8m-pose](https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8m-pose.pt) | 640                   | 65,0                  | 88,8               | 456,3                          | 2,00                                | 26,4               | 81,0              |
        | [YOLOv8l-pose](https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8l-pose.pt) | 640                   | 67,6                  | 90,0               | 784,5                          | 2,59                                | 44,4               | 168,6             |
        | [YOLOv8x-pose](https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8x-pose.pt) | 640                   | 69,2                  | 90,2               | 1607,1                         | 3,73                                | 69,4               | 263,2             |
        | [YOLOv8x-pose-p6](https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8x-pose-p6.pt) | 1280                  | 71,6                  | 91,2               | 4088,7                         | 10,04                               | 99,1               | 1066,4            |

## Utilisation

Vous pouvez utiliser YOLOv8 pour des tâches de détection d'objets en utilisant le package pip Ultralytics. Voici un exemple de code montrant comment utiliser les modèles YOLOv8 pour l'inférence :

!!! Exemple ""

    Cet exemple fournit un code d'inférence simple pour YOLOv8. Pour plus d'options, y compris la manipulation des résultats d'inférence, consultez le mode [Predict](../modes/predict.md). Pour utiliser YOLOv8 avec d'autres modes, consultez les modes [Train](../modes/train.md), [Val](../modes/val.md) et [Export](../modes/export.md).

    === "Python"

        Les modèles pré-entraînés PyTorch `*.pt`, ainsi que les fichiers de configuration `*.yaml`, peuvent être transmis à la classe `YOLO()` pour créer une instance de modèle en python :

        ```python
        from ultralytics import YOLO

        # Charger un modèle YOLOv8n pré-entraîné sur COCO
        model = YOLO('yolov8n.pt')

        # Afficher les informations sur le modèle (facultatif)
        model.info()

        # Entraîner le modèle sur l'ensemble de données d'exemple COCO8 pendant 100 époques
        results = model.train(data='coco8.yaml', epochs=100, imgsz=640)

        # Effectuer une inférence avec le modèle YOLOv8n sur l'image 'bus.jpg'
        results = model('path/to/bus.jpg')
        ```

    === "CLI"

        Des commandes CLI sont disponibles pour exécuter directement les modèles :

        ```bash
        # Charger un modèle YOLOv8n pré-entraîné sur COCO et l'entraîner sur l'ensemble de données d'exemple COCO8 pendant 100 époques
        yolo train model=yolov8n.pt data=coco8.yaml epochs=100 imgsz=640

        # Charger un modèle YOLOv8n pré-entraîné sur COCO et effectuer une inférence sur l'image 'bus.jpg'
        yolo predict model=yolov8n.pt source=path/to/bus.jpg
        ```

## Citations et remerciements

Si vous utilisez le modèle YOLOv8 ou tout autre logiciel de ce référentiel dans votre travail, veuillez le citer selon le format suivant :

!!! Note ""

    === "BibTeX"

        ```bibtex
        @software{yolov8_ultralytics,
          author = {Glenn Jocher et Ayush Chaurasia et Jing Qiu},
          title = {Ultralytics YOLOv8},
          version = {8.0.0},
          year = {2023},
          url = {https://github.com/ultralytics/ultralytics},
          orcid = {0000-0001-5950-6979, 0000-0002-7603-6750, 0000-0003-3783-7069},
          license = {AGPL-3.0}
        }
        ```

Veuillez noter que le DOI est en attente et sera ajouté à la citation une fois disponible. L'utilisation du logiciel est conforme à la licence AGPL-3.0.
