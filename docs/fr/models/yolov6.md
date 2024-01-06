---
comments: true
description: Explorez Meituan YOLOv6, un modèle de détection d'objets à la pointe de la technologie offrant un équilibre entre vitesse et précision. Plongez-vous dans les fonctionnalités, les modèles pré-entraînés et l'utilisation de Python.
keywords: Meituan YOLOv6, détection d'objets, Ultralytics, YOLOv6 docs, Bi-directional Concatenation, Anchor-Aided Training, modèles pré-entraînés, applications en temps réel
---

# Meituan YOLOv6

## Vue d'ensemble

[Meituan](https://about.meituan.com/) YOLOv6 est un détecteur d'objets de pointe qui offre un équilibre remarquable entre vitesse et précision, ce qui en fait un choix populaire pour les applications en temps réel. Ce modèle introduit plusieurs améliorations remarquables sur son architecture et son schéma d'entraînement, notamment la mise en œuvre d'un module de concaténation bidirectionnelle (BiC), d'une stratégie d'entraînement assistée par ancrage (AAT) et d'une conception améliorée de l'épine dorsale et du cou pour une précision de pointe sur l'ensemble de données COCO.

![Meituan YOLOv6](https://user-images.githubusercontent.com/26833433/240750495-4da954ce-8b3b-41c4-8afd-ddb74361d3c2.png)
![Exemple d'image du modèle](https://user-images.githubusercontent.com/26833433/240750557-3e9ec4f0-0598-49a8-83ea-f33c91eb6d68.png)
**Aperçu de YOLOv6.** Diagramme de l'architecture du modèle montrant les composants du réseau redessinés et les stratégies d'entraînement qui ont conduit à d'importantes améliorations des performances. (a) L'épine dorsale de YOLOv6 (N et S sont indiqués). Notez que pour M/L, RepBlocks est remplacé par CSPStackRep. (b) La structure d'un module BiC. (c) Un bloc SimCSPSPPF. ([source](https://arxiv.org/pdf/2301.05586.pdf)).

### Caractéristiques principales

- **Module de concaténation bidirectionnelle (BiC) :** YOLOv6 introduit un module BiC dans le cou du détecteur, améliorant les signaux de localisation et offrant des gains de performance avec une dégradation de vitesse négligeable.
- **Stratégie d'entraînement assistée par ancrage (AAT) :** Ce modèle propose AAT pour profiter des avantages des paradigmes basés sur ancrage et sans ancrage sans compromettre l'efficacité de l'inférence.
- **Conception améliorée de l'épine dorsale et du cou :** En approfondissant YOLOv6 pour inclure une autre étape dans l'épine dorsale et le cou, ce modèle atteint des performances de pointe sur l'ensemble de données COCO avec une entrée haute résolution.
- **Stratégie d'autodistillation :** Une nouvelle stratégie d'autodistillation est mise en œuvre pour améliorer les performances des modèles plus petits de YOLOv6, en améliorant la branche de régression auxiliaire pendant l'entraînement et en la supprimant lors de l'inférence afin d'éviter une baisse notable de la vitesse.

## Métriques de performance

YOLOv6 propose différents modèles pré-entraînés avec différentes échelles :

- YOLOv6-N : 37,5 % de précision sur COCO val2017 à 1187 FPS avec le GPU NVIDIA Tesla T4.
- YOLOv6-S : 45,0 % de précision à 484 FPS.
- YOLOv6-M : 50,0 % de précision à 226 FPS.
- YOLOv6-L : 52,8 % de précision à 116 FPS.
- YOLOv6-L6 : Précision de pointe en temps réel.

YOLOv6 propose également des modèles quantifiés pour différentes précisions et des modèles optimisés pour les plates-formes mobiles.

## Exemples d'utilisation

Cet exemple fournit des exemples simples d'entraînement et d'inférence de YOLOv6. Pour une documentation complète sur ces exemples et d'autres [modes](../modes/index.md), consultez les pages de documentation [Predict](../modes/predict.md), [Train](../modes/train.md), [Val](../modes/val.md) et [Export](../modes/export.md).

!!! Example "Exemple"

    === "Python"

        Les modèles pré-entraînés PyTorch `*.pt`, ainsi que les fichiers de configuration `*.yaml`, peuvent être utilisés pour créer une instance de modèle en python en utilisant la classe `YOLO()` :

        ```python
        from ultralytics import YOLO

        # Créer un modèle YOLOv6n à partir de zéro
        model = YOLO('yolov6n.yaml')

        # Afficher les informations sur le modèle (facultatif)
        model.info()

        # Entraîner le modèle sur l'ensemble de données d'exemple COCO8 pendant 100 epochs
        results = model.train(data='coco8.yaml', epochs=100, imgsz=640)

        # Effectuer une inférence avec le modèle YOLOv6n sur l'image 'bus.jpg'
        results = model('path/to/bus.jpg')
        ```

    === "CLI"

        Des commandes CLI sont disponibles pour exécuter directement les modèles :

        ```bash
        # Créer un modèle YOLOv6n à partir de zéro et l'entraîner sur l'ensemble de données d'exemple COCO8 pendant 100 epochs
        yolo train model=yolov6n.yaml data=coco8.yaml epochs=100 imgsz=640

        # Créer un modèle YOLOv6n à partir de zéro et effectuer une inférence sur l'image 'bus.jpg'
        yolo predict model=yolov6n.yaml source=path/to/bus.jpg
        ```

## Tâches et modes pris en charge

La série YOLOv6 propose une gamme de modèles, chacun optimisé pour la [détection d'objets](../tasks/detect.md) haute performance. Ces modèles répondent à des besoins computationnels et des exigences de précision variables, ce qui les rend polyvalents pour une large gamme d'applications.

| Type de modèle | Modèles pré-entraînés | Tâches prises en charge                  | Inférence | Validation | Entraînement | Export |
|----------------|-----------------------|------------------------------------------|-----------|------------|--------------|--------|
| YOLOv6-N       | `yolov6-n.pt`         | [Détection d'objets](../tasks/detect.md) | ✅         | ✅          | ✅            | ✅      |
| YOLOv6-S       | `yolov6-s.pt`         | [Détection d'objets](../tasks/detect.md) | ✅         | ✅          | ✅            | ✅      |
| YOLOv6-M       | `yolov6-m.pt`         | [Détection d'objets](../tasks/detect.md) | ✅         | ✅          | ✅            | ✅      |
| YOLOv6-L       | `yolov6-l.pt`         | [Détection d'objets](../tasks/detect.md) | ✅         | ✅          | ✅            | ✅      |
| YOLOv6-L6      | `yolov6-l6.pt`        | [Détection d'objets](../tasks/detect.md) | ✅         | ✅          | ✅            | ✅      |

Ce tableau fournit un aperçu détaillé des variantes du modèle YOLOv6, mettant en évidence leurs capacités dans les tâches de détection d'objets et leur compatibilité avec différents modes opérationnels tels que l'[Inférence](../modes/predict.md), la [Validation](../modes/val.md), l'[Entraînement](../modes/train.md) et l'[Export](../modes/export.md). Cette prise en charge complète permet aux utilisateurs de tirer pleinement parti des capacités des modèles YOLOv6 dans un large éventail de scénarios de détection d'objets.

## Citations et remerciements

Nous tenons à remercier les auteurs pour leur contribution importante dans le domaine de la détection d'objets en temps réel :

!!! Quote ""

    === "BibTeX"

        ```bibtex
        @misc{li2023yolov6,
              title={YOLOv6 v3.0: A Full-Scale Reloading},
              author={Chuyi Li and Lulu Li and Yifei Geng and Hongliang Jiang and Meng Cheng and Bo Zhang and Zaidan Ke and Xiaoming Xu and Xiangxiang Chu},
              year={2023},
              eprint={2301.05586},
              archivePrefix={arXiv},
              primaryClass={cs.CV}
        }
        ```

    Le document original de YOLOv6 peut être consulté sur [arXiv](https://arxiv.org/abs/2301.05586). Les auteurs ont rendu leur travail accessible au public, et le code source peut être consulté sur [GitHub](https://github.com/meituan/YOLOv6). Nous apprécions leurs efforts pour faire avancer le domaine et rendre leur travail accessible à la communauté plus large.
