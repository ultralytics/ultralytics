---
comments: true
description: Obtenez un aperçu des modèles YOLOv3, YOLOv3-Ultralytics et YOLOv3u. Apprenez-en davantage sur leurs fonctionnalités clés, leur utilisation et les tâches prises en charge pour la détection d'objets.
keywords: YOLOv3, YOLOv3-Ultralytics, YOLOv3u, Détection d'objets, Inférence, Entraînement, Ultralytics
---

# YOLOv3, YOLOv3-Ultralytics et YOLOv3u

## Aperçu

Ce document présente un aperçu de trois modèles de détection d'objets étroitement liés, à savoir [YOLOv3](https://pjreddie.com/darknet/yolo/), [YOLOv3-Ultralytics](https://github.com/ultralytics/yolov3) et [YOLOv3u](https://github.com/ultralytics/ultralytics).

1. **YOLOv3**: Il s'agit de la troisième version de l'algorithme de détection d'objets You Only Look Once (YOLO). Initiée par Joseph Redmon, YOLOv3 a amélioré ses prédécesseurs en introduisant des fonctionnalités telles que des prédictions à plusieurs échelles et trois tailles différentes de noyaux de détection.

2. **YOLOv3-Ultralytics**: Il s'agit de l'implémentation par Ultralytics du modèle YOLOv3. Il reproduit l'architecture d'origine de YOLOv3 et offre des fonctionnalités supplémentaires, telles que la prise en charge de plusieurs modèles pré-entraînés et des options de personnalisation plus faciles.

3. **YOLOv3u**: Il s'agit d'une version mise à jour de YOLOv3-Ultralytics qui intègre la nouvelle tête de détection sans ancrage et sans objectivité utilisée dans les modèles YOLOv8. YOLOv3u conserve la même architecture de base et de cou de YOLOv3, mais avec la nouvelle tête de détection de YOLOv8.

![Ultralytics YOLOv3](https://raw.githubusercontent.com/ultralytics/assets/main/yolov3/banner-yolov3.png)

## Caractéristiques clés

- **YOLOv3**: A introduit l'utilisation de trois échelles différentes pour la détection, en tirant parti de trois tailles différentes de noyaux de détection : 13x13, 26x26 et 52x52. Cela a considérablement amélioré la précision de la détection pour les objets de différentes tailles. De plus, YOLOv3 a ajouté des fonctionnalités telles que des prédictions multi-étiquettes pour chaque boîte englobante et un meilleur réseau d'extraction de caractéristiques.

- **YOLOv3-Ultralytics**: L'implémentation d'Ultralytics de YOLOv3 offre les mêmes performances que le modèle d'origine, mais propose également un support supplémentaire pour plus de modèles pré-entraînés, des méthodes d'entraînement supplémentaires et des options de personnalisation plus faciles. Cela le rend plus polyvalent et convivial pour les applications pratiques.

- **YOLOv3u**: Ce modèle mis à jour intègre la nouvelle tête de détection sans ancrage et sans objectivité de YOLOv8. En éliminant le besoin de boîtes d'ancrage prédéfinies et de scores d'objectivité, cette conception de tête de détection peut améliorer la capacité du modèle à détecter des objets de différentes tailles et formes. Cela rend YOLOv3u plus robuste et précis pour les tâches de détection d'objets.

## Tâches et modes pris en charge

Les modèles de la série YOLOv3, notamment YOLOv3, YOLOv3-Ultralytics et YOLOv3u, sont spécialement conçus pour les tâches de détection d'objets. Ces modèles sont réputés pour leur efficacité dans divers scénarios réels, alliant précision et rapidité. Chaque variante propose des fonctionnalités et des optimisations uniques, les rendant adaptés à une gamme d'applications.

Les trois modèles prennent en charge un ensemble complet de modes, garantissant ainsi leur polyvalence à différentes étapes du déploiement et du développement du modèle. Ces modes comprennent [Inférence](../modes/predict.md), [Validation](../modes/val.md), [Entraînement](../modes/train.md) et [Export](../modes/export.md), offrant aux utilisateurs un ensemble complet d'outils pour une détection d'objets efficace.

| Type de modèle     | Tâches prises en charge                  | Inférence | Validation | Entraînement | Export |
|--------------------|------------------------------------------|-----------|------------|--------------|--------|
| YOLOv3             | [Détection d'objets](../tasks/detect.md) | ✅         | ✅          | ✅            | ✅      |
| YOLOv3-Ultralytics | [Détection d'objets](../tasks/detect.md) | ✅         | ✅          | ✅            | ✅      |
| YOLOv3u            | [Détection d'objets](../tasks/detect.md) | ✅         | ✅          | ✅            | ✅      |

Ce tableau offre un aperçu rapide des capacités de chaque variante de YOLOv3, mettant en évidence leur polyvalence et leur pertinence pour diverses tâches et modes opérationnels dans les flux de travail de détection d'objets.

## Exemples d'utilisation

Cet exemple présente des exemples simples d'entraînement et d'inférence de YOLOv3. Pour une documentation complète sur ces exemples et d'autres [modes](../modes/index.md), consultez les pages de documentation sur [Predict](../modes/predict.md), [Train](../modes/train.md), [Val](../modes/val.md) et [Export](../modes/export.md).

!!! Example "Exemple"

    === "Python"

        Les modèles pré-entraînés PyTorch `*.pt`, ainsi que les fichiers de configuration `*.yaml`, peuvent être transmis à la classe `YOLO()` pour créer une instance de modèle en Python :

        ```python
        from ultralytics import YOLO

        # Charger un modèle YOLOv3n pré-entraîné avec COCO
        model = YOLO('yolov3n.pt')

        # Afficher les informations sur le modèle (facultatif)
        model.info()

        # Entraîner le modèle sur l'ensemble de données d'exemple COCO8 pendant 100 époques
        results = model.train(data='coco8.yaml', epochs=100, imgsz=640)

        # Exécuter l'inférence avec le modèle YOLOv3n sur l'image 'bus.jpg'
        results = model('path/to/bus.jpg')
        ```

    === "CLI"

        Des commandes CLI sont disponibles pour exécuter directement les modèles :

        ```bash
        # Charger un modèle YOLOv3n pré-entraîné avec COCO et l'entraîner sur l'ensemble de données d'exemple COCO8 pendant 100 époques
        yolo train model=yolov3n.pt data=coco8.yaml epochs=100 imgsz=640

        # Charger un modèle YOLOv3n pré-entraîné avec COCO et exécuter l'inférence sur l'image 'bus.jpg'
        yolo predict model=yolov3n.pt source=path/to/bus.jpg
        ```

## Citations et remerciements

Si vous utilisez YOLOv3 dans le cadre de vos recherches, veuillez citer les articles originaux sur YOLO et le référentiel YOLOv3 d'Ultralytics :

!!! Quote ""

    === "BibTeX"

        ```bibtex
        @article{redmon2018yolov3,
          title={YOLOv3: An Incremental Improvement},
          author={Redmon, Joseph and Farhadi, Ali},
          journal={arXiv preprint arXiv:1804.02767},
          year={2018}
        }
        ```

Merci à Joseph Redmon et Ali Farhadi pour le développement du YOLOv3 original.
