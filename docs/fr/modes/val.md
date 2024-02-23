---
comments: true
description: Guide de validation des modèles YOLOv8. Apprenez à évaluer la performance de vos modèles YOLO en utilisant les paramètres de validation et les métriques avec des exemples en Python et en CLI.
keywords: Ultralytics, YOLO Docs, YOLOv8, validation, évaluation de modèle, hyperparamètres, précision, métriques, Python, CLI
---

# Validation des modèles avec Ultralytics YOLO

<img width="1024" src="https://github.com/ultralytics/assets/raw/main/yolov8/banner-integrations.png" alt="Écosystème Ultralytics YOLO et intégrations">

## Introduction

La validation est une étape cruciale dans le pipeline d'apprentissage automatique, vous permettant d'évaluer la qualité de vos modèles entraînés. Le mode Val dans Ultralytics YOLOv8 offre une gamme robuste d'outils et de métriques pour évaluer la performance de vos modèles de détection d'objets. Ce guide sert de ressource complète pour comprendre comment utiliser efficacement le mode Val pour assurer que vos modèles sont à la fois précis et fiables.

## Pourquoi valider avec Ultralytics YOLO ?

Voici pourquoi l'utilisation du mode Val de YOLOv8 est avantageuse :

- **Précision :** Obtenez des métriques précises telles que mAP50, mAP75 et mAP50-95 pour évaluer de manière exhaustive votre modèle.
- **Convenance :** Utilisez des fonctionnalités intégrées qui se souviennent des paramètres d'entraînement, simplifiant ainsi le processus de validation.
- **Flexibilité :** Validez votre modèle avec les mêmes jeux de données ou des jeux différents et des tailles d'image variées.
- **Réglage des hyperparamètres :** Utilisez les métriques de validation pour peaufiner votre modèle pour de meilleures performances.

### Caractéristiques clés du mode Val

Voici les fonctionnalités notables offertes par le mode Val de YOLOv8 :

- **Paramètres Automatisés :** Les modèles se souviennent de leurs configurations d'entraînement pour une validation simple.
- **Support Multi-métrique :** Évaluez votre modèle en fonction d'une gamme de métriques de précision.
- **CLI et API Python :** Choisissez entre l'interface en ligne de commande ou l'API Python en fonction de vos préférences pour la validation.
- **Compatibilité des Données :** Fonctionne de manière transparente avec les jeux de données utilisés pendant la phase d'entraînement ainsi qu'avec les jeux personnalisés.

!!! Tip "Conseil"

    * Les modèles YOLOv8 se souviennent automatiquement de leurs paramètres d'entraînement, vous pouvez donc facilement valider un modèle à la même taille d'image et sur le jeu de données original avec juste `yolo val model=yolov8n.pt` ou `model('yolov8n.pt').val()`

## Exemples d'utilisation

Validez la précision du modèle YOLOv8n entraîné sur le jeu de données COCO128. Aucun argument n'a besoin d'être passé car le `modèle` conserve ses `données` d'entraînement et arguments comme attributs du modèle. Consultez la section des arguments ci-dessous pour une liste complète des arguments d'exportation.

!!! Example "Exemple"

    === "Python"

        ```python
        from ultralytics import YOLO

        # Charger un modèle
        model = YOLO('yolov8n.pt')  # charger un modèle officiel
        model = YOLO('chemin/vers/meilleur.pt')  # charger un modèle personnalisé

        # Valider le modèle
        metrics = model.val()  # pas besoin d'arguments, jeu de données et paramètres mémorisés
        metrics.box.map    # map50-95
        metrics.box.map50  # map50
        metrics.box.map75  # map75
        metrics.box.maps   # une liste contenant map50-95 de chaque catégorie
        ```
    === "CLI"

        ```bash
        yolo detect val model=yolov8n.pt  # val modèle officiel
        yolo detect val model=chemin/vers/meilleur.pt  # val modèle personnalisé
        ```

## Arguments

Les paramètres de validation pour les modèles YOLO font référence aux divers hyperparamètres et configurations utilisés pour évaluer la performance du modèle sur un jeu de données de validation. Ces paramètres peuvent affecter la performance, la vitesse et la précision du modèle. Certains paramètres de validation YOLO courants incluent la taille du lot, la fréquence à laquelle la validation est effectuée pendant l'entraînement et les métriques utilisées pour évaluer la performance du modèle. D'autres facteurs pouvant affecter le processus de validation incluent la taille et la composition du jeu de données de validation et la tâche spécifique pour laquelle le modèle est utilisé. Il est important de régler et d'expérimenter soigneusement ces paramètres pour s'assurer que le modèle fonctionne bien sur le jeu de données de validation et pour détecter et prévenir le surajustement.

| Clé           | Valeur  | Description                                                                                    |
|---------------|---------|------------------------------------------------------------------------------------------------|
| `data`        | `None`  | chemin vers le fichier de données, par exemple coco128.yaml                                    |
| `imgsz`       | `640`   | taille des images d'entrée en tant qu'entier                                                   |
| `batch`       | `16`    | nombre d'images par lot (-1 pour AutoBatch)                                                    |
| `save_json`   | `False` | sauvegarder les résultats dans un fichier JSON                                                 |
| `save_hybrid` | `False` | sauvegarder la version hybride des étiquettes (étiquettes + prédictions supplémentaires)       |
| `conf`        | `0.001` | seuil de confiance de l'objet pour la détection                                                |
| `iou`         | `0.6`   | seuil d'intersection sur union (IoU) pour la NMS                                               |
| `max_det`     | `300`   | nombre maximum de détections par image                                                         |
| `half`        | `True`  | utiliser la précision moitié (FP16)                                                            |
| `device`      | `None`  | appareil sur lequel exécuter, par exemple cuda device=0/1/2/3 ou device=cpu                    |
| `dnn`         | `False` | utiliser OpenCV DNN pour l'inférence ONNX                                                      |
| `plots`       | `False` | afficher les graphiques lors de la formation                                                   |
| `rect`        | `False` | val rectangulaire avec chaque lot regroupé pour un minimum de rembourrage                      |
| `split`       | `val`   | fraction du jeu de données à utiliser pour la validation, par exemple 'val', 'test' ou 'train' |
|
