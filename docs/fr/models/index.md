---
comments: true
description: Explorez la gamme diversifiée de modèles de la famille YOLO, SAM, MobileSAM, FastSAM, YOLO-NAS et RT-DETR pris en charge par Ultralytics. Commencez avec des exemples pour l'utilisation CLI et Python.
keywords: Ultralytics, documentation, YOLO, SAM, MobileSAM, FastSAM, YOLO-NAS, RT-DETR, modèles, architectures, Python, CLI
---

# Modèles pris en charge par Ultralytics

Bienvenue dans la documentation des modèles d'Ultralytics ! Nous offrons un soutien pour une large gamme de modèles, chacun étant adapté à des tâches spécifiques comme [la détection d'objets](../tasks/detect.md), [la segmentation d'instance](../tasks/segment.md), [la classification d'images](../tasks/classify.md), [l'estimation de pose](../tasks/pose.md), et [le suivi multi-objets](../modes/track.md). Si vous êtes intéressé à contribuer avec votre architecture de modèle à Ultralytics, consultez notre [Guide de Contribution](../../help/contributing.md).

!!! Note "Remarque"

    🚧 Notre documentation dans différentes langues est actuellement en construction, et nous travaillons dur pour l'améliorer. Merci de votre patience ! 🙏

## Modèles en vedette

Voici quelques-uns des modèles clés pris en charge :

1. **[YOLOv3](yolov3.md)** : La troisième itération de la famille de modèles YOLO, initialement par Joseph Redmon, connue pour ses capacités de détection d'objets en temps réel efficaces.
2. **[YOLOv4](yolov4.md)** : Une mise à jour native darknet de YOLOv3, publiée par Alexey Bochkovskiy en 2020.
3. **[YOLOv5](yolov5.md)** : Une version améliorée de l'architecture YOLO par Ultralytics, offrant de meilleures performances et compromis de vitesse par rapport aux versions précédentes.
4. **[YOLOv6](yolov6.md)** : Publié par [Meituan](https://about.meituan.com/) en 2022, et utilisé dans beaucoup de ses robots de livraison autonomes.
5. **[YOLOv7](yolov7.md)** : Modèles YOLO mis à jour publiés en 2022 par les auteurs de YOLOv4.
6. **[YOLOv8](yolov8.md) NOUVEAU 🚀**: La dernière version de la famille YOLO, présentant des capacités améliorées telles que la segmentation d'instance, l'estimation de pose/points clés et la classification.
7. **[Segment Anything Model (SAM)](sam.md)** : Le modèle Segment Anything Model (SAM) de Meta.
8. **[Mobile Segment Anything Model (MobileSAM)](mobile-sam.md)** : MobileSAM pour applications mobiles, développé par l'Université de Kyung Hee.
9. **[Fast Segment Anything Model (FastSAM)](fast-sam.md)** : FastSAM par le Image & Video Analysis Group, Institute of Automation, Chinese Academy of Sciences.
10. **[YOLO-NAS](yolo-nas.md)** : Modèles de Recherche d'Architecture Neuronale YOLO (NAS).
11. **[Realtime Detection Transformers (RT-DETR)](rtdetr.md)** : Modèles du Transformateur de Détection en Temps Réel (RT-DETR) de PaddlePaddle de Baidu.

<p align="center">
  <br>
  <iframe width="720" height="405" src="https://www.youtube.com/embed/MWq1UxqTClU?si=nHAW-lYDzrz68jR0"
    title="Lecteur vidéo YouTube" frameborder="0"
    allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share"
    allowfullscreen>
  </iframe>
  <br>
  <strong>Regardez :</strong> Exécutez les modèles YOLO d'Ultralytics en seulement quelques lignes de code.
</p>

## Pour Commencer : Exemples d'Utilisation

Cet exemple fournit des exemples simples d'entraînement et d'inférence YOLO. Pour une documentation complète sur ces [modes](../modes/index.md) et d'autres, consultez les pages de documentation [Prédire](../modes/predict.md), [Entraîner](../modes/train.md), [Val](../modes/val.md) et [Exporter](../modes/export.md).

Notez que l'exemple ci-dessous concerne les modèles [Detect](../tasks/detect.md) YOLOv8 pour la détection d'objets. Pour des tâches supplémentaires prises en charge, voir les documentations [Segmenter](../tasks/segment.md), [Classifier](../tasks/classify.md) et [Poser](../tasks/pose.md).

!!! Example "Exemple"

    === "Python"

        Des modèles pré-entraînés PyTorch `*.pt` ainsi que des fichiers de configuration `*.yaml` peuvent être passés aux classes `YOLO()`, `SAM()`, `NAS()` et `RTDETR()` pour créer une instance de modèle en Python :

        ```python
        from ultralytics import YOLO

        # Charger un modèle YOLOv8n pré-entraîné sur COCO
        model = YOLO('yolov8n.pt')

        # Afficher les informations du modèle (optionnel)
        model.info()

        # Entraîner le modèle sur le jeu de données exemple COCO8 pendant 100 époques
        results = model.train(data='coco8.yaml', epochs=100, imgsz=640)

        # Exécuter l'inférence avec le modèle YOLOv8n sur l'image 'bus.jpg'
        results = model('path/to/bus.jpg')
        ```

    === "CLI"

        Des commandes CLI sont disponibles pour exécuter directement les modèles :

        ```bash
        # Charger un modèle YOLOv8n pré-entraîné sur COCO et l'entraîner sur le jeu de données exemple COCO8 pendant 100 époques
        yolo train model=yolov8n.pt data=coco8.yaml epochs=100 imgsz=640

        # Charger un modèle YOLOv8n pré-entraîné sur COCO et exécuter l'inférence sur l'image 'bus.jpg'
        yolo predict model=yolov8n.pt source=path/to/bus.jpg
        ```

## Contribution de Nouveaux Modèles

Vous êtes intéressé à contribuer votre modèle à Ultralytics ? Génial ! Nous sommes toujours ouverts à l'expansion de notre portefeuille de modèles.

1. **Forkez le Référentiel** : Commencez par forker le [référentiel GitHub d'Ultralytics](https://github.com/ultralytics/ultralytics).

2. **Clonez Votre Fork** : Clonez votre fork sur votre machine locale et créez une nouvelle branche pour travailler dessus.

3. **Implémentez Votre Modèle** : Ajoutez votre modèle en suivant les normes et directives de codage fournies dans notre [Guide de Contribution](../../help/contributing.md).

4. **Testez Rigoureusement** : Assurez-vous de tester votre modèle de manière rigoureuse, à la fois isolément et comme partie du pipeline.

5. **Créez une Pull Request** : Une fois que vous êtes satisfait de votre modèle, créez une pull request au répertoire principal pour examen.

6. **Revue de Code & Fusion** : Après examen, si votre modèle répond à nos critères, il sera fusionné dans le répertoire principal.

Pour des étapes détaillées, consultez notre [Guide de Contribution](../../help/contributing.md).
