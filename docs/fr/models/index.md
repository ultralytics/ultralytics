---
comments: true
description: Explorez la diversité des modèles YOLO, SAM, MobileSAM, FastSAM, YOLO-NAS et RT-DETR pris en charge par Ultralytics. Commencez avec des exemples d'utilisation pour CLI et Python.
keywords: Ultralytics, documentation, YOLO, SAM, MobileSAM, FastSAM, YOLO-NAS, RT-DETR, modèles, architectures, Python, CLI
---

# Modèles pris en charge par Ultralytics

Bienvenue dans la documentation des modèles d'Ultralytics ! Nous proposons une prise en charge d'une large gamme de modèles, chacun adapté à des tâches spécifiques comme [la détection d'objets](../tasks/detect.md), [la segmentation d'instances](../tasks/segment.md), [la classification d'images](../tasks/classify.md), [l'estimation de posture](../tasks/pose.md) et [le suivi multi-objets](../modes/track.md). Si vous souhaitez contribuer avec votre architecture de modèle à Ultralytics, consultez notre [Guide de Contribution](../../help/contributing.md).

!!! Note "Note"

    🚧 Notre documentation multilingue est actuellement en construction et nous travaillons activement à l'améliorer. Merci de votre patience ! 🙏

## Modèles en vedette

Voici quelques-uns des modèles clés pris en charge :

1. **[YOLOv3](../../models/yolov3.md)** : La troisième itération de la famille de modèles YOLO, originellement par Joseph Redmon, reconnue pour ses capacités de détection d'objets en temps réel efficaces.
2. **[YOLOv4](../../models/yolov4.md)** : Une mise à jour de YOLOv3 native de darknet, publiée par Alexey Bochkovskiy en 2020.
3. **[YOLOv5](../../models/yolov5.md)** : Une version améliorée de l'architecture YOLO par Ultralytics, offrant de meilleurs compromis de performance et de vitesse par rapport aux versions précédentes.
4. **[YOLOv6](../../models/yolov6.md)** : Publié par [Meituan](https://about.meituan.com/) en 2022, et utilisé dans de nombreux robots de livraison autonomes de l'entreprise.
5. **[YOLOv7](../../models/yolov7.md)** : Modèles YOLO mis à jour et sortis en 2022 par les auteurs de YOLOv4.
6. **[YOLOv8](../../models/yolov8.md)** : La dernière version de la famille YOLO, avec des capacités améliorées telles que la segmentation d’instances, l'estimation de pose/points clés, et la classification.
7. **[Segment Anything Model (SAM)](../../models/sam.md)** : Le modèle Segment Anything Model (SAM) de Meta.
8. **[Mobile Segment Anything Model (MobileSAM)](../../models/mobile-sam.md)** : MobileSAM pour les applications mobiles, par l'Université de Kyung Hee.
9. **[Fast Segment Anything Model (FastSAM)](../../models/fast-sam.md)** : FastSAM par le groupe d’Analyse Image et Vidéo, Institut d'Automatisation, Académie Chinoise des Sciences.
10. **[YOLO-NAS](../../models/yolo-nas.md)** : Modèles YOLO Neural Architecture Search (NAS).
11. **[Realtime Detection Transformers (RT-DETR)](../../models/rtdetr.md)** : Modèles de Realtime Detection Transformer (RT-DETR) de Baidu's PaddlePaddle.

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

## Pour commencer : Exemples d'utilisation

!!! Example "Exemple"

    === "Python"

        Les modèles préentrainés `*.pt` ainsi que les fichiers de configuration `*.yaml` peuvent être passés aux classes `YOLO()`, `SAM()`, `NAS()` et `RTDETR()` pour créer une instance du modèle en Python :

        ```python
        from ultralytics import YOLO

        # Charger un modèle YOLOv8n préentrainé sur COCO
        model = YOLO('yolov8n.pt')

        # Afficher les informations du modèle (optionnel)
        model.info()

        # Entraîner le modèle sur l'exemple de jeu de données COCO8 pendant 100 époques
        results = model.train(data='coco8.yaml', epochs=100, imgsz=640)

        # Exécuter l'inférence avec le modèle YOLOv8n sur l'image 'bus.jpg'
        results = model('path/to/bus.jpg')
        ```

    === "CLI"

        Des commandes CLI sont disponibles pour exécuter directement les modèles :

        ```bash
        # Charger un modèle YOLOv8n préentrainé sur COCO et l'entraîner sur l'exemple de jeu de données COCO8 pendant 100 époques
        yolo train model=yolov8n.pt data=coco8.yaml epochs=100 imgsz=640

        # Charger un modèle YOLOv8n préentrainé sur COCO et exécuter l'inférence sur l'image 'bus.jpg'
        yolo predict model=yolov8n.pt source=path/to/bus.jpg
        ```

## Contribuer de nouveaux modèles

Intéressé à contribuer votre modèle à Ultralytics ? Super ! Nous sommes toujours ouverts à l'expansion de notre portefeuille de modèles.

1. **Forker le Répertoire** : Commencez par forker le [répertoire GitHub d'Ultralytics](https://github.com/ultralytics/ultralytics).

2. **Cloner Votre Fork** : Clonez votre fork sur votre machine locale et créez une nouvelle branche pour travailler dessus.

3. **Implémenter Votre Modèle** : Ajoutez votre modèle en suivant les standards et directives de codage fournis dans notre [Guide de Contribution](../../help/contributing.md).

4. **Tester Rigoureusement** : Assurez-vous de tester votre modèle de manière rigoureuse, à la fois isolément et en tant que partie du pipeline.

5. **Créer une Pull Request** : Une fois que vous êtes satisfait de votre modèle, créez une demandе de tirage (pull request) vers le répertoire principal pour examen.

6. **Revue de Code & Fusion** : Après la revue, si votre modèle répond à nos critères, il sera fusionné dans le répertoire principal.

Pour des étapes détaillées, consultez notre [Guide de Contribution](../../help/contributing.md).
