---
comments: true
description: Découvrez le modèle Segment Anything (SAM) de pointe d'Ultralytics permettant la segmentation d'images en temps réel. Apprenez-en davantage sur sa segmentation promptable, ses performances hors échantillon et comment l'utiliser.
keywords: Ultralytics, segmentation d'image, Segment Anything Model, SAM, SA-1B dataset, performances en temps réel, transfert hors échantillon, détection d'objets, analyse d'images, apprentissage automatique
---

# Segment Anything Model (SAM)

Bienvenue à la pointe de la segmentation d'image avec le modèle Segment Anything, ou SAM. Ce modèle révolutionnaire a changé la donne en introduisant la segmentation d'image promptable avec des performances en temps réel, établissant de nouvelles normes dans le domaine.

## Introduction à SAM : Le modèle Segment Anything

Le modèle Segment Anything, ou SAM, est un modèle de segmentation d'image de pointe qui permet une segmentation promptable, offrant une polyvalence inégalée dans les tâches d'analyse d'image. SAM forme le cœur de l'initiative Segment Anything, un projet innovant qui introduit un modèle, une tâche et un jeu de données novateurs pour la segmentation d'images.

La conception avancée de SAM lui permet de s'adapter à de nouvelles distributions et tâches d'images sans connaissance préalable, une fonctionnalité connue sous le nom de transfert hors échantillon. Entraîné sur le vaste ensemble de données [SA-1B](https://ai.facebook.com/datasets/segment-anything/), qui contient plus d'un milliard de masques répartis sur 11 millions d'images soigneusement sélectionnées, SAM a affiché des performances hors échantillon impressionnantes, dépassant les résultats entièrement supervisés précédents dans de nombreux cas.

![Image d'échantillon de jeu de données](https://user-images.githubusercontent.com/26833433/238056229-0e8ffbeb-f81a-477e-a490-aff3d82fd8ce.jpg)
Exemple d'images avec des masques superposés provenant de notre nouveau jeu de données, SA-1B. SA-1B contient 11 millions d'images diverses, haute résolution, autorisées et protégeant la vie privée, ainsi que 1,1 milliard de masques de segmentation de haute qualité. Ces masques ont été annotés entièrement automatiquement par SAM, et comme le confirment des évaluations humaines et de nombreux tests, leur qualité et leur diversité sont élevées. Les images sont regroupées par nombre de masques par image pour la visualisation (il y a environ 100 masques par image en moyenne).

## Caractéristiques clés du modèle Segment Anything (SAM)

- **Tâche de segmentation promptable :** SAM a été conçu en gardant à l'esprit une tâche de segmentation promptable, ce qui lui permet de générer des masques de segmentation valides à partir de n'importe quelle indication donnée, telle que des indices spatiaux ou des indices textuels identifiant un objet.
- **Architecture avancée :** Le modèle Segment Anything utilise un puissant encodeur d'images, un encodeur de prompt et un décodeur de masques léger. Cette architecture unique permet une invitation flexible, un calcul de masques en temps réel et une prise en compte de l'ambiguïté dans les tâches de segmentation.
- **Le jeu de données SA-1B :** Introduit par le projet Segment Anything, le jeu de données SA-1B comprend plus d'un milliard de masques sur 11 millions d'images. En tant que plus grand jeu de données de segmentation à ce jour, il offre à SAM une source de données d'entraînement diversifiée et à grande échelle.
- **Performances hors échantillon :** SAM affiche des performances hors échantillon exceptionnelles dans diverses tâches de segmentation, ce qui en fait un outil prêt à l'emploi pour des applications diverses nécessitant un minimum d'ingénierie de prompt.

Pour une analyse approfondie du modèle Segment Anything et du jeu de données SA-1B, veuillez visiter le [site web Segment Anything](https://segment-anything.com) et consulter l'article de recherche [Segment Anything](https://arxiv.org/abs/2304.02643).

## Modèles disponibles, tâches prises en charge et modes d'exploitation

Ce tableau présente les modèles disponibles avec leurs poids pré-entraînés spécifiques, les tâches qu'ils prennent en charge et leur compatibilité avec différents modes d'exploitation tels que [Inférence](../modes/predict.md), [Validation](../modes/val.md), [Entraînement](../modes/train.md) et [Exportation](../modes/export.md), indiqués par des emojis ✅ pour les modes pris en charge et des emojis ❌ pour les modes non pris en charge.

| Type de modèle | Poids pré-entraînés | Tâches prises en charge                        | Inférence | Validation | Entraînement | Exportation |
|----------------|---------------------|------------------------------------------------|-----------|------------|--------------|-------------|
| SAM de base    | `sam_b.pt`          | [Segmentation d'instance](../tasks/segment.md) | ✅         | ❌          | ❌            | ✅           |
| SAM large      | `sam_l.pt`          | [Segmentation d'instance](../tasks/segment.md) | ✅         | ❌          | ❌            | ✅           |

## Comment utiliser SAM : Polyvalence et puissance dans la segmentation d'images

Le modèle Segment Anything peut être utilisé pour une multitude de tâches secondaires qui vont au-delà de ses données d'entraînement. Cela comprend la détection des contours, la génération de propositions d'objets, la segmentation d'instances et la prédiction préliminaire texte-à-masque. Grâce à l'ingénierie de prompts, SAM peut s'adapter rapidement à de nouvelles tâches et distributions de données de manière sans apprentissage, ce qui en fait un outil polyvalent et puissant pour tous vos besoins en matière de segmentation d'images.

### Exemple de prédiction SAM

!!! Example "Segmentation avec des prompts"

    Segmenter l'image avec des prompts donnés.

    === "Python"

        ```python
        from ultralytics import SAM

        # Charger un modèle
        model = SAM('sam_b.pt')

        # Afficher les informations sur le modèle (facultatif)
        model.info()

        # Exécuter l'inférence avec un prompt de zones de délimitation
        model('ultralytics/assets/zidane.jpg', bboxes=[439, 437, 524, 709])

        # Exécuter l'inférence avec un prompt de points
        model('ultralytics/assets/zidane.jpg', points=[900, 370], labels=[1])
        ```

!!! Example "Segmenter tout"

    Segmenter toute l'image.

    === "Python"

        ```python
        from ultralytics import SAM

        # Charger un modèle
        model = SAM('sam_b.pt')

        # Afficher les informations sur le modèle (facultatif)
        model.info()

        # Exécuter l'inférence
        model('path/to/image.jpg')
        ```

    === "CLI"

        ```bash
        # Exécuter l'inférence avec un modèle SAM
        yolo predict model=sam_b.pt source=path/to/image.jpg
        ```

- La logique ici est de segmenter toute l'image si vous ne passez aucun prompt (bboxes/points/masks).

!!! Example "Exemple SAMPredictor"

    De cette manière, vous pouvez définir l'image une fois et exécuter l'inférence des prompts plusieurs fois sans exécuter l'encodeur d'image plusieurs fois.

    === "Inférence avec des prompts"

        ```python
        from ultralytics.models.sam import Predictor as SAMPredictor

        # Créer un SAMPredictor
        overrides = dict(conf=0.25, task='segment', mode='predict', imgsz=1024, model="mobile_sam.pt")
        predictor = SAMPredictor(overrides=overrides)

        # Définir l'image
        predictor.set_image("ultralytics/assets/zidane.jpg")  # définir avec un fichier image
        predictor.set_image(cv2.imread("ultralytics/assets/zidane.jpg"))  # définir avec np.ndarray
        results = predictor(bboxes=[439, 437, 524, 709])
        results = predictor(points=[900, 370], labels=[1])

        # Réinitialiser l'image
        predictor.reset_image()
        ```

    Segmenter toute l'image avec des arguments supplémentaires.

    === "Segmenter tout"

        ```python
        from ultralytics.models.sam import Predictor as SAMPredictor

        # Créer un SAMPredictor
        overrides = dict(conf=0.25, task='segment', mode='predict', imgsz=1024, model="mobile_sam.pt")
        predictor = SAMPredictor(overrides=overrides)

        # Segmenter avec des arguments supplémentaires
        results = predictor(source="ultralytics/assets/zidane.jpg", crop_n_layers=1, points_stride=64)
        ```

- Plus d'arguments supplémentaires pour `Segmenter tout` voir la référence [`Predictor/generate`](../../../reference/models/sam/predict.md).

## Comparaison de SAM avec YOLOv8

Nous comparons ici le plus petit modèle SAM de Meta, SAM-b, avec le plus petit modèle de segmentation d'Ultralytics, [YOLOv8n-seg](../tasks/segment.md) :

| Modèle                                                       | Taille                            | Paramètres                  | Vitesse (CPU)                       |
|--------------------------------------------------------------|-----------------------------------|-----------------------------|-------------------------------------|
| SAM-b - Meta's SAM-b                                         | 358 Mo                            | 94,7 M                      | 51096 ms/im                         |
| [MobileSAM](mobile-sam.md)                                   | 40,7 Mo                           | 10,1 M                      | 46122 ms/im                         |
| [FastSAM-s](fast-sam.md) with YOLOv8 backbone                | 23,7 Mo                           | 11,8 M                      | 115 ms/im                           |
| YOLOv8n-seg - Ultralytics [YOLOv8n-seg](../tasks/segment.md) | **6,7 Mo** (53,4 fois plus petit) | **3,4 M** (27,9 fois moins) | **59 ms/im** (866 fois plus rapide) |

Cette comparaison montre les différences d'ordre de grandeur dans les tailles et les vitesses des modèles. Alors que SAM présente des fonctionnalités uniques pour la segmentation automatique, il ne rivalise pas directement avec les modèles de segmentation YOLOv8, qui sont plus petits, plus rapides et plus efficaces.

Tests effectués sur un MacBook Apple M2 de 2023 avec 16 Go de RAM. Pour reproduire ce test :

!!! Example "Exemple"

    === "Python"
        ```python
        from ultralytics import FastSAM, SAM, YOLO

        # Profiler SAM-b
        modèle = SAM('sam_b.pt')
        modèle.info()
        modèle('ultralytics/assets')

        # Profiler MobileSAM
        modèle = SAM('mobile_sam.pt')
        modèle.info()
        modèle('ultralytics/assets')

        # Profiler FastSAM-s
        modèle = FastSAM('FastSAM-s.pt')
        modèle.info()
        modèle('ultralytics/assets')

        # Profiler YOLOv8n-seg
        modèle = YOLO('yolov8n-seg.pt')
        modèle.info()
        modèle('ultralytics/assets')
        ```

## Annotation automatique : Un moyen rapide d'obtenir des jeux de données de segmentation

L'annotation automatique est une fonctionnalité clé de SAM, permettant aux utilisateurs de générer un [jeu de données de segmentation](https://docs.ultralytics.com/datasets/segment) à l'aide d'un modèle de détection pré-entraîné. Cette fonctionnalité permet une annotation rapide et précise d'un grand nombre d'images, en contournant la nécessité d'une annotation manuelle chronophage.

### Générez votre jeu de données de segmentation à l'aide d'un modèle de détection

Pour annoter automatiquement votre jeu de données avec le framework Ultralytics, utilisez la fonction `auto_annotate` comme indiqué ci-dessous :

!!! Example "Exemple"

    === "Python"
        ```python
        from ultralytics.data.annotator import auto_annotate

        auto_annotate(data="path/to/images", det_model="yolov8x.pt", sam_model='sam_b.pt')
        ```

| Argument   | Type                 | Description                                                                                                            | Default      |
|------------|----------------------|------------------------------------------------------------------------------------------------------------------------|--------------|
| data       | str                  | Chemin d'accès à un dossier contenant les images à annoter.                                                            |              |
| det_model  | str, optionnel       | Modèle de détection pré-entraîné YOLO. Par défaut, 'yolov8x.pt'.                                                       | 'yolov8x.pt' |
| sam_model  | str, optionnel       | Modèle de segmentation pré-entraîné SAM. Par défaut, 'sam_b.pt'.                                                       | 'sam_b.pt'   |
| device     | str, optionnel       | Appareil sur lequel exécuter les modèles. Par défaut, une chaîne vide (CPU ou GPU, si disponible).                     |              |
| output_dir | str, None, optionnel | Répertoire pour enregistrer les résultats annotés. Par défaut, un dossier 'labels' dans le même répertoire que 'data'. | None         |

La fonction `auto_annotate` prend en compte le chemin de vos images, avec des arguments optionnels pour spécifier les modèles de détection et de segmentation SAM pré-entraînés, l'appareil sur lequel exécuter les modèles et le répertoire de sortie pour enregistrer les résultats annotés.

L'annotation automatique avec des modèles pré-entraînés peut réduire considérablement le temps et les efforts nécessaires pour créer des jeux de données de segmentation de haute qualité. Cette fonctionnalité est particulièrement bénéfique pour les chercheurs et les développeurs travaillant avec de grandes collections d'images, car elle leur permet de se concentrer sur le développement et l'évaluation des modèles plutôt que sur l'annotation manuelle.

## Citations et remerciements

Si vous trouvez SAM utile dans vos travaux de recherche ou de développement, veuillez envisager de citer notre article :

!!! Quote ""

    === "BibTeX"

        ```bibtex
        @misc{kirillov2023segment,
              title={Segment Anything},
              author={Alexander Kirillov and Eric Mintun and Nikhila Ravi and Hanzi Mao and Chloe Rolland and Laura Gustafson and Tete Xiao and Spencer Whitehead and Alexander C. Berg and Wan-Yen Lo and Piotr Dollár and Ross Girshick},
              year={2023},
              eprint={2304.02643},
              archivePrefix={arXiv},
              primaryClass={cs.CV}
        }
        ```

Nous tenons à exprimer notre gratitude à Meta AI pour la création et la maintenance de cette ressource précieuse pour la communauté de la vision par ordinateur.

*keywords: Segment Anything, Segment Anything Model, SAM, Meta SAM, segmentation d'image, segmentation promptable, performances hors échantillon, jeu de données SA-1B, architecture avancée, annotation automatique, Ultralytics, modèles pré-entraînés, SAM de base, SAM large, segmentation d'instance, vision par ordinateur, IA, intelligence artificielle, apprentissage automatique, annotation de données, masques de segmentation, modèle de détection, modèle de détection YOLO, bibtex, Meta AI.*
