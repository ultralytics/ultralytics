---
comments: true
description: Découvrez une documentation détaillée sur YOLO-NAS, un modèle de détection d'objets supérieur. Apprenez-en davantage sur ses fonctionnalités, les modèles pré-entraînés, son utilisation avec l'API Python d'Ultralytics, et bien plus encore.
keywords: YOLO-NAS, Deci AI, détection d'objets, apprentissage profond, recherche architecturale neuronale, API Python d'Ultralytics, modèle YOLO, modèles pré-entraînés, quantification, optimisation, COCO, Objects365, Roboflow 100
---

# YOLO-NAS

## Aperçu

Développé par Deci AI, YOLO-NAS est un modèle de détection d'objets révolutionnaire. Il est le fruit d'une technologie avancée de recherche architecturale neuronale, minutieusement conçu pour pallier les limitations des précédents modèles YOLO. Avec des améliorations significatives en matière de prise en charge de la quantification et de compromis entre précision et latence, YOLO-NAS représente une avancée majeure en matière de détection d'objets.

![Exemple de modèle](https://learnopencv.com/wp-content/uploads/2023/05/yolo-nas_COCO_map_metrics.png)
**Aperçu de YOLO-NAS**. YOLO-NAS utilise des blocs adaptés à la quantification et une quantification sélective pour des performances optimales. Le modèle, une fois converti en version quantifiée INT8, présente une baisse de précision minimale, ce qui constitue une amélioration significative par rapport aux autres modèles. Ces avancées aboutissent à une architecture supérieure offrant des capacités de détection d'objets inégalées et des performances exceptionnelles.

### Fonctionnalités clés

- **Bloc de base compatible avec la quantification:** YOLO-NAS introduit un nouveau bloc de base adapté à la quantification, ce qui permet de pallier l'une des principales limitations des précédents modèles YOLO.
- **Entraînement sophistiqué et quantification:** YOLO-NAS utilise des schémas d'entraînement avancés et une quantification après l'entraînement pour améliorer les performances.
- **Optimisation AutoNAC et pré-entraînement:** YOLO-NAS utilise l'optimisation AutoNAC et est pré-entraîné sur des ensembles de données renommés tels que COCO, Objects365 et Roboflow 100. Ce pré-entraînement le rend extrêmement adapté aux tâches de détection d'objets ultérieures dans des environnements de production.

## Modèles pré-entraînés

Découvrez la puissance de la détection d'objets de nouvelle génération avec les modèles YOLO-NAS pré-entraînés fournis par Ultralytics. Ces modèles sont conçus pour offrir des performances exceptionnelles en termes de vitesse et de précision. Choisissez parmi une variété d'options adaptées à vos besoins spécifiques :

| Modèle           | mAP   | Latence (ms) |
|------------------|-------|--------------|
| YOLO-NAS S       | 47.5  | 3.21         |
| YOLO-NAS M       | 51.55 | 5.85         |
| YOLO-NAS L       | 52.22 | 7.87         |
| YOLO-NAS S INT-8 | 47.03 | 2.36         |
| YOLO-NAS M INT-8 | 51.0  | 3.78         |
| YOLO-NAS L INT-8 | 52.1  | 4.78         |

Chaque variante de modèle est conçue pour offrir un équilibre entre la précision moyenne (mAP) et la latence, vous permettant ainsi d'optimiser vos tâches de détection d'objets en termes de performance et de vitesse.

## Exemples d'utilisation

Ultralytics a rendu les modèles YOLO-NAS faciles à intégrer dans vos applications Python grâce à notre package Python `ultralytics`. Le package fournit une interface conviviale pour simplifier le processus.

Les exemples suivants montrent comment utiliser les modèles YOLO-NAS avec le package `ultralytics` pour l'inférence et la validation :

### Exemples d'inférence et de validation

Dans cet exemple, nous validons YOLO-NAS-s sur l'ensemble de données COCO8.

!!! Example "Exemple"

    Cet exemple fournit un code simple pour l'inférence et la validation de YOLO-NAS. Pour gérer les résultats de l'inférence, consultez le mode [Predict](../modes/predict.md). Pour utiliser YOLO-NAS avec des modes supplémentaires, consultez [Val](../modes/val.md) et [Export](../modes/export.md). L'entraînement n'est pas pris en charge pour YOLO-NAS avec le package `ultralytics`.

    === "Python"

        Il est possible de passer des modèles pré-entraînés `*.pt` de PyTorch à la classe `NAS()` pour créer une instance de modèle en Python :

        ```python
        from ultralytics import NAS

        # Charger un modèle YOLO-NAS-s pré-entraîné sur COCO
        model = NAS('yolo_nas_s.pt')

        # Afficher les informations sur le modèle (facultatif)
        model.info()

        # Valider le modèle sur l'ensemble de données COCO8
        results = model.val(data='coco8.yaml')

        # Effectuer une inférence avec le modèle YOLO-NAS-s sur l'image 'bus.jpg'
        results = model('path/to/bus.jpg')
        ```

    === "CLI"

        Des commandes CLI sont disponibles pour exécuter directement les modèles :

        ```bash
        # Charger un modèle YOLO-NAS-s pré-entraîné sur COCO et valider ses performances sur l'ensemble de données COCO8
        yolo val model=yolo_nas_s.pt data=coco8.yaml

        # Charger un modèle YOLO-NAS-s pré-entraîné sur COCO et effectuer une inférence sur l'image 'bus.jpg'
        yolo predict model=yolo_nas_s.pt source=path/to/bus.jpg
        ```

## Tâches et modes pris en charge

Nous proposons trois variantes des modèles YOLO-NAS : Small (s), Medium (m) et Large (l). Chaque variante est conçue pour répondre à des besoins computationnels et de performances différents :

- **YOLO-NAS-s** : Optimisé pour les environnements où les ressources computationnelles sont limitées mais l'efficacité est primordiale.
- **YOLO-NAS-m** : Offre une approche équilibrée, adaptée à la détection d'objets polyvalente avec une précision accrue.
- **YOLO-NAS-l** : Adapté aux scénarios nécessitant la plus haute précision, où les ressources computationnelles sont moins contraignantes.

Voici un aperçu détaillé de chaque modèle, comprenant des liens vers leurs poids pré-entraînés, les tâches qu'ils prennent en charge et leur compatibilité avec différents modes opérationnels.

| Type de modèle | Poids pré-entraînés                                                                           | Tâches prises en charge                  | Inférence | Validation | Entraînement | Export |
|----------------|-----------------------------------------------------------------------------------------------|------------------------------------------|-----------|------------|--------------|--------|
| YOLO-NAS-s     | [yolo_nas_s.pt](https://github.com/ultralytics/assets/releases/download/v0.0.0/yolo_nas_s.pt) | [Détection d'objets](../tasks/detect.md) | ✅         | ✅          | ❌            | ✅      |
| YOLO-NAS-m     | [yolo_nas_m.pt](https://github.com/ultralytics/assets/releases/download/v0.0.0/yolo_nas_m.pt) | [Détection d'objets](../tasks/detect.md) | ✅         | ✅          | ❌            | ✅      |
| YOLO-NAS-l     | [yolo_nas_l.pt](https://github.com/ultralytics/assets/releases/download/v0.0.0/yolo_nas_l.pt) | [Détection d'objets](../tasks/detect.md) | ✅         | ✅          | ❌            | ✅      |

## Citations et remerciements

Si vous utilisez YOLO-NAS dans vos travaux de recherche ou de développement, veuillez citer SuperGradients :

!!! Quote ""

    === "BibTeX"

        ```bibtex
        @misc{supergradients,
              doi = {10.5281/ZENODO.7789328},
              url = {https://zenodo.org/record/7789328},
              author = {Aharon,  Shay and {Louis-Dupont} and {Ofri Masad} and Yurkova,  Kate and {Lotem Fridman} and {Lkdci} and Khvedchenya,  Eugene and Rubin,  Ran and Bagrov,  Natan and Tymchenko,  Borys and Keren,  Tomer and Zhilko,  Alexander and {Eran-Deci}},
              title = {Super-Gradients},
              publisher = {GitHub},
              journal = {GitHub repository},
              year = {2021},
        }
        ```

Nous exprimons notre gratitude à l'équipe [Super-Gradients](https://github.com/Deci-AI/super-gradients/) de Deci AI pour ses efforts dans la création et la maintenance de cette précieuse ressource pour la communauté de la vision par ordinateur. Nous sommes convaincus que YOLO-NAS, avec son architecture innovante et ses capacités de détection d'objets supérieures, deviendra un outil essentiel pour les développeurs et les chercheurs.

*keywords: YOLO-NAS, Deci AI, détection d'objets, apprentissage profond, recherche architecturale neuronale, API Python d'Ultralytics, modèle YOLO, SuperGradients, modèles pré-entraînés, bloc de base compatible avec la quantification, schémas d'entraînement avancés, quantification après l'entraînement, optimisation AutoNAC, COCO, Objects365, Roboflow 100*
