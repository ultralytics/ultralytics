---
comments: true
description: Découvrez YOLOv5u, une version améliorée du modèle YOLOv5 offrant un meilleur compromis entre précision et vitesse, ainsi que de nombreux modèles pré-entraînés pour diverses tâches de détection d'objets.
keywords: YOLOv5u, détection d'objets, modèles pré-entraînés, Ultralytics, inférence, validation, YOLOv5, YOLOv8, sans ancre, sans objectivité, applications temps réel, apprentissage automatique
---

# YOLOv5

## Présentation

YOLOv5u représente une avancée dans les méthodologies de détection d'objets. Originaire de l'architecture fondamentale du modèle [YOLOv5](https://github.com/ultralytics/yolov5) développé par Ultralytics, YOLOv5u intègre la division sans ancre et sans objectivité, une fonctionnalité précédemment introduite dans les modèles [YOLOv8](yolov8.md). Cette adaptation affine l'architecture du modèle, ce qui conduit à un meilleur compromis entre précision et vitesse dans les tâches de détection d'objets. Compte tenu des résultats empiriques et des fonctionnalités dérivées, YOLOv5u offre une alternative efficace pour ceux qui recherchent des solutions robustes à la fois pour la recherche et les applications pratiques.

![YOLOv5 Ultralytics](https://raw.githubusercontent.com/ultralytics/assets/main/yolov5/v70/splash.png)

## Principales fonctionnalités

- **Division sans ancre Ultralytics :** Les modèles de détection d'objets traditionnels reposent sur des boîtes d'ancrage prédéfinies pour prédire les emplacements des objets. Cependant, YOLOv5u modernise cette approche. En adoptant une division sans ancre Ultralytics, il garantit un mécanisme de détection plus flexible et adaptatif, ce qui améliore les performances dans divers scénarios.

- **Bon compromis entre précision et vitesse optimisée :** La vitesse et la précision sont souvent opposées. Mais YOLOv5u remet en question ce compromis. Il offre un équilibre calibré, garantissant des détections en temps réel sans compromettre la précision. Cette fonctionnalité est particulièrement précieuse pour les applications qui demandent des réponses rapides, comme les véhicules autonomes, la robotique et l'analyse vidéo en temps réel.

- **Variété de modèles pré-entraînés :** Comprendre que différentes tâches nécessitent différents ensembles d'outils, YOLOv5u propose une pléthore de modèles pré-entraînés. Que vous vous concentriez sur l'inférence, la validation ou l'entraînement, un modèle sur mesure vous attend. Cette variété garantit que vous n'utilisez pas une solution universelle, mais un modèle spécifiquement ajusté à votre défi unique.

## Tâches et modes pris en charge

Les modèles YOLOv5u, avec divers poids pré-entraînés, excellent dans les tâches de [détection d'objets](../tasks/detect.md). Ils prennent en charge une gamme complète de modes, ce qui les rend adaptés à diverses applications, du développement au déploiement.

| Type de modèle | Poids pré-entraînés                                                                                                         | Tâche                                    | Inférence | Validation | Entraînement | Export |
|----------------|-----------------------------------------------------------------------------------------------------------------------------|------------------------------------------|-----------|------------|--------------|--------|
| YOLOv5u        | `yolov5nu`, `yolov5su`, `yolov5mu`, `yolov5lu`, `yolov5xu`, `yolov5n6u`, `yolov5s6u`, `yolov5m6u`, `yolov5l6u`, `yolov5x6u` | [Détection d'objets](../tasks/detect.md) | ✅         | ✅          | ✅            | ✅      |

Ce tableau fournit un aperçu détaillé des variantes de modèles YOLOv5u, mettant en évidence leur applicabilité dans les tâches de détection d'objets et leur prise en charge de divers modes opérationnels tels que [Inférence](../modes/predict.md), [Validation](../modes/val.md), [Entraînement](../modes/train.md) et [Exportation](../modes/export.md). Cette prise en charge complète garantit que les utilisateurs peuvent exploiter pleinement les capacités des modèles YOLOv5u dans un large éventail de scénarios de détection d'objets.

## Métriques de performance

!!! Performance

    === "Détection"

    Consultez la [documentation sur la détection](https://docs.ultralytics.com/tasks/detect/) pour des exemples d'utilisation avec ces modèles formés sur [COCO](https://docs.ultralytics.com/datasets/detect/coco/), qui comprennent 80 classes pré-entraînées.

    | Modèle                                                                                     | YAML                                                                                                           | taille<br><sup>(pixels) | mAP<sup>val<br>50-95 | Vitesse<br><sup>CPU ONNX<br>(ms) | Vitesse<br><sup>A100 TensorRT<br>(ms) | params<br><sup>(M) | FLOPs<br><sup>(B) |
    |-------------------------------------------------------------------------------------------|----------------------------------------------------------------------------------------------------------------|-----------------------|----------------------|--------------------------------|-------------------------------------|--------------------|-------------------|
    | [yolov5nu.pt](https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov5nu.pt) | [yolov5n.yaml](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/models/v5/yolov5.yaml)     | 640                   | 34,3                 | 73,6                           | 1,06                                | 2,6                | 7,7               |
    | [yolov5su.pt](https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov5su.pt) | [yolov5s.yaml](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/models/v5/yolov5.yaml)     | 640                   | 43,0                 | 120,7                          | 1,27                                | 9,1                | 24,0              |
    | [yolov5mu.pt](https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov5mu.pt) | [yolov5m.yaml](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/models/v5/yolov5.yaml)     | 640                   | 49,0                 | 233,9                          | 1,86                                | 25,1               | 64,2              |
    | [yolov5lu.pt](https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov5lu.pt) | [yolov5l.yaml](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/models/v5/yolov5.yaml)     | 640                   | 52,2                 | 408,4                          | 2,50                                | 53,2               | 135,0             |
    | [yolov5xu.pt](https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov5xu.pt) | [yolov5x.yaml](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/models/v5/yolov5.yaml)     | 640                   | 53,2                 | 763,2                          | 3,81                                | 97,2               | 246,4             |
    |                                                                                           |                                                                                                                |                       |                      |                                |                                     |                    |                   |
    | [yolov5n6u.pt](https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov5n6u.pt) | [yolov5n6.yaml](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/models/v5/yolov5-p6.yaml) | 1280                  | 42,1                 | 211,0                          | 1,83                                | 4,3                | 7,8               |
    | [yolov5s6u.pt](https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov5s6u.pt) | [yolov5s6.yaml](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/models/v5/yolov5-p6.yaml) | 1280                  | 48,6                 | 422,6                          | 2,34                                | 15,3               | 24,6              |
    | [yolov5m6u.pt](https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov5m6u.pt) | [yolov5m6.yaml](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/models/v5/yolov5-p6.yaml) | 1280                  | 53,6                 | 810,9                          | 4,36                                | 41,2               | 65,7              |
    | [yolov5l6u.pt](https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov5l6u.pt) | [yolov5l6.yaml](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/models/v5/yolov5-p6.yaml) | 1280                  | 55,7                 | 1470,9                         | 5,47                                | 86,1               | 137,4             |
    | [yolov5x6u.pt](https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov5x6u.pt) | [yolov5x6.yaml](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/models/v5/yolov5-p6.yaml) | 1280                  | 56,8                 | 2436,5                         | 8,98                                | 155,4              | 250,7             |

## Exemples d'utilisation

Cet exemple présente des exemples simples d'entraînement et d'inférence YOLOv5. Pour une documentation complète sur ces exemples et d'autres [modes](../modes/index.md), consultez les pages de documentation [Predict](../modes/predict.md), [Train](../modes/train.md), [Val](../modes/val.md) et [Export](../modes/export.md).

!!! Example "Exemple"

    === "Python"

        Les modèles PyTorch pré-entraînés `*.pt` ainsi que les fichiers de configuration `*.yaml` peuvent être passés à la classe `YOLO()` pour créer une instance de modèle en python :

        ```python
        from ultralytics import YOLO

        # Charger un modèle YOLOv5n pré-entraîné sur COCO
        model = YOLO('yolov5n.pt')

        # Afficher les informations sur le modèle (facultatif)
        model.info()

        # Former le modèle sur l'ensemble de données d'exemple COCO8 pendant 100 époques
        results = model.train(data='coco8.yaml', epochs=100, imgsz=640)

        # Exécuter l'inférence avec le modèle YOLOv5n sur l'image 'bus.jpg'
        results = model('path/to/bus.jpg')
        ```

    === "CLI"

        Des commandes CLI sont disponibles pour exécuter directement les modèles :

        ```bash
        # Charger un modèle YOLOv5n pré-entraîné sur COCO et l'entraîner sur l'ensemble de données d'exemple COCO8 pendant 100 époques
        yolo train model=yolov5n.pt data=coco8.yaml epochs=100 imgsz=640

        # Charger un modèle YOLOv5n pré-entraîné sur COCO et exécuter l'inférence sur l'image 'bus.jpg'
        yolo predict model=yolov5n.pt source=path/to/bus.jpg
        ```

## Citations et remerciements

Si vous utilisez YOLOv5 ou YOLOv5u dans vos recherches, veuillez citer le référentiel Ultralytics YOLOv5 comme suit :

!!! Quote ""

    === "BibTeX"
        ```bibtex
        @software{yolov5,
          title = {Ultralytics YOLOv5},
          author = {Glenn Jocher},
          year = {2020},
          version = {7.0},
          license = {AGPL-3.0},
          url = {https://github.com/ultralytics/yolov5},
          doi = {10.5281/zenodo.3908559},
          orcid = {0000-0001-5950-6979}
        }
        ```

Veuillez noter que les modèles YOLOv5 sont fournis sous les licences [AGPL-3.0](https://github.com/ultralytics/ultralytics/blob/main/LICENSE) et [Enterprise](https://ultralytics.com/license).
