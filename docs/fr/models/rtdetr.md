---
comments: true
description: Découvrez les fonctionnalités et les avantages de RT-DETR de Baidu, un détecteur d'objets en temps réel efficace et adaptable grâce aux Vision Transformers, incluant des modèles pré-entraînés.
keywords: RT-DETR, Baidu, Vision Transformers, détection d'objets, performance en temps réel, CUDA, TensorRT, sélection de requêtes informée par IoU, Ultralytics, API Python, PaddlePaddle
---

# RT-DETR de Baidu : un détecteur d'objets en temps réel basé sur les Vision Transformers

## Présentation

Le Real-Time Detection Transformer (RT-DETR), développé par Baidu, est un détecteur d'objets de pointe de bout en bout qui offre des performances en temps réel tout en maintenant une grande précision. Il exploite la puissance des Vision Transformers (ViT) pour traiter efficacement les caractéristiques multiscalaires en dissociant l'interaction intra-échelle et la fusion inter-échelles. RT-DETR est hautement adaptable, permettant un ajustement flexible de la vitesse d'inférence en utilisant différentes couches de décodeur sans nécessiter de nouvelle formation. Le modèle est performant sur des infrastructures accélérées telles que CUDA avec TensorRT, surpassant de nombreux autres détecteurs d'objets en temps réel.

![Exemple d'image du modèle](https://user-images.githubusercontent.com/26833433/238963168-90e8483f-90aa-4eb6-a5e1-0d408b23dd33.png)
**Vue d'ensemble du RT-DETR de Baidu.** Le diagramme d'architecture du modèle RT-DETR montre les trois dernières étapes du réseau {S3, S4, S5} comme entrée de l'encodeur. L'encodeur hybride efficace transforme les caractéristiques multiscalaires en une séquence de caractéristiques d'image grâce à l'interaction à l'intérieur de l'échelle (AIFI - *Adeptation of Intra-scale Feature Interaction*) et au module de fusion inter-échelles (CCFM - *Cross-scale Context-aware Feature Fusion Module*). La sélection de requêtes informée par IoU est utilisée pour sélectionner un nombre fixe de caractéristiques d'image pour servir de requêtes d'objets initiales pour le décodeur. Enfin, le décodeur avec des têtes de prédictions auxiliaires optimise de manière itérative les requêtes d'objets pour générer des boîtes et des scores de confiance ([source](https://arxiv.org/pdf/2304.08069.pdf)).

### Fonctionnalités principales

- **Encodeur hybride efficace :** RT-DETR de Baidu utilise un encodeur hybride efficace qui traite les caractéristiques multiscalaires en dissociant l'interaction intra-échelle et la fusion inter-échelles. Cette conception unique basée sur les Vision Transformers réduit les coûts de calcul et permet une détection d'objets en temps réel.
- **Sélection de requêtes informée par IoU :** RT-DETR de Baidu améliore l'initialisation des requêtes d'objets en utilisant une sélection de requêtes informée par IoU. Cela permet au modèle de se concentrer sur les objets les plus pertinents de la scène, améliorant ainsi la précision de la détection.
- **Vitesse d'inférence adaptable :** RT-DETR de Baidu prend en charge des ajustements flexibles de la vitesse d'inférence en utilisant différentes couches de décodeur sans nécessiter de nouvelle formation. Cette adaptabilité facilite l'application pratique dans différents scénarios de détection d'objets en temps réel.

## Modèles pré-entraînés

L'API Python Ultralytics fournit des modèles pré-entraînés RT-DETR de PaddlePaddle avec différentes échelles :

- RT-DETR-L : 53,0 % de précision moyenne (AP) sur COCO val2017, 114 images par seconde (FPS) sur GPU T4
- RT-DETR-X : 54,8 % de précision moyenne (AP) sur COCO val2017, 74 images par seconde (FPS) sur GPU T4

## Exemples d'utilisation

Cet exemple présente des exemples simples d'entraînement et d'inférence avec RT-DETRR. Pour une documentation complète sur ceux-ci et d'autres [modes](../modes/index.md), consultez les pages de documentation [Predict](../modes/predict.md),  [Train](../modes/train.md), [Val](../modes/val.md) et [Export](../modes/export.md).

!!! Example "Exemple"

    === "Python"

        ```python
        from ultralytics import RTDETR

        # Charger un modèle RT-DETR-l pré-entraîné sur COCO
        model = RTDETR('rtdetr-l.pt')

        # Afficher des informations sur le modèle (facultatif)
        model.info()

        # Entraîner le modèle sur l'ensemble de données d'exemple COCO8 pendant 100 époques
        results = model.train(data='coco8.yaml', epochs=100, imgsz=640)

        # Effectuer une inférence avec le modèle RT-DETR-l sur l'image 'bus.jpg'
        results = model('path/to/bus.jpg')
        ```

    === "CLI"

        ```bash
        # Charger un modèle RT-DETR-l pré-entraîné sur COCO et l'entraîner sur l'ensemble de données d'exemple COCO8 pendant 100 époques
        yolo train model=rtdetr-l.pt data=coco8.yaml epochs=100 imgsz=640

        # Charger un modèle RT-DETR-l pré-entraîné sur COCO et effectuer une inférence sur l'image 'bus.jpg'
        yolo predict model=rtdetr-l.pt source=path/to/bus.jpg
        ```

## Tâches et modes pris en charge

Ce tableau présente les types de modèles, les poids pré-entraînés spécifiques, les tâches prises en charge par chaque modèle et les différents modes ([Train](../modes/train.md), [Val](../modes/val.md), [Predict](../modes/predict.md), [Export](../modes/export.md)) pris en charge, indiqués par des emojis ✅.

| Type de modèle      | Poids pré-entraînés | Tâches prises en charge                  | Inférence | Validation | Entraînement | Export |
|---------------------|---------------------|------------------------------------------|-----------|------------|--------------|--------|
| RT-DETR Large       | `rtdetr-l.pt`       | [Détection d'objets](../tasks/detect.md) | ✅         | ✅          | ✅            | ✅      |
| RT-DETR Extra-Large | `rtdetr-x.pt`       | [Détection d'objets](../tasks/detect.md) | ✅         | ✅          | ✅            | ✅      |

## Citations et Remerciements

Si vous utilisez RT-DETR de Baidu dans votre travail de recherche ou de développement, veuillez citer l'[article original](https://arxiv.org/abs/2304.08069) :

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

Nous tenons à remercier Baidu et l'équipe [PaddlePaddle](https://github.com/PaddlePaddle/PaddleDetection) pour la création et la maintenance de cette précieuse ressource pour la communauté de la vision par ordinateur. Leur contribution au domaine avec le développement du détecteur d'objets en temps réel basé sur les Vision Transformers, RT-DETR, est grandement appréciée.

*keywords: RT-DETR, Transformer, ViT, Vision Transformers, RT-DETR de Baidu, PaddlePaddle, Modèles PaddlePaddle RT-DETR pré-entraînés, utilisation de RT-DETR de Baidu, API Python Ultralytics, détection d'objets en temps réel*
