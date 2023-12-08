---
comments: true
description: Découvrez notre guide détaillé sur YOLOv4, un détecteur d'objets en temps réel de pointe. Comprenez ses points forts architecturaux, ses fonctionnalités innovantes et des exemples d'application.
keywords: ultralytics, YOLOv4, détection d'objets, réseau neuronal, détection en temps réel, détecteur d'objets, apprentissage automatique
---

# YOLOv4: Détection d'Objets Rapide et Précise

Bienvenue sur la page de documentation d'Ultralytics pour YOLOv4, un détecteur d'objets en temps réel de pointe lancé en 2020 par Alexey Bochkovskiy sur [https://github.com/AlexeyAB/darknet](https://github.com/AlexeyAB/darknet). YOLOv4 est conçu pour offrir un équilibre optimal entre vitesse et précision, en en faisant un excellent choix pour de nombreuses applications.

![Schéma d'architecture de YOLOv4](https://user-images.githubusercontent.com/26833433/246185689-530b7fe8-737b-4bb0-b5dd-de10ef5aface.png)
**Schéma d'architecture de YOLOv4**. Présentant la conception détaillée du réseau de YOLOv4, comprenant les composants backbone, neck et head, ainsi que leurs couches interconnectées pour une détection d'objets en temps réel optimale.

## Introduction

YOLOv4 signifie You Only Look Once version 4. Il s'agit d'un modèle de détection d'objets en temps réel développé pour remédier aux limitations des versions précédentes de YOLO comme [YOLOv3](yolov3.md) et d'autres modèles de détection d'objets. Contrairement à d'autres détecteurs d'objets basés sur des réseaux neuronaux convolutifs (CNN), YOLOv4 n'est pas seulement applicable aux systèmes de recommandation, mais aussi à la gestion de processus autonomes et à la réduction de l'entrée humaine. Son utilisation sur des unités de traitement graphique (GPU) conventionnelles permet une utilisation massive à un prix abordable, et il est conçu pour fonctionner en temps réel sur un GPU conventionnel tout en ne nécessitant qu'un seul de ces GPU pour l'entraînement.

## Architecture

YOLOv4 utilise plusieurs fonctionnalités innovantes qui travaillent ensemble pour optimiser ses performances. Celles-ci incluent les connexions résiduelles pondérées (WRC), les connexions partielles à travers les étapes (CSP), la normalisation mini-batch traversée (CmBN), l'entraînement auto-antagoniste (SAT), l'activation Mish, l'augmentation des données en mosaïque, la régularisation DropBlock et la perte CIoU. Ces fonctionnalités sont combinées pour obtenir des résultats de pointe.

Un détecteur d'objets typique est composé de plusieurs parties, notamment l'entrée, le backbone, le neck et le head. Le backbone de YOLOv4 est pré-entraîné sur ImageNet et est utilisé pour prédire les classes et les boîtes englobantes des objets. Le backbone peut provenir de plusieurs modèles, notamment VGG, ResNet, ResNeXt ou DenseNet. La partie "neck" du détecteur est utilisée pour collecter des cartes de caractéristiques à partir de différentes étapes et comprend généralement plusieurs chemins "bottom-up" et plusieurs chemins "top-down". La partie "head" est ce qui est utilisé pour faire les détections et classifications finales des objets.

## Ensemble de Bonus

YOLOv4 utilise également des méthodes appelées "ensemble de bonus", qui sont des techniques permettant d'améliorer la précision du modèle lors de l'entraînement sans augmenter le coût de l'inférence. L'augmentation de données est une technique commune de l'ensemble de bonus utilisée dans la détection d'objets, qui augmente la variabilité des images d'entrée pour améliorer la robustesse du modèle. Quelques exemples d'augmentation de données incluent les distorsions photométriques (ajustement de la luminosité, du contraste, de la teinte, de la saturation et du bruit d'une image) et les distorsions géométriques (ajout d'échelle aléatoire, de recadrage, de retournement et de rotation). Ces techniques aident le modèle à mieux généraliser à différents types d'images.

## Fonctionnalités et Performances

YOLOv4 est conçu pour une vitesse et une précision optimales dans la détection d'objets. L'architecture de YOLOv4 comprend CSPDarknet53 en tant que backbone, PANet en tant que neck et YOLOv3 en tant que detection head. Ce design permet à YOLOv4 de réaliser une détection d'objets à une vitesse impressionnante, ce qui le rend adapté aux applications en temps réel. YOLOv4 excelle également en précision, atteignant des résultats de pointe dans les benchmarks de détection d'objets.

## Exemples d'Utilisation

Au moment de la rédaction de ce document, Ultralytics ne prend pas en charge les modèles YOLOv4. Par conséquent, les utilisateurs intéressés par l'utilisation de YOLOv4 devront consulter directement le référentiel GitHub de YOLOv4 pour les instructions d'installation et d'utilisation.

Voici un bref aperçu des étapes typiques que vous pourriez suivre pour utiliser YOLOv4 :

1. Rendez-vous sur le référentiel GitHub de YOLOv4 : [https://github.com/AlexeyAB/darknet](https://github.com/AlexeyAB/darknet).

2. Suivez les instructions fournies dans le fichier README pour l'installation. Cela implique généralement de cloner le référentiel, d'installer les dépendances nécessaires et de configurer les variables d'environnement nécessaires.

3. Une fois l'installation terminée, vous pouvez entraîner et utiliser le modèle selon les instructions d'utilisation fournies dans le référentiel. Cela implique généralement la préparation de votre ensemble de données, la configuration des paramètres du modèle, l'entraînement du modèle, puis l'utilisation du modèle entraîné pour effectuer la détection d'objets.

Veuillez noter que les étapes spécifiques peuvent varier en fonction de votre cas d'utilisation spécifique et de l'état actuel du référentiel YOLOv4. Il est donc fortement recommandé de se référer directement aux instructions fournies dans le référentiel GitHub de YOLOv4.

Nous regrettons tout inconvénient que cela pourrait causer et nous nous efforcerons de mettre à jour ce document avec des exemples d'utilisation pour Ultralytics une fois que le support de YOLOv4 sera implémenté.

## Conclusion

YOLOv4 est un modèle de détection d'objets puissant et efficace qui concilie vitesse et précision. Son utilisation de fonctionnalités uniques et de techniques "ensemble de bonus" lors de l'entraînement lui permet de réaliser d'excellentes performances dans les tâches de détection d'objets en temps réel. YOLOv4 peut être entraîné et utilisé par n'importe qui disposant d'un GPU conventionnel, le rendant accessible et pratique pour un large éventail d'applications.

## Citations et Remerciements

Nous tenons à remercier les auteurs de YOLOv4 pour leurs contributions importantes dans le domaine de la détection d'objets en temps réel :

!!! Quote ""

    === "BibTeX"

        ```bibtex
        @misc{bochkovskiy2020yolov4,
              title={YOLOv4: Optimal Speed and Accuracy of Object Detection},
              author={Alexey Bochkovskiy and Chien-Yao Wang and Hong-Yuan Mark Liao},
              year={2020},
              eprint={2004.10934},
              archivePrefix={arXiv},
              primaryClass={cs.CV}
        }
        ```

L'article original de YOLOv4 peut être consulté sur [arXiv](https://arxiv.org/pdf/2004.10934.pdf). Les auteurs ont rendu leur travail accessible au public, et le code source peut être consulté sur [GitHub](https://github.com/AlexeyAB/darknet). Nous apprécions leurs efforts pour faire progresser le domaine et rendre leur travail accessible à la communauté élargie.
