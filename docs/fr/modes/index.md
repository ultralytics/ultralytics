---
comments: true
description: De l'entraînement au suivi, exploitez au mieux YOLOv8 d'Ultralytics. Obtenez des aperçus et des exemples pour chaque mode pris en charge, y compris la validation, l'exportation et le benchmarking.
keywords: Ultralytics, YOLOv8, Machine Learning, Détection d'objets, Entraînement, Validation, Prédiction, Exportation, Suivi, Benchmarking
---

# Modes Ultralytics YOLOv8

<img width="1024" src="https://github.com/ultralytics/assets/raw/main/yolov8/banner-integrations.png" alt="Écosystème Ultralytics YOLO et intégrations">

## Introduction

Ultralytics YOLOv8 n'est pas simplement un autre modèle de détection d'objets ; c'est un cadre polyvalent conçu pour couvrir l'intégralité du cycle de vie des modèles d'apprentissage automatique — de l'ingestion de données et l'entraînement des modèles à la validation, le déploiement et le suivi en conditions réelles. Chaque mode remplit un objectif spécifique et est conçu pour vous offrir la flexibilité et l'efficacité nécessaires pour différentes tâches et cas d'utilisation.

<p align="center">
  <br>
  <iframe width="720" height="405" src="https://www.youtube.com/embed/j8uQc0qB91s?si=dhnGKgqvs7nPgeaM"
    title="Lecteur vidéo YouTube" frameborder="0"
    allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share"
    allowfullscreen>
  </iframe>
  <br>
  <strong>Regardez :</strong> Tutoriel sur les modes Ultralytics : Entraînement, Validation, Prédiction, Exportation & Benchmark.
</p>

### Aperçu des Modes

Comprendre les différents **modes** pris en charge par Ultralytics YOLOv8 est crucial pour tirer le maximum de vos modèles :

- **Mode d'entraînement (Train)** : Affinez votre modèle sur des jeux de données personnalisés ou préchargés.
- **Mode de validation (Val)** : Un contrôle post-entraînement pour évaluer la performance du modèle.
- **Mode de prédiction (Predict)** : Déployez la puissance prédictive de votre modèle sur des données du monde réel.
- **Mode d'exportation (Export)** : Préparez votre modèle au déploiement dans différents formats.
- **Mode de suivi (Track)** : Étendez votre modèle de détection d'objets à des applications de suivi en temps réel.
- **Mode benchmark (Benchmark)** : Analysez la vitesse et la précision de votre modèle dans divers environnements de déploiement.

Ce guide complet vise à vous donner un aperçu et des informations pratiques sur chaque mode, en vous aidant à exploiter tout le potentiel de YOLOv8.

## [Entraînement (Train)](train.md)

Le mode d'entraînement est utilisé pour entraîner un modèle YOLOv8 sur un jeu de données personnalisé. Dans ce mode, le modèle est entraîné en utilisant le jeu de données et les hyperparamètres spécifiés. Le processus d'entraînement implique l'optimisation des paramètres du modèle afin qu'il puisse prédire avec précision les classes et les emplacements des objets dans une image.

[Exemples d'entraînement](train.md){ .md-button }

## [Validation (Val)](val.md)

Le mode de validation est utilisé pour valider un modèle YOLOv8 après qu'il ait été entraîné. Dans ce mode, le modèle est évalué sur un ensemble de validation pour mesurer sa précision et sa capacité de généralisation. Ce mode peut être utilisé pour ajuster les hyperparamètres du modèle afin d'améliorer ses performances.

[Exemples de validation](val.md){ .md-button }

## [Prédiction (Predict)](predict.md)

Le mode de prédiction est utilisé pour faire des prédictions à l'aide d'un modèle YOLOv8 entraîné sur de nouvelles images ou vidéos. Dans ce mode, le modèle est chargé à partir d'un fichier de checkpoint, et l'utilisateur peut fournir des images ou vidéos pour effectuer l'inférence. Le modèle prédit les classes et les emplacements des objets dans les images ou vidéos fournies.

[Exemples de prédiction](predict.md){ .md-button }

## [Exportation (Export)](export.md)

Le mode d'exportation est utilisé pour exporter un modèle YOLOv8 dans un format pouvant être utilisé pour le déploiement. Dans ce mode, le modèle est converti dans un format pouvant être utilisé par d'autres applications logicielles ou dispositifs matériels. Ce mode est pratique pour déployer le modèle dans des environnements de production.

[Exemples d'exportation](export.md){ .md-button }

## [Suivi (Track)](track.md)

Le mode de suivi est utilisé pour suivre des objets en temps réel à l'aide d'un modèle YOLOv8. Dans ce mode, le modèle est chargé à partir d'un fichier de checkpoint, et l'utilisateur peut fournir un flux vidéo en direct pour effectuer le suivi d'objets en temps réel. Ce mode est utile pour des applications telles que les systèmes de surveillance ou les voitures autonomes.

[Exemples de suivi](track.md){ .md-button }

## [Benchmark (Benchmark)](benchmark.md)

Le mode benchmark est utilisé pour profiler la vitesse et la précision de divers formats d'exportation pour YOLOv8. Les benchmarks fournissent des informations sur la taille du format exporté, ses métriques `mAP50-95` (pour la détection d'objets, la segmentation et la pose) ou `accuracy_top5` (pour la classification), et le temps d'inférence en millisecondes par image pour différents formats d'exportation comme ONNX, OpenVINO, TensorRT et autres. Ces informations peuvent aider les utilisateurs à choisir le format d'export optimal pour leur cas d'utilisation spécifique en fonction de leurs exigences de vitesse et de précision.

[Exemples de benchmark](benchmark.md){ .md-button }
