---
comments: true
description: Guide étape par étape pour entraîner des modèles YOLOv8 avec Ultralytics YOLO incluant des exemples d'entraînement mono-GPU et multi-GPU
keywords: Ultralytics, YOLOv8, YOLO, détection d'objet, mode entraînement, jeu de données personnalisé, entraînement GPU, multi-GPU, hyperparamètres, exemples CLI, exemples Python
---

# Entraînement de modèles avec Ultralytics YOLO

<img width="1024" src="https://github.com/ultralytics/assets/raw/main/yolov8/banner-integrations.png" alt="Ultralytics YOLO écosystème et intégrations">

## Introduction

L'entraînement d'un modèle d'apprentissage profond implique de lui fournir des données et d'ajuster ses paramètres afin qu'il puisse faire des prédictions précises. Le mode Entraînement de Ultralytics YOLOv8 est conçu pour un entraînement efficace et performant de modèles de détection d'objets, en utilisant pleinement les capacités du matériel moderne. Ce guide vise à couvrir tous les détails nécessaires pour commencer à entraîner vos propres modèles en utilisant l'ensemble robuste de fonctionnalités de YOLOv8.

<p align="center">
  <br>
  <iframe width="720" height="405" src="https://www.youtube.com/embed/LNwODJXcvt4?si=7n1UvGRLSd9p5wKs"
    title="Lecteur vidéo YouTube" frameborder="0"
    allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share"
    allowfullscreen>
  </iframe>
  <br>
  <strong>Regardez :</strong> Comment entraîner un modèle YOLOv8 sur votre jeu de données personnalisé dans Google Colab.
</p>

## Pourquoi choisir Ultralytics YOLO pour l'entraînement ?

Voici quelques raisons convaincantes de choisir le mode Entraînement de YOLOv8 :

- **Efficacité :** Optimisez l'utilisation de votre matériel, que vous soyez sur une configuration mono-GPU ou que vous échelonnier sur plusieurs GPUs.
- **Polyvalence :** Entraînez sur des jeux de données personnalisés en plus de ceux déjà disponibles comme COCO, VOC et ImageNet.
- **Convivialité :** Interfaces CLI et Python simples mais puissantes pour une expérience d'entraînement directe.
- **Flexibilité des hyperparamètres :** Un large éventail d'hyperparamètres personnalisables pour peaufiner les performances du modèle.

### Principales caractéristiques du mode Entraînement

Voici quelques caractéristiques remarquables du mode Entraînement de YOLOv8 :

- **Téléchargement automatique de jeux de données :** Les jeux de données standards comme COCO, VOC et ImageNet sont téléchargés automatiquement lors de la première utilisation.
- **Support multi-GPU :** Échelonnez vos efforts de formation de manière fluide sur plusieurs GPUs pour accélérer le processus.
- **Configuration des hyperparamètres :** La possibilité de modifier les hyperparamètres via des fichiers de configuration YAML ou des arguments CLI.
- **Visualisation et suivi :** Suivi en temps réel des métriques d'entraînement et visualisation du processus d'apprentissage pour de meilleures perspectives.

!!! Tip "Astuce"

    * Les jeux de données YOLOv8 comme COCO, VOC, ImageNet et bien d'autres se téléchargent automatiquement lors de la première utilisation, par exemple `yolo train data=coco.yaml`

## Exemples d'utilisation

Entraînez YOLOv8n sur le jeu de données COCO128 pendant 100 époques avec une taille d'image de 640. Le dispositif d'entraînement peut être spécifié à l'aide de l'argument `device`. Si aucun argument n'est passé, le GPU `device=0` sera utilisé s'il est disponible, sinon `device=cpu` sera utilisé. Consultez la section Arguments ci-dessous pour obtenir une liste complète des arguments d'entraînement.

!!! Example "Exemple d'entraînement mono-GPU et CPU"

    Le dispositif est déterminé automatiquement. Si un GPU est disponible, il sera utilisé, sinon l'entraînement commencera sur CPU.

    === "Python"

        ```python
        from ultralytics import YOLO

        # Charger un modèle
        model = YOLO('yolov8n.yaml')  # construire un nouveau modèle à partir de YAML
        model = YOLO('yolov8n.pt')  # charger un modèle préentraîné (recommandé pour l'entraînement)
        model = YOLO('yolov8n.yaml').load('yolov8n.pt')  # construire à partir de YAML et transférer les poids

        # Entraîner le modèle
        results = model.train(data='coco128.yaml', epochs=100, imgsz=640)
        ```

    === "CLI"

        ```bash
        # Construire un nouveau modèle à partir de YAML et commencer l'entraînement à partir de zéro
        yolo detect train data=coco128.yaml model=yolov8n.yaml epochs=100 imgsz=640

        # Commencer l'entraînement à partir d'un modèle préentraîné *.pt
        yolo detect train data=coco128.yaml model=yolov8n.pt epochs=100 imgsz=640

        # Construire un nouveau modèle à partir de YAML, transférer les poids préentraînés et commencer l'entraînement
        yolo detect train data=coco128.yaml model=yolov8n.yaml pretrained=yolov8n.pt epochs=100 imgsz=640
        ```

### Entraînement multi-GPU

L'entraînement multi-GPU permet une utilisation plus efficace des ressources matérielles disponibles en répartissant la charge d'entraînement sur plusieurs GPUs. Cette fonctionnalité est disponible via l'API Python et l'interface de ligne de commande. Pour activer l'entraînement multi-GPU, spécifiez les ID des dispositifs GPU que vous souhaitez utiliser.

!!! Example "Exemple d'entraînement multi-GPU"

    Pour s'entraîner avec 2 GPUs, les dispositifs CUDA 0 et 1, utilisez les commandes suivantes. Développez à des GPUs supplémentaires selon le besoin.

    === "Python"

        ```python
        from ultralytics import YOLO

        # Charger un modèle
        model = YOLO('yolov8n.pt')  # charger un modèle préentraîné (recommandé pour l'entraînement)

        # Entraîner le modèle avec 2 GPUs
        results = model.train(data='coco128.yaml', epochs=100, imgsz=640, device=[0, 1])
        ```

    === "CLI"

        ```bash
        # Commencer l'entraînement à partir d'un modèle préentraîné *.pt en utilisant les GPUs 0 et 1
        yolo detect train data=coco128.yaml model=yolov8n.pt epochs=100 imgsz=640 device=0,1
        ```

### Entraînement MPS avec Apple M1 et M2

Avec le support pour les puces Apple M1 et M2 intégré dans les modèles Ultralytics YOLO, il est maintenant possible d'entraîner vos modèles sur des dispositifs utilisant le puissant framework Metal Performance Shaders (MPS). Le MPS offre un moyen performant d'exécuter des tâches de calcul et de traitement d'image sur le silicium personnalisé d'Apple.

Pour activer l'entraînement sur les puces Apple M1 et M2, vous devez spécifier 'mps' comme votre dispositif lors du lancement du processus d'entraînement. Voici un exemple de la manière dont vous pourriez le faire en Python et via la ligne de commande :

!!! Example "Exemple d'entraînement MPS"

    === "Python"

        ```python
        from ultralytics import YOLO

        # Charger un modèle
        model = YOLO('yolov8n.pt')  # charger un modèle préentraîné (recommandé pour l'entraînement)

        # Entraîner le modèle avec MPS
        results = model.train(data='coco128.yaml', epochs=100, imgsz=640, device='mps')
        ```

    === "CLI"

        ```bash
        # Commencer l'entraînement à partir d'un modèle préentraîné *.pt avec MPS
        yolo detect train data=coco128.yaml model=yolov8n.pt epochs=100 imgsz=640 device=mps
        ```

Tout en exploitant la puissance de calcul des puces M1/M2, cela permet un traitement plus efficace des tâches d'entraînement. Pour des conseils plus détaillés et des options de configuration avancée, veuillez consulter la [documentation MPS de PyTorch](https://pytorch.org/docs/stable/notes/mps.html).

## Journalisation

Lors de l'entraînement d'un modèle YOLOv8, il peut être précieux de suivre la performance du modèle au fil du temps. C'est là que la journalisation entre en jeu. YOLO d'Ultralytics prend en charge trois types de journaux - Comet, ClearML et TensorBoard.

Pour utiliser un journal, sélectionnez-le dans le menu déroulant ci-dessus et exécutez-le. Le journal choisi sera installé et initialisé.

### Comet

[Comet](https://www.comet.ml/site/) est une plateforme qui permet aux scientifiques de données et aux développeurs de suivre, comparer, expliquer et optimiser les expériences et les modèles. Elle offre des fonctionnalités telles que le suivi en temps réel des mesures, les différences de code et le suivi des hyperparamètres.

Pour utiliser Comet :

!!! Example "Exemple"

    === "Python"
        ```python
        # pip install comet_ml
        import comet_ml

        comet_ml.init()
        ```

N'oubliez pas de vous connecter à votre compte Comet sur leur site web et d'obtenir votre clé API. Vous devrez ajouter cela à vos variables d'environnement ou à votre script pour enregistrer vos expériences.

### ClearML

[ClearML](https://www.clear.ml/) est une plateforme open source qui automatise le suivi des expériences et aide à partager efficacement les ressources. Elle est conçue pour aider les équipes à gérer, exécuter et reproduire leur travail en ML plus efficacement.

Pour utiliser ClearML :

!!! Example "Exemple"

    === "Python"
        ```python
        # pip install clearml
        import clearml

        clearml.browser_login()
        ```

Après avoir exécuté ce script, vous devrez vous connecter à votre compte ClearML sur le navigateur et authentifier votre session.

### TensorBoard

[TensorBoard](https://www.tensorflow.org/tensorboard) est un ensemble d'outils de visualisation pour TensorFlow. Il vous permet de visualiser votre graphique TensorFlow, de tracer des mesures quantitatives sur l'exécution de votre graphique et de montrer des données supplémentaires comme des images qui le traversent.

Pour utiliser TensorBoard dans [Google Colab](https://colab.research.google.com/github/ultralytics/ultralytics/blob/main/examples/tutorial.ipynb) :

!!! Example "Exemple"

    === "CLI"
        ```bash
        load_ext tensorboard
        tensorboard --logdir ultralytics/runs  # remplacer par le répertoire 'runs'
        ```

Pour utiliser TensorBoard localement, exécutez la commande ci-dessous et consultez les résultats à l'adresse http://localhost:6006/.

!!! Example "Exemple"

    === "CLI"
        ```bash
        tensorboard --logdir ultralytics/runs  # remplacer par le répertoire 'runs'
        ```

Cela chargera TensorBoard et le dirigera vers le répertoire où vos journaux d'entraînement sont sauvegardés.

Après avoir configuré votre journal, vous pouvez ensuite poursuivre l'entraînement de votre modèle. Toutes les métriques d'entraînement seront automatiquement enregistrées sur votre plateforme choisie, et vous pourrez accéder à ces journaux pour surveiller les performances de votre modèle au fil du temps, comparer différents modèles et identifier les domaines d'amélioration.
