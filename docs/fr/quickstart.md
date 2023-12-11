---
comments: true
description: Explorez diverses méthodes pour installer Ultralytics en utilisant pip, conda, git et Docker. Apprenez comment utiliser Ultralytics avec l'interface en ligne de commande ou au sein de vos projets Python.
keywords: installation d'Ultralytics, pip install Ultralytics, Docker install Ultralytics, interface en ligne de commande Ultralytics, interface Python Ultralytics
---

## Installer Ultralytics

Ultralytics propose diverses méthodes d'installation, y compris pip, conda et Docker. Installez YOLOv8 via le package `ultralytics` avec pip pour obtenir la dernière version stable ou en clonant le [répertoire GitHub d'Ultralytics](https://github.com/ultralytics/ultralytics) pour la version la plus récente. Docker peut être utilisé pour exécuter le package dans un conteneur isolé, évitant l'installation locale.

!!! Example "Installer"

    === "Installation avec Pip (recommandé)"
        Installez le package `ultralytics` en utilisant pip, ou mettez à jour une installation existante en exécutant `pip install -U ultralytics`. Visitez l'Index des Packages Python (PyPI) pour plus de détails sur le package `ultralytics` : [https://pypi.org/project/ultralytics/](https://pypi.org/project/ultralytics/).

        [![Version PyPI](https://badge.fury.io/py/ultralytics.svg)](https://badge.fury.io/py/ultralytics) [![Téléchargements](https://static.pepy.tech/badge/ultralytics)](https://pepy.tech/project/ultralytics)

        ```bash
        # Installer le package ultralytics depuis PyPI
        pip install ultralytics
        ```

        Vous pouvez également installer le package `ultralytics` directement depuis le [répertoire GitHub](https://github.com/ultralytics/ultralytics). Cela peut être utile si vous voulez la version de développement la plus récente. Assurez-vous d'avoir l'outil en ligne de commande Git installé sur votre système. La commande `@main` installe la branche `main` et peut être modifiée pour une autre branche, p. ex. `@my-branch`, ou supprimée entièrement pour revenir par défaut à la branche `main`.

        ```bash
        # Installer le package ultralytics depuis GitHub
        pip install git+https://github.com/ultralytics/ultralytics.git@main
        ```


    === "Installation avec Conda"
        Conda est un gestionnaire de packages alternatif à pip qui peut également être utilisé pour l'installation. Visitez Anaconda pour plus de détails à [https://anaconda.org/conda-forge/ultralytics](https://anaconda.org/conda-forge/ultralytics). Le répertoire feedstock d'Ultralytics pour la mise à jour du package conda est sur [https://github.com/conda-forge/ultralytics-feedstock/](https://github.com/conda-forge/ultralytics-feedstock/).


        [![Recette Conda](https://img.shields.io/badge/recipe-ultralytics-green.svg)](https://anaconda.org/conda-forge/ultralytics) [![Téléchargements Conda](https://img.shields.io/conda/dn/conda-forge/ultralytics.svg)](https://anaconda.org/conda-forge/ultralytics) [![Version Conda](https://img.shields.io/conda/vn/conda-forge/ultralytics.svg)](https://anaconda.org/conda-forge/ultralytics) [![Plateformes Conda](https://img.shields.io/conda/pn/conda-forge/ultralytics.svg)](https://anaconda.org/conda-forge/ultralytics)

        ```bash
        # Installer le package ultralytics en utilisant conda
        conda install -c conda-forge ultralytics
        ```

        !!! Note "Note"

            Si vous installez dans un environnement CUDA, la meilleure pratique est d'installer `ultralytics`, `pytorch` et `pytorch-cuda` dans la même commande pour permettre au gestionnaire de package conda de résoudre les conflits, ou bien d'installer `pytorch-cuda` en dernier pour lui permettre de remplacer le package `pytorch` spécifique aux CPU si nécessaire.
            ```bash
            # Installer tous les packages ensemble en utilisant conda
            conda install -c pytorch -c nvidia -c conda-forge pytorch torchvision pytorch-cuda=11.8 ultralytics
            ```

        ### Image Docker Conda

        Les images Docker Conda d'Ultralytics sont également disponibles sur [DockerHub](https://hub.docker.com/r/ultralytics/ultralytics). Ces images sont basées sur [Miniconda3](https://docs.conda.io/projects/miniconda/en/latest/) et constituent un moyen simple de commencer à utiliser `ultralytics` dans un environnement Conda.

        ```bash
        # Définir le nom de l'image comme variable
        t=ultralytics/ultralytics:latest-conda

        # Télécharger la dernière image ultralytics de Docker Hub
        sudo docker pull $t

        # Exécuter l'image ultralytics dans un conteneur avec support GPU
        sudo docker run -it --ipc=host --gpus all $t  # tous les GPUs
        sudo docker run -it --ipc=host --gpus '"device=2,3"' $t  # spécifier les GPUs
        ```

    === "Clone Git"
        Clonez le répertoire `ultralytics` si vous êtes intéressé par la contribution au développement ou si vous souhaitez expérimenter avec le dernier code source. Après le clonage, naviguez dans le répertoire et installez le package en mode éditable `-e` en utilisant pip.
        ```bash
        # Cloner le répertoire ultralytics
        git clone https://github.com/ultralytics/ultralytics

        # Naviguer vers le répertoire cloné
        cd ultralytics

        # Installer le package en mode éditable pour le développement
        pip install -e .
        ```

Voir le fichier [requirements.txt](https://github.com/ultralytics/ultralytics/blob/main/requirements.txt) d'`ultralytics` pour une liste des dépendances. Notez que tous les exemples ci-dessus installent toutes les dépendances requises.

<p align="center">
  <br>
  <iframe width="720" height="405" src="https://www.youtube.com/embed/_a7cVL9hqnk"
    title="YouTube video player" frameborder="0"
    allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share"
    allowfullscreen>
  </iframe>
  <br>
  <strong>Watch:</strong> Ultralytics YOLO Quick Start Guide
</p>

!!! astuce "Conseil"

    Les prérequis de PyTorch varient selon le système d'exploitation et les exigences CUDA, donc il est recommandé d'installer PyTorch en premier en suivant les instructions sur [https://pytorch.org/get-started/locally](https://pytorch.org/get-started/locally).

    <a href="https://pytorch.org/get-started/locally/">
        <img width="800" alt="Instructions d'installation de PyTorch" src="https://user-images.githubusercontent.com/26833433/228650108-ab0ec98a-b328-4f40-a40d-95355e8a84e3.png">
    </a>

## Utiliser Ultralytics avec CLI

L'interface en ligne de commande (CLI) d'Ultralytics permet l'utilisation de commandes simples en une seule ligne sans nécessiter d'environnement Python. La CLI ne requiert pas de personnalisation ou de code Python. Vous pouvez simplement exécuter toutes les tâches depuis le terminal avec la commande `yolo`. Consultez le [Guide CLI](/../usage/cli.md) pour en savoir plus sur l'utilisation de YOLOv8 depuis la ligne de commande.

!!! Example "Exemple"

    === "Syntaxe"

        Les commandes `yolo` d'Ultralytics utilisent la syntaxe suivante :
        ```bash
        yolo TÂCHE MODE ARGS

        Où   TÂCHE (facultatif) est l'une de [detect, segment, classify]
             MODE (obligatoire) est l'un de [train, val, predict, export, track]
             ARGS (facultatif) sont n'importe quel nombre de paires personnalisées 'arg=valeur' comme 'imgsz=320' qui remplacent les valeurs par défaut.
        ```
        Voyez tous les ARGS dans le [Guide de Configuration](/../usage/cfg.md) complet ou avec `yolo cfg`

    === "Entraînement"

        Entraînez un modèle de détection pour 10 epochs avec un learning_rate initial de 0.01
        ```bash
        yolo train data=coco128.yaml model=yolov8n.pt epochs=10 lr0=0.01
        ```

    === "Prédiction"

        Prédisez une vidéo YouTube en utilisant un modèle de segmentation pré-entraîné à une taille d'image de 320 :
        ```bash
        yolo predict model=yolov8n-seg.pt source='https://youtu.be/LNwODJXcvt4' imgsz=320
        ```

    === "Validation"

        Validez un modèle de détection pré-entraîné avec un batch-size de 1 et une taille d'image de 640 :
        ```bash
        yolo val model=yolov8n.pt data=coco128.yaml batch=1 imgsz=640
        ```

    === "Exportation"

        Exportez un modèle de classification YOLOv8n au format ONNX à une taille d'image de 224 par 128 (pas de TÂCHE requise)
        ```bash
        yolo export model=yolov8n-cls.pt format=onnx imgsz=224,128
        ```

    === "Spécial"

        Exécutez des commandes spéciales pour voir la version, afficher les paramètres, effectuer des vérifications et plus encore :
        ```bash
        yolo help
        yolo checks
        yolo version
        yolo settings
        yolo copy-cfg
        yolo cfg
        ```

!!! Warning "Avertissement"

    Les arguments doivent être passés sous forme de paires `arg=val`, séparés par un signe égal `=` et délimités par des espaces ` ` entre les paires. N'utilisez pas de préfixes d'arguments `--` ou de virgules `,` entre les arguments.

    - `yolo predict model=yolov8n.pt imgsz=640 conf=0.25` &nbsp; ✅
    - `yolo predict model yolov8n.pt imgsz 640 conf 0.25` &nbsp; ❌
    - `yolo predict --model yolov8n.pt --imgsz 640 --conf 0.25` &nbsp; ❌

[Guide CLI](/../usage/cli.md){ .md-button }

## Utiliser Ultralytics avec Python

L'interface Python de YOLOv8 permet une intégration transparente dans vos projets Python, facilitant le chargement, l'exécution et le traitement de la sortie du modèle. Conçue avec simplicité et facilité d'utilisation à l'esprit, l'interface Python permet aux utilisateurs de mettre en œuvre rapidement la détection d'objets, la segmentation et la classification dans leurs projets. Cela fait de l'interface Python de YOLOv8 un outil inestimable pour quiconque cherche à intégrer ces fonctionnalités dans ses projets Python.

Par exemple, les utilisateurs peuvent charger un modèle, l'entraîner, évaluer ses performances sur un set de validation, et même l'exporter au format ONNX avec seulement quelques lignes de code. Consultez le [Guide Python](/../usage/python.md) pour en savoir plus sur l'utilisation de YOLOv8 au sein de vos projets Python.

!!! Example "Exemple"

    ```python
    from ultralytics import YOLO

    # Créer un nouveau modèle YOLO à partir de zéro
    model = YOLO('yolov8n.yaml')

    # Charger un modèle YOLO pré-entraîné (recommandé pour l'entraînement)
    model = YOLO('yolov8n.pt')

    # Entraîner le modèle en utilisant le jeu de données 'coco128.yaml' pour 3 epochs
    résultats = model.train(data='coco128.yaml', epochs=3)

    # Évaluer la performance du modèle sur le set de validation
    résultats = model.val()

    # Effectuer la détection d'objets sur une image en utilisant le modèle
    résultats = model('https://ultralytics.com/images/bus.jpg')

    # Exporter le modèle au format ONNX
    succès = model.export(format='onnx')
    ```

[Guide Python](/../usage/python.md){.md-button .md-button--primary}
