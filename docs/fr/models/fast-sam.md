---
comments: true
description: Découvrez FastSAM, une solution basée sur les réseaux de neurones à convolution (CNN) pour la segmentation d'objets en temps réel dans les images. Interaction utilisateur améliorée, efficacité computationnelle et adaptabilité à différentes tâches de vision.
keywords: FastSAM, apprentissage automatique, solution basée sur les CNN, segmentation d'objets, solution en temps réel, Ultralytics, tâches de vision, traitement d'images, applications industrielles, interaction utilisateur
---

# Fast Segment Anything Model (FastSAM)

Le Fast Segment Anything Model (FastSAM) est une solution basée sur les réseaux de neurones à convolution (CNN) en temps réel pour la tâche Segment Anything. Cette tâche est conçue pour segmenter n'importe quel objet dans une image en fonction de différentes interactions utilisateur possibles. FastSAM réduit considérablement les demandes computationnelles tout en maintenant des performances compétitives, ce qui en fait un choix pratique pour diverses tâches de vision.

![Vue d'ensemble de l'architecture du Fast Segment Anything Model (FastSAM)](https://user-images.githubusercontent.com/26833433/248551984-d98f0f6d-7535-45d0-b380-2e1440b52ad7.jpg)

## Vue d'ensemble

FastSAM est conçu pour remédier aux limitations du [Segment Anything Model (SAM)](sam.md), un modèle Transformer lourd nécessitant des ressources computationnelles importantes. FastSAM découpe la tâche de segmentation en deux étapes séquentielles : la segmentation de toutes les instances et la sélection guidée par une invitation. La première étape utilise [YOLOv8-seg](../tasks/segment.md) pour produire les masques de segmentation de toutes les instances de l'image. Dans la deuxième étape, il génère la région d'intérêt correspondant à l'invitation.

## Fonctionnalités clés

1. **Solution en temps réel :** En exploitant l'efficacité computationnelle des CNN, FastSAM fournit une solution en temps réel pour la tâche Segment Anything, ce qui en fait une solution précieuse pour les applications industrielles nécessitant des résultats rapides.

2. **Efficacité et performances :** FastSAM offre une réduction significative des demandes computationnelles et des ressources sans compromettre la qualité des performances. Il atteint des performances comparables à SAM, mais avec une réduction drastique des ressources computationnelles, ce qui permet une application en temps réel.

3. **Segmentation guidée par une invitation :** FastSAM peut segmenter n'importe quel objet dans une image, guidé par différentes invitations d'interaction utilisateur possibles, offrant ainsi flexibilité et adaptabilité dans différents scénarios.

4. **Basé sur YOLOv8-seg :** FastSAM est basé sur [YOLOv8-seg](../tasks/segment.md), un détecteur d'objets équipé d'une branche de segmentation d'instances. Cela lui permet de produire efficacement les masques de segmentation de toutes les instances dans une image.

5. **Résultats concurrentiels sur les bancs d'essai :** Dans la tâche de proposition d'objets sur MS COCO, FastSAM obtient des scores élevés à une vitesse significativement plus rapide que [SAM](sam.md) sur une seule NVIDIA RTX 3090, démontrant ainsi son efficacité et sa capacité.

6. **Applications pratiques :** Cette approche propose une nouvelle solution pratique pour un grand nombre de tâches de vision à une vitesse très élevée, des dizaines ou des centaines de fois plus rapide que les méthodes actuelles.

7. **Faisabilité de la compression du modèle :** FastSAM démontre la faisabilité d'une voie qui peut réduire considérablement l'effort computationnel en introduisant une contrainte artificielle dans la structure, ouvrant ainsi de nouvelles possibilités pour l'architecture de modèles de grande taille pour les tâches de vision générales.

## Modèles disponibles, tâches prises en charge et modes d'exploitation

Ce tableau présente les modèles disponibles avec leurs poids pré-entraînés spécifiques, les tâches qu'ils prennent en charge et leur compatibilité avec différents modes d'exploitation tels que [Inférence](../modes/predict.md), [Validation](../modes/val.md), [Entraînement](../modes/train.md) et [Exportation](../modes/export.md), indiqués par des emojis ✅ pour les modes pris en charge et des emojis ❌ pour les modes non pris en charge.

| Type de modèle | Poids pré-entraînés | Tâches prises en charge                         | Inférence | Validation | Entraînement | Exportation |
|----------------|---------------------|-------------------------------------------------|-----------|------------|--------------|-------------|
| FastSAM-s      | `FastSAM-s.pt`      | [Segmentation d'instances](../tasks/segment.md) | ✅         | ❌          | ❌            | ✅           |
| FastSAM-x      | `FastSAM-x.pt`      | [Segmentation d'instances](../tasks/segment.md) | ✅         | ❌          | ❌            | ✅           |

## Exemples d'utilisation

Les modèles FastSAM sont faciles à intégrer dans vos applications Python. Ultralytics propose une API Python conviviale et des commandes CLI pour simplifier le développement.

### Utilisation de la prédiction

Pour effectuer une détection d'objets sur une image, utilisez la méthode `Predict` comme indiqué ci-dessous :

!!! Example "Exemple"

    === "Python"
        ```python
        from ultralytics import FastSAM
        from ultralytics.models.fastsam import FastSAMPrompt

        # Définir une source d'inférence
        source = 'chemin/vers/bus.jpg'

        # Créer un modèle FastSAM
        model = FastSAM('FastSAM-s.pt')  # ou FastSAM-x.pt

        # Effectuer une inférence sur une image
        everything_results = model(source, device='cpu', retina_masks=True, imgsz=1024, conf=0.4, iou=0.9)

        # Préparer un objet Processus Invitation
        prompt_process = FastSAMPrompt(source, everything_results, device='cpu')

        # Invitation Everything
        ann = prompt_process.everything_prompt()

        # Bbox shape par défaut [0,0,0,0] -> [x1,y1,x2,y2]
        ann = prompt_process.box_prompt(bbox=[200, 200, 300, 300])

        # Invitation Text
        ann = prompt_process.text_prompt(text='une photo d\'un chien')

        # Invitation Point
        # points par défaut [[0,0]] [[x1,y1],[x2,y2]]
        # point_label par défaut [0] [1,0] 0:fond, 1:premier plan
        ann = prompt_process.point_prompt(points=[[200, 200]], pointlabel=[1])
        prompt_process.plot(annotations=ann, output='./')
        ```

    === "CLI"
        ```bash
        # Charger un modèle FastSAM et segmenter tout avec
        yolo segment predict model=FastSAM-s.pt source=chemin/vers/bus.jpg imgsz=640
        ```

Cet exemple démontre la simplicité du chargement d'un modèle pré-entraîné et de l'exécution d'une prédiction sur une image.

### Utilisation de la validation

La validation du modèle sur un ensemble de données peut être effectuée de la manière suivante :

!!! Example "Exemple"

    === "Python"
        ```python
        from ultralytics import FastSAM

        # Créer un modèle FastSAM
        model = FastSAM('FastSAM-s.pt')  # ou FastSAM-x.pt

        # Valider le modèle
        results = model.val(data='coco8-seg.yaml')
        ```

    === "CLI"
        ```bash
        # Charger un modèle FastSAM et le valider sur l'ensemble de données d'exemple COCO8 avec une taille d'image de 640 pixels
        yolo segment val model=FastSAM-s.pt data=coco8.yaml imgsz=640
        ```

Veuillez noter que FastSAM ne prend en charge que la détection et la segmentation d'une seule classe d'objet. Cela signifie qu'il reconnaîtra et segmentera tous les objets comme étant de la même classe. Par conséquent, lors de la préparation de l'ensemble de données, vous devez convertir tous les identifiants de catégorie d'objet en 0.

## Utilisation officielle de FastSAM

FastSAM est également disponible directement à partir du dépôt [https://github.com/CASIA-IVA-Lab/FastSAM](https://github.com/CASIA-IVA-Lab/FastSAM). Voici un bref aperçu des étapes typiques que vous pourriez suivre pour utiliser FastSAM :

### Installation

1. Clonez le dépôt FastSAM :
   ```shell
   git clone https://github.com/CASIA-IVA-Lab/FastSAM.git
   ```

2. Créez et activez un environnement Conda avec Python 3.9 :
   ```shell
   conda create -n FastSAM python=3.9
   conda activate FastSAM
   ```

3. Accédez au dépôt cloné et installez les packages requis :
   ```shell
   cd FastSAM
   pip install -r requirements.txt
   ```

4. Installez le modèle CLIP :
   ```shell
   pip install git+https://github.com/openai/CLIP.git
   ```

### Exemple d'utilisation

1. Téléchargez un [point de contrôle de modèle](https://drive.google.com/file/d/1m1sjY4ihXBU1fZXdQ-Xdj-mDltW-2Rqv/view?usp=sharing).

2. Utilisez FastSAM pour l'inférence. Exemples de commandes :

    - Segmentez tout dans une image :
      ```shell
      python Inference.py --model_path ./weights/FastSAM.pt --img_path ./images/dogs.jpg
      ```

    - Segmentez des objets spécifiques à l'aide de l'invitation de texte :
      ```shell
      python Inference.py --model_path ./weights/FastSAM.pt --img_path ./images/dogs.jpg --text_prompt "le chien jaune"
      ```

    - Segmentez des objets dans un rectangle englobant (fournir les coordonnées du rectangle au format xywh) :
      ```shell
      python Inference.py --model_path ./weights/FastSAM.pt --img_path ./images/dogs.jpg --box_prompt "[570,200,230,400]"
      ```

    - Segmentez des objets à proximité de points spécifiques :
      ```shell
      python Inference.py --model_path ./weights/FastSAM.pt --img_path ./images/dogs.jpg --point_prompt "[[520,360],[620,300]]" --point_label "[1,0]"
      ```

De plus, vous pouvez essayer FastSAM via une [démonstration Colab](https://colab.research.google.com/drive/1oX14f6IneGGw612WgVlAiy91UHwFAvr9?usp=sharing) ou sur la [démonstration Web HuggingFace](https://huggingface.co/spaces/An-619/FastSAM) pour une expérience visuelle.

## Citations et remerciements

Nous tenons à remercier les auteurs de FastSAM pour leurs contributions importantes dans le domaine de la segmentation d'instances en temps réel :

!!! Quote ""

    === "BibTeX"

      ```bibtex
      @misc{zhao2023fast,
            title={Fast Segment Anything},
            author={Xu Zhao and Wenchao Ding and Yongqi An and Yinglong Du and Tao Yu and Min Li and Ming Tang and Jinqiao Wang},
            year={2023},
            eprint={2306.12156},
            archivePrefix={arXiv},
            primaryClass={cs.CV}
      }
      ```

Le document original FastSAM peut être consulté sur [arXiv](https://arxiv.org/abs/2306.12156). Les auteurs ont rendu leur travail accessible au public, et le code source peut être consulté sur [GitHub](https://github.com/CASIA-IVA-Lab/FastSAM). Nous apprécions leurs efforts pour faire avancer le domaine et rendre leur travail accessible à la communauté dans son ensemble.
