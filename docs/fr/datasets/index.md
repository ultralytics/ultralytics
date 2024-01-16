---
comments: true
description: Explorez divers ensembles de données de vision par ordinateur pris en charge par Ultralytics pour la détection d'objets, la segmentation, l'estimation de la pose, la classification d'images et le suivi multi-objets.
keywords: vision par ordinateur, ensembles de données, Ultralytics, YOLO, détection d'objets, segmentation d'instance, estimation de la pose, classification d'images, suivi multi-objets
---

# Aperçu des ensembles de données

Ultralytics fournit un soutien pour divers ensembles de données pour faciliter les tâches de vision par ordinateur telles que la détection, la segmentation d'instance, l'estimation de la pose, la classification et le suivi multi-objets. Ci-dessous se trouve une liste des principaux ensembles de données Ultralytics, suivie d'un résumé de chaque tâche de vision par ordinateur et des ensembles de données respectifs.

!!! Note "Note"

    🚧 Notre documentation multilingue est actuellement en cours de construction et nous travaillons dur pour l'améliorer. Merci de votre patience ! 🙏

## [Ensembles de données de détection](../../datasets/detect/index.md)

La détection d'objets par boîte englobante est une technique de vision par ordinateur qui consiste à détecter et localiser des objets dans une image en dessinant une boîte englobante autour de chaque objet.

- [Argoverse](../../datasets/detect/argoverse.md) : Un ensemble de données contenant des données de suivi 3D et de prévision de mouvement dans des environnements urbains avec des annotations détaillées.
- [COCO](../../datasets/detect/coco.md) : Un ensemble de données de grande échelle conçu pour la détection d'objets, la segmentation et l'annotation avec plus de 200K images étiquetées.
- [COCO8](../../datasets/detect/coco8.md) : Contient les 4 premières images de COCO train et COCO val, adaptées pour des tests rapides.
- [Global Wheat 2020](../../datasets/detect/globalwheat2020.md) : Un ensemble de données d'images de têtes de blé recueillies dans le monde entier pour les tâches de détection et de localisation d'objets.
- [Objects365](../../datasets/detect/objects365.md) : Un ensemble de données de grande qualité et à grande échelle pour la détection d'objets avec 365 catégories d'objets et plus de 600K images annotées.
- [OpenImagesV7](../../datasets/detect/open-images-v7.md) : Un ensemble de données complet de Google avec 1.7M d'images d'entraînement et 42k images de validation.
- [SKU-110K](../../datasets/detect/sku-110k.md) : Un ensemble de données mettant en vedette la détection d'objets denses dans les environnements de vente au détail avec plus de 11K images et 1.7 million de boîtes englobantes.
- [VisDrone](../../datasets/detect/visdrone.md) : Un ensemble de données contenant des données de détection d'objets et de suivi multi-objets à partir d'images capturées par drone avec plus de 10K images et séquences vidéo.
- [VOC](../../datasets/detect/voc.md) : L'ensemble de données de classes d'objets visuels Pascal (VOC) pour la détection d'objets et la segmentation avec 20 classes d'objets et plus de 11K images.
- [xView](../../datasets/detect/xview.md) : Un ensemble de données pour la détection d'objets dans l'imagerie aérienne avec 60 catégories d'objets et plus d'un million d'objets annotés.

## [Ensembles de données de segmentation d'instance](../../datasets/segment/index.md)

La segmentation d'instance est une technique de vision par ordinateur qui consiste à identifier et localiser des objets dans une image au niveau des pixels.

- [COCO](../../datasets/segment/coco.md) : Un ensemble de données de grande échelle conçu pour la détection d'objets, la segmentation et les tâches d'annotation avec plus de 200K images étiquetées.
- [COCO8-seg](../../datasets/segment/coco8-seg.md) : Un ensemble de données plus petit pour les tâches de segmentation d'instance, contenant un sous-ensemble de 8 images COCO avec des annotations de segmentation.

## [Estimation de pose](../../datasets/pose/index.md)

L'estimation de la pose est une technique utilisée pour déterminer la pose de l'objet par rapport à la caméra ou au système de coordonnées mondial.

- [COCO](../../datasets/pose/coco.md) : Un ensemble de données de grande échelle avec des annotations de poses humaines conçu pour les tâches d'estimation de la pose.
- [COCO8-pose](../../datasets/pose/coco8-pose.md) : Un ensemble de données plus petit pour les tâches d'estimation de la pose, contenant un sous-ensemble de 8 images COCO avec des annotations de pose humaine.
- [Tiger-pose](../../datasets/pose/tiger-pose.md) : Un ensemble de données compact composé de 263 images centrées sur les tigres, annotées avec 12 points par tigre pour les tâches d'estimation de la pose.

## [Classification](../../datasets/classify/index.md)

La classification d'images est une tâche de vision par ordinateur qui implique de catégoriser une image dans une ou plusieurs classes ou catégories prédéfinies en fonction de son contenu visuel.

- [Caltech 101](../../datasets/classify/caltech101.md) : Un ensemble de données contenant des images de 101 catégories d'objets pour les tâches de classification d'images.
- [Caltech 256](../../datasets/classify/caltech256.md) : Une version étendue de Caltech 101 avec 256 catégories d'objets et des images plus complexes.
- [CIFAR-10](../../datasets/classify/cifar10.md) : Un ensemble de données de 60K images couleur 32x32 réparties en 10 classes, avec 6K images par classe.
- [CIFAR-100](../../datasets/classify/cifar100.md) : Une version étendue de CIFAR-10 avec 100 catégories d'objets et 600 images par classe.
- [Fashion-MNIST](../../datasets/classify/fashion-mnist.md) : Un ensemble de données composé de 70 000 images en niveaux de gris de 10 catégories de mode pour les tâches de classification d'images.
- [ImageNet](../../datasets/classify/imagenet.md) : Un ensemble de données à grande échelle pour la détection d'objets et la classification d'images avec plus de 14 millions d'images et 20 000 catégories.
- [ImageNet-10](../../datasets/classify/imagenet10.md) : Un sous-ensemble plus petit d'ImageNet avec 10 catégories pour des expériences et des tests plus rapides.
- [Imagenette](../../datasets/classify/imagenette.md) : Un sous-ensemble plus petit d'ImageNet qui contient 10 classes facilement distinctes pour un entraînement et des tests plus rapides.
- [Imagewoof](../../datasets/classify/imagewoof.md) : Un sous-ensemble d'ImageNet plus difficile contenant 10 catégories de races de chiens pour les tâches de classification d'images.
- [MNIST](../../datasets/classify/mnist.md) : Un ensemble de données de 70 000 images en niveaux de gris de chiffres manuscrits pour les tâches de classification d'images.

## [Boîtes Englobantes Orientées (OBB)](../../datasets/obb/index.md)

Les Boîtes Englobantes Orientées (OBB) sont une méthode en vision par ordinateur pour détecter des objets inclinés dans les images en utilisant des boîtes englobantes rotatives, souvent appliquée à l'imagerie aérienne et satellite.

- [DOTAv2](../../datasets/obb/dota-v2.md) : Un ensemble de données d'imagerie aérienne populaire avec 1.7 million d'instances et 11 268 images.

## [Suivi Multi-Objets](../../datasets/track/index.md)

Le suivi multi-objets est une technique de vision par ordinateur qui consiste à détecter et suivre plusieurs objets dans le temps dans une séquence vidéo.

- [Argoverse](../../datasets/detect/argoverse.md) : Un ensemble de données contenant des données de suivi 3D et de prévision de mouvement dans des environnements urbains avec des annotations détaillées pour les tâches de suivi multi-objets.
- [VisDrone](../../datasets/detect/visdrone.md) : Un ensemble de données contenant des données de détection d'objets et de suivi multi-objets à partir d'images capturées par drone avec plus de 10K images et séquences vidéo.

## Contribuer de Nouveaux Ensembles de Données

Contribuer un nouvel ensemble de données implique plusieurs étapes pour s'assurer qu'il s'aligne bien avec l'infrastructure existante. Voici les étapes nécessaires :

### Étapes pour Contribuer un Nouvel Ensemble de Données

1. **Collecter des Images** : Rassemblez les images qui appartiennent à l'ensemble de données. Celles-ci pourraient être collectées à partir de différentes sources, telles que des bases de données publiques ou votre propre collection.

2. **Annoter des Images** : Annotez ces images avec des boîtes englobantes, des segments ou des points clés, en fonction de la tâche.

3. **Exporter des Annotations** : Convertissez ces annotations au format de fichier YOLO `*.txt` pris en charge par Ultralytics.

4. **Organiser l'Ensemble de Données** : Rangez votre ensemble de données dans la bonne structure de dossiers. Vous devriez avoir des répertoires de niveau supérieur `train/` et `val/`, et à l'intérieur de chacun, un sous-répertoire `images/` et `labels/`.

    ```
    dataset/
    ├── train/
    │   ├── images/
    │   └── labels/
    └── val/
        ├── images/
        └── labels/
    ```

5. **Créer un Fichier `data.yaml`** : Dans le répertoire racine de votre ensemble de données, créez un fichier `data.yaml` qui décrit l'ensemble de données, les classes et les autres informations nécessaires.

6. **Optimiser les Images (Optionnel)** : Si vous souhaitez réduire la taille de l'ensemble de données pour un traitement plus efficace, vous pouvez optimiser les images en utilisant le code ci-dessous. Ceci n'est pas requis, mais recommandé pour des tailles d'ensemble de données plus petites et des vitesses de téléchargement plus rapides.

7. **Zipper l'Ensemble de Données** : Compressez le dossier complet de l'ensemble de données dans un fichier zip.

8. **Documenter et PR** : Créez une page de documentation décrivant votre ensemble de données et comment il s'intègre dans le cadre existant. Après cela, soumettez une Pull Request (PR). Référez-vous aux [lignes directrices de contribution Ultralytics](https://docs.ultralytics.com/help/contributing) pour plus de détails sur la manière de soumettre une PR.

### Exemple de Code pour Optimiser et Zipper un Ensemble de Données

!!! Example "Optimiser et Zipper un Ensemble de Données"

    === "Python"

    ```python
    from pathlib import Path
    from ultralytics.data.utils import compress_one_image
    from ultralytics.utils.downloads import zip_directory

    # Définir le répertoire de l'ensemble de données
    path = Path('chemin/vers/ensemble-de-données')

    # Optimiser les images dans l'ensemble de données (optionnel)
    for f in path.rglob('*.jpg'):
        compress_one_image(f)

    # Zipper l'ensemble de données dans 'chemin/vers/ensemble-de-données.zip'
    zip_directory(path)
    ```

En suivant ces étapes, vous pouvez contribuer un nouvel ensemble de données qui s'intègre bien avec la structure existante d'Ultralytics.
