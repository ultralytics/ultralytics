---
comments: true
description: En savoir plus sur MobileSAM, son implémentation, la comparaison avec SAM d'origine, et comment le télécharger et le tester dans le cadre de l'environnement Ultralytics. Améliorez vos applications mobiles dès aujourd'hui.
keywords: MobileSAM, Ultralytics, SAM, applications mobiles, Arxiv, GPU, API, encodeur d'image, décodeur de masque, téléchargement de modèle, méthode de test
---

![Logo MobileSAM](https://github.com/ChaoningZhang/MobileSAM/blob/master/assets/logo2.png?raw=true)

# Segmenter N'importe Quoi sur Mobile (MobileSAM)

Le document MobileSAM est maintenant disponible sur [arXiv](https://arxiv.org/pdf/2306.14289.pdf).

Une démonstration de MobileSAM exécutée sur un processeur CPU est accessible via ce [lien de démonstration](https://huggingface.co/spaces/dhkim2810/MobileSAM). Les performances sur un CPU Mac i5 prennent environ 3 secondes. Sur la démo de Hugging Face, l'interface ainsi que les CPU moins performants contribuent à une réponse plus lente, mais cela continue de fonctionner efficacement.

MobileSAM est implémenté dans divers projets, notamment [Grounding-SAM](https://github.com/IDEA-Research/Grounded-Segment-Anything), [AnyLabeling](https://github.com/vietanhdev/anylabeling), et [Segment Anything en 3D](https://github.com/Jumpat/SegmentAnythingin3D).

MobileSAM est entraîné sur un seul GPU avec un ensemble de données de 100 000 images (1% des images originales) en moins d'une journée. Le code de cet entraînement sera disponible à l'avenir.

## Modèles Disponibles, Tâches Prises en Charge et Modes d'Utilisation

Ce tableau présente les modèles disponibles avec leurs poids pré-entraînés spécifiques, les tâches qu'ils prennent en charge, et leur compatibilité avec les différents modes d'utilisation tels que [Inférence](../modes/predict.md), [Validation](../modes/val.md), [Entraînement](../modes/train.md) et [Export](../modes/export.md), indiqués par les emojis ✅ pour les modes pris en charge et ❌ pour les modes non pris en charge.

| Type de Modèle | Poids Pré-entraînés | Tâches Prises en Charge                         | Inférence | Validation | Entraînement | Export |
|----------------|---------------------|-------------------------------------------------|-----------|------------|--------------|--------|
| MobileSAM      | `mobile_sam.pt`     | [Segmentation d'Instances](../tasks/segment.md) | ✅         | ❌          | ❌            | ✅      |

## Passage de SAM à MobileSAM

Étant donné que MobileSAM conserve le même pipeline que SAM d'origine, nous avons incorporé le pré-traitement, le post-traitement et toutes les autres interfaces de l'original. Par conséquent, ceux qui utilisent actuellement SAM d'origine peuvent passer à MobileSAM avec un effort minimal.

MobileSAM a des performances comparables à celles de SAM d'origine et conserve le même pipeline à l'exception d'un changement dans l'encodeur d'image. Plus précisément, nous remplaçons l'encodeur d'image lourd original ViT-H (632M) par un encodeur Tiny-ViT plus petit (5M). Sur un seul GPU, MobileSAM fonctionne à environ 12 ms par image : 8 ms sur l'encodeur d'image et 4 ms sur le décodeur de masque.

Le tableau suivant présente une comparaison des encodeurs d'image basés sur ViT :

| Encodeur d'Image | SAM d'Origine | MobileSAM |
|------------------|---------------|-----------|
| Paramètres       | 611M          | 5M        |
| Vitesse          | 452 ms        | 8 ms      |

SAM d'origine et MobileSAM utilisent tous deux le même décodeur de masque basé sur une instruction :

| Décodeur de Masque | SAM d'Origine | MobileSAM |
|--------------------|---------------|-----------|
| Paramètres         | 3.876M        | 3.876M    |
| Vitesse            | 4 ms          | 4 ms      |

Voici une comparaison du pipeline complet :

| Pipeline Complet (Enc+Dec) | SAM d'Origine | MobileSAM |
|----------------------------|---------------|-----------|
| Paramètres                 | 615M          | 9.66M     |
| Vitesse                    | 456 ms        | 12 ms     |

Les performances de MobileSAM et de SAM d'origine sont démontrées en utilisant à la fois un point et une boîte comme instructions.

![Image avec un Point comme Instruction](https://raw.githubusercontent.com/ChaoningZhang/MobileSAM/master/assets/mask_box.jpg?raw=true)

![Image avec une Boîte comme Instruction](https://raw.githubusercontent.com/ChaoningZhang/MobileSAM/master/assets/mask_box.jpg?raw=true)

Avec ses performances supérieures, MobileSAM est environ 5 fois plus petit et 7 fois plus rapide que FastSAM actuel. Plus de détails sont disponibles sur la [page du projet MobileSAM](https://github.com/ChaoningZhang/MobileSAM).

## Test de MobileSAM dans Ultralytics

Tout comme SAM d'origine, nous proposons une méthode de test simple dans Ultralytics, comprenant des modes pour les instructions Point et Boîte.

### Téléchargement du modèle

Vous pouvez télécharger le modèle [ici](https://github.com/ChaoningZhang/MobileSAM/blob/master/weights/mobile_sam.pt).

### Instruction Point

!!! Example "Exemple"

    === "Python"
        ```python
        from ultralytics import SAM

        # Chargement du modèle
        model = SAM('mobile_sam.pt')

        # Prédiction d'un segment à partir d'une instruction Point
        model.predict('ultralytics/assets/zidane.jpg', points=[900, 370], labels=[1])
        ```

### Instruction Boîte

!!! Example "Exemple"

    === "Python"
        ```python
        from ultralytics import SAM

        # Chargement du modèle
        model = SAM('mobile_sam.pt')

        # Prédiction d'un segment à partir d'une instruction Boîte
        model.predict('ultralytics/assets/zidane.jpg', bboxes=[439, 437, 524, 709])
        ```

Nous avons mis en œuvre `MobileSAM` et `SAM` en utilisant la même API. Pour plus d'informations sur l'utilisation, veuillez consulter la [page SAM](sam.md).

## Citations et Remerciements

Si vous trouvez MobileSAM utile dans vos travaux de recherche ou de développement, veuillez envisager de citer notre document :

!!! Quote ""

    === "BibTeX"

        ```bibtex
        @article{mobile_sam,
          title={Faster Segment Anything: Towards Lightweight SAM for Mobile Applications},
          author={Zhang, Chaoning and Han, Dongshen and Qiao, Yu and Kim, Jung Uk and Bae, Sung Ho and Lee, Seungkyu and Hong, Choong Seon},
          journal={arXiv preprint arXiv:2306.14289},
          year={2023}
        }
