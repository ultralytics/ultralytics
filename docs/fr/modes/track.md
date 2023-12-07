---
comments: true
description: Apprenez à utiliser Ultralytics YOLO pour le suivi d'objets dans les flux vidéo. Guides pour utiliser différents traceurs et personnaliser les configurations de traceurs.
keywords: Ultralytics, YOLO, suivi d'objets, flux vidéo, BoT-SORT, ByteTrack, guide Python, guide CLI
---

# Suivi Multi-Objets avec Ultralytics YOLO

<img width="1024" src="https://user-images.githubusercontent.com/26833433/243418637-1d6250fd-1515-4c10-a844-a32818ae6d46.png" alt="Exemples de suivi multi-objets">

Le suivi d'objets dans le domaine de l'analyse vidéo est une tâche essentielle qui non seulement identifie l'emplacement et la classe des objets à l'intérieur de l'image, mais maintient également un identifiant unique pour chaque objet détecté au fur et à mesure que la vidéo progresse. Les applications sont illimitées, allant de la surveillance et de la sécurité à l'analytique sportive en temps réel.

## Pourquoi Choisir Ultralytics YOLO pour le Suivi d'Objet ?

La sortie des traceurs Ultralytics est cohérente avec la détection standard d'objets mais apporte la valeur ajoutée des identifiants d'objets. Cela facilite le suivi des objets dans les flux vidéo et effectue des analyses subséquentes. Voici pourquoi vous devriez envisager d'utiliser Ultralytics YOLO pour vos besoins de suivi d'objet :

- **Efficacité :** Traitez les flux vidéo en temps réel sans compromettre la précision.
- **Flexibilité :** Prend en charge de multiples algorithmes de suivi et configurations.
- **Facilité d'Utilisation :** API Python simple et options CLI pour une intégration et un déploiement rapides.
- **Personnalisabilité :** Facile à utiliser avec des modèles YOLO entraînés sur mesure, permettant une intégration dans des applications spécifiques au domaine.

<p align="center">
  <br>
  <iframe width="720" height="405" src="https://www.youtube.com/embed/hHyHmOtmEgs?si=VNZtXmm45Nb9s-N-"
    title="Lecteur vidéo YouTube" frameborder="0"
    allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share"
    allowfullscreen>
  </iframe>
  <br>
  <strong>Regardez :</strong> Détection et suivi d'objets avec Ultralytics YOLOv8.
</p>

## Applications dans le Monde Réel

|                                                        Transport                                                         |                                                       Distribution                                                       |                                                       Aquaculture                                                       |
|:------------------------------------------------------------------------------------------------------------------------:|:------------------------------------------------------------------------------------------------------------------------:|:-----------------------------------------------------------------------------------------------------------------------:|
| ![Suivi de véhicules](https://github.com/RizwanMunawar/ultralytics/assets/62513924/ee6e6038-383b-4f21-ac29-b2a1c7d386ab) | ![Suivi de personnes](https://github.com/RizwanMunawar/ultralytics/assets/62513924/93bb4ee2-77a0-4e4e-8eb6-eb8f527f0527) | ![Suivi de poissons](https://github.com/RizwanMunawar/ultralytics/assets/62513924/a5146d0f-bfa8-4e0a-b7df-3c1446cd8142) |
|                                                    Suivi de Véhicules                                                    |                                                    Suivi de Personnes                                                    |                                                    Suivi de Poissons                                                    |

## Caractéristiques en Bref

Ultralytics YOLO étend ses fonctionnalités de détection d'objets pour fournir un suivi d'objets robuste et polyvalent :

- **Suivi en Temps Réel :** Suivi fluide d'objets dans des vidéos à fréquence d'images élevée.
- **Prise en Charge de Multiples Traceurs :** Choisissez parmi une variété d'algorithmes de suivi éprouvés.
- **Configurations de Traceurs Personnalisables :** Adaptez l'algorithme de suivi pour répondre à des exigences spécifiques en réglant divers paramètres.

## Traceurs Disponibles

Ultralytics YOLO prend en charge les algorithmes de suivi suivants. Ils peuvent être activés en passant le fichier de configuration YAML correspondant tel que `tracker=tracker_type.yaml` :

* [BoT-SORT](https://github.com/NirAharon/BoT-SORT) - Utilisez `botsort.yaml` pour activer ce traceur.
* [ByteTrack](https://github.com/ifzhang/ByteTrack) - Utilisez `bytetrack.yaml` pour activer ce traceur.

Le traceur par défaut est BoT-SORT.

## Suivi

Pour exécuter le traceur sur des flux vidéo, utilisez un modèle Detect, Segment ou Pose formé tel que YOLOv8n, YOLOv8n-seg et YOLOv8n-pose.

!!! Example "Exemple"

    === "Python"

        ```python
        from ultralytics import YOLO

        # Charger un modèle officiel ou personnalisé
        model = YOLO('yolov8n.pt')  # Charger un modèle Detect officiel
        model = YOLO('yolov8n-seg.pt')  # Charger un modèle Segment officiel
        model = YOLO('yolov8n-pose.pt')  # Charger un modèle Pose officiel
        model = YOLO('chemin/vers/best.pt')  # Charger un modèle entraîné personnalisé

        # Effectuer le suivi avec le modèle
        results = model.track(source="https://youtu.be/LNwODJXcvt4", show=True)  # Suivi avec le traceur par défaut
        results = model.track(source="https://youtu.be/LNwODJXcvt4", show=True, tracker="bytetrack.yaml")  # Suivi avec le traceur ByteTrack
        ```

    === "CLI"

        ```bash
        # Effectuer le suivi avec divers modèles en utilisant l'interface en ligne de commande
        yolo track model=yolov8n.pt source="https://youtu.be/LNwODJXcvt4"  # Modèle Detect officiel
        yolo track model=yolov8n-seg.pt source="https://youtu.be/LNwODJXcvt4"  # Modèle Segment officiel
        yolo track model=yolov8n-pose.pt source="https://youtu.be/LNwODJXcvt4"  # Modèle Pose officiel
        yolo track model=chemin/vers/best.pt source="https://youtu.be/LNwODJXcvt4"  # Modèle entraîné personnalisé

        # Suivi en utilisant le traceur ByteTrack
        yolo track model=chemin/vers/best.pt tracker="bytetrack.yaml"
        ```

Comme on peut le voir dans l'utilisation ci-dessus, le suivi est disponible pour tous les modèles Detect, Segment et Pose exécutés sur des vidéos ou des sources de diffusion.

## Configuration

### Arguments de Suivi

La configuration du suivi partage des propriétés avec le mode Prédiction, telles que `conf`, `iou`, et `show`. Pour des configurations supplémentaires, référez-vous à la page [Predict](https://docs.ultralytics.com/modes/predict/) du modèle.

!!! Example "Exemple"

    === "Python"

        ```python
        from ultralytics import YOLO

        # Configurer les paramètres de suivi et exécuter le traceur
        model = YOLO('yolov8n.pt')
        results = model.track(source="https://youtu.be/LNwODJXcvt4", conf=0.3, iou=0.5, show=True)
        ```

    === "CLI"

        ```bash
        # Configurer les paramètres de suivi et exécuter le traceur en utilisant l'interface en ligne de commande
        yolo track model=yolov8n.pt source="https://youtu.be/LNwODJXcvt4" conf=0.3, iou=0.5 show
        ```

### Sélection du Traceur

Ultralytics vous permet également d'utiliser un fichier de configuration de traceur modifié. Pour cela, faites simplement une copie d'un fichier de configuration de traceur (par exemple, `custom_tracker.yaml`) à partir de [ultralytics/cfg/trackers](https://github.com/ultralytics/ultralytics/tree/main/ultralytics/cfg/trackers) et modifiez toute configuration (à l'exception du `tracker_type`) selon vos besoins.

!!! Example "Exemple"

    === "Python"

        ```python
        from ultralytics import YOLO

        # Charger le modèle et exécuter le traceur avec un fichier de configuration personnalisé
        model = YOLO('yolov8n.pt')
        results = model.track(source="https://youtu.be/LNwODJXcvt4", tracker='custom_tracker.yaml')
        ```

    === "CLI"

        ```bash
        # Charger le modèle et exécuter le traceur avec un fichier de configuration personnalisé en utilisant l'interface en ligne de commande
        yolo track model=yolov8n.pt source="https://youtu.be/LNwODJXcvt4" tracker='custom_tracker.yaml'
        ```

Pour une liste complète des arguments de suivi, référez-vous à la page [ultralytics/cfg/trackers](https://github.com/ultralytics/ultralytics/tree/main/ultralytics/cfg/trackers).

## Exemples Python

### Boucle de Persistance des Pistes

Voici un script Python utilisant OpenCV (`cv2`) et YOLOv8 pour exécuter le suivi d'objet sur des images vidéo. Ce script suppose toujours que vous avez déjà installé les packages nécessaires (`opencv-python` et `ultralytics`). L'argument `persist=True` indique au traceur que l'image ou la trame actuelle est la suivante dans une séquence et s'attend à ce que les pistes de l'image précédente soient présentes dans l'image actuelle.

!!! Example "Boucle for streaming avec suivi"

    ```python
    import cv2
    from ultralytics import YOLO

    # Charger le modèle YOLOv8
    model = YOLO('yolov8n.pt')

    # Ouvrir le fichier vidéo
    video_path = "chemin/vers/video.mp4"
    cap = cv2.VideoCapture(video_path)

    # Parcourir les images vidéo
    while cap.isOpened():
        # Lire une image de la vidéo
        success, frame = cap.read()

        if success:
            # Exécuter le suivi YOLOv8 sur l'image, en persistant les pistes entre les images
            results = model.track(frame, persist=True)

            # Visualiser les résultats sur l'image
            annotated_frame = results[0].plot()

            # Afficher l'image annotée
            cv2.imshow("Suivi YOLOv8", annotated_frame)

            # Interrompre la boucle si 'q' est pressée
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        else:
            # Interrompre la boucle si la fin de la vidéo est atteinte
            break

    # Relâcher l'objet de capture vidéo et fermer la fenêtre d'affichage
    cap.release()
    cv2.destroyAllWindows()
    ```

Veuillez noter le changement de `model(frame)` à `model.track(frame)`, qui active le suivi d'objet à la place de la simple détection. Ce script modifié exécutera le traceur sur chaque image de la vidéo, visualisera les résultats et les affichera dans une fenêtre. La boucle peut être quittée en appuyant sur 'q'.

## Contribuer de Nouveaux Traceurs

Êtes-vous compétent en suivi multi-objets et avez-vous réussi à implémenter ou adapter un algorithme de suivi avec Ultralytics YOLO ? Nous vous invitons à contribuer à notre section Traceurs sur [ultralytics/cfg/trackers](https://github.com/ultralytics/ultralytics/tree/main/ultralytics/cfg/trackers) ! Vos applications et solutions dans le monde réel pourraient être inestimables pour les utilisateurs travaillant sur des tâches de suivi.

En contribuant à cette section, vous aidez à élargir l'éventail des solutions de suivi disponibles au sein du cadre Ultralytics YOLO, ajoutant une autre couche de fonctionnalité et d'utilité pour la communauté.

Pour initier votre contribution, veuillez vous référer à notre [Guide de Contribution](https://docs.ultralytics.com/help/contributing) pour des instructions complètes sur la soumission d'une Pull Request (PR) 🛠️. Nous sommes impatients de voir ce que vous apportez à la table !

Ensemble, améliorons les capacités de suivi de l'écosystème Ultralytics YOLO 🙏 !
