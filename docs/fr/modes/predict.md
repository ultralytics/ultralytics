---
comments: true
description: Découvrez comment utiliser le mode de prédiction YOLOv8 pour diverses tâches. Apprenez sur différentes sources d'inférence comme des images, vidéos et formats de données.
keywords: Ultralytics, YOLOv8, mode de prédiction, sources d'inférence, tâches de prédiction, mode streaming, traitement d'images, traitement vidéo, apprentissage automatique, IA
---

# Prédiction de Modèle avec Ultralytics YOLO

<img width="1024" src="https://github.com/ultralytics/assets/raw/main/yolov8/banner-integrations.png" alt="Écosystème et intégrations Ultralytics YOLO">

## Introduction

Dans l'univers de l'apprentissage automatique et de la vision par ordinateur, le processus de donner du sens aux données visuelles est appelé 'inférence' ou 'prédiction'. Ultralytics YOLOv8 propose une fonctionnalité puissante connue sous le nom de **mode de prédiction** adapté pour l'inférence en temps réel et haute performance sur une large gamme de sources de données.

<p align="center">
  <br>
  <iframe width="720" height="405" src="https://www.youtube.com/embed/QtsI0TnwDZs?si=ljesw75cMO2Eas14"
    title="Lecteur vidéo YouTube" frameborder="0"
    allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share"
    allowfullscreen>
  </iframe>
  <br>
  <strong>Regardez :</strong> Comment Extraire les Sorties du Modèle Ultralytics YOLOv8 pour des Projets Personnalisés.
</p>

## Applications Réelles

|                                                               Fabrication                                                               |                                                                 Sports                                                                  |                                                                Sécurité                                                                |
|:---------------------------------------------------------------------------------------------------------------------------------------:|:---------------------------------------------------------------------------------------------------------------------------------------:|:--------------------------------------------------------------------------------------------------------------------------------------:|
| ![Détection des Pièces de Véhicules](https://github.com/RizwanMunawar/ultralytics/assets/62513924/a0f802a8-0776-44cf-8f17-93974a4a28a1) | ![Détection des Joueurs de Football](https://github.com/RizwanMunawar/ultralytics/assets/62513924/7d320e1f-fc57-4d7f-a691-78ee579c3442) | ![Détection de Chutes de Personnes](https://github.com/RizwanMunawar/ultralytics/assets/62513924/86437c4a-3227-4eee-90ef-9efb697bdb43) |
|                                                    Détection des Pièces de Véhicules                                                    |                                                    Détection des Joueurs de Football                                                    |                                                    Détection de Chutes de Personnes                                                    |

## Pourquoi Utiliser Ultralytics YOLO pour l'Inférence ?

Voici pourquoi vous devriez considérer le mode de prédiction YOLOv8 pour vos besoins variés en inférence :

- **Polyvalence :** Capable de faire des inférences sur des images, des vidéos et même des flux en direct.
- **Performance :** Conçu pour le traitement en temps réel à grande vitesse sans sacrifier la précision.
- **Facilité d'Utilisation :** Interfaces Python et CLI intuitives pour un déploiement et des tests rapides.
- **Très Personnalisable :** Divers paramètres et réglages pour ajuster le comportement d'inférence du modèle selon vos besoins spécifiques.

### Caractéristiques Clés du Mode de Prédiction

Le mode de prédiction YOLOv8 est conçu pour être robuste et polyvalent, avec des fonctionnalités telles que :

- **Compatibilité avec Plusieurs Sources de Données :** Que vos données soient sous forme d'images individuelles, d'une collection d'images, de fichiers vidéo ou de flux vidéo en temps réel, le mode de prédiction répond à vos besoins.
- **Mode Streaming :** Utilisez la fonctionnalité de streaming pour générer un générateur efficace en termes de mémoire d'objets `Results`. Activez-le en réglant `stream=True` dans la méthode d'appel du prédicteur.
- **Traitement par Lots :** La capacité de traiter plusieurs images ou trames vidéo dans un seul lot, accélérant ainsi le temps d'inférence.
- **Facile à Intégrer :** S'intègre facilement dans les pipelines de données existants et autres composants logiciels, grâce à son API souple.

Les modèles YOLO d'Ultralytics renvoient soit une liste d'objets `Results` Python, soit un générateur Python efficace en termes de mémoire d'objets `Results` lorsque `stream=True` est passé au modèle pendant l'inférence :

!!! Example "Prédire"

    === "Renvoie une liste avec `stream=False`"
        ```python
        from ultralytics import YOLO

        # Charger un modèle
        model = YOLO('yolov8n.pt')  # modèle YOLOv8n pré-entraîné

        # Exécuter une inférence par lots sur une liste d'images
        results = model(['im1.jpg', 'im2.jpg'])  # renvoie une liste d'objets Results

        # Traiter la liste des résultats
        for result in results:
            boxes = result.boxes  # Objet Boxes pour les sorties bbox
            masks = result.masks  # Objet Masks pour les masques de segmentation
            keypoints = result.keypoints  # Objet Keypoints pour les sorties de pose
            probs = result.probs  # Objet Probs pour les sorties de classification
        ```

    === "Renvoie un générateur avec `stream=True`"
        ```python
        from ultralytics import YOLO

        # Charger un modèle
        model = YOLO('yolov8n.pt')  # modèle YOLOv8n pré-entraîné

        # Exécuter une inférence par lots sur une liste d'images
        results = model(['im1.jpg', 'im2.jpg'], stream=True)  # renvoie un générateur d'objets Results

        # Traiter le générateur de résultats
        for result in results:
            boxes = result.boxes  # Objet Boxes pour les sorties bbox
            masks = result.masks  # Objet Masks pour les masques de segmentation
            keypoints = result.keypoints  # Objet Keypoints pour les sorties de pose
            probs = result.probs  # Objet Probs pour les sorties de classification
        ```

## Sources d'Inférence

YOLOv8 peut traiter différents types de sources d'entrée pour l'inférence, comme illustré dans le tableau ci-dessous. Les sources incluent des images statiques, des flux vidéos et divers formats de données. Le tableau indique également si chaque source peut être utilisée en mode streaming avec l'argument `stream=True` ✅. Le mode streaming est bénéfique pour traiter des vidéos ou des flux en direct car il crée un générateur de résultats au lieu de charger tous les cadres en mémoire.

!!! astuce "Astuce"

    Utilisez `stream=True` pour traiter des vidéos longues ou des jeux de données volumineux afin de gérer efficacement la mémoire. Quand `stream=False`, les résultats pour tous les cadres ou points de données sont stockés en mémoire, ce qui peut rapidement s'accumuler et provoquer des erreurs de mémoire insuffisante pour de grandes entrées. En revanche, `stream=True` utilise un générateur, qui ne garde que les résultats du cadre ou point de données actuel en mémoire, réduisant considérablement la consommation de mémoire et prévenant les problèmes de mémoire insuffisante.

| Source          | Argument                                   | Type            | Notes                                                                                                                        |
|-----------------|--------------------------------------------|-----------------|------------------------------------------------------------------------------------------------------------------------------|
| image           | `'image.jpg'`                              | `str` ou `Path` | Fichier image unique.                                                                                                        |
| URL             | `'https://ultralytics.com/images/bus.jpg'` | `str`           | URL vers une image.                                                                                                          |
| capture d'écran | `'screen'`                                 | `str`           | Prendre une capture d'écran.                                                                                                 |
| PIL             | `Image.open('im.jpg')`                     | `PIL.Image`     | Format HWC avec canaux RGB.                                                                                                  |
| OpenCV          | `cv2.imread('im.jpg')`                     | `np.ndarray`    | Format HWC avec canaux BGR `uint8 (0-255)`.                                                                                  |
| numpy           | `np.zeros((640,1280,3))`                   | `np.ndarray`    | Format HWC avec canaux BGR `uint8 (0-255)`.                                                                                  |
| torch           | `torch.zeros(16,3,320,640)`                | `torch.Tensor`  | Format BCHW avec canaux RGB `float32 (0.0-1.0)`.                                                                             |
| CSV             | `'sources.csv'`                            | `str` ou `Path` | Fichier CSV contenant des chemins vers des images, vidéos ou répertoires.                                                    |
| vidéo ✅         | `'video.mp4'`                              | `str` ou `Path` | Fichier vidéo dans des formats comme MP4, AVI, etc.                                                                          |
| répertoire ✅    | `'chemin/'`                                | `str` ou `Path` | Chemin vers un répertoire contenant des images ou des vidéos.                                                                |
| motif global ✅  | `'chemin/*.jpg'`                           | `str`           | Motif glob pour faire correspondre plusieurs fichiers. Utilisez le caractère `*` comme joker.                                |
| YouTube ✅       | `'https://youtu.be/LNwODJXcvt4'`           | `str`           | URL vers une vidéo YouTube.                                                                                                  |
| flux ✅          | `'rtsp://exemple.com/media.mp4'`           | `str`           | URL pour des protocoles de streaming comme RTSP, RTMP, TCP, ou une adresse IP.                                               |
| multi-flux ✅    | `'liste.streams'`                          | `str` ou `Path` | Fichier texte `*.streams` avec une URL de flux par ligne, c'est-à-dire que 8 flux s'exécuteront avec une taille de lot de 8. |

Ci-dessous des exemples de code pour utiliser chaque type de source :

!!! Example "Sources de prédiction"

    === "image"
        Exécutez une inférence sur un fichier image.
        ```python
        from ultralytics import YOLO

        # Charger un modèle YOLOv8n pré-entraîné
        model = YOLO('yolov8n.pt')

        # Définir le chemin vers le fichier image
        source = 'chemin/vers/image.jpg'

        # Exécuter une inférence sur la source
        results = model(source)  # liste d'objets Results
        ```

    === "capture d'écran"
        Exécutez une inférence sur le contenu actuel de l'écran sous forme de capture d'écran.
        ```python
        from ultralytics import YOLO

        # Charger un modèle YOLOv8n pré-entraîné
        model = YOLO('yolov8n.pt')

        # Définir la capture d'écran actuelle comme source
        source = 'screen'

        # Exécuter une inférence sur la source
        results = model(source)  # liste d'objets Results
        ```

    === "URL"
        Exécutez une inférence sur une image ou vidéo hébergée à distance via URL.
        ```python
        from ultralytics import YOLO

        # Charger un modèle YOLOv8n pré-entraîné
        model = YOLO('yolov8n.pt')

        # Définir l'URL d'une image ou vidéo distante
        source = 'https://ultralytics.com/images/bus.jpg'

        # Exécuter une inférence sur la source
        results = model(source)  # liste d'objets Results
        ```

    === "PIL"
        Exécutez une inférence sur une image ouverte avec la bibliothèque Python Imaging Library (PIL).
        ```python
        from PIL import Image
        from ultralytics import YOLO

        # Charger un modèle YOLOv8n pré-entraîné
        model = YOLO('yolov8n.pt')

        # Ouvrir une image avec PIL
        source = Image.open('chemin/vers/image.jpg')

        # Exécuter une inférence sur la source
        results = model(source)  # liste d'objets Results
        ```

    === "OpenCV"
        Exécutez une inférence sur une image lue avec OpenCV.
        ```python
        import cv2
        from ultralytics import YOLO

        # Charger un modèle YOLOv8n pré-entraîné
        model = YOLO('yolov8n.pt')

        # Lire une image avec OpenCV
        source = cv2.imread('chemin/vers/image.jpg')

        # Exécuter une inférence sur la source
        results = model(source)  # liste d'objets Results
        ```

    === "numpy"
        Exécutez une inférence sur une image représentée sous forme de tableau numpy.
        ```python
        import numpy as np
        from ultralytics import YOLO

        # Charger un modèle YOLOv8n pré-entraîné
        model = YOLO('yolov8n.pt')

        # Créer un tableau numpy aléatoire de forme HWC (640, 640, 3) avec des valeurs dans l'intervalle [0, 255] et de type uint8
        source = np.random.randint(low=0, high=255, size=(640, 640, 3), dtype='uint8')

        # Exécuter une inférence sur la source
        results = model(source)  # liste d'objets Results
        ```

    === "torch"
        Exécutez une inférence sur une image représentée sous forme de tenseur PyTorch.
        ```python
        import torch
        from ultralytics import YOLO

        # Charger un modèle YOLOv8n pré-entraîné
        model = YOLO('yolov8n.pt')

        # Créer un tenseur aléatoire torch de forme BCHW (1, 3, 640, 640) avec des valeurs dans l'intervalle [0, 1] et de type float32
        source = torch.rand(1, 3, 640, 640, dtype=torch.float32)

        # Exécuter une inférence sur la source
        results = model(source)  # liste d'objets Results
