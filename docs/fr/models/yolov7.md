---
comments: true
description: Découvrez le YOLOv7, un détecteur d'objets en temps réel. Comprenez sa vitesse supérieure, son impressionnante précision et son accent unique sur l'optimisation bag-of-freebies entraînable.
keywords: YOLOv7, détecteur d'objets en temps réel, état de l'art, Ultralytics, jeu de données MS COCO, ré-paramétrisation du modèle, affectation des étiquettes dynamiques, mise à l'échelle étendue, mise à l'échelle composée
---

# YOLOv7 : Bag-of-Freebies Entraînable

YOLOv7 est un détecteur d'objets en temps réel à la pointe de la technologie qui surpasse tous les détecteurs d'objets connus en termes de vitesse et de précision, dans une plage de 5 FPS à 160 FPS. Il présente la précision la plus élevée (56,8% AP) parmi tous les détecteurs d'objets en temps réel connus avec un FPS de 30 ou plus sur GPU V100. De plus, YOLOv7 surpasse les autres détecteurs d'objets tels que YOLOR, YOLOX, Scaled-YOLOv4, YOLOv5 et bien d'autres en termes de vitesse et de précision. Le modèle est entraîné à partir de zéro sur le jeu de données MS COCO, sans utiliser d'autres jeux de données ou de poids pré-entraînés. Le code source de YOLOv7 est disponible sur GitHub.

![Comparaison de YOLOv7 avec les détecteurs d'objets SOTA](https://github.com/ultralytics/ultralytics/assets/26833433/5e1e0420-8122-4c79-b8d0-2860aa79af92)
**Comparaison des détecteurs d'objets de pointe.
** À partir des résultats du Tableau 2, nous savons que la méthode proposée présente le meilleur compromis vitesse-précision dans l'ensemble. Si nous comparons YOLOv7-tiny-SiLU avec YOLOv5-N (r6.1), notre méthode est 127 FPS plus rapide et plus précise de 10,7% en AP. De plus, YOLOv7 atteint 51,4% d'AP à une fréquence d'images de 161 FPS, tandis que PPYOLOE-L avec la même AP atteint seulement 78 FPS. En termes d'utilisation des paramètres, YOLOv7 consomme 41% de moins que PPYOLOE-L. Si nous comparons YOLOv7-X avec une vitesse d'inférence de 114 FPS à YOLOv5-L (r6.1) avec une vitesse d'inférence de 99 FPS, YOLOv7-X peut améliorer l'AP de 3,9%. Si YOLOv7-X est comparé à YOLOv5-X (r6.1) de taille similaire, la vitesse d'inférence de YOLOv7-X est de 31 FPS plus rapide. De plus, en termes de nombre de paramètres et de calculs, YOLOv7-X réduit de 22% les paramètres et de 8% les calculs par rapport à YOLOv5-X (r6.1), mais améliore l'AP de 2,2% ([Source](https://arxiv.org/pdf/2207.02696.pdf)).

## Aperçu

La détection d'objets en temps réel est un composant important de nombreux systèmes de vision par ordinateur, notamment le suivi multi-objets, la conduite autonome, la robotique et l'analyse d'images médicales. Ces dernières années, le développement de la détection d'objets en temps réel s'est concentré sur la conception d'architectures efficaces et l'amélioration de la vitesse d'inférence des CPU, des GPU et des unités de traitement neuronal (NPU) dans différentes configurations. YOLOv7 prend en charge les GPU mobiles et les appareils GPU, de l'edge au cloud.

Contrairement aux détecteurs d'objets en temps réel traditionnels qui se concentrent sur l'optimisation de l'architecture, YOLOv7 introduit une approche axée sur l'optimisation du processus d'entraînement. Cela comprend des modules et des méthodes d'optimisation conçus pour améliorer la précision de la détection d'objets sans augmenter le coût de l'inférence, un concept connu sous le nom de "bag-of-freebies entraînable".

## Fonctionnalités Principales

YOLOv7 propose plusieurs fonctionnalités principales :

1. **Ré-paramétrisation du Modèle** : YOLOv7 propose un modèle re-paramétré planifié, qui est une stratégie applicable aux couches de différents réseaux avec le concept de propagation des gradients.

2. **Affectation Dynamique des Étiquettes** : La formation du modèle avec des couches de sortie multiples présente un nouveau problème : "Comment attribuer des cibles dynamiques aux sorties des différentes branches ?" Pour résoudre ce problème, YOLOv7 introduit une nouvelle méthode d'affectation des étiquettes appelée affectation des étiquettes guidée en cascade de grossières à fines.

3. **Mise à l'Échelle Étendue et Composée** : YOLOv7 propose des méthodes de "mise à l'échelle étendue" et de "mise à l'échelle composée" pour le détecteur d'objets en temps réel, qui permettent d'utiliser efficacement les paramètres et les calculs.

4. **Efficacité** : La méthode proposée par YOLOv7 permet de réduire efficacement environ 40% des paramètres et 50% des calculs du détecteur d'objets en temps réel de pointe, tout en offrant une vitesse d'inférence plus rapide et une plus grande précision de détection.

## Exemples d'Utilisation

Au moment de la rédaction de cet article, Ultralytics ne prend pas en charge les modèles YOLOv7. Par conséquent, tout utilisateur intéressé par l'utilisation de YOLOv7 devra se référer directement au dépôt GitHub de YOLOv7 pour obtenir les instructions d'installation et d'utilisation.

Voici un bref aperçu des étapes typiques que vous pourriez suivre pour utiliser YOLOv7 :

1. Rendez-vous sur le dépôt GitHub de YOLOv7 : [https://github.com/WongKinYiu/yolov7](https://github.com/WongKinYiu/yolov7).

2. Suivez les instructions fournies dans le fichier README pour l'installation. Cela implique généralement de cloner le dépôt, d'installer les dépendances nécessaires et de configurer les variables d'environnement nécessaires.

3. Une fois l'installation terminée, vous pouvez entraîner et utiliser le modèle selon les instructions d'utilisation fournies dans le dépôt. Cela implique généralement la préparation de votre ensemble de données, la configuration des paramètres du modèle, l'entraînement du modèle, puis l'utilisation du modèle entraîné pour effectuer la détection d'objets.

Veuillez noter que les étapes spécifiques peuvent varier en fonction de votre cas d'utilisation spécifique et de l'état actuel du dépôt YOLOv7. Par conséquent, il est fortement recommandé de vous reporter directement aux instructions fournies dans le dépôt GitHub de YOLOv7.

Nous nous excusons pour tout inconvénient que cela pourrait causer et nous nous efforcerons de mettre à jour ce document avec des exemples d'utilisation pour Ultralytics une fois la prise en charge de YOLOv7 mise en place.

## Citations et Remerciements

Nous tenons à remercier les auteurs de YOLOv7 pour leurs contributions significatives dans le domaine de la détection d'objets en temps réel :

!!! Quote ""

    === "BibTeX"

        ```bibtex
        @article{wang2022yolov7,
          title={{YOLOv7}: Trainable bag-of-freebies sets new state-of-the-art for real-time object detectors},
          author={Wang, Chien-Yao and Bochkovskiy, Alexey and Liao, Hong-Yuan Mark},
          journal={arXiv preprint arXiv:2207.02696},
          year={2022}
        }
        ```

Le document original de YOLOv7 peut être consulté sur [arXiv](https://arxiv.org/pdf/2207.02696.pdf). Les auteurs ont rendu leur travail accessible au public, et le code source peut être consulté sur [GitHub](https://github.com/WongKinYiu/yolov7). Nous apprécions leurs efforts pour faire avancer le domaine et rendre leur travail accessible à la communauté élargie.
