---
comments: true
description: Guide étape par étape sur l'exportation de vos modèles YOLOv8 vers divers formats tels que ONNX, TensorRT, CoreML et plus encore pour le déploiement. Explorez maintenant !.
keywords: YOLO, YOLOv8, Ultralytics, Exportation de modèle, ONNX, TensorRT, CoreML, TensorFlow SavedModel, OpenVINO, PyTorch, exporter un modèle
---

# Exportation de modèle avec Ultralytics YOLO

<img width="1024" src="https://github.com/ultralytics/assets/raw/main/yolov8/banner-integrations.png" alt="Écosystème et intégrations Ultralytics YOLO">

## Introduction

L'objectif ultime de l'entraînement d'un modèle est de le déployer pour des applications dans le monde réel. Le mode d'exportation de Ultralytics YOLOv8 offre une large gamme d'options pour exporter votre modèle entraîné dans différents formats, le rendant déployable sur diverses plateformes et appareils. Ce guide complet vise à vous guider à travers les nuances de l'exportation de modèles, en montrant comment atteindre une compatibilité et des performances maximales.

<p align="center">
  <br>
  <iframe width="720" height="405" src="https://www.youtube.com/embed/WbomGeoOT_k?si=aGmuyooWftA0ue9X"
    title="Lecteur de vidéo YouTube" frameborder="0"
    allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share"
    allowfullscreen>
  </iframe>
  <br>
  <strong>Regardez :</strong> Comment exporter un modèle Ultralytics YOLOv8 entraîné personnalisé et effectuer une inférence en direct sur webcam.
</p>

## Pourquoi choisir le mode d'exportation YOLOv8 ?

- **Polyvalence :** Exportation vers plusieurs formats, y compris ONNX, TensorRT, CoreML et plus encore.
- **Performance :** Gagnez jusqu'à 5 fois la vitesse d'une GPU avec TensorRT et 3 fois la vitesse d'une CPU avec ONNX ou OpenVINO.
- **Compatibilité :** Rendez votre modèle universellement déployable sur de nombreux environnements matériels et logiciels.
- **Facilité d'utilisation :** Interface en ligne de commande (CLI) et API Python simples pour une exportation rapide et directe du modèle.

### Caractéristiques clés du mode d'exportation

Voici quelques-unes des fonctionnalités remarquables :

- **Exportation en un clic :** Commandes simples pour exporter vers différents formats.
- **Exportation groupée :** Exportez des modèles capables d'inférence par lot.
- **Inférence optimisée :** Les modèles exportés sont optimisés pour des temps d'inférence plus rapides.
- **Vidéos tutorielles :** Guides détaillés et tutoriels pour une expérience d'exportation fluide.

!!! astuce "Conseil"

    * Exportez vers ONNX ou OpenVINO pour une accélération de la CPU jusqu'à 3 fois.
    * Exportez vers TensorRT pour une accélération de la GPU jusqu'à 5 fois.

## Exemples d'utilisation

Exportez un modèle YOLOv8n vers un format différent tel que ONNX ou TensorRT. Voir la section Arguments ci-dessous pour une liste complète des arguments d'exportation.

!!! Example "Exemple"

    === "Python"

        ```python
        from ultralytics import YOLO

        # Charger un modèle
        model = YOLO('yolov8n.pt')  # chargez un modèle officiel
        model = YOLO('path/to/best.pt')  # chargez un modèle entraîné personnalisé

        # Exporter le modèle
        model.export(format='onnx')
        ```
    === "CLI"

        ```bash
        yolo export model=yolov8n.pt format=onnx  # exporter modèle officiel
        yolo export model=path/to/best.pt format=onnx  # exporter modèle entraîné personnalisé
        ```

## Arguments

Les paramètres d'exportation pour les modèles YOLO se réfèrent aux diverses configurations et options utilisées pour sauvegarder ou exporter le modèle pour utilisation dans d'autres environnements ou plateformes. Ces paramètres peuvent affecter la performance, la taille et la compatibilité du modèle avec différents systèmes. Certains paramètres d'exportation YOLO courants incluent le format du fichier modèle exporté (par exemple, ONNX, TensorFlow SavedModel), le dispositif sur lequel le modèle sera exécuté (par exemple, CPU, GPU), et la présence de fonctionnalités supplémentaires telles que des masques ou des étiquettes multiples par boîte. D'autres facteurs qui peuvent affecter le processus d'exportation incluent la tâche spécifique pour laquelle le modèle est utilisé et les exigences ou contraintes de l'environnement ou de la plateforme cible. Il est important de considérer et de configurer ces paramètres avec soin pour s'assurer que le modèle exporté est optimisé pour le cas d'utilisation visé et peut être utilisé efficacement dans l'environnement cible.

| Clé         | Valeur          | Description                                                                      |
|-------------|-----------------|----------------------------------------------------------------------------------|
| `format`    | `'torchscript'` | format vers lequel exporter                                                      |
| `imgsz`     | `640`           | taille d'image sous forme scalaire ou liste (h, w), par ex. (640, 480)           |
| `keras`     | `False`         | utilisez Keras pour l'exportation TensorFlow SavedModel                          |
| `optimize`  | `False`         | TorchScript : optimisation pour mobile                                           |
| `half`      | `False`         | quantification FP16                                                              |
| `int8`      | `False`         | quantification INT8                                                              |
| `dynamic`   | `False`         | ONNX/TensorRT : axes dynamiques                                                  |
| `simplify`  | `False`         | ONNX/TensorRT : simplifier le modèle                                             |
| `opset`     | `None`          | ONNX : version de l'ensemble d'opérations (facultatif, par défaut à la dernière) |
| `workspace` | `4`             | TensorRT : taille de l'espace de travail (GB)                                    |
| `nms`       | `False`         | CoreML : ajout de la NMS                                                         |

## Formats d'exportation

Les formats d'exportation disponibles pour YOLOv8 sont dans le tableau ci-dessous. Vous pouvez exporter vers n'importe quel format en utilisant l'argument `format`, par ex. `format='onnx'` ou `format='engine'`.

| Format                                                             | Argument `format` | Modèle                    | Métadonnées | Arguments                                           |
|--------------------------------------------------------------------|-------------------|---------------------------|-------------|-----------------------------------------------------|
| [PyTorch](https://pytorch.org/)                                    | -                 | `yolov8n.pt`              | ✅           | -                                                   |
| [TorchScript](https://pytorch.org/docs/stable/jit.html)            | `torchscript`     | `yolov8n.torchscript`     | ✅           | `imgsz`, `optimize`                                 |
| [ONNX](https://onnx.ai/)                                           | `onnx`            | `yolov8n.onnx`            | ✅           | `imgsz`, `half`, `dynamic`, `simplify`, `opset`     |
| [OpenVINO](https://docs.openvino.ai/latest/index.html)             | `openvino`        | `yolov8n_openvino_model/` | ✅           | `imgsz`, `half`                                     |
| [TensorRT](https://developer.nvidia.com/tensorrt)                  | `engine`          | `yolov8n.engine`          | ✅           | `imgsz`, `half`, `dynamic`, `simplify`, `workspace` |
| [CoreML](https://github.com/apple/coremltools)                     | `coreml`          | `yolov8n.mlpackage`       | ✅           | `imgsz`, `half`, `int8`, `nms`                      |
| [TF SavedModel](https://www.tensorflow.org/guide/saved_model)      | `saved_model`     | `yolov8n_saved_model/`    | ✅           | `imgsz`, `keras`                                    |
| [TF GraphDef](https://www.tensorflow.org/api_docs/python/tf/Graph) | `pb`              | `yolov8n.pb`              | ❌           | `imgsz`                                             |
| [TF Lite](https://www.tensorflow.org/lite)                         | `tflite`          | `yolov8n.tflite`          | ✅           | `imgsz`, `half`, `int8`                             |
| [TF Edge TPU](https://coral.ai/docs/edgetpu/models-intro/)         | `edgetpu`         | `yolov8n_edgetpu.tflite`  | ✅           | `imgsz`                                             |
| [TF.js](https://www.tensorflow.org/js)                             | `tfjs`            | `yolov8n_web_model/`      | ✅           | `imgsz`                                             |
| [PaddlePaddle](https://github.com/PaddlePaddle)                    | `paddle`          | `yolov8n_paddle_model/`   | ✅           | `imgsz`                                             |
| [ncnn](https://github.com/Tencent/ncnn)                            | `ncnn`            | `yolov8n_ncnn_model/`     | ✅           | `imgsz`, `half`                                     |
