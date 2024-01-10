---
comments: true
description: Apprenez à utiliser les modèles de segmentation d'instance avec Ultralytics YOLO. Instructions pour la formation, la validation, la prédiction d'image et l'exportation de modèle.
keywords: yolov8, segmentation d'instance, Ultralytics, jeu de données COCO, segmentation d'image, détection d'objet, formation de modèle, validation de modèle, prédiction d'image, exportation de modèle
---

# Segmentation d'Instance

<img width="1024" src="https://user-images.githubusercontent.com/26833433/243418644-7df320b8-098d-47f1-85c5-26604d761286.png" alt="Exemples de segmentation d'instance">

La segmentation d'instance va plus loin que la détection d'objet et implique d'identifier des objets individuels dans une image et de les segmenter du reste de l'image.

Le résultat d'un modèle de segmentation d'instance est un ensemble de masques ou de contours qui délimitent chaque objet dans l'image, accompagnés d'étiquettes de classe et de scores de confiance pour chaque objet. La segmentation d'instance est utile lorsque vous avez besoin de savoir non seulement où se trouvent les objets dans une image, mais aussi quelle est leur forme exacte.

<p align="center">
  <br>
  <iframe width="720" height="405" src="https://www.youtube.com/embed/o4Zd-IeMlSY?si=37nusCzDTd74Obsp"
    title="Lecteur vidéo YouTube" frameborder="0"
    allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share"
    allowfullscreen>
  </iframe>
  <br>
  <strong>Regarder :</strong> Exécutez la Segmentation avec le Modèle Ultralytics YOLOv8 Pré-Entraîné en Python.
</p>

!!! astuce "Astuce"

    Les modèles YOLOv8 Segment utilisent le suffixe `-seg`, par exemple `yolov8n-seg.pt` et sont pré-entraînés sur [COCO](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/datasets/coco.yaml).

## [Modèles](https://github.com/ultralytics/ultralytics/tree/main/ultralytics/cfg/models/v8)

Les modèles Segment pré-entraînés YOLOv8 sont indiqués ici. Les modèles Detect, Segment et Pose sont pré-entraînés sur le jeu de données [COCO](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/datasets/coco.yaml), tandis que les modèles Classify sont pré-entraînés sur le jeu de données [ImageNet](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/datasets/ImageNet.yaml).

Les [modèles](https://github.com/ultralytics/ultralytics/tree/main/ultralytics/cfg/models) se téléchargent automatiquement depuis la dernière [version](https://github.com/ultralytics/assets/releases) Ultralytics lors de la première utilisation.

| Modèle                                                                                       | Taille<br><sup>(pixels) | mAP<sup>boîte<br>50-95 | mAP<sup>masque<br>50-95 | Vitesse<br><sup>CPU ONNX<br>(ms) | Vitesse<br><sup>A100 TensorRT<br>(ms) | Paramètres<br><sup>(M) | FLOPs<br><sup>(B) |
|----------------------------------------------------------------------------------------------|-------------------------|------------------------|-------------------------|----------------------------------|---------------------------------------|------------------------|-------------------|
| [YOLOv8n-seg](https://github.com/ultralytics/assets/releases/download/v8.1.0/yolov8n-seg.pt) | 640                     | 36.7                   | 30.5                    | 96.1                             | 1.21                                  | 3.4                    | 12.6              |
| [YOLOv8s-seg](https://github.com/ultralytics/assets/releases/download/v8.1.0/yolov8s-seg.pt) | 640                     | 44.6                   | 36.8                    | 155.7                            | 1.47                                  | 11.8                   | 42.6              |
| [YOLOv8m-seg](https://github.com/ultralytics/assets/releases/download/v8.1.0/yolov8m-seg.pt) | 640                     | 49.9                   | 40.8                    | 317.0                            | 2.18                                  | 27.3                   | 110.2             |
| [YOLOv8l-seg](https://github.com/ultralytics/assets/releases/download/v8.1.0/yolov8l-seg.pt) | 640                     | 52.3                   | 42.6                    | 572.4                            | 2.79                                  | 46.0                   | 220.5             |
| [YOLOv8x-seg](https://github.com/ultralytics/assets/releases/download/v8.1.0/yolov8x-seg.pt) | 640                     | 53.4                   | 43.4                    | 712.1                            | 4.02                                  | 71.8                   | 344.1             |

- Les valeurs **mAP<sup>val</sup>** sont pour un seul modèle à une seule échelle sur le jeu de données [COCO val2017](http://cocodataset.org).
  <br>Pour reproduire, utilisez `yolo val segment data=coco.yaml device=0`
- **Vitesse** moyennée sur les images COCO val en utilisant une instance [Amazon EC2 P4d](https://aws.amazon.com/ec2/instance-types/p4/).
  <br>Pour reproduire, utilisez `yolo val segment data=coco128-seg.yaml batch=1 device=0|cpu`

## Formation

Entraînez YOLOv8n-seg sur le jeu de données COCO128-seg pendant 100 époques à la taille d'image 640. Pour une liste complète des arguments disponibles, consultez la page [Configuration](/../usage/cfg.md).

!!! Example "Exemple"

    === "Python"

        ```python
        from ultralytics import YOLO

        # Charger un modèle
        model = YOLO('yolov8n-seg.yaml')  # construire un nouveau modèle à partir du YAML
        model = YOLO('yolov8n-seg.pt')  # charger un modèle pré-entraîné (recommandé pour la formation)
        model = YOLO('yolov8n-seg.yaml').load('yolov8n.pt')  # construire à partir du YAML et transférer les poids

        # Entraîner le modèle
        résultats = model.train(data='coco128-seg.yaml', epochs=100, imgsz=640)
        ```
    === "CLI"

        ```bash
        # Construire un nouveau modèle à partir du YAML et commencer la formation à partir de zéro
        yolo segment train data=coco128-seg.yaml model=yolov8n-seg.yaml epochs=100 imgsz=640

        # Commencer la formation à partir d'un modèle *.pt pré-entraîné
        yolo segment train data=coco128-seg.yaml model=yolov8n-seg.pt epochs=100 imgsz=640

        # Construire un nouveau modèle à partir du YAML, transférer les poids pré-entraînés et commencer la formation
        yolo segment train data=coco128-seg.yaml model=yolov8n-seg.yaml pretrained=yolov8n-seg.pt epochs=100 imgsz=640
        ```

### Format des données

Le format des données de segmentation YOLO peut être trouvé en détail dans le [Guide du Jeu de Données](../../../datasets/segment/index.md). Pour convertir votre jeu de données existant à partir d'autres formats (comme COCO, etc.) au format YOLO, veuillez utiliser l'outil [JSON2YOLO](https://github.com/ultralytics/JSON2YOLO) par Ultralytics.

## Validation

Validez la précision du modèle YOLOv8n-seg entraîné sur le jeu de données COCO128-seg. Aucun argument n'est nécessaire car le `modèle`
conserve ses données de formation et ses arguments comme attributs du modèle.

!!! Example "Exemple"

    === "Python"

        ```python
        from ultralytics import YOLO

        # Charger un modèle
        model = YOLO('yolov8n-seg.pt')  # charger un modèle officiel
        model = YOLO('chemin/vers/le/meilleur.pt')  # charger un modèle personnalisé

        # Valider le modèle
        métriques = model.val()  # aucun argument nécessaire, jeu de données et paramètres mémorisés
        métriques.box.map    # map50-95(B)
        métriques.box.map50  # map50(B)
        métriques.box.map75  # map75(B)
        métriques.box.maps   # une liste contient map50-95(B) de chaque catégorie
        métriques.seg.map    # map50-95(M)
        métriques.seg.map50  # map50(M)
        métriques.seg.map75  # map75(M)
        métriques.seg.maps   # une liste contient map50-95(M) de chaque catégorie
        ```
    === "CLI"

        ```bash
        yolo segment val model=yolov8n-seg.pt  # valider le modèle officiel
        yolo segment val model=chemin/vers/le/meilleur.pt  # valider le modèle personnalisé
        ```

## Prédiction

Utilisez un modèle YOLOv8n-seg entraîné pour effectuer des prédictions sur des images.

!!! Example "Exemple"

    === "Python"

        ```python
        from ultralytics import YOLO

        # Charger un modèle
        model = YOLO('yolov8n-seg.pt')  # charger un modèle officiel
        model = YOLO('chemin/vers/le/meilleur.pt')  # charger un modèle personnalisé

        # Prédire avec le modèle
        résultats = model('https://ultralytics.com/images/bus.jpg')  # prédire sur une image
        ```
    === "CLI"

        ```bash
        yolo segment predict model=yolov8n-seg.pt source='https://ultralytics.com/images/bus.jpg'  # prédire avec le modèle officiel
        yolo segment predict model=chemin/vers/le/meilleur.pt source='https://ultralytics.com/images/bus.jpg'  # prédire avec le modèle personnalisé
        ```

Voir les détails complets du mode `predict` sur la page [Predict](https://docs.ultralytics.com/modes/predict/).

## Exportation

Exportez un modèle YOLOv8n-seg vers un format différent comme ONNX, CoreML, etc.

!!! Example "Exemple"

    === "Python"

        ```python
        from ultralytics import YOLO

        # Charger un modèle
        model = YOLO('yolov8n-seg.pt')  # charger un modèle officiel
        model = YOLO('chemin/vers/le/meilleur.pt')  # charger un modèle entraîné personnalisé

        # Exporter le modèle
        model.export(format='onnx')
        ```
    === "CLI"

        ```bash
        yolo export model=yolov8n-seg.pt format=onnx  # exporter le modèle officiel
        yolo export model=chemin/vers/le/meilleur.pt format=onnx  # exporter le modèle entraîné personnalisé
        ```

Les formats d'exportation YOLOv8-seg disponibles sont dans le tableau ci-dessous. Vous pouvez prédire ou valider directement sur les modèles exportés, par exemple `yolo predict model=yolov8n-seg.onnx`. Des exemples d'utilisation sont présentés pour votre modèle après l'exportation.

| Format                                                             | Argument `format` | Modèle                        | Métadonnées | Arguments                                           |
|--------------------------------------------------------------------|-------------------|-------------------------------|-------------|-----------------------------------------------------|
| [PyTorch](https://pytorch.org/)                                    | -                 | `yolov8n-seg.pt`              | ✅           | -                                                   |
| [TorchScript](https://pytorch.org/docs/stable/jit.html)            | `torchscript`     | `yolov8n-seg.torchscript`     | ✅           | `imgsz`, `optimize`                                 |
| [ONNX](https://onnx.ai/)                                           | `onnx`            | `yolov8n-seg.onnx`            | ✅           | `imgsz`, `half`, `dynamic`, `simplify`, `opset`     |
| [OpenVINO](https://docs.openvino.ai/latest/index.html)             | `openvino`        | `yolov8n-seg_openvino_model/` | ✅           | `imgsz`, `half`                                     |
| [TensorRT](https://developer.nvidia.com/tensorrt)                  | `engine`          | `yolov8n-seg.engine`          | ✅           | `imgsz`, `half`, `dynamic`, `simplify`, `workspace` |
| [CoreML](https://github.com/apple/coremltools)                     | `coreml`          | `yolov8n-seg.mlpackage`       | ✅           | `imgsz`, `half`, `int8`, `nms`                      |
| [TF SavedModel](https://www.tensorflow.org/guide/saved_model)      | `saved_model`     | `yolov8n-seg_saved_model/`    | ✅           | `imgsz`, `keras`                                    |
| [TF GraphDef](https://www.tensorflow.org/api_docs/python/tf/Graph) | `pb`              | `yolov8n-seg.pb`              | ❌           | `imgsz`                                             |
| [TF Lite](https://www.tensorflow.org/lite)                         | `tflite`          | `yolov8n-seg.tflite`          | ✅           | `imgsz`, `half`, `int8`                             |
| [TF Edge TPU](https://coral.ai/docs/edgetpu/models-intro/)         | `edgetpu`         | `yolov8n-seg_edgetpu.tflite`  | ✅           | `imgsz`                                             |
| [TF.js](https://www.tensorflow.org/js)                             | `tfjs`            | `yolov8n-seg_web_model/`      | ✅           | `imgsz`                                             |
| [PaddlePaddle](https://github.com/PaddlePaddle)                    | `paddle`          | `yolov8n-seg_paddle_model/`   | ✅           | `imgsz`                                             |
| [ncnn](https://github.com/Tencent/ncnn)                            | `ncnn`            | `yolov8n-seg_ncnn_model/`     | ✅           | `imgsz`, `half`                                     |

Voir les détails complets d'`export` sur la page [Export](https://docs.ultralytics.com/modes/export/).
