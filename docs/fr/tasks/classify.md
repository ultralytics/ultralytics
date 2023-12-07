---
comments: true
description: Apprenez-en davantage sur les modèles de classification d'images YOLOv8 Classify. Obtenez des informations détaillées sur la liste des modèles pré-entraînés et comment entraîner, valider, prédire et exporter des modèles.
keywords: Ultralytics, YOLOv8, Classification d'images, Modèles pré-entraînés, YOLOv8n-cls, Entraînement, Validation, Prédiction, Exportation de modèles
---

# Classification d'images

<img width="1024" src="https://user-images.githubusercontent.com/26833433/243418606-adf35c62-2e11-405d-84c6-b84e7d013804.png" alt="Exemples de classification d'images">

La classification d'images est la tâche la plus simple des trois et consiste à classer une image entière dans l'une d'un ensemble de classes prédéfinies.

Le résultat d'un classificateur d'images est une étiquette de classe unique et un score de confiance. La classification d'images est utile lorsque vous avez besoin de savoir seulement à quelle classe appartient une image et que vous n'avez pas besoin de connaître l'emplacement des objets de cette classe ou leur forme exacte.

!!! Tip "Astuce"

    Les modèles YOLOv8 Classify utilisent le suffixe `-cls`, par exemple `yolov8n-cls.pt` et sont pré-entraînés sur [ImageNet](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/datasets/ImageNet.yaml).

## [Modèles](https://github.com/ultralytics/ultralytics/tree/main/ultralytics/cfg/models/v8)

Les modèles Classify pré-entraînés YOLOv8 sont présentés ici. Les modèles Detect, Segment et Pose sont pré-entraînés sur le dataset [COCO](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/datasets/coco.yaml), tandis que les modèles Classify sont pré-entraînés sur le dataset [ImageNet](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/datasets/ImageNet.yaml).

Les [modèles](https://github.com/ultralytics/ultralytics/tree/main/ultralytics/cfg/models) se téléchargent automatiquement depuis la dernière version Ultralytics [release](https://github.com/ultralytics/assets/releases) lors de la première utilisation.

| Modèle                                                                                       | taille<br><sup>(pixels) | acc<br><sup>top1 | acc<br><sup>top5 | Vitesse<br><sup>CPU ONNX<br>(ms) | Vitesse<br><sup>A100 TensorRT<br>(ms) | params<br><sup>(M) | FLOPs<br><sup>(B) à 640 |
|----------------------------------------------------------------------------------------------|-------------------------|------------------|------------------|----------------------------------|---------------------------------------|--------------------|-------------------------|
| [YOLOv8n-cls](https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n-cls.pt) | 224                     | 66.6             | 87.0             | 12.9                             | 0.31                                  | 2.7                | 4.3                     |
| [YOLOv8s-cls](https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8s-cls.pt) | 224                     | 72.3             | 91.1             | 23.4                             | 0.35                                  | 6.4                | 13.5                    |
| [YOLOv8m-cls](https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8m-cls.pt) | 224                     | 76.4             | 93.2             | 85.4                             | 0.62                                  | 17.0               | 42.7                    |
| [YOLOv8l-cls](https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8l-cls.pt) | 224                     | 78.0             | 94.1             | 163.0                            | 0.87                                  | 37.5               | 99.7                    |
| [YOLOv8x-cls](https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8x-cls.pt) | 224                     | 78.4             | 94.3             | 232.0                            | 1.01                                  | 57.4               | 154.8                   |

- Les valeurs **acc** sont les précisions des modèles sur le jeu de données de validation d'[ImageNet](https://www.image-net.org/).
  <br>Pour reproduire : `yolo val classify data=path/to/ImageNet device=0`
- Les **vitesses** sont calculées sur les images de validation d'ImageNet à l'aide d'une instance [Amazon EC2 P4d](https://aws.amazon.com/ec2/instance-types/p4/).
  <br>Pour reproduire : `yolo val classify data=path/to/ImageNet batch=1 device=0|cpu`

## Entraînement

Entraînez le modèle YOLOv8n-cls sur le dataset MNIST160 pendant 100 époques avec une taille d'image de 64. Pour une liste complète des arguments disponibles, consultez la page [Configuration](/../usage/cfg.md).

!!! Example "Exemple"

    === "Python"

        ```python
        from ultralytics import YOLO

        # Charger un modèle
        model = YOLO('yolov8n-cls.yaml')  # construire un nouveau modèle à partir du YAML
        model = YOLO('yolov8n-cls.pt')  # charger un modèle pré-entraîné (recommandé pour l'entraînement)
        model = YOLO('yolov8n-cls.yaml').load('yolov8n-cls.pt')  # construire à partir du YAML et transférer les poids

        # Entraîner le modèle
        results = model.train(data='mnist160', epochs=100, imgsz=64)
        ```

    === "CLI"

        ```bash
        # Construire un nouveau modèle à partir du YAML et commencer l'entraînement à partir de zéro
        yolo classify train data=mnist160 model=yolov8n-cls.yaml epochs=100 imgsz=64

        # Commencer l'entraînement à partir d'un modèle *.pt pré-entraîné
        yolo classify train data=mnist160 model=yolov8n-cls.pt epochs=100 imgsz=64

        # Construire un nouveau modèle à partir du YAML, transférer les poids pré-entraînés et commencer l'entraînement
        yolo classify train data=mnist160 model=yolov8n-cls.yaml pretrained=yolov8n-cls.pt epochs=100 imgsz=64
        ```

### Format du dataset

Le format du dataset de classification YOLO peut être trouvé en détails dans le [Guide des Datasets](../../../datasets/classify/index.md).

## Validation

Validez la précision du modèle YOLOv8n-cls entraîné sur le dataset MNIST160. Aucun argument n'est nécessaire car le `modèle` conserve ses données d'entraînement et arguments en tant qu'attributs du modèle.

!!! Example "Exemple"

    === "Python"

        ```python
        from ultralytics import YOLO

        # Charger un modèle
        model = YOLO('yolov8n-cls.pt')  # charger un modèle officiel
        model = YOLO('path/to/best.pt')  # charger un modèle personnalisé

        # Valider le modèle
        metrics = model.val()  # aucun argument nécessaire, les données et les paramètres sont mémorisés
        metrics.top1   # précision top 1
        metrics.top5   # précision top 5
        ```
    === "CLI"

        ```bash
        yolo classify val model=yolov8n-cls.pt  # valider le modèle officiel
        yolo classify val model=path/to/best.pt  # valider le modèle personnalisé
        ```

## Prédiction

Utilisez un modèle YOLOv8n-cls entraîné pour exécuter des prédictions sur des images.

!!! Example "Exemple"

    === "Python"

        ```python
        from ultralytics import YOLO

        # Charger un modèle
        model = YOLO('yolov8n-cls.pt')  # charger un modèle officiel
        model = YOLO('path/to/best.pt')  # charger un modèle personnalisé

        # Prédire avec le modèle
        results = model('https://ultralytics.com/images/bus.jpg')  # prédire sur une image
        ```
    === "CLI"

        ```bash
        yolo classify predict model=yolov8n-cls.pt source='https://ultralytics.com/images/bus.jpg'  # prédiction avec le modèle officiel
        yolo classify predict model=path/to/best.pt source='https://ultralytics.com/images/bus.jpg'  # prédiction avec le modèle personnalisé
        ```

Voir les détails complets du mode `predict` sur la page [Prédire](https://docs.ultralytics.com/modes/predict/).

## Exportation

Exportez un modèle YOLOv8n-cls dans un format différent comme ONNX, CoreML, etc.

!!! Example "Exemple"

    === "Python"

        ```python
        from ultralytics import YOLO

        # Charger un modèle
        model = YOLO('yolov8n-cls.pt')  # charger un modèle officiel
        model = YOLO('path/to/best.pt')  # charger un modèle entraîné personnalisé

        # Exporter le modèle
        model.export(format='onnx')
        ```
    === "CLI"

        ```bash
        yolo export model=yolov8n-cls.pt format=onnx  # exporter le modèle officiel
        yolo export model=path/to/best.pt format=onnx  # exporter le modèle entraîné personnalisé
        ```

Les formats d'exportation disponibles pour YOLOv8-cls sont présentés dans le tableau ci-dessous. Vous pouvez prédire ou valider directement sur les modèles exportés, par exemple `yolo predict model=yolov8n-cls.onnx`. Des exemples d'utilisation sont présentés pour votre modèle une fois l'exportation terminée.

| Format                                                             | Argument `format` | Modèle                        | Métadonnées | Arguments                                           |
|--------------------------------------------------------------------|-------------------|-------------------------------|-------------|-----------------------------------------------------|
| [PyTorch](https://pytorch.org/)                                    | -                 | `yolov8n-cls.pt`              | ✅           | -                                                   |
| [TorchScript](https://pytorch.org/docs/stable/jit.html)            | `torchscript`     | `yolov8n-cls.torchscript`     | ✅           | `imgsz`, `optimize`                                 |
| [ONNX](https://onnx.ai/)                                           | `onnx`            | `yolov8n-cls.onnx`            | ✅           | `imgsz`, `half`, `dynamic`, `simplify`, `opset`     |
| [OpenVINO](https://docs.openvino.ai/latest/index.html)             | `openvino`        | `yolov8n-cls_openvino_model/` | ✅           | `imgsz`, `half`                                     |
| [TensorRT](https://developer.nvidia.com/tensorrt)                  | `engine`          | `yolov8n-cls.engine`          | ✅           | `imgsz`, `half`, `dynamic`, `simplify`, `workspace` |
| [CoreML](https://github.com/apple/coremltools)                     | `coreml`          | `yolov8n-cls.mlpackage`       | ✅           | `imgsz`, `half`, `int8`, `nms`                      |
| [TF SavedModel](https://www.tensorflow.org/guide/saved_model)      | `saved_model`     | `yolov8n-cls_saved_model/`    | ✅           | `imgsz`, `keras`                                    |
| [TF GraphDef](https://www.tensorflow.org/api_docs/python/tf/Graph) | `pb`              | `yolov8n-cls.pb`              | ❌           | `imgsz`                                             |
| [TF Lite](https://www.tensorflow.org/lite)                         | `tflite`          | `yolov8n-cls.tflite`          | ✅           | `imgsz`, `half`, `int8`                             |
| [TF Edge TPU](https://coral.ai/docs/edgetpu/models-intro/)         | `edgetpu`         | `yolov8n-cls_edgetpu.tflite`  | ✅           | `imgsz`                                             |
| [TF.js](https://www.tensorflow.org/js)                             | `tfjs`            | `yolov8n-cls_web_model/`      | ✅           | `imgsz`                                             |
| [PaddlePaddle](https://github.com/PaddlePaddle)                    | `paddle`          | `yolov8n-cls_paddle_model/`   | ✅           | `imgsz`                                             |
| [ncnn](https://github.com/Tencent/ncnn)                            | `ncnn`            | `yolov8n-cls_ncnn_model/`     | ✅           | `imgsz`, `half`                                     |

Voir les détails complets de l'`exportation` sur la page [Export](https://docs.ultralytics.com/modes/export/).
