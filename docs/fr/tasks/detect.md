---
comments: true
description: Documentation officielle pour YOLOv8 par Ultralytics. Apprenez comment entraîner, valider, prédire et exporter des modèles dans différents formats. Incluant des statistiques de performances détaillées.
keywords: YOLOv8, Ultralytics, détection d'objets, modèles pré-entraînés, entraînement, validation, prédiction, exportation de modèles, COCO, ImageNet, PyTorch, ONNX, CoreML
---

# Détection d'Objets

<img width="1024" src="https://user-images.githubusercontent.com/26833433/243418624-5785cb93-74c9-4541-9179-d5c6782d491a.png" alt="Exemples de détection d'objets">

La détection d'objets est une tâche qui implique l'identification de l'emplacement et de la classe des objets dans une image ou un flux vidéo.

La sortie d'un détecteur d'objets est un ensemble de boîtes englobantes qui entourent les objets de l'image, accompagnées de libellés de classe et de scores de confiance pour chaque boîte. La détection d'objets est un bon choix lorsque vous avez besoin d'identifier des objets d'intérêt dans une scène, mais que vous n'avez pas besoin de connaître exactement où se trouve l'objet ou sa forme exacte.

<p align="center">
  <br>
  <iframe width="720" height="405" src="https://www.youtube.com/embed/5ku7npMrW40?si=6HQO1dDXunV8gekh"
    title="Lecteur vidéo YouTube" frameborder="0"
    allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share"
    allowfullscreen>
  </iframe>
  <br>
  <strong>Regardez :</strong> Détection d'Objets avec le Modèle Pré-entraîné Ultralytics YOLOv8.
</p>

!!! Tip "Conseil"

    Les modèles Detect YOLOv8 sont les modèles YOLOv8 par défaut, c.-à-d. `yolov8n.pt` et sont pré-entraînés sur le jeu de données [COCO](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/datasets/coco.yaml).

## [Modèles](https://github.com/ultralytics/ultralytics/tree/main/ultralytics/cfg/models/v8)

Les modèles pré-entraînés Detect YOLOv8 sont présentés ici. Les modèles Detect, Segment, et Pose sont pré-entraînés sur le jeu de données [COCO](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/datasets/coco.yaml), tandis que les modèles Classify sont pré-entraînés sur le jeu de données [ImageNet](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/datasets/ImageNet.yaml).

[Les modèles](https://github.com/ultralytics/ultralytics/tree/main/ultralytics/cfg/models) se téléchargent automatiquement à partir de la dernière [version](https://github.com/ultralytics/assets/releases) d'Ultralytics lors de la première utilisation.

| Modèle                                                                               | Taille<br><sup>(pixels) | mAP<sup>val<br>50-95 | Vitesse<br><sup>CPU ONNX<br>(ms) | Vitesse<br><sup>A100 TensorRT<br>(ms) | Paramètres<br><sup>(M) | FLOPs<br><sup>(B) |
|--------------------------------------------------------------------------------------|-------------------------|----------------------|----------------------------------|---------------------------------------|------------------------|-------------------|
| [YOLOv8n](https://github.com/ultralytics/assets/releases/download/v8.1.0/yolov8n.pt) | 640                     | 37.3                 | 80.4                             | 0.99                                  | 3.2                    | 8.7               |
| [YOLOv8s](https://github.com/ultralytics/assets/releases/download/v8.1.0/yolov8s.pt) | 640                     | 44.9                 | 128.4                            | 1.20                                  | 11.2                   | 28.6              |
| [YOLOv8m](https://github.com/ultralytics/assets/releases/download/v8.1.0/yolov8m.pt) | 640                     | 50.2                 | 234.7                            | 1.83                                  | 25.9                   | 78.9              |
| [YOLOv8l](https://github.com/ultralytics/assets/releases/download/v8.1.0/yolov8l.pt) | 640                     | 52.9                 | 375.2                            | 2.39                                  | 43.7                   | 165.2             |
| [YOLOv8x](https://github.com/ultralytics/assets/releases/download/v8.1.0/yolov8x.pt) | 640                     | 53.9                 | 479.1                            | 3.53                                  | 68.2                   | 257.8             |

- Les valeurs de **mAP<sup>val</sup>** sont pour un seul modèle à une seule échelle sur le jeu de données [COCO val2017](https://cocodataset.org).
  <br>Reproductible avec `yolo val detect data=coco.yaml device=0`
- La **Vitesse** est moyennée sur les images COCO val en utilisant une instance [Amazon EC2 P4d](https://aws.amazon.com/fr/ec2/instance-types/p4/).
  <br>Reproductible avec `yolo val detect data=coco128.yaml batch=1 device=0|cpu`

## Entraînement

Entraînez le modèle YOLOv8n sur le jeu de données COCO128 pendant 100 époques à la taille d'image de 640. Pour une liste complète des arguments disponibles, consultez la page [Configuration](/../usage/cfg.md).

!!! Example "Exemple"

    === "Python"

        ```python
        from ultralytics import YOLO

        # Charger un modèle
        model = YOLO('yolov8n.yaml')  # construire un nouveau modèle à partir de YAML
        model = YOLO('yolov8n.pt')  # charger un modèle pré-entraîné (recommandé pour l'entraînement)
        model = YOLO('yolov8n.yaml').load('yolov8n.pt')  # construire à partir de YAML et transférer les poids

        # Entraîner le modèle
        results = model.train(data='coco128.yaml', epochs=100, imgsz=640)
        ```
    === "CLI"

        ```bash
        # Construire un nouveau modèle à partir de YAML et commencer l'entraînement à partir de zéro
        yolo detect train data=coco128.yaml model=yolov8n.yaml epochs=100 imgsz=640

        # Commencer l'entraînement à partir d'un modèle *.pt pré-entraîné
        yolo detect train data=coco128.yaml model=yolov8n.pt epochs=100 imgsz=640

        # Construire un nouveau modèle à partir de YAML, transférer les poids pré-entraînés et commencer l'entraînement
        yolo detect train data=coco128.yaml model=yolov8n.yaml pretrained=yolov8n.pt epochs=100 imgsz=640
        ```

### Format des données

Le format des jeux de données de détection YOLO est détaillé dans le [Guide des Jeux de Données](../../../datasets/detect/index.md). Pour convertir votre jeu de données existant depuis d'autres formats (comme COCO, etc.) vers le format YOLO, veuillez utiliser l'outil [JSON2YOLO](https://github.com/ultralytics/JSON2YOLO) par Ultralytics.

## Validation

Validez la précision du modèle YOLOv8n entraîné sur le jeu de données COCO128. Aucun argument n'est nécessaire puisque le `modèle` conserve ses `données` d'entraînement et arguments en tant qu'attributs du modèle.

!!! Example "Exemple"

    === "Python"

        ```python
        from ultralytics import YOLO

        # Charger un modèle
        model = YOLO('yolov8n.pt')  # charger un modèle officiel
        model = YOLO('chemin/vers/best.pt')  # charger un modèle personnalisé

        # Valider le modèle
        metrics = model.val()  # pas d'arguments nécessaires, jeu de données et paramètres enregistrés
        metrics.box.map    # map50-95
        metrics.box.map50  # map50
        metrics.box.map75  # map75
        metrics.box.maps   # une liste contenant map50-95 de chaque catégorie
        ```
    === "CLI"

        ```bash
        yolo detect val model=yolov8n.pt  # valider le modèle officiel
        yolo detect val model=chemin/vers/best.pt  # valider le modèle personnalisé
        ```

## Prédiction

Utilisez un modèle YOLOv8n entraîné pour exécuter des prédictions sur des images.

!!! Example "Exemple"

    === "Python"

        ```python
        from ultralytics import YOLO

        # Charger un modèle
        model = YOLO('yolov8n.pt')  # charger un modèle officiel
        model = YOLO('chemin/vers/best.pt')  # charger un modèle personnalisé

        # Prédire avec le modèle
        results = model('https://ultralytics.com/images/bus.jpg')  # prédire sur une image
        ```
    === "CLI"

        ```bash
        yolo detect predict model=yolov8n.pt source='https://ultralytics.com/images/bus.jpg'  # prédire avec le modèle officiel
        yolo detect predict model=chemin/vers/best.pt source='https://ultralytics.com/images/bus.jpg'  # prédire avec le modèle personnalisé
        ```

Consultez les détails complets du mode `predict` sur la page [Prédire](https://docs.ultralytics.com/modes/predict/).

## Exportation

Exportez un modèle YOLOv8n dans un format différent tel que ONNX, CoreML, etc.

!!! Example "Exemple"

    === "Python"

        ```python
        from ultralytics import YOLO

        # Charger un modèle
        model = YOLO('yolov8n.pt')  # charger un modèle officiel
        model = YOLO('chemin/vers/best.pt')  # charger un modèle entraîné personnalisé

        # Exporter le modèle
        model.export(format='onnx')
        ```
    === "CLI"

        ```bash
        yolo export model=yolov8n.pt format=onnx  # exporter le modèle officiel
        yolo export model=chemin/vers/best.pt format=onnx  # exporter le modèle entraîné personnalisé
        ```

Les formats d'exportation YOLOv8 disponibles sont présentés dans le tableau ci-dessous. Vous pouvez directement prédire ou valider sur des modèles exportés, c'est-à-dire `yolo predict model=yolov8n.onnx`. Des exemples d'utilisation sont présentés pour votre modèle après l'exportation complète.

| Format                                                               | Argument `format` | Modèle                    | Métadonnées | Arguments                                           |
|----------------------------------------------------------------------|-------------------|---------------------------|-------------|-----------------------------------------------------|
| [PyTorch](https://pytorch.org/)                                      | -                 | `yolov8n.pt`              | ✅           | -                                                   |
| [TorchScript](https://pytorch.org/docs/stable/jit.html)              | `torchscript`     | `yolov8n.torchscript`     | ✅           | `imgsz`, `optimize`                                 |
| [ONNX](https://onnx.ai/)                                             | `onnx`            | `yolov8n.onnx`            | ✅           | `imgsz`, `half`, `dynamic`, `simplify`, `opset`     |
| [OpenVINO](https://docs.openvino.ai/latest/index.html)               | `openvino`        | `yolov8n_openvino_model/` | ✅           | `imgsz`, `half`                                     |
| [TensorRT](https://developer.nvidia.com/tensorrt)                    | `engine`          | `yolov8n.engine`          | ✅           | `imgsz`, `half`, `dynamic`, `simplify`, `workspace` |
| [CoreML](https://github.com/apple/coremltools)                       | `coreml`          | `yolov8n.mlpackage`       | ✅           | `imgsz`, `half`, `int8`, `nms`                      |
| [Modèle TF Enregistré](https://www.tensorflow.org/guide/saved_model) | `saved_model`     | `yolov8n_saved_model/`    | ✅           | `imgsz`, `keras`                                    |
| [GraphDef TF](https://www.tensorflow.org/api_docs/python/tf/Graph)   | `pb`              | `yolov8n.pb`              | ❌           | `imgsz`                                             |
| [TF Lite](https://www.tensorflow.org/lite)                           | `tflite`          | `yolov8n.tflite`          | ✅           | `imgsz`, `half`, `int8`                             |
| [TPU Edge TF](https://coral.ai/docs/edgetpu/models-intro/)           | `edgetpu`         | `yolov8n_edgetpu.tflite`  | ✅           | `imgsz`                                             |
| [TF.js](https://www.tensorflow.org/js)                               | `tfjs`            | `yolov8n_web_model/`      | ✅           | `imgsz`                                             |
| [PaddlePaddle](https://github.com/PaddlePaddle)                      | `paddle`          | `yolov8n_paddle_model/`   | ✅           | `imgsz`                                             |
| [ncnn](https://github.com/Tencent/ncnn)                              | `ncnn`            | `yolov8n_ncnn_model/`     | ✅           | `imgsz`, `half`                                     |

Consultez tous les détails `export` sur la page [Exporter](https://docs.ultralytics.com/modes/export/).
