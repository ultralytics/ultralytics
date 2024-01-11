---
comments: true
description: Apprenez à utiliser Ultralytics YOLOv8 pour des tâches d'estimation de pose. Trouvez des modèles pré-entraînés, apprenez à entraîner, valider, prédire et exporter vos propres modèles.
keywords: Ultralytics, YOLO, YOLOv8, estimation de pose, détection de points clés, détection d'objet, modèles pré-entraînés, apprentissage automatique, intelligence artificielle
---

# Estimation de Pose

![Estimation de pose exemples](https://user-images.githubusercontent.com/26833433/243418616-9811ac0b-a4a7-452a-8aba-484ba32bb4a8.png)

L'estimation de pose est une tâche qui consiste à identifier l'emplacement de points spécifiques dans une image, souvent appelés points clés. Ces points clés peuvent représenter différentes parties de l'objet telles que les articulations, les repères ou d'autres caractéristiques distinctives. L'emplacement des points clés est généralement représenté par un ensemble de coordonnées 2D `[x, y]` ou 3D `[x, y, visible]`.

La sortie d'un modèle d'estimation de pose est un ensemble de points représentant les points clés sur un objet dans l'image, généralement accompagnés des scores de confiance pour chaque point. L'estimation de pose est un bon choix lorsque vous avez besoin d'identifier des parties spécifiques d'un objet dans une scène, et leur emplacement les uns par rapport aux autres.

![Regardez : Estimation de Pose avec Ultralytics YOLOv8](https://www.youtube.com/embed/Y28xXQmju64?si=pCY4ZwejZFu6Z4kZ)

!!! astuce "Conseil"

    Les modèles YOLOv8 _pose_ utilisent le suffixe `-pose`, c'est-à-dire `yolov8n-pose.pt`. Ces modèles sont entraînés sur le jeu de données [COCO keypoints](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/datasets/coco-pose.yaml) et conviennent à une variété de tâches d'estimation de pose.

## [Modèles](https://github.com/ultralytics/ultralytics/tree/main/ultralytics/cfg/models/v8)

Les modèles Pose pré-entraînés YOLOv8 sont montrés ici. Les modèles Detect, Segment et Pose sont pré-entraînés sur le jeu de données [COCO](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/datasets/coco.yaml), tandis que les modèles Classify sont pré-entraînés sur le jeu de données [ImageNet](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/datasets/ImageNet.yaml).

Les [Modèles](https://github.com/ultralytics/ultralytics/tree/main/ultralytics/cfg/models) se téléchargent automatiquement à partir de la dernière version d'Ultralytics [release](https://github.com/ultralytics/assets/releases) lors de la première utilisation.

| Modèle                                                                                               | taille<br><sup>(pixels) | mAP<sup>pose<br>50-95 | mAP<sup>pose<br>50 | Vitesse<br><sup>CPU ONNX<br>(ms) | Vitesse<br><sup>A100 TensorRT<br>(ms) | params<br><sup>(M) | FLOPs<br><sup>(B) |
|------------------------------------------------------------------------------------------------------|-------------------------|-----------------------|--------------------|----------------------------------|---------------------------------------|--------------------|-------------------|
| [YOLOv8n-pose](https://github.com/ultralytics/assets/releases/download/v8.1.0/yolov8n-pose.pt)       | 640                     | 50.4                  | 80.1               | 131.8                            | 1.18                                  | 3.3                | 9.2               |
| [YOLOv8s-pose](https://github.com/ultralytics/assets/releases/download/v8.1.0/yolov8s-pose.pt)       | 640                     | 60.0                  | 86.2               | 233.2                            | 1.42                                  | 11.6               | 30.2              |
| [YOLOv8m-pose](https://github.com/ultralytics/assets/releases/download/v8.1.0/yolov8m-pose.pt)       | 640                     | 65.0                  | 88.8               | 456.3                            | 2.00                                  | 26.4               | 81.0              |
| [YOLOv8l-pose](https://github.com/ultralytics/assets/releases/download/v8.1.0/yolov8l-pose.pt)       | 640                     | 67.6                  | 90.0               | 784.5                            | 2.59                                  | 44.4               | 168.6             |
| [YOLOv8x-pose](https://github.com/ultralytics/assets/releases/download/v8.1.0/yolov8x-pose.pt)       | 640                     | 69.2                  | 90.2               | 1607.1                           | 3.73                                  | 69.4               | 263.2             |
| [YOLOv8x-pose-p6](https://github.com/ultralytics/assets/releases/download/v8.1.0/yolov8x-pose-p6.pt) | 1280                    | 71.6                  | 91.2               | 4088.7                           | 10.04                                 | 99.1               | 1066.4            |

- Les valeurs de **mAP<sup>val</sup>** sont pour un seul modèle à une seule échelle sur le jeu de données [COCO Keypoints val2017](http://cocodataset.org).
  <br>Reproduire avec `yolo val pose data=coco-pose.yaml device=0`
- La **vitesse** moyenne sur les images de validation COCO en utilisant une instance [Amazon EC2 P4d](https://aws.amazon.com/ec2/instance-types/p4/).
  <br>Reproduire avec `yolo val pose data=coco8-pose.yaml batch=1 device=0|cpu`

## Entraînement

Entraînez un modèle YOLOv8-pose sur le jeu de données COCO128-pose.

!!! Example "Exemple"

    === "Python"

        ```python
        from ultralytics import YOLO

        # Charger un modèle
        model = YOLO('yolov8n-pose.yaml')  # construire un nouveau modèle à partir du YAML
        model = YOLO('yolov8n-pose.pt')    # charger un modèle pré-entraîné (recommandé pour l'entraînement)
        model = YOLO('yolov8n-pose.yaml').load('yolov8n-pose.pt')  # construire à partir du YAML et transférer les poids

        # Entraîner le modèle
        résultats = model.train(data='coco8-pose.yaml', epochs=100, imgsz=640)
        ```
    === "CLI"

        ```bash
        # Construire un nouveau modèle à partir du YAML et commencer l'entraînement à partir de zéro
        yolo pose train data=coco8-pose.yaml model=yolov8n-pose.yaml epochs=100 imgsz=640

        # Commencer l'entraînement à partir d'un modèle *.pt pré-entraîné
        yolo pose train data=coco8-pose.yaml model=yolov8n-pose.pt epochs=100 imgsz=640

        # Construire un nouveau modèle à partir du YAML, transférer les poids pré-entraînés et commencer l'entraînement
        yolo pose train data=coco8-pose.yaml model=yolov8n-pose.yaml pretrained=yolov8n-pose.pt epochs=100 imgsz=640
        ```

### Format du jeu de données

Le format du jeu de données YOLO pose peut être trouvé en détail dans le [Guide des jeux de données](../../../datasets/pose/index.md). Pour convertir votre jeu de données existant à partir d'autres formats (comme COCO, etc.) vers le format YOLO, veuillez utiliser l'outil [JSON2YOLO](https://github.com/ultralytics/JSON2YOLO) d'Ultralytics.

## Val

Validez la précision du modèle YOLOv8n-pose entraîné sur le jeu de données COCO128-pose. Aucun argument n'est nécessaire car le `modèle` conserve ses données d'entraînement et arguments en tant qu'attributs du modèle.

!!! Example "Exemple"

    === "Python"

        ```python
        from ultralytics import YOLO

        # Charger un modèle
        model = YOLO('yolov8n-pose.pt')     # charger un modèle officiel
        model = YOLO('chemin/vers/best.pt')  # charger un modèle personnalisé

        # Valider le modèle
        métriques = model.val()  # aucun argument nécessaire, jeu de données et paramètres mémorisés
        métriques.box.map    # map50-95
        métriques.box.map50  # map50
        métriques.box.map75  # map75
        métriques.box.maps   # une liste contenant map50-95 de chaque catégorie
        ```
    === "CLI"

        ```bash
        yolo pose val model=yolov8n-pose.pt  # val modèle officiel
        yolo pose val model=chemin/vers/best.pt  # val modèle personnalisé
        ```

## Prédiction

Utilisez un modèle YOLOv8n-pose entraîné pour exécuter des prédictions sur des images.

!!! Example "Exemple"

    === "Python"

        ```python
        from ultralytics import YOLO

        # Charger un modèle
        model = YOLO('yolov8n-pose.pt')     # charger un modèle officiel
        model = YOLO('chemin/vers/best.pt')  # charger un modèle personnalisé

        # Prédire avec le modèle
        résultats = model('https://ultralytics.com/images/bus.jpg')  # prédire sur une image
        ```
    === "CLI"

        ```bash
        yolo pose predict model=yolov8n-pose.pt source='https://ultralytics.com/images/bus.jpg'  # prédire avec modèle officiel
        yolo pose predict model=chemin/vers/best.pt source='https://ultralytics.com/images/bus.jpg'  # prédire avec modèle personnalisé
        ```

Consultez les détails complets du mode `predict` sur la page [Prédire](https://docs.ultralytics.com/modes/predict/).

## Exportation

Exportez un modèle YOLOv8n Pose dans un autre format tel que ONNX, CoreML, etc.

!!! Example "Exemple"

    === "Python"

        ```python
        from ultralytics import YOLO

        # Charger un modèle
        model = YOLO('yolov8n-pose.pt')      # charger un modèle officiel
        model = YOLO('chemin/vers/best.pt')   # charger un modèle personnalisé entraîné

        # Exporter le modèle
        model.export(format='onnx')
        ```
    === "CLI"

        ```bash
        yolo export model=yolov8n-pose.pt format=onnx  # exporter modèle officiel
        yolo export model=chemin/vers/best.pt format=onnx  # exporter modèle personnalisé entraîné
        ```

Les formats d'exportation YOLOv8-pose disponibles sont dans le tableau ci-dessous. Vous pouvez prédire ou valider directement sur des modèles exportés, par exemple `yolo predict model=yolov8n-pose.onnx`. Des exemples d'utilisation sont montrés pour votre modèle après la fin de l'exportation.

| Format                                                             | Argument `format` | Modèle                         | Métadonnées | Arguments                                           |
|--------------------------------------------------------------------|-------------------|--------------------------------|-------------|-----------------------------------------------------|
| [PyTorch](https://pytorch.org/)                                    | -                 | `yolov8n-pose.pt`              | ✅           | -                                                   |
| [TorchScript](https://pytorch.org/docs/stable/jit.html)            | `torchscript`     | `yolov8n-pose.torchscript`     | ✅           | `imgsz`, `optimize`                                 |
| [ONNX](https://onnx.ai/)                                           | `onnx`            | `yolov8n-pose.onnx`            | ✅           | `imgsz`, `half`, `dynamic`, `simplify`, `opset`     |
| [OpenVINO](https://docs.openvino.ai/latest/index.html)             | `openvino`        | `yolov8n-pose_openvino_model/` | ✅           | `imgsz`, `half`                                     |
| [TensorRT](https://developer.nvidia.com/tensorrt)                  | `engine`          | `yolov8n-pose.engine`          | ✅           | `imgsz`, `half`, `dynamic`, `simplify`, `workspace` |
| [CoreML](https://github.com/apple/coremltools)                     | `coreml`          | `yolov8n-pose.mlpackage`       | ✅           | `imgsz`, `half`, `int8`, `nms`                      |
| [TF SavedModel](https://www.tensorflow.org/guide/saved_model)      | `saved_model`     | `yolov8n-pose_saved_model/`    | ✅           | `imgsz`, `keras`                                    |
| [TF GraphDef](https://www.tensorflow.org/api_docs/python/tf/Graph) | `pb`              | `yolov8n-pose.pb`              | ❌           | `imgsz`                                             |
| [TF Lite](https://www.tensorflow.org/lite)                         | `tflite`          | `yolov8n-pose.tflite`          | ✅           | `imgsz`, `half`, `int8`                             |
| [TF Edge TPU](https://coral.ai/docs/edgetpu/models-intro/)         | `edgetpu`         | `yolov8n-pose_edgetpu.tflite`  | ✅           | `imgsz`                                             |
| [TF.js](https://www.tensorflow.org/js)                             | `tfjs`            | `yolov8n-pose_web_model/`      | ✅           | `imgsz`                                             |
| [PaddlePaddle](https://github.com/PaddlePaddle)                    | `paddle`          | `yolov8n-pose_paddle_model/`   | ✅           | `imgsz`                                             |
| [ncnn](https://github.com/Tencent/ncnn)                            | `ncnn`            | `yolov8n-pose_ncnn_model/`     | ✅           | `imgsz`, `half`                                     |

Consultez les détails complets de `export` sur la page [Exporter](https://docs.ultralytics.com/modes/export/).
