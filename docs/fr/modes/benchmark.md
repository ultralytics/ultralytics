---
comments: true
description: Apprenez comment profiler la vitesse et l'exactitude de YOLOv8 à travers divers formats d'exportation ; obtenez des insights sur les métriques mAP50-95, accuracy_top5 et plus.
keywords: Ultralytics, YOLOv8, benchmarking, profilage de vitesse, profilage de précision, mAP50-95, accuracy_top5, ONNX, OpenVINO, TensorRT, formats d'exportation YOLO
---

# Benchmarking de Modèles avec Ultralytics YOLO

<img width="1024" src="https://github.com/ultralytics/assets/raw/main/yolov8/banner-integrations.png" alt="Écosystème Ultralytics YOLO et intégrations">

## Introduction

Une fois votre modèle entraîné et validé, l'étape logique suivante est d'évaluer ses performances dans divers scénarios du monde réel. Le mode benchmark dans Ultralytics YOLOv8 répond à cet objectif en fournissant un cadre robuste pour évaluer la vitesse et l'exactitude de votre modèle sur une gamme de formats d'exportation.

## Pourquoi le Benchmarking est-il Crucial ?

- **Décisions Éclairées :** Obtenez des insights sur les arbitrages entre la vitesse et l'exactitude.
- **Allocation des Ressources :** Comprenez comment les différents formats d'exportation se comportent sur différents matériels.
- **Optimisation :** Découvrez quel format d'exportation offre la meilleure performance pour votre cas d'utilisation spécifique.
- **Efficacité des Coûts :** Utilisez les ressources matérielles plus efficacement en vous basant sur les résultats des benchmarks.

### Mesures Clés en Mode Benchmark

- **mAP50-95 :** Pour la détection d'objets, la segmentation et l'estimation de pose.
- **accuracy_top5 :** Pour la classification d'images.
- **Temps d'Inférence :** Temps pris pour chaque image en millisecondes.

### Formats d'Exportation Supportés

- **ONNX :** Pour une performance optimale sur CPU.
- **TensorRT :** Pour une efficacité maximale sur GPU.
- **OpenVINO :** Pour l'optimisation du matériel Intel.
- **CoreML, TensorFlow SavedModel, et Plus :** Pour des besoins variés de déploiement.

!!! astuce "Conseil"

    * Exportez vers ONNX ou OpenVINO pour un gain de vitesse CPU jusqu'à 3x.
    * Exportez vers TensorRT pour un gain de vitesse GPU jusqu'à 5x.

## Exemples d'Utilisation

Exécutez les benchmarks YOLOv8n sur tous les formats d'exportation supportés, y compris ONNX, TensorRT, etc. Consultez la section Arguments ci-dessous pour une liste complète des arguments d'exportation.

!!! Example "Exemple"

    === "Python"

        ```python
        from ultralytics.utils.benchmarks import benchmark

        # Benchmark sur GPU
        benchmark(model='yolov8n.pt', data='coco8.yaml', imgsz=640, half=False, device=0)
        ```
    === "CLI"

        ```bash
        yolo benchmark model=yolov8n.pt data='coco8.yaml' imgsz=640 half=False device=0
        ```

## Arguments

Des arguments tels que `model`, `data`, `imgsz`, `half`, `device` et `verbose` offrent aux utilisateurs la flexibilité d'ajuster précisément les benchmarks à leurs besoins spécifiques et de comparer facilement les performances de différents formats d'exportation.

| Clé       | Valeur  | Description                                                                           |
|-----------|---------|---------------------------------------------------------------------------------------|
| `model`   | `None`  | chemin vers le fichier modèle, par ex. yolov8n.pt, yolov8n.yaml                       |
| `data`    | `None`  | chemin vers le YAML référençant le dataset de benchmarking (sous l'étiquette `val`)   |
| `imgsz`   | `640`   | taille de l'image comme scalaire ou liste (h, w), par ex. (640, 480)                  |
| `half`    | `False` | quantification FP16                                                                   |
| `int8`    | `False` | quantification INT8                                                                   |
| `device`  | `None`  | appareil sur lequel exécuter, par ex. appareil cuda=0 ou device=0,1,2,3 ou device=cpu |
| `verbose` | `False` | ne pas continuer en cas d'erreur (bool), ou seuil de plancher val (float)             |

## Formats d'Exportation

Les benchmarks tenteront de s'exécuter automatiquement sur tous les formats d'exportation possibles ci-dessous.

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

Consultez les détails complets sur `export` dans la page [Export](https://docs.ultralytics.com/modes/export/).
