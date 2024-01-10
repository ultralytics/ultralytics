---
comments: true
description: Aprende a utilizar Ultralytics YOLOv8 para tareas de estimación de pose. Encuentra modelos preentrenados, aprende a entrenar, validar, predecir y exportar tus propios modelos.
keywords: Ultralytics, YOLO, YOLOv8, estimación de pose, detección de puntos clave, detección de objetos, modelos preentrenados, aprendizaje automático, inteligencia artificial
---

# Estimación de Pose

<img width="1024" src="https://user-images.githubusercontent.com/26833433/243418616-9811ac0b-a4a7-452a-8aba-484ba32bb4a8.png" alt="Ejemplos de estimación de pose">

La estimación de pose es una tarea que implica identificar la ubicación de puntos específicos en una imagen, comúnmente referidos como puntos clave. Estos puntos clave pueden representar varias partes del objeto, como articulaciones, puntos de referencia u otras características distintivas. La ubicación de los puntos clave generalmente se representa como un conjunto de coordenadas 2D `[x, y]` o 3D `[x, y, visible]`.

La salida de un modelo de estimación de pose es un conjunto de puntos que representan los puntos clave en un objeto de la imagen, generalmente junto con las puntuaciones de confianza para cada punto. La estimación de pose es una buena opción cuando se necesita identificar partes específicas de un objeto en una escena y su ubicación relativa entre ellas.

<p align="center">
  <br>
  <iframe width="720" height="405" src="https://www.youtube.com/embed/Y28xXQmju64?si=pCY4ZwejZFu6Z4kZ"
    title="Reproductor de video de YouTube" frameborder="0"
    allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share"
    allowfullscreen>
  </iframe>
  <br>
  <strong>Ver:</strong> Estimación de Pose con Ultralytics YOLOv8.
</p>

!!! Tip "Consejo"

    Los modelos _pose_ YOLOv8 utilizan el sufijo `-pose`, por ejemplo, `yolov8n-pose.pt`. Estos modelos están entrenados en el conjunto de datos [COCO keypoints](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/datasets/coco-pose.yaml) y son adecuados para una variedad de tareas de estimación de pose.

## [Modelos](https://github.com/ultralytics/ultralytics/tree/main/ultralytics/cfg/models/v8)

Aquí se muestran los modelos preentrenados de YOLOv8 Pose. Los modelos Detect, Segment y Pose están preentrenados en el conjunto de datos [COCO](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/datasets/coco.yaml), mientras que los modelos Classify están preentrenados en el conjunto de datos [ImageNet](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/datasets/ImageNet.yaml).

Los [modelos](https://github.com/ultralytics/ultralytics/tree/main/ultralytics/cfg/models) se descargan automáticamente desde el último lanzamiento de Ultralytics [release](https://github.com/ultralytics/assets/releases) en el primer uso.

| Modelo                                                                                               | tamaño<br><sup>(píxeles) | mAP<sup>pose<br>50-95 | mAP<sup>pose<br>50 | Velocidad<br><sup>CPU ONNX<br>(ms) | Velocidad<br><sup>A100 TensorRT<br>(ms) | parámetros<br><sup>(M) | FLOPs<br><sup>(B) |
|------------------------------------------------------------------------------------------------------|--------------------------|-----------------------|--------------------|------------------------------------|-----------------------------------------|------------------------|-------------------|
| [YOLOv8n-pose](https://github.com/ultralytics/assets/releases/download/v8.1.0/yolov8n-pose.pt)       | 640                      | 50.4                  | 80.1               | 131.8                              | 1.18                                    | 3.3                    | 9.2               |
| [YOLOv8s-pose](https://github.com/ultralytics/assets/releases/download/v8.1.0/yolov8s-pose.pt)       | 640                      | 60.0                  | 86.2               | 233.2                              | 1.42                                    | 11.6                   | 30.2              |
| [YOLOv8m-pose](https://github.com/ultralytics/assets/releases/download/v8.1.0/yolov8m-pose.pt)       | 640                      | 65.0                  | 88.8               | 456.3                              | 2.00                                    | 26.4                   | 81.0              |
| [YOLOv8l-pose](https://github.com/ultralytics/assets/releases/download/v8.1.0/yolov8l-pose.pt)       | 640                      | 67.6                  | 90.0               | 784.5                              | 2.59                                    | 44.4                   | 168.6             |
| [YOLOv8x-pose](https://github.com/ultralytics/assets/releases/download/v8.1.0/yolov8x-pose.pt)       | 640                      | 69.2                  | 90.2               | 1607.1                             | 3.73                                    | 69.4                   | 263.2             |
| [YOLOv8x-pose-p6](https://github.com/ultralytics/assets/releases/download/v8.1.0/yolov8x-pose-p6.pt) | 1280                     | 71.6                  | 91.2               | 4088.7                             | 10.04                                   | 99.1                   | 1066.4            |

- Los valores de **mAP<sup>val</sup>** son para un solo modelo a una sola escala en el conjunto de datos [COCO Keypoints val2017](http://cocodataset.org).
  <br>Reproducir con `yolo val pose data=coco-pose.yaml device=0`
- **Velocidad** promediada sobre imágenes COCO val usando una instancia [Amazon EC2 P4d](https://aws.amazon.com/ec2/instance-types/p4/).
  <br>Reproducir con `yolo val pose data=coco8-pose.yaml batch=1 device=0|cpu`

## Entrenar

Entrena un modelo YOLOv8-pose en el conjunto de datos COCO128-pose.

!!! Example "Ejemplo"

    === "Python"

        ```python
        from ultralytics import YOLO

        # Cargar un modelo
        model = YOLO('yolov8n-pose.yaml')  # construir un nuevo modelo desde YAML
        model = YOLO('yolov8n-pose.pt')  # cargar un modelo preentrenado (recomendado para entrenar)
        model = YOLO('yolov8n-pose.yaml').load('yolov8n-pose.pt')  # construir desde YAML y transferir los pesos

        # Entrenar el modelo
        results = model.train(data='coco8-pose.yaml', epochs=100, imgsz=640)
        ```
    === "CLI"

        ```bash
        # Construir un nuevo modelo desde YAML y comenzar entrenamiento desde cero
        yolo pose train data=coco8-pose.yaml model=yolov8n-pose.yaml epochs=100 imgsz=640

        # Empezar entrenamiento desde un modelo *.pt preentrenado
        yolo pose train data=coco8-pose.yaml model=yolov8n-pose.pt epochs=100 imgsz=640

        # Construir un nuevo modelo desde YAML, transferir pesos preentrenados y comenzar entrenamiento
        yolo pose train data=coco8-pose.yaml model=yolov8n-pose.yaml pretrained=yolov8n-pose.pt epochs=100 imgsz=640
        ```

### Formato del conjunto de datos

El formato del conjunto de datos de pose de YOLO se puede encontrar en detalle en la [Guía de Conjuntos de Datos](../../../datasets/pose/index.md). Para convertir tu conjunto de datos existente de otros formatos (como COCO, etc.) al formato de YOLO, usa la herramienta [JSON2YOLO](https://github.com/ultralytics/JSON2YOLO) de Ultralytics.

## Validar

Valida la precisión del modelo YOLOv8n-pose entrenado en el conjunto de datos COCO128-pose. No es necesario pasar ningún argumento ya que el `modelo` mantiene sus `datos` de entrenamiento y argumentos como atributos del modelo.

!!! Example "Ejemplo"

    === "Python"

        ```python
        from ultralytics import YOLO

        # Cargar un modelo
        model = YOLO('yolov8n-pose.pt')  # cargar un modelo oficial
        model = YOLO('path/to/best.pt')  # cargar un modelo personalizado

        # Validar el modelo
        metrics = model.val()  # no se necesitan argumentos, el conjunto de datos y configuraciones se recuerdan
        metrics.box.map    # map50-95
        metrics.box.map50  # map50
        metrics.box.map75  # map75
        metrics.box.maps   # una lista contiene map50-95 de cada categoría
        ```
    === "CLI"

        ```bash
        yolo pose val model=yolov8n-pose.pt  # modelo oficial de val
        yolo pose val model=path/to/best.pt  # modelo personalizado de val
        ```

## Predecir

Usa un modelo YOLOv8n-pose entrenado para realizar predicciones en imágenes.

!!! Example "Ejemplo"

    === "Python"

        ```python
        from ultralytics import YOLO

        # Cargar un modelo
        model = YOLO('yolov8n-pose.pt')  # cargar un modelo oficial
        model = YOLO('path/to/best.pt')  # cargar un modelo personalizado

        # Predecir con el modelo
        results = model('https://ultralytics.com/images/bus.jpg')  # predecir en una imagen
        ```
    === "CLI"

        ```bash
        yolo pose predict model=yolov8n-pose.pt source='https://ultralytics.com/images/bus.jpg'  # predecir con modelo oficial
        yolo pose predict model=path/to/best.pt source='https://ultralytics.com/images/bus.jpg'  # predecir con modelo personalizado
        ```

Consulta los detalles completos del modo `predict` en la página de [Predicción](https://docs.ultralytics.com/modes/predict/).

## Exportar

Exporta un modelo YOLOv8n Pose a un formato diferente como ONNX, CoreML, etc.

!!! Example "Ejemplo"

    === "Python"

        ```python
        from ultralytics import YOLO

        # Cargar un modelo
        model = YOLO('yolov8n-pose.pt')  # cargar un modelo oficial
        model = YOLO('path/to/best.pt')  # cargar un modelo entrenado personalizado

        # Exportar el modelo
        model.export(format='onnx')
        ```
    === "CLI"

        ```bash
        yolo export model=yolov8n-pose.pt format=onnx  # exportar modelo oficial
        yolo export model=path/to/best.pt format=onnx  # exportar modelo entrenado personalizado
        ```

Los formatos de exportación de YOLOv8-pose disponibles se muestran en la tabla a continuación. Puedes predecir o validar directamente en modelos exportados, por ejemplo, `yolo predict model=yolov8n-pose.onnx`. Los ejemplos de uso se muestran para tu modelo después de que la exportación se completa.

| Formato                                                            | Argumento `format` | Modelo                         | Metadatos | Argumentos                                                    |
|--------------------------------------------------------------------|--------------------|--------------------------------|-----------|---------------------------------------------------------------|
| [PyTorch](https://pytorch.org/)                                    | -                  | `yolov8n-pose.pt`              | ✅         | -                                                             |
| [TorchScript](https://pytorch.org/docs/stable/jit.html)            | `torchscript`      | `yolov8n-pose.torchscript`     | ✅         | `imgsz`, `optimize`                                           |
| [ONNX](https://onnx.ai/)                                           | `onnx`             | `yolov8n-pose.onnx`            | ✅         | `imgsz`, `half`, `dinámico`, `simplify`, `opset`              |
| [OpenVINO](https://docs.openvino.ai/latest/index.html)             | `openvino`         | `yolov8n-pose_openvino_model/` | ✅         | `imgsz`, `half`                                               |
| [TensorRT](https://developer.nvidia.com/tensorrt)                  | `engine`           | `yolov8n-pose.engine`          | ✅         | `imgsz`, `half`, `dinámico`, `simplify`, `espacio de trabajo` |
| [CoreML](https://github.com/apple/coremltools)                     | `coreml`           | `yolov8n-pose.mlpackage`       | ✅         | `imgsz`, `half`, `int8`, `nms`                                |
| [TF SavedModel](https://www.tensorflow.org/guide/saved_model)      | `saved_model`      | `yolov8n-pose_saved_model/`    | ✅         | `imgsz`, `keras`                                              |
| [TF GraphDef](https://www.tensorflow.org/api_docs/python/tf/Graph) | `pb`               | `yolov8n-pose.pb`              | ❌         | `imgsz`                                                       |
| [TF Lite](https://www.tensorflow.org/lite)                         | `tflite`           | `yolov8n-pose.tflite`          | ✅         | `imgsz`, `half`, `int8`                                       |
| [TF Edge TPU](https://coral.ai/docs/edgetpu/models-intro/)         | `edgetpu`          | `yolov8n-pose_edgetpu.tflite`  | ✅         | `imgsz`                                                       |
| [TF.js](https://www.tensorflow.org/js)                             | `tfjs`             | `yolov8n-pose_web_model/`      | ✅         | `imgsz`                                                       |
| [PaddlePaddle](https://github.com/PaddlePaddle)                    | `paddle`           | `yolov8n-pose_paddle_model/`   | ✅         | `imgsz`                                                       |
| [ncnn](https://github.com/Tencent/ncnn)                            | `ncnn`             | `yolov8n-pose_ncnn_model/`     | ✅         | `imgsz`, `half`                                               |

Consulta los detalles completos del modo `export` en la página de [Exportación](https://docs.ultralytics.com/modes/export/).
