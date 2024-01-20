---
comments: true
description: Aprenda sobre los modelos de clasificación de imágenes YOLOv8 Classify. Obtenga información detallada sobre la Lista de Modelos Preentrenados y cómo Entrenar, Validar, Predecir y Exportar modelos.
keywords: Ultralytics, YOLOv8, Clasificación de imágenes, Modelos preentrenados, YOLOv8n-cls, Entrenamiento, Validación, Predicción, Exportación de modelos
---

# Clasificación de Imágenes

<img width="1024" src="https://user-images.githubusercontent.com/26833433/243418606-adf35c62-2e11-405d-84c6-b84e7d013804.png" alt="Ejemplos de clasificación de imágenes">

La clasificación de imágenes es la tarea más sencilla de las tres y consiste en clasificar una imagen completa en una de un conjunto de clases predefinidas.

La salida de un clasificador de imágenes es una única etiqueta de clase y una puntuación de confianza. La clasificación de imágenes es útil cuando solo necesita saber a qué clase pertenece una imagen y no necesita conocer dónde están ubicados los objetos de esa clase o cuál es su forma exacta.

!!! Tip "Consejo"

    Los modelos YOLOv8 Classify utilizan el sufijo `-cls`, por ejemplo, `yolov8n-cls.pt` y están preentrenados en [ImageNet](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/datasets/ImageNet.yaml).

## [Modelos](https://github.com/ultralytics/ultralytics/tree/main/ultralytics/cfg/models/v8)

Los modelos Classify preentrenados YOLOv8 se muestran aquí. Los modelos Detect, Segment y Pose están preentrenados en el conjunto de datos [COCO](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/datasets/coco.yaml), mientras que los modelos Classify están preentrenados en el conjunto de datos [ImageNet](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/datasets/ImageNet.yaml).

Los [modelos](https://github.com/ultralytics/ultralytics/tree/main/ultralytics/cfg/models) se descargan automáticamente desde el último [lanzamiento](https://github.com/ultralytics/assets/releases) de Ultralytics en el primer uso.

| Modelo                                                                                       | Tamaño<br><sup>(píxeles) | Exactitud<br><sup>top1 | Exactitud<br><sup>top5 | Velocidad<br><sup>CPU ONNX<br>(ms) | Velocidad<br><sup>A100 TensorRT<br>(ms) | Parámetros<br><sup>(M) | FLOPs<br><sup>(B) en 640 |
|----------------------------------------------------------------------------------------------|--------------------------|------------------------|------------------------|------------------------------------|-----------------------------------------|------------------------|--------------------------|
| [YOLOv8n-cls](https://github.com/ultralytics/assets/releases/download/v8.1.0/yolov8n-cls.pt) | 224                      | 66.6                   | 87.0                   | 12.9                               | 0.31                                    | 2.7                    | 4.3                      |
| [YOLOv8s-cls](https://github.com/ultralytics/assets/releases/download/v8.1.0/yolov8s-cls.pt) | 224                      | 72.3                   | 91.1                   | 23.4                               | 0.35                                    | 6.4                    | 13.5                     |
| [YOLOv8m-cls](https://github.com/ultralytics/assets/releases/download/v8.1.0/yolov8m-cls.pt) | 224                      | 76.4                   | 93.2                   | 85.4                               | 0.62                                    | 17.0                   | 42.7                     |
| [YOLOv8l-cls](https://github.com/ultralytics/assets/releases/download/v8.1.0/yolov8l-cls.pt) | 224                      | 78.0                   | 94.1                   | 163.0                              | 0.87                                    | 37.5                   | 99.7                     |
| [YOLOv8x-cls](https://github.com/ultralytics/assets/releases/download/v8.1.0/yolov8x-cls.pt) | 224                      | 78.4                   | 94.3                   | 232.0                              | 1.01                                    | 57.4                   | 154.8                    |

- Los valores de **Exactitud** son las precisiones de los modelos en el conjunto de datos de validación de [ImageNet](https://www.image-net.org/).
  <br>Para reproducir usar `yolo val classify data=path/to/ImageNet device=0`
- **Velocidad** promediada sobre imágenes de validación de ImageNet usando una instancia de [Amazon EC2 P4d](https://aws.amazon.com/ec2/instance-types/p4/)
  <br>Para reproducir usar `yolo val classify data=path/to/ImageNet batch=1 device=0|cpu`

## Entrenamiento

Entrena el modelo YOLOv8n-cls en el conjunto de datos MNIST160 durante 100 épocas con un tamaño de imagen de 64. Para obtener una lista completa de argumentos disponibles, consulte la página de [Configuración](/../usage/cfg.md).

!!! Example "Ejemplo"

    === "Python"

        ```python
        from ultralytics import YOLO

        # Cargar un modelo
        model = YOLO('yolov8n-cls.yaml')  # construir un nuevo modelo desde YAML
        model = YOLO('yolov8n-cls.pt')  # cargar un modelo preentrenado (recomendado para entrenamiento)
        model = YOLO('yolov8n-cls.yaml').load('yolov8n-cls.pt')  # construir desde YAML y transferir pesos

        # Entrenar el modelo
        results = model.train(data='mnist160', epochs=100, imgsz=64)
        ```

    === "CLI"

        ```bash
        # Construir un nuevo modelo desde YAML y empezar entrenamiento desde cero
        yolo classify train data=mnist160 model=yolov8n-cls.yaml epochs=100 imgsz=64

        # Empezar entrenamiento desde un modelo *.pt preentrenado
        yolo classify train data=mnist160 model=yolov8n-cls.pt epochs=100 imgsz=64

        # Construir un nuevo modelo desde YAML, transferir pesos preentrenados e iniciar entrenamiento
        yolo classify train data=mnist160 model=yolov8n-cls.yaml pretrained=yolov8n-cls.pt epochs=100 imgsz=64
        ```

### Formato del conjunto de datos

El formato del conjunto de datos de clasificación YOLO puede encontrarse en detalle en la [Guía de Conjuntos de Datos](../../../datasets/classify/index.md).

## Validación

Validar la exactitud del modelo YOLOv8n-cls entrenado en el conjunto de datos MNIST160. No es necesario pasar ningún argumento ya que el `modelo` retiene su `data` y argumentos como atributos del modelo.

!!! Example "Ejemplo"

    === "Python"

        ```python
        from ultralytics import YOLO

        # Cargar un modelo
        model = YOLO('yolov8n-cls.pt')  # cargar un modelo oficial
        model = YOLO('path/to/best.pt')  # cargar un modelo personalizado

        # Validar el modelo
        metrics = model.val()  # no se necesitan argumentos, el conjunto de datos y configuraciones se recuerdan
        metrics.top1   # precisión top1
        metrics.top5   # precisión top5
        ```
    === "CLI"

        ```bash
        yolo classify val model=yolov8n-cls.pt  # validar modelo oficial
        yolo classify val model=path/to/best.pt  # validar modelo personalizado
        ```

## Predicción

Usar un modelo YOLOv8n-cls entrenado para realizar predicciones en imágenes.

!!! Example "Ejemplo"

    === "Python"

        ```python
        from ultralytics import YOLO

        # Cargar un modelo
        model = YOLO('yolov8n-cls.pt')  # cargar un modelo oficial
        model = YOLO('path/to/best.pt')  # cargar un modelo personalizado

        # Predecir con el modelo
        results = model('https://ultralytics.com/images/bus.jpg')  # predecir en una imagen
        ```
    === "CLI"

        ```bash
        yolo classify predict model=yolov8n-cls.pt source='https://ultralytics.com/images/bus.jpg'  # predecir con modelo oficial
        yolo classify predict model=path/to/best.pt source='https://ultralytics.com/images/bus.jpg'  # predecir con modelo personalizado
        ```

Ver detalles completos del modo `predict` en la página de [Predicción](https://docs.ultralytics.com/modes/predict/).

## Exportación

Exportar un modelo YOLOv8n-cls a un formato diferente como ONNX, CoreML, etc.

!!! Example "Ejemplo"

    === "Python"

        ```python
        from ultralytics import YOLO

        # Cargar un modelo
        model = YOLO('yolov8n-cls.pt')  # cargar un modelo oficial
        model = YOLO('path/to/best.pt')  # cargar un modelo entrenado personalizado

        # Exportar el modelo
        model.export(format='onnx')
        ```
    === "CLI"

        ```bash
        yolo export model=yolov8n-cls.pt format=onnx  # exportar modelo oficial
        yolo export model=path/to/best.pt format=onnx  # exportar modelo entrenado personalizado
        ```

Los formatos de exportación disponibles para YOLOv8-cls se encuentran en la tabla a continuación. Puede predecir o validar directamente en modelos exportados, por ejemplo, `yolo predict model=yolov8n-cls.onnx`. Ejemplos de uso se muestran para su modelo después de que se completa la exportación.

| Formato                                                            | Argumento `format` | Modelo                        | Metadatos | Argumentos                                          |
|--------------------------------------------------------------------|--------------------|-------------------------------|-----------|-----------------------------------------------------|
| [PyTorch](https://pytorch.org/)                                    | -                  | `yolov8n-cls.pt`              | ✅         | -                                                   |
| [TorchScript](https://pytorch.org/docs/stable/jit.html)            | `torchscript`      | `yolov8n-cls.torchscript`     | ✅         | `imgsz`, `optimize`                                 |
| [ONNX](https://onnx.ai/)                                           | `onnx`             | `yolov8n-cls.onnx`            | ✅         | `imgsz`, `half`, `dynamic`, `simplify`, `opset`     |
| [OpenVINO](https://docs.openvino.ai/latest/index.html)             | `openvino`         | `yolov8n-cls_openvino_model/` | ✅         | `imgsz`, `half`                                     |
| [TensorRT](https://developer.nvidia.com/tensorrt)                  | `engine`           | `yolov8n-cls.engine`          | ✅         | `imgsz`, `half`, `dynamic`, `simplify`, `workspace` |
| [CoreML](https://github.com/apple/coremltools)                     | `coreml`           | `yolov8n-cls.mlpackage`       | ✅         | `imgsz`, `half`, `int8`, `nms`                      |
| [TF SavedModel](https://www.tensorflow.org/guide/saved_model)      | `saved_model`      | `yolov8n-cls_saved_model/`    | ✅         | `imgsz`, `keras`                                    |
| [TF GraphDef](https://www.tensorflow.org/api_docs/python/tf/Graph) | `pb`               | `yolov8n-cls.pb`              | ❌         | `imgsz`                                             |
| [TF Lite](https://www.tensorflow.org/lite)                         | `tflite`           | `yolov8n-cls.tflite`          | ✅         | `imgsz`, `half`, `int8`                             |
| [TF Edge TPU](https://coral.ai/docs/edgetpu/models-intro/)         | `edgetpu`          | `yolov8n-cls_edgetpu.tflite`  | ✅         | `imgsz`                                             |
| [TF.js](https://www.tensorflow.org/js)                             | `tfjs`             | `yolov8n-cls_web_model/`      | ✅         | `imgsz`                                             |
| [PaddlePaddle](https://github.com/PaddlePaddle)                    | `paddle`           | `yolov8n-cls_paddle_model/`   | ✅         | `imgsz`                                             |
| [ncnn](https://github.com/Tencent/ncnn)                            | `ncnn`             | `yolov8n-cls_ncnn_model/`     | ✅         | `imgsz`, `half`                                     |

Vea detalles completos de `exportación` en la página de [Exportación](https://docs.ultralytics.com/modes/export/).
