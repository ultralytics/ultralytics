---
comments: true
description: Documentación oficial de YOLOv8 de Ultralytics. Aprende a entrenar, validar, predecir y exportar modelos en varios formatos. Incluyendo estadísticas detalladas de rendimiento.
keywords: YOLOv8, Ultralytics, detección de objetos, modelos preentrenados, entrenamiento, validación, predicción, exportación de modelos, COCO, ImageNet, PyTorch, ONNX, CoreML
---

# Detección de Objetos

<img width="1024" src="https://user-images.githubusercontent.com/26833433/243418624-5785cb93-74c9-4541-9179-d5c6782d491a.png" alt="Ejemplos de detección de objetos">

La detección de objetos es una tarea que implica identificar la ubicación y clase de objetos en una imagen o flujo de video.

La salida de un detector de objetos es un conjunto de cajas delimitadoras que encierran a los objetos en la imagen, junto con etiquetas de clase y puntajes de confianza para cada caja. La detección de objetos es una buena opción cuando necesitas identificar objetos de interés en una escena, pero no necesitas saber exactamente dónde se encuentra el objeto o su forma exacta.

<p align="center">
  <br>
  <iframe width="720" height="405" src="https://www.youtube.com/embed/5ku7npMrW40?si=6HQO1dDXunV8gekh"
    title="Reproductor de video YouTube" frameborder="0"
    allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share"
    permitir pantalla completa>
  </iframe>
  <br>
  <strong>Ver:</strong> Detección de Objetos con Modelo Preentrenado YOLOv8 de Ultralytics.
</p>

!!! Tip "Consejo"

    Los modelos YOLOv8 Detect son los modelos predeterminados de YOLOv8, es decir, `yolov8n.pt` y están preentrenados en [COCO](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/datasets/coco.yaml).

## [Modelos](https://github.com/ultralytics/ultralytics/tree/main/ultralytics/cfg/models/v8)

Los modelos preentrenados de YOLOv8 Detect se muestran aquí. Los modelos de Detect, Segment y Pose están preentrenados en el conjunto de datos [COCO](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/datasets/coco.yaml), mientras que los modelos de Classify están preentrenados en el conjunto de datos [ImageNet](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/datasets/ImageNet.yaml).

Los [modelos](https://github.com/ultralytics/ultralytics/tree/main/ultralytics/cfg/models) se descargan automáticamente desde el último lanzamiento de Ultralytics [release](https://github.com/ultralytics/assets/releases) en el primer uso.

| Modelo                                                                               | tamaño<br><sup>(píxeles) | mAP<sup>val<br>50-95 | Velocidad<br><sup>CPU ONNX<br>(ms) | Velocidad<br><sup>A100 TensorRT<br>(ms) | parámetros<br><sup>(M) | FLOPs<br><sup>(B) |
|--------------------------------------------------------------------------------------|--------------------------|----------------------|------------------------------------|-----------------------------------------|------------------------|-------------------|
| [YOLOv8n](https://github.com/ultralytics/assets/releases/download/v8.1.0/yolov8n.pt) | 640                      | 37.3                 | 80.4                               | 0.99                                    | 3.2                    | 8.7               |
| [YOLOv8s](https://github.com/ultralytics/assets/releases/download/v8.1.0/yolov8s.pt) | 640                      | 44.9                 | 128.4                              | 1.20                                    | 11.2                   | 28.6              |
| [YOLOv8m](https://github.com/ultralytics/assets/releases/download/v8.1.0/yolov8m.pt) | 640                      | 50.2                 | 234.7                              | 1.83                                    | 25.9                   | 78.9              |
| [YOLOv8l](https://github.com/ultralytics/assets/releases/download/v8.1.0/yolov8l.pt) | 640                      | 52.9                 | 375.2                              | 2.39                                    | 43.7                   | 165.2             |
| [YOLOv8x](https://github.com/ultralytics/assets/releases/download/v8.1.0/yolov8x.pt) | 640                      | 53.9                 | 479.1                              | 3.53                                    | 68.2                   | 257.8             |

- Los valores de **mAP<sup>val</sup>** son para un solo modelo a una sola escala en el conjunto de datos [COCO val2017](http://cocodataset.org).
  <br>Reproduce utilizando `yolo val detect data=coco.yaml device=0`
- La **Velocidad** es el promedio sobre las imágenes de COCO val utilizando una instancia [Amazon EC2 P4d](https://aws.amazon.com/ec2/instance-types/p4/).
  <br>Reproduce utilizando `yolo val detect data=coco128.yaml batch=1 device=0|cpu`

## Entrenamiento

Entrena a YOLOv8n en el conjunto de datos COCO128 durante 100 épocas a tamaño de imagen 640. Para una lista completa de argumentos disponibles, consulta la página [Configuración](/../usage/cfg.md).

!!! Example "Ejemplo"

    === "Python"

        ```python
        from ultralytics import YOLO

        # Cargar un modelo
        model = YOLO('yolov8n.yaml')  # construye un nuevo modelo desde YAML
        model = YOLO('yolov8n.pt')  # carga un modelo preentrenado (recomendado para entrenamiento)
        model = YOLO('yolov8n.yaml').load('yolov8n.pt')  # construye desde YAML y transfiere los pesos

        # Entrenar el modelo
        results = model.train(data='coco128.yaml', epochs=100, imgsz=640)
        ```
    === "CLI"

        ```bash
        # Construir un nuevo modelo desde YAML y comenzar entrenamiento desde cero
        yolo detect train data=coco128.yaml model=yolov8n.yaml epochs=100 imgsz=640

        # Comenzar entrenamiento desde un modelo *.pt preentrenado
        yolo detect train data=coco128.yaml model=yolov8n.pt epochs=100 imgsz=640

        # Construir un nuevo modelo desde YAML, transferir pesos preentrenados y comenzar entrenamiento
        yolo detect train data=coco128.yaml model=yolov8n.yaml pretrained=yolov8n.pt epochs=100 imgsz=640
        ```

### Formato del conjunto de datos

El formato del conjunto de datos de detección de YOLO se puede encontrar en detalle en la [Guía de Conjuntos de Datos](../../../datasets/detect/index.md). Para convertir tu conjunto de datos existente desde otros formatos (como COCO, etc.) al formato YOLO, por favor usa la herramienta [JSON2YOLO](https://github.com/ultralytics/JSON2YOLO) de Ultralytics.

## Validación

Valida la precisión del modelo YOLOv8n entrenado en el conjunto de datos COCO128. No es necesario pasar ningún argumento, ya que el `modelo` retiene sus datos de `entrenamiento` y argumentos como atributos del modelo.

!!! Example "Ejemplo"

    === "Python"

        ```python
        from ultralytics import YOLO

        # Cargar un modelo
        model = YOLO('yolov8n.pt')  # cargar un modelo oficial
        model = YOLO('ruta/a/mejor.pt')  # cargar un modelo personalizado

        # Validar el modelo
        metrics = model.val()  # sin argumentos necesarios, el conjunto de datos y configuraciones se recuerdan
        metrics.box.map    # map50-95
        metrics.box.map50  # map50
        metrics.box.map75  # map75
        metrics.box.maps   # una lista contiene map50-95 de cada categoría
        ```
    === "CLI"

        ```bash
        yolo detect val model=yolov8n.pt  # validar modelo oficial
        yolo detect val model=ruta/a/mejor.pt  # validar modelo personalizado
        ```

## Predicción

Utiliza un modelo YOLOv8n entrenado para realizar predicciones en imágenes.

!!! Example "Ejemplo"

    === "Python"

        ```python
        from ultralytics import YOLO

        # Cargar un modelo
        model = YOLO('yolov8n.pt')  # cargar un modelo oficial
        model = YOLO('ruta/a/mejor.pt')  # cargar un modelo personalizado

        # Predecir con el modelo
        results = model('https://ultralytics.com/images/bus.jpg')  # predecir en una imagen
        ```
    === "CLI"

        ```bash
        yolo detect predict model=yolov8n.pt source='https://ultralytics.com/images/bus.jpg'  # predecir con modelo oficial
        yolo detect predict model=ruta/a/mejor.pt source='https://ultralytics.com/images/bus.jpg'  # predecir con modelo personalizado
        ```

Consulta los detalles completos del modo `predict` en la página [Predicción](https://docs.ultralytics.com/modes/predict/).

## Exportación

Exporta un modelo YOLOv8n a un formato diferente como ONNX, CoreML, etc.

!!! Example "Ejemplo"

    === "Python"

        ```python
        from ultralytics import YOLO

        # Cargar un modelo
        model = YOLO('yolov8n.pt')  # cargar un modelo oficial
        model = YOLO('ruta/a/mejor.pt')  # cargar un modelo entrenado personalizado

        # Exportar el modelo
        model.export(format='onnx')
        ```
    === "CLI"

        ```bash
        yolo export model=yolov8n.pt format=onnx  # exportar modelo oficial
        yolo export model=ruta/a/mejor.pt format=onnx  # exportar modelo entrenado personalizado
        ```

Los formatos de exportación de YOLOv8 disponibles se encuentran en la tabla a continuación. Puedes predecir o validar directamente en modelos exportados, es decir, `yolo predict model=yolov8n.onnx`. Ejemplos de uso se muestran para tu modelo después de que la exportación se completa.

| Formato                                                            | Argumento `format` | Modelo                     | Metadata | Argumentos                                                             |
|--------------------------------------------------------------------|--------------------|----------------------------|----------|------------------------------------------------------------------------|
| [PyTorch](https://pytorch.org/)                                    | -                  | `yolov8n.pt`               | ✅        | -                                                                      |
| [TorchScript](https://pytorch.org/docs/stable/jit.html)            | `torchscript`      | `yolov8n.torchscript`      | ✅        | `imgsz`, `optimizar`                                                   |
| [ONNX](https://onnx.ai/)                                           | `onnx`             | `yolov8n.onnx`             | ✅        | `imgsz`, `mitad`, `dinámico`, `simplificar`, `conjunto de operaciones` |
| [OpenVINO](https://docs.openvino.ai/latest/index.html)             | `openvino`         | `modelo_yolov8n_openvino/` | ✅        | `imgsz`, `mitad`                                                       |
| [TensorRT](https://developer.nvidia.com/tensorrt)                  | `engine`           | `yolov8n.engine`           | ✅        | `imgsz`, `mitad`, `dinámico`, `simplificar`, `espacio de trabajo`      |
| [CoreML](https://github.com/apple/coremltools)                     | `coreml`           | `yolov8n.mlpackage`        | ✅        | `imgsz`, `mitad`, `int8`, `nms`                                        |
| [TF SavedModel](https://www.tensorflow.org/guide/saved_model)      | `saved_model`      | `modelo_guardado_yolov8n/` | ✅        | `imgsz`, `keras`                                                       |
| [TF GraphDef](https://www.tensorflow.org/api_docs/python/tf/Graph) | `pb`               | `yolov8n.pb`               | ❌        | `imgsz`                                                                |
| [TF Lite](https://www.tensorflow.org/lite)                         | `tflite`           | `yolov8n.tflite`           | ✅        | `imgsz`, `mitad`, `int8`                                               |
| [TF Edge TPU](https://coral.ai/docs/edgetpu/models-intro/)         | `edgetpu`          | `yolov8n_edgetpu.tflite`   | ✅        | `imgsz`                                                                |
| [TF.js](https://www.tensorflow.org/js)                             | `tfjs`             | `modelo_web_yolov8n/`      | ✅        | `imgsz`                                                                |
| [PaddlePaddle](https://github.com/PaddlePaddle)                    | `paddle`           | `modelo_yolov8n_paddle/`   | ✅        | `imgsz`                                                                |
| [ncnn](https://github.com/Tencent/ncnn)                            | `ncnn`             | `modelo_ncnn_yolov8n/`     | ✅        | `imgsz`, `mitad`                                                       |

Consulta los detalles completos de la `exportación` en la página [Exportar](https://docs.ultralytics.com/modes/export/).
