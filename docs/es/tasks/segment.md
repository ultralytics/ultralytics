---
comments: true
description: Aprende a utilizar modelos de segmentación de instancias con Ultralytics YOLO. Instrucciones sobre entrenamiento, validación, predicción de imágenes y exportación de modelos.
keywords: yolov8, segmentación de instancias, Ultralytics, conjunto de datos COCO, segmentación de imágenes, detección de objetos, entrenamiento de modelos, validación de modelos, predicción de imágenes, exportación de modelos.
---

# Segmentación de Instancias

<img width="1024" src="https://user-images.githubusercontent.com/26833433/243418644-7df320b8-098d-47f1-85c5-26604d761286.png" alt="Ejemplos de segmentación de instancias">

La segmentación de instancias va un paso más allá de la detección de objetos e implica identificar objetos individuales en una imagen y segmentarlos del resto de la imagen.

La salida de un modelo de segmentación de instancias es un conjunto de máscaras o contornos que delimitan cada objeto en la imagen, junto con etiquetas de clase y puntajes de confianza para cada objeto. La segmentación de instancias es útil cuando necesitas saber no solo dónde están los objetos en una imagen, sino también cuál es su forma exacta.

<p align="center">
  <br>
  <iframe width="720" height="405" src="https://www.youtube.com/embed/o4Zd-IeMlSY?si=37nusCzDTd74Obsp"
    title="Reproductor de video de YouTube" frameborder="0"
    allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share"
    allowfullscreen>
  </iframe>
  <br>
  <strong>Mira:</strong> Ejecuta la Segmentación con el Modelo Ultralytics YOLOv8 Preentrenado en Python.
</p>

!!! Tip "Consejo"

    Los modelos YOLOv8 Segment utilizan el sufijo `-seg`, es decir, `yolov8n-seg.pt` y están preentrenados en el [COCO](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/datasets/coco.yaml).

## [Modelos](https://github.com/ultralytics/ultralytics/tree/main/ultralytics/cfg/models/v8)

Aquí se muestran los modelos Segment preentrenados YOLOv8. Los modelos Detect, Segment y Pose están preentrenados en el conjunto de datos [COCO](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/datasets/coco.yaml), mientras que los modelos Classify están preentrenados en el conjunto de datos [ImageNet](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/datasets/ImageNet.yaml).

Los [Modelos](https://github.com/ultralytics/ultralytics/tree/main/ultralytics/cfg/models) se descargan automáticamente desde el último lanzamiento de Ultralytics [release](https://github.com/ultralytics/assets/releases) en su primer uso.

| Modelo                                                                                       | Tamaño<br><sup>(píxeles) | mAP<sup>caja<br>50-95 | mAP<sup>máscara<br>50-95 | Velocidad<br><sup>CPU ONNX<br>(ms) | Velocidad<br><sup>A100 TensorRT<br>(ms) | Parámetros<br><sup>(M) | FLOPs<br><sup>(B) |
|----------------------------------------------------------------------------------------------|--------------------------|-----------------------|--------------------------|------------------------------------|-----------------------------------------|------------------------|-------------------|
| [YOLOv8n-seg](https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n-seg.pt) | 640                      | 36.7                  | 30.5                     | 96.1                               | 1.21                                    | 3.4                    | 12.6              |
| [YOLOv8s-seg](https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8s-seg.pt) | 640                      | 44.6                  | 36.8                     | 155.7                              | 1.47                                    | 11.8                   | 42.6              |
| [YOLOv8m-seg](https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8m-seg.pt) | 640                      | 49.9                  | 40.8                     | 317.0                              | 2.18                                    | 27.3                   | 110.2             |
| [YOLOv8l-seg](https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8l-seg.pt) | 640                      | 52.3                  | 42.6                     | 572.4                              | 2.79                                    | 46.0                   | 220.5             |
| [YOLOv8x-seg](https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8x-seg.pt) | 640                      | 53.4                  | 43.4                     | 712.1                              | 4.02                                    | 71.8                   | 344.1             |

- Los valores **mAP<sup>val</sup>** son para un único modelo a una única escala en el conjunto de datos [COCO val2017](http://cocodataset.org).
  <br>Reproducir utilizando `yolo val segment data=coco.yaml device=0`
- La **Velocidad** promediada sobre imágenes de COCO val utilizando una instancia de [Amazon EC2 P4d](https://aws.amazon.com/ec2/instance-types/p4/).
  <br>Reproducir utilizando `yolo val segment data=coco128-seg.yaml batch=1 device=0|cpu`

## Entrenamiento

Entrena el modelo YOLOv8n-seg en el conjunto de datos COCO128-seg durante 100 épocas con tamaño de imagen de 640. Para una lista completa de argumentos disponibles, consulta la página de [Configuración](/../usage/cfg.md).

!!! Example "Ejemplo"

    === "Python"

        ```python
        from ultralytics import YOLO

        # Cargar un modelo
        model = YOLO('yolov8n-seg.yaml')  # construir un nuevo modelo desde YAML
        model = YOLO('yolov8n-seg.pt')  # cargar un modelo preentrenado (recomendado para entrenamiento)
        model = YOLO('yolov8n-seg.yaml').load('yolov8n.pt')  # construir desde YAML y transferir pesos

        # Entrenar el modelo
        results = model.train(data='coco128-seg.yaml', epochs=100, imgsz=640)
        ```
    === "CLI"

        ```bash
        # Construir un nuevo modelo desde YAML y comenzar a entrenar desde cero
        yolo segment train data=coco128-seg.yaml model=yolov8n-seg.yaml epochs=100 imgsz=640

        # Comenzar a entrenar desde un modelo *.pt preentrenado
        yolo segment train data=coco128-seg.yaml model=yolov8n-seg.pt epochs=100 imgsz=640

        # Construir un nuevo modelo desde YAML, transferir pesos preentrenados y comenzar a entrenar
        yolo segment train data=coco128-seg.yaml model=yolov8n-seg.yaml pretrained=yolov8n-seg.pt epochs=100 imgsz=640
        ```

### Formato del conjunto de datos

El formato del conjunto de datos de segmentación YOLO puede encontrarse detallado en la [Guía de Conjuntos de Datos](../../../datasets/segment/index.md). Para convertir tu conjunto de datos existente de otros formatos (como COCO, etc.) al formato YOLO, utiliza la herramienta [JSON2YOLO](https://github.com/ultralytics/JSON2YOLO) de Ultralytics.

## Validación

Valida la precisión del modelo YOLOv8n-seg entrenado en el conjunto de datos COCO128-seg. No es necesario pasar ningún argumento ya que el `modelo` retiene sus `datos` de entrenamiento y argumentos como atributos del modelo.

!!! Example "Ejemplo"

    === "Python"

        ```python
        from ultralytics import YOLO

        # Cargar un modelo
        model = YOLO('yolov8n-seg.pt')  # cargar un modelo oficial
        model = YOLO('ruta/a/mejor.pt')  # cargar un modelo personalizado

        # Validar el modelo
        metrics = model.val()  # no se necesitan argumentos, el conjunto de datos y configuraciones se recuerdan
        metrics.box.map    # map50-95(B)
        metrics.box.map50  # map50(B)
        metrics.box.map75  # map75(B)
        metrics.box.maps   # una lista contiene map50-95(B) de cada categoría
        metrics.seg.map    # map50-95(M)
        metrics.seg.map50  # map50(M)
        metrics.seg.map75  # map75(M)
        metrics.seg.maps   # una lista contiene map50-95(M) de cada categoría
        ```
    === "CLI"

        ```bash
        yolo segment val model=yolov8n-seg.pt  # validar el modelo oficial
        yolo segment val model=ruta/a/mejor.pt  # validar el modelo personalizado
        ```

## Predicción

Usa un modelo YOLOv8n-seg entrenado para realizar predicciones en imágenes.

!!! Example "Ejemplo"

    === "Python"

        ```python
        from ultralytics import YOLO

        # Cargar un modelo
        model = YOLO('yolov8n-seg.pt')  # cargar un modelo oficial
        model = YOLO('ruta/a/mejor.pt')  # cargar un modelo personalizado

        # Predecir con el modelo
        results = model('https://ultralytics.com/images/bus.jpg')  # predecir en una imagen
        ```
    === "CLI"

        ```bash
        yolo segment predict model=yolov8n-seg.pt source='https://ultralytics.com/images/bus.jpg'  # predecir con el modelo oficial
        yolo segment predict model=ruta/a/mejor.pt source='https://ultralytics.com/images/bus.jpg'  # predecir con el modelo personalizado
        ```

Consulta todos los detalles del modo `predict` en la página de [Predicción](https://docs.ultralytics.com/modes/predict/).

## Exportación

Exporta un modelo YOLOv8n-seg a un formato diferente como ONNX, CoreML, etc.

!!! Example "Ejemplo"

    === "Python"

        ```python
        from ultralytics import YOLO

        # Cargar un modelo
        model = YOLO('yolov8n-seg.pt')  # cargar un modelo oficial
        model = YOLO('ruta/a/mejor.pt')  # cargar un modelo entrenado personalizado

        # Exportar el modelo
        model.export(format='onnx')
        ```
    === "CLI"

        ```bash
        yolo export model=yolov8n-seg.pt format=onnx  # exportar el modelo oficial
        yolo export model=ruta/a/mejor.pt format=onnx  # exportar el modelo entrenado personalizado
        ```

Los formatos disponibles para exportar YOLOv8-seg se muestran en la tabla a continuación. Puedes predecir o validar directamente en modelos exportados, es decir, `yolo predict model=yolov8n-seg.onnx`. Se muestran ejemplos de uso para tu modelo después de que se completa la exportación.

| Formato                                                            | Argumento `format` | Modelo                        | Metadatos | Argumentos                                          |
|--------------------------------------------------------------------|--------------------|-------------------------------|-----------|-----------------------------------------------------|
| [PyTorch](https://pytorch.org/)                                    | -                  | `yolov8n-seg.pt`              | ✅         | -                                                   |
| [TorchScript](https://pytorch.org/docs/stable/jit.html)            | `torchscript`      | `yolov8n-seg.torchscript`     | ✅         | `imgsz`, `optimize`                                 |
| [ONNX](https://onnx.ai/)                                           | `onnx`             | `yolov8n-seg.onnx`            | ✅         | `imgsz`, `half`, `dynamic`, `simplify`, `opset`     |
| [OpenVINO](https://docs.openvino.ai/latest/index.html)             | `openvino`         | `yolov8n-seg_openvino_model/` | ✅         | `imgsz`, `half`                                     |
| [TensorRT](https://developer.nvidia.com/tensorrt)                  | `engine`           | `yolov8n-seg.engine`          | ✅         | `imgsz`, `half`, `dynamic`, `simplify`, `workspace` |
| [CoreML](https://github.com/apple/coremltools)                     | `coreml`           | `yolov8n-seg.mlpackage`       | ✅         | `imgsz`, `half`, `int8`, `nms`                      |
| [TF SavedModel](https://www.tensorflow.org/guide/saved_model)      | `saved_model`      | `yolov8n-seg_saved_model/`    | ✅         | `imgsz`, `keras`                                    |
| [TF GraphDef](https://www.tensorflow.org/api_docs/python/tf/Graph) | `pb`               | `yolov8n-seg.pb`              | ❌         | `imgsz`                                             |
| [TF Lite](https://www.tensorflow.org/lite)                         | `tflite`           | `yolov8n-seg.tflite`          | ✅         | `imgsz`, `half`, `int8`                             |
| [TF Edge TPU](https://coral.ai/docs/edgetpu/models-intro/)         | `edgetpu`          | `yolov8n-seg_edgetpu.tflite`  | ✅         | `imgsz`                                             |
| [TF.js](https://www.tensorflow.org/js)                             | `tfjs`             | `yolov8n-seg_web_model/`      | ✅         | `imgsz`                                             |
| [PaddlePaddle](https://github.com/PaddlePaddle)                    | `paddle`           | `yolov8n-seg_paddle_model/`   | ✅         | `imgsz`                                             |
| [ncnn](https://github.com/Tencent/ncnn)                            | `ncnn`             | `yolov8n-seg_ncnn_model/`     | ✅         | `imgsz`, `half`                                     |

Consulta todos los detalles del modo `export` en la página de [Exportación](https://docs.ultralytics.com/modes/export/).
