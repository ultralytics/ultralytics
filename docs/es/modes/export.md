---
comments: true
description: Guía paso a paso sobre cómo exportar sus modelos YOLOv8 a varios formatos como ONNX, TensorRT, CoreML y más para su despliegue. ¡Explora ahora!.
keywords: YOLO, YOLOv8, Ultralytics, Exportación de modelos, ONNX, TensorRT, CoreML, TensorFlow SavedModel, OpenVINO, PyTorch, exportar modelo
---

# Exportación de Modelos con Ultralytics YOLO

<img width="1024" src="https://github.com/ultralytics/assets/raw/main/yolov8/banner-integrations.png" alt="Ecosistema de Ultralytics YOLO e integraciones">

## Introducción

El objetivo final de entrenar un modelo es desplegarlo para aplicaciones en el mundo real. El modo exportación en Ultralytics YOLOv8 ofrece una gama versátil de opciones para exportar tu modelo entrenado a diferentes formatos, haciéndolo desplegable en varias plataformas y dispositivos. Esta guía integral pretende guiarte a través de los matices de la exportación de modelos, mostrando cómo lograr la máxima compatibilidad y rendimiento.

<p align="center">
  <br>
  <iframe width="720" height="405" src="https://www.youtube.com/embed/WbomGeoOT_k?si=aGmuyooWftA0ue9X"
    title="Reproductor de vídeo de YouTube" frameborder="0"
    allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share"
    allowfullscreen>
  </iframe>
  <br>
  <strong>Ver:</strong> Cómo Exportar un Modelo Entrenado Personalizado de Ultralytics YOLOv8 y Ejecutar Inferencia en Vivo en la Webcam.
</p>

## ¿Por Qué Elegir el Modo Exportación de YOLOv8?

- **Versatilidad:** Exporta a múltiples formatos incluyendo ONNX, TensorRT, CoreML y más.
- **Rendimiento:** Acelera hasta 5 veces la velocidad en GPU con TensorRT y 3 veces en CPU con ONNX o OpenVINO.
- **Compatibilidad:** Hacer que tu modelo sea universalmente desplegable en numerosos entornos de hardware y software.
- **Facilidad de Uso:** Interfaz de línea de comandos simple y API de Python para una exportación de modelos rápida y sencilla.

### Características Clave del Modo de Exportación

Aquí tienes algunas de las funcionalidades destacadas:

- **Exportación con Un Solo Clic:** Comandos simples para exportar a diferentes formatos.
- **Exportación por Lotes:** Exporta modelos capaces de inferencia por lotes.
- **Inferencia Optimizada:** Los modelos exportados están optimizados para tiempos de inferencia más rápidos.
- **Vídeos Tutoriales:** Guías y tutoriales en profundidad para una experiencia de exportación fluida.

!!! Tip "Consejo"

    * Exporta a ONNX u OpenVINO para acelerar la CPU hasta 3 veces.
    * Exporta a TensorRT para acelerar la GPU hasta 5 veces.

## Ejemplos de Uso

Exporta un modelo YOLOv8n a un formato diferente como ONNX o TensorRT. Consulta la sección Argumentos más abajo para una lista completa de argumentos de exportación.

!!! Example "Ejemplo"

    === "Python"

        ```python
        from ultralytics import YOLO

        # Carga un modelo
        model = YOLO('yolov8n.pt')  # carga un modelo oficial
        model = YOLO('path/to/best.pt')  # carga un modelo entrenado personalizado

        # Exporta el modelo
        model.export(format='onnx')
        ```
    === "CLI"

        ```bash
        yolo export model=yolov8n.pt format=onnx  # exporta modelo oficial
        yolo export model=path/to/best.pt format=onnx  # exporta modelo entrenado personalizado
        ```

## Argumentos

Los ajustes de exportación para modelos YOLO se refieren a las diversas configuraciones y opciones utilizadas para guardar o exportar el modelo para su uso en otros entornos o plataformas. Estos ajustes pueden afectar el rendimiento del modelo, su tamaño y su compatibilidad con diferentes sistemas. Algunos ajustes comunes de exportación de YOLO incluyen el formato del archivo del modelo exportado (p. ej., ONNX, TensorFlow SavedModel), el dispositivo en el que se ejecutará el modelo (p. ej., CPU, GPU) y la presencia de características adicionales como máscaras o múltiples etiquetas por caja. Otros factores que pueden afectar el proceso de exportación incluyen la tarea específica para la que se está utilizando el modelo y los requisitos o limitaciones del entorno o plataforma objetivo. Es importante considerar y configurar cuidadosamente estos ajustes para asegurar que el modelo exportado está optimizado para el caso de uso previsto y se pueda utilizar eficazmente en el entorno objetivo.

| Llave       | Valor           | Descripción                                                     |
|-------------|-----------------|-----------------------------------------------------------------|
| `format`    | `'torchscript'` | formato al que exportar                                         |
| `imgsz`     | `640`           | tamaño de imagen como escalar o lista (h, w), p. ej. (640, 480) |
| `keras`     | `False`         | usu Keras para la exportación de TF SavedModel                  |
| `optimize`  | `False`         | TorchScript: optimizar para móvil                               |
| `half`      | `False`         | cuantificación FP16                                             |
| `int8`      | `False`         | cuantificación INT8                                             |
| `dynamic`   | `False`         | ONNX/TensorRT: ejes dinámicos                                   |
| `simplify`  | `False`         | ONNX/TensorRT: simplificar modelo                               |
| `opset`     | `None`          | ONNX: versión de opset (opcional, por defecto la más reciente)  |
| `workspace` | `4`             | TensorRT: tamaño del espacio de trabajo (GB)                    |
| `nms`       | `False`         | CoreML: añadir NMS                                              |

## Formatos de Exportación

Los formatos de exportación disponibles de YOLOv8 están en la tabla a continuación. Puedes exportar a cualquier formato usando el argumento `format`, por ejemplo, `format='onnx'` o `format='engine'`.

| Formato                                                            | Argumento `format` | Modelo                    | Metadatos | Argumentos                                          |
|--------------------------------------------------------------------|--------------------|---------------------------|-----------|-----------------------------------------------------|
| [PyTorch](https://pytorch.org/)                                    | -                  | `yolov8n.pt`              | ✅         | -                                                   |
| [TorchScript](https://pytorch.org/docs/stable/jit.html)            | `torchscript`      | `yolov8n.torchscript`     | ✅         | `imgsz`, `optimize`                                 |
| [ONNX](https://onnx.ai/)                                           | `onnx`             | `yolov8n.onnx`            | ✅         | `imgsz`, `half`, `dynamic`, `simplify`, `opset`     |
| [OpenVINO](https://docs.openvino.ai/latest/index.html)             | `openvino`         | `yolov8n_openvino_model/` | ✅         | `imgsz`, `half`                                     |
| [TensorRT](https://developer.nvidia.com/tensorrt)                  | `engine`           | `yolov8n.engine`          | ✅         | `imgsz`, `half`, `dynamic`, `simplify`, `workspace` |
| [CoreML](https://github.com/apple/coremltools)                     | `coreml`           | `yolov8n.mlpackage`       | ✅         | `imgsz`, `half`, `int8`, `nms`                      |
| [TF SavedModel](https://www.tensorflow.org/guide/saved_model)      | `saved_model`      | `yolov8n_saved_model/`    | ✅         | `imgsz`, `keras`                                    |
| [TF GraphDef](https://www.tensorflow.org/api_docs/python/tf/Graph) | `pb`               | `yolov8n.pb`              | ❌         | `imgsz`                                             |
| [TF Lite](https://www.tensorflow.org/lite)                         | `tflite`           | `yolov8n.tflite`          | ✅         | `imgsz`, `half`, `int8`                             |
| [TF Edge TPU](https://coral.ai/docs/edgetpu/models-intro/)         | `edgetpu`          | `yolov8n_edgetpu.tflite`  | ✅         | `imgsz`                                             |
| [TF.js](https://www.tensorflow.org/js)                             | `tfjs`             | `yolov8n_web_model/`      | ✅         | `imgsz`                                             |
| [PaddlePaddle](https://github.com/PaddlePaddle)                    | `paddle`           | `yolov8n_paddle_model/`   | ✅         | `imgsz`                                             |
| [ncnn](https://github.com/Tencent/ncnn)                            | `ncnn`             | `yolov8n_ncnn_model/`     | ✅         | `imgsz`, `half`                                     |
