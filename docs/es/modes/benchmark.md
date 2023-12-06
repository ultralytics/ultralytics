---
comments: true
description: Aprenda cómo perfilar la velocidad y exactitud de YOLOv8 en varios formatos de exportación; obtenga perspectivas sobre las métricas mAP50-95, accuracy_top5 y más.
keywords: Ultralytics, YOLOv8, benchmarking, perfilado de velocidad, perfilado de exactitud, mAP50-95, accuracy_top5, ONNX, OpenVINO, TensorRT, formatos de exportación YOLO
---

# Model Benchmarking con Ultralytics YOLO

<img width="1024" src="https://github.com/ultralytics/assets/raw/main/yolov8/banner-integrations.png" alt="Ecosistema e integraciones de Ultralytics YOLO">

## Introducción

Una vez que su modelo está entrenado y validado, el siguiente paso lógico es evaluar su rendimiento en varios escenarios del mundo real. El modo benchmark en Ultralytics YOLOv8 cumple con este propósito proporcionando un marco sólido para valorar la velocidad y exactitud de su modelo a través de una gama de formatos de exportación.

## ¿Por Qué Es Crucial el Benchmarking?

- **Decisiones Informadas:** Obtenga perspectivas sobre el equilibrio entre velocidad y precisión.
- **Asignación de Recursos:** Entienda cómo diferentes formatos de exportación se desempeñan en diferentes hardware.
- **Optimización:** Aprenda cuál formato de exportación ofrece el mejor rendimiento para su caso de uso específico.
- **Eficiencia de Costo:** Haga un uso más eficiente de los recursos de hardware basado en los resultados del benchmark.

### Métricas Clave en el Modo Benchmark

- **mAP50-95:** Para detección de objetos, segmentación y estimación de pose.
- **accuracy_top5:** Para clasificación de imágenes.
- **Tiempo de Inferencia:** Tiempo tomado para cada imagen en milisegundos.

### Formatos de Exportación Soportados

- **ONNX:** Para un rendimiento óptimo de CPU
- **TensorRT:** Para la máxima eficiencia de GPU
- **OpenVINO:** Para la optimización en hardware de Intel
- **CoreML, TensorFlow SavedModel y Más:** Para necesidades de despliegue diversas.

!!! Tip "Consejo"

    * Exporte a ONNX o OpenVINO para acelerar la velocidad de CPU hasta 3 veces.
    * Exporte a TensorRT para acelerar la velocidad de GPU hasta 5 veces.

## Ejemplos de Uso

Ejecute benchmarks de YOLOv8n en todos los formatos de exportación soportados incluyendo ONNX, TensorRT, etc. Vea la sección de Argumentos a continuación para una lista completa de argumentos de exportación.

!!! Example "Ejemplo"

    === "Python"

        ```python
        from ultralytics.utils.benchmarks import benchmark

        # Benchmark en GPU
        benchmark(model='yolov8n.pt', data='coco8.yaml', imgsz=640, half=False, device=0)
        ```
    === "CLI"

        ```bash
        yolo benchmark model=yolov8n.pt data='coco8.yaml' imgsz=640 half=False device=0
        ```

## Argumentos

Argumentos como `model`, `data`, `imgsz`, `half`, `device`, y `verbose` proporcionan a los usuarios la flexibilidad de ajustar los benchmarks a sus necesidades específicas y comparar el rendimiento de diferentes formatos de exportación con facilidad.

| Clave     | Valor   | Descripción                                                                                              |
|-----------|---------|----------------------------------------------------------------------------------------------------------|
| `model`   | `None`  | ruta al archivo del modelo, es decir, yolov8n.pt, yolov8n.yaml                                           |
| `data`    | `None`  | ruta a YAML que referencia el conjunto de datos de benchmarking (bajo la etiqueta `val`)                 |
| `imgsz`   | `640`   | tamaño de imagen como escalar o lista (h, w), es decir, (640, 480)                                       |
| `half`    | `False` | cuantificación FP16                                                                                      |
| `int8`    | `False` | cuantificación INT8                                                                                      |
| `device`  | `None`  | dispositivo en el que se ejecutará, es decir, dispositivo cuda=0 o dispositivo=0,1,2,3 o dispositivo=cpu |
| `verbose` | `False` | no continuar en caso de error (bool), o umbral de piso de valor (float)                                  |

## Formatos de Exportación

Los benchmarks intentarán ejecutarse automáticamente en todos los posibles formatos de exportación a continuación.

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

Vea los detalles completos de `export` en la página [Export](https://docs.ultralytics.com/modes/export/).
