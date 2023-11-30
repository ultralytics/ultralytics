---
comments: true
description: Guía para validar modelos YOLOv8. Aprenda a evaluar el rendimiento de sus modelos YOLO utilizando configuraciones y métricas de validación con ejemplos en Python y CLI.
keywords: Ultralytics, Documentación YOLO, YOLOv8, validación, evaluación de modelos, hiperparámetros, precisión, métricas, Python, CLI
---

# Validación de modelos con Ultralytics YOLO

<img width="1024" src="https://github.com/ultralytics/assets/raw/main/yolov8/banner-integrations.png" alt="Ecosistema e integraciones de Ultralytics YOLO">

## Introducción

La validación es un paso crítico en el flujo de trabajo de aprendizaje automático, permitiéndole evaluar la calidad de sus modelos entrenados. El modo Val en Ultralytics YOLOv8 proporciona un robusto conjunto de herramientas y métricas para evaluar el rendimiento de sus modelos de detección de objetos. Esta guía sirve como un recurso completo para comprender cómo utilizar efectivamente el modo Val para asegurar que sus modelos sean precisos y confiables.

## ¿Por qué validar con Ultralytics YOLO?

Estas son las ventajas de usar el modo Val de YOLOv8:

- **Precisión:** Obtenga métricas precisas como mAP50, mAP75 y mAP50-95 para evaluar de manera integral su modelo.
- **Comodidad:** Utilice funciones integradas que recuerdan los ajustes de entrenamiento, simplificando el proceso de validación.
- **Flexibilidad:** Valide su modelo con el mismo conjunto de datos o diferentes conjuntos de datos y tamaños de imagen.
- **Ajuste de Hiperparámetros:** Use las métricas de validación para ajustar su modelo y mejorar el rendimiento.

### Características principales del modo Val

Estas son las funcionalidades notables ofrecidas por el modo Val de YOLOv8:

- **Configuraciones Automatizadas:** Los modelos recuerdan sus configuraciones de entrenamiento para una validación sencilla.
- **Soporte de Múltiples Métricas:** Evalúe su modelo basado en una gama de métricas de precisión.
- **CLI y API de Python:** Elija entre la interfaz de línea de comandos o API de Python basada en su preferencia para validación.
- **Compatibilidad de Datos:** Funciona sin problemas con conjuntos de datos utilizados durante la fase de entrenamiento así como con conjuntos de datos personalizados.

!!! Tip "Consejo"

    * Los modelos YOLOv8 recuerdan automáticamente sus ajustes de entrenamiento, así que puede validar un modelo en el mismo tamaño de imagen y en el conjunto de datos original fácilmente con solo `yolo val model=yolov8n.pt` o `model('yolov8n.pt').val()`

## Ejemplos de Uso

Valide la precisión del modelo YOLOv8n entrenado en el conjunto de datos COCO128. No es necesario pasar ningún argumento ya que el `modelo` retiene sus `datos` de entrenamiento y argumentos como atributos del modelo. Vea la sección de Argumentos a continuación para una lista completa de argumentos de exportación.

!!! Example "Ejemplo"

    === "Python"

        ```python
        from ultralytics import YOLO

        # Cargar un modelo
        model = YOLO('yolov8n.pt')  # cargar un modelo oficial
        model = YOLO('ruta/a/best.pt')  # cargar un modelo personalizado

        # Validar el modelo
        metrics = model.val()  # no se necesitan argumentos, el conjunto de datos y ajustes se recuerdan
        metrics.box.map    # map50-95
        metrics.box.map50  # map50
        metrics.box.map75  # map75
        metrics.box.maps   # una lista que contiene map50-95 de cada categoría
        ```
    === "CLI"

        ```bash
        yolo detect val model=yolov8n.pt  # val model oficial
        yolo detect val model=ruta/a/best.pt  # val model personalizado
        ```

## Argumentos

Los ajustes de validación para modelos YOLO se refieren a los diversos hiperparámetros y configuraciones utilizados para evaluar el rendimiento del modelo en un conjunto de datos de validación. Estos ajustes pueden afectar el rendimiento, la velocidad y la precisión del modelo. Algunos ajustes comunes de validación YOLO incluyen el tamaño del lote, la frecuencia con la que se realiza la validación durante el entrenamiento y las métricas utilizadas para evaluar el rendimiento del modelo. Otros factores que pueden afectar el proceso de validación incluyen el tamaño y la composición del conjunto de datos de validación y la tarea específica para la que se utiliza el modelo. Es importante ajustar y experimentar cuidadosamente con estos ajustes para asegurarse de que el modelo esté funcionando bien en el conjunto de datos de validación y para detectar y prevenir el sobreajuste.

| Clave         | Valor   | Descripción                                                                                       |
|---------------|---------|---------------------------------------------------------------------------------------------------|
| `data`        | `None`  | ruta al archivo de datos, por ejemplo coco128.yaml                                                |
| `imgsz`       | `640`   | tamaño de las imágenes de entrada como entero                                                     |
| `batch`       | `16`    | número de imágenes por lote (-1 para AutoBatch)                                                   |
| `save_json`   | `False` | guardar resultados en archivo JSON                                                                |
| `save_hybrid` | `False` | guardar versión híbrida de las etiquetas (etiquetas + predicciones adicionales)                   |
| `conf`        | `0.001` | umbral de confianza del objeto para detección                                                     |
| `iou`         | `0.6`   | umbral de Intersección sobre Unión (IoU) para NMS                                                 |
| `max_det`     | `300`   | número máximo de detecciones por imagen                                                           |
| `half`        | `True`  | usar precisión de punto flotante de media preción (FP16)                                          |
| `device`      | `None`  | dispositivo en el que se ejecuta, por ejemplo dispositivo cuda=0/1/2/3 o dispositivo=cpu          |
| `dnn`         | `False` | utilizar OpenCV DNN para inferencia ONNX                                                          |
| `plots`       | `False` | mostrar gráficos durante el entrenamiento                                                         |
| `rect`        | `False` | val rectangular con cada lote compilado para el mínimo relleno                                    |
| `split`       | `val`   | división del conjunto de datos a utilizar para la validación, por ejemplo 'val', 'test' o 'train' |
|
