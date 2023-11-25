---
comments: true
description: Desde el entrenamiento hasta el seguimiento, aprovecha al máximo YOLOv8 con Ultralytics. Obtén información y ejemplos para cada modo compatible incluyendo validación, exportación y evaluación comparativa.
keywords: Ultralytics, YOLOv8, Aprendizaje Automático, Detección de Objetos, Entrenamiento, Validación, Predicción, Exportación, Seguimiento, Benchmarking
---

# Modos de Ultralytics YOLOv8

<img width="1024" src="https://github.com/ultralytics/assets/raw/main/yolov8/banner-integrations.png" alt="Ecosistema Ultralytics YOLO e integraciones">

## Introducción

Ultralytics YOLOv8 no es solo otro modelo de detección de objetos; es un marco de trabajo versátil diseñado para cubrir todo el ciclo de vida de los modelos de aprendizaje automático, desde la ingesta de datos y el entrenamiento del modelo hasta la validación, implementación y seguimiento en el mundo real. Cada modo sirve para un propósito específico y está diseñado para ofrecerte la flexibilidad y eficiencia necesarias para diferentes tareas y casos de uso.

<p align="center">
  <br>
  <iframe width="720" height="405" src="https://www.youtube.com/embed/j8uQc0qB91s?si=dhnGKgqvs7nPgeaM"
    title="Reproductor de video de YouTube" frameborder="0"
    allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share"
    allowfullscreen>
  </iframe>
  <br>
  <strong>Mira:</strong> Tutorial de Modos Ultralytics: Entrenar, Validar, Predecir, Exportar y Hacer Benchmarking.
</p>

### Modos a Primera Vista

Comprender los diferentes **modos** que soporta Ultralytics YOLOv8 es crítico para sacar el máximo provecho a tus modelos:

- **Modo Entrenar (Train)**: Afina tu modelo en conjuntos de datos personalizados o pre-cargados.
- **Modo Validar (Val)**: Un punto de control post-entrenamiento para validar el rendimiento del modelo.
- **Modo Predecir (Predict)**: Libera el poder predictivo de tu modelo en datos del mundo real.
- **Modo Exportar (Export)**: Prepara tu modelo para la implementación en varios formatos.
- **Modo Seguir (Track)**: Extiende tu modelo de detección de objetos a aplicaciones de seguimiento en tiempo real.
- **Modo Benchmark (Benchmark)**: Analiza la velocidad y precisión de tu modelo en diversos entornos de implementación.

Esta guía completa tiene como objetivo proporcionarte una visión general y conocimientos prácticos de cada modo, ayudándote a aprovechar todo el potencial de YOLOv8.

## [Entrenar (Train)](train.md)

El modo Entrenar se utiliza para entrenar un modelo YOLOv8 en un conjunto de datos personalizado. En este modo, el modelo se entrena utilizando el conjunto de datos y los hiperparámetros especificados. El proceso de entrenamiento implica optimizar los parámetros del modelo para que pueda predecir con precisión las clases y ubicaciones de los objetos en una imagen.

[Ejemplos de Entrenamiento](train.md){ .md-button }

## [Validar (Val)](val.md)

El modo Validar se usa para validar un modelo YOLOv8 después de haber sido entrenado. En este modo, el modelo se evalúa en un conjunto de validación para medir su precisión y rendimiento de generalización. Este modo se puede usar para ajustar los hiperparámetros del modelo y mejorar su rendimiento.

[Ejemplos de Validación](val.md){ .md-button }

## [Predecir (Predict)](predict.md)

El modo Predecir se utiliza para realizar predicciones usando un modelo YOLOv8 entrenado en imágenes o videos nuevos. En este modo, el modelo se carga desde un archivo de punto de control, y el usuario puede proporcionar imágenes o videos para realizar inferencias. El modelo predice las clases y ubicaciones de los objetos en las imágenes o videos de entrada.

[Ejemplos de Predicción](predict.md){ .md-button }

## [Exportar (Export)](export.md)

El modo Exportar se utiliza para exportar un modelo YOLOv8 a un formato que se pueda usar para la implementación. En este modo, el modelo se convierte a un formato que puede ser utilizado por otras aplicaciones de software o dispositivos de hardware. Este modo es útil al implementar el modelo en entornos de producción.

[Ejemplos de Exportación](export.md){ .md-button }

## [Seguir (Track)](track.md)

El modo Seguir se usa para rastrear objetos en tiempo real utilizando un modelo YOLOv8. En este modo, el modelo se carga desde un archivo de punto de control, y el usuario puede proporcionar un flujo de video en vivo para realizar seguimiento de objetos en tiempo real. Este modo es útil para aplicaciones como sistemas de vigilancia o coches autónomos.

[Ejemplos de Seguimiento](track.md){ .md-button }

## [Benchmark (Benchmark)](benchmark.md)

El modo Benchmark se utiliza para perfilar la velocidad y precisión de varios formatos de exportación de YOLOv8. Los benchmarks proporcionan información sobre el tamaño del formato de exportación, sus métricas de `mAP50-95` (para detección de objetos, segmentación y pose) o métricas de `accuracy_top5` (para clasificación), y el tiempo de inferencia en milisegundos por imagen a través de varios formatos de exportación como ONNX, OpenVINO, TensorRT y otros. Esta información puede ayudar a los usuarios a elegir el formato de exportación óptimo para su caso de uso específico, basado en sus requerimientos de velocidad y precisión.

[Ejemplos de Benchmarking](benchmark.md){ .md-button }
