---
comments: true
description: Aprenda sobre las tareas fundamentales de visión por computadora que YOLOv8 puede realizar, incluyendo detección, segmentación, clasificación y estimación de pose. Comprenda sus usos en sus proyectos de IA.
keywords: Ultralytics, YOLOv8, Detección, Segmentación, Clasificación, Estimación de Pose, Marco de IA, Tareas de Visión por Computadora
---

# Tareas de Ultralytics YOLOv8

<br>
<img width="1024" src="https://raw.githubusercontent.com/ultralytics/assets/main/im/banner-tasks.png" alt="Tareas soportadas por Ultralytics YOLO">

YOLOv8 es un marco de trabajo de IA que soporta múltiples **tareas** de visión por computadora. El marco puede usarse para realizar [detección](detect.md), [segmentación](segment.md), [clasificación](classify.md) y estimación de [pose](pose.md). Cada una de estas tareas tiene un objetivo y caso de uso diferente.

!!! Note "Nota"

    🚧 Nuestra documentación multilenguaje está actualmente en construcción y estamos trabajando arduamente para mejorarla. ¡Gracias por su paciencia! 🙏

<p align="center">
  <br>
  <iframe width="720" height="405" src="https://www.youtube.com/embed/NAs-cfq9BDw"
    title="Reproductor de video YouTube" frameborder="0"
    allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share"
    allowfullscreen>
  </iframe>
  <br>
  <strong>Mire:</strong> Explore las Tareas de Ultralytics YOLO: Detección de Objetos, Segmentación, Seguimiento y Estimación de Pose.
</p>

## [Detección](detect.md)

La detección es la tarea principal soportada por YOLOv8. Implica detectar objetos en una imagen o cuadro de video y dibujar cuadros delimitadores alrededor de ellos. Los objetos detectados se clasifican en diferentes categorías basadas en sus características. YOLOv8 puede detectar múltiples objetos en una sola imagen o cuadro de video con alta precisión y velocidad.

[Ejemplos de Detección](detect.md){ .md-button }

## [Segmentación](segment.md)

La segmentación es una tarea que implica segmentar una imagen en diferentes regiones basadas en el contenido de la imagen. A cada región se le asigna una etiqueta basada en su contenido. Esta tarea es útil en aplicaciones tales como segmentación de imágenes y imágenes médicas. YOLOv8 utiliza una variante de la arquitectura U-Net para realizar la segmentación.

[Ejemplos de Segmentación](segment.md){ .md-button }

## [Clasificación](classify.md)

La clasificación es una tarea que implica clasificar una imagen en diferentes categorías. YOLOv8 puede usarse para clasificar imágenes basadas en su contenido. Utiliza una variante de la arquitectura EfficientNet para realizar la clasificación.

[Ejemplos de Clasificación](classify.md){ .md-button }

## [Pose](pose.md)

La detección de pose/puntos clave es una tarea que implica detectar puntos específicos en una imagen o cuadro de video. Estos puntos se conocen como puntos clave y se utilizan para rastrear el movimiento o la estimación de la pose. YOLOv8 puede detectar puntos clave en una imagen o cuadro de video con alta precisión y velocidad.

[Ejemplos de Pose](pose.md){ .md-button }

## Conclusión

YOLOv8 soporta múltiples tareas, incluyendo detección, segmentación, clasificación y detección de puntos clave. Cada una de estas tareas tiene diferentes objetivos y casos de uso. Al entender las diferencias entre estas tareas, puede elegir la tarea adecuada para su aplicación de visión por computadora.
