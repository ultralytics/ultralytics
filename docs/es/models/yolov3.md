---
comments: true
description: Obtén una descripción general de YOLOv3, YOLOv3-Ultralytics y YOLOv3u. Aprende sobre sus características clave, uso y tareas admitidas para la detección de objetos.
keywords: YOLOv3, YOLOv3-Ultralytics, YOLOv3u, Detección de objetos, Inferencia, Entrenamiento, Ultralytics
---

# YOLOv3, YOLOv3-Ultralytics y YOLOv3u

## Descripción general

Este documento presenta una descripción general de tres modelos de detección de objetos estrechamente relacionados, conocidos como [YOLOv3](https://pjreddie.com/darknet/yolo/), [YOLOv3-Ultralytics](https://github.com/ultralytics/yolov3) y [YOLOv3u](https://github.com/ultralytics/ultralytics).

1. **YOLOv3:** Esta es la tercera versión del algoritmo de detección de objetos You Only Look Once (YOLO). Originalmente desarrollado por Joseph Redmon, YOLOv3 mejoró a sus predecesores al introducir características como predicciones multiescala y tres tamaños diferentes de núcleos de detección.

2. **YOLOv3-Ultralytics:** Esta es la implementación de YOLOv3 realizada por Ultralytics. Reproduce la arquitectura original de YOLOv3 y ofrece funcionalidades adicionales, como soporte para más modelos pre-entrenados y opciones de personalización más fáciles.

3. **YOLOv3u:** Esta es una versión actualizada de YOLOv3-Ultralytics que incorpora la cabeza dividida sin anclaje y sin objeto utilizada en los modelos YOLOv8. YOLOv3u mantiene la misma arquitectura de columna vertebral y cuello que YOLOv3, pero con la cabeza de detección actualizada de YOLOv8.

![Ultralytics YOLOv3](https://raw.githubusercontent.com/ultralytics/assets/main/yolov3/banner-yolov3.png)

## Características clave

- **YOLOv3:** Introdujo el uso de tres escalas diferentes para la detección, aprovechando tres tamaños diferentes de núcleos de detección: 13x13, 26x26 y 52x52. Esto mejoró significativamente la precisión de detección para objetos de diferentes tamaños. Además, YOLOv3 añadió características como predicciones con múltiples etiquetas para cada cuadro delimitador y una mejor red extractora de características.

- **YOLOv3-Ultralytics:** La implementación de Ultralytics de YOLOv3 proporciona el mismo rendimiento que el modelo original, pero cuenta con soporte adicional para más modelos pre-entrenados, métodos de entrenamiento adicionales y opciones de personalización más fáciles. Esto lo hace más versátil y fácil de usar para aplicaciones prácticas.

- **YOLOv3u:** Este modelo actualizado incorpora la cabeza dividida sin anclaje y sin objeto de YOLOv8. Al eliminar la necesidad de cajas de anclaje predefinidas y puntuaciones de objeto, este diseño de cabeza de detección puede mejorar la capacidad del modelo para detectar objetos de diferentes tamaños y formas. Esto hace que YOLOv3u sea más robusto y preciso para tareas de detección de objetos.

## Tareas y modos admitidos

La serie YOLOv3, que incluye YOLOv3, YOLOv3-Ultralytics y YOLOv3u, está diseñada específicamente para tareas de detección de objetos. Estos modelos son reconocidos por su eficacia en diversos escenarios del mundo real, equilibrando precisión y velocidad. Cada variante ofrece características y optimizaciones únicas, lo que los hace adecuados para una variedad de aplicaciones.

Los tres modelos admiten un conjunto completo de modos, asegurando versatilidad en diversas etapas del despliegue y desarrollo del modelo. Estos modos incluyen [Inferencia](../modes/predict.md), [Validación](../modes/val.md), [Entrenamiento](../modes/train.md) y [Exportación](../modes/export.md), proporcionando a los usuarios un conjunto completo de herramientas para una detección de objetos efectiva.

| Tipo de modelo     | Tareas admitidas                           | Inferencia | Validación | Entrenamiento | Exportación |
|--------------------|--------------------------------------------|------------|------------|---------------|-------------|
| YOLOv3             | [Detección de objetos](../tasks/detect.md) | ✅          | ✅          | ✅             | ✅           |
| YOLOv3-Ultralytics | [Detección de objetos](../tasks/detect.md) | ✅          | ✅          | ✅             | ✅           |
| YOLOv3u            | [Detección de objetos](../tasks/detect.md) | ✅          | ✅          | ✅             | ✅           |

Esta tabla proporciona una visión rápida de las capacidades de cada variante de YOLOv3, destacando su versatilidad y aptitud para diversas tareas y modos operativos en flujos de trabajo de detección de objetos.

## Ejemplos de uso

Este ejemplo proporciona ejemplos sencillos de entrenamiento e inferencia de YOLOv3. Para obtener documentación completa sobre estos y otros [modos](../modes/index.md), consulta las páginas de documentación de [Predict](../modes/predict.md), [Train](../modes/train.md), [Val](../modes/val.md) y [Export](../modes/export.md).

!!! Example "Ejemplo"

    === "Python"

        Los modelos pre-entrenados de PyTorch en archivos `*.pt`, así como los archivos de configuración `*.yaml`, se pueden pasar a la clase `YOLO()` para crear una instancia del modelo en Python:

        ```python
        from ultralytics import YOLO

        # Cargar un modelo YOLOv3n pre-entrenado en COCO
        model = YOLO('yolov3n.pt')

        # Mostrar información del modelo (opcional)
        model.info()

        # Entrenar el modelo en el conjunto de datos de ejemplo COCO8 durante 100 épocas
        results = model.train(data='coco8.yaml', epochs=100, imgsz=640)

        # Ejecutar inferencia con el modelo YOLOv3n en la imagen 'bus.jpg'
        results = model('path/to/bus.jpg')
        ```

    === "CLI"

        Hay comandos de CLI disponibles para ejecutar directamente los modelos:

        ```bash
        # Cargar un modelo YOLOv3n pre-entrenado en COCO y entrenarlo en el conjunto de datos de ejemplo COCO8 durante 100 épocas
        yolo train model=yolov3n.pt data=coco8.yaml epochs=100 imgsz=640

        # Cargar un modelo YOLOv3n pre-entrenado en COCO y ejecutar inferencia en la imagen 'bus.jpg'
        yolo predict model=yolov3n.pt source=path/to/bus.jpg
        ```

## Citaciones y agradecimientos

Si utilizas YOLOv3 en tu investigación, por favor, cita los artículos originales de YOLO y el repositorio de YOLOv3 de Ultralytics:

!!! Quote ""

    === "BibTeX"

        ```bibtex
        @article{redmon2018yolov3,
          title={YOLOv3: An Incremental Improvement},
          author={Redmon, Joseph and Farhadi, Ali},
          journal={arXiv preprint arXiv:1804.02767},
          year={2018}
        }
        ```

Gracias a Joseph Redmon y Ali Farhadi por desarrollar YOLOv3 original.
