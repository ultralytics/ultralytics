---
comments: true
description: Explore la amplia gama de modelos de la familia YOLO, SAM, MobileSAM, FastSAM, YOLO-NAS y RT-DETR soportados por Ultralytics. Comienza con ejemplos para el uso tanto de CLI como de Python.
keywords: Ultralytics, documentación, YOLO, SAM, MobileSAM, FastSAM, YOLO-NAS, RT-DETR, modelos, arquitecturas, Python, CLI
---

# Modelos soportados por Ultralytics

¡Bienvenido a la documentación de modelos de Ultralytics! Ofrecemos soporte para una amplia gama de modelos, cada uno adaptado a tareas específicas como [detección de objetos](../tasks/detect.md), [segmentación de instancias](../tasks/segment.md), [clasificación de imágenes](../tasks/classify.md), [estimación de posturas](../tasks/pose.md), y [seguimiento de múltiples objetos](../modes/track.md). Si estás interesado en contribuir con tu arquitectura de modelo a Ultralytics, consulta nuestra [Guía de Contribución](../../help/contributing.md).

!!! Note "Nota"

    🚧 Estamos trabajando arduamente para mejorar nuestra documentación en varios idiomas actualmente en construcción. ¡Gracias por tu paciencia! 🙏

## Modelos destacados

Aquí están algunos de los modelos clave soportados:

1. **[YOLOv3](yolov3.md)**: La tercera iteración de la familia de modelos YOLO, original de Joseph Redmon, conocida por su capacidad de detección de objetos en tiempo real eficientemente.
2. **[YOLOv4](yolov4.md)**: Una actualización nativa de darknet para YOLOv3, lanzada por Alexey Bochkovskiy en 2020.
3. **[YOLOv5](yolov5.md)**: Una versión mejorada de la arquitectura YOLO por Ultralytics, ofreciendo un mejor rendimiento y compromiso de velocidad comparado con versiones anteriores.
4. **[YOLOv6](yolov6.md)**: Lanzado por [Meituan](https://about.meituan.com/) en 2022, y utilizado en muchos de los robots de entrega autónomos de la compañía.
5. **[YOLOv7](yolov7.md)**: Modelos YOLO actualizados lanzados en 2022 por los autores de YOLOv4.
6. **[YOLOv8](yolov8.md) NUEVO 🚀**: La última versión de la familia YOLO, con capacidades mejoradas como segmentación de instancias, estimación de posturas/puntos clave y clasificación.
7. **[Modelo Segment Anything (SAM)](sam.md)**: Modelo Segment Anything (SAM) de Meta.
8. **[Mobile Segment Anything Model (MobileSAM)](mobile-sam.md)**: MobileSAM para aplicaciones móviles, por la Universidad de Kyung Hee.
9. **[Fast Segment Anything Model (FastSAM)](fast-sam.md)**: FastSAM por el Grupo de Análisis de Imagen y Video, Instituto de Automatización, Academia China de Ciencias.
10. **[YOLO-NAS](yolo-nas.md)**: Modelos YOLO de Búsqueda de Arquitectura Neural (NAS).
11. **[Transformadores de Detección en Tiempo Real (RT-DETR)](rtdetr.md)**: Modelos de Transformador de Detección en Tiempo Real (RT-DETR) de Baidu's PaddlePaddle.

<p align="center">
  <br>
  <iframe width="720" height="405" src="https://www.youtube.com/embed/MWq1UxqTClU?si=nHAW-lYDzrz68jR0"
    title="Reproductor de video de YouTube" frameborder="0"
    allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share"
    allowfullscreen>
  </iframe>
  <br>
  <strong>Mira:</strong> Ejecuta modelos YOLO de Ultralytics en solo unas pocas líneas de código.
</p>

## Empezando: Ejemplos de Uso

Este ejemplo proporciona ejemplos simples de entrenamiento e inferencia YOLO. Para la documentación completa de estos y otros [modos](../modes/index.md), consulta las páginas de documentación de [Predict](../modes/predict.md), [Train](../modes/train.md), [Val](../modes/val.md) y [Export](../modes/export.md).

Nota que el siguiente ejemplo es para los modelos YOLOv8 [Detect](../tasks/detect.md) para detección de objetos. Para tareas adicionales soportadas, consulta la documentación de [Segment](../tasks/segment.md), [Classify](../tasks/classify.md) y [Pose](../tasks/pose.md).

!!! Example "Ejemplo"

    === "Python"

        Los modelos pre-entrenados `*.pt` de PyTorch así como los archivos de configuración `*.yaml` se pueden pasar a las clases `YOLO()`, `SAM()`, `NAS()` y `RTDETR()` para crear una instancia de modelo en Python:

        ```python
        from ultralytics import YOLO

        # Cargar un modelo YOLOv8n preentrenado en COCO
        model = YOLO('yolov8n.pt')

        # Mostrar información del modelo (opcional)
        model.info()

        # Entrenar el modelo en el conjunto de datos de ejemplo COCO8 durante 100 épocas
        results = model.train(data='coco8.yaml', epochs=100, imgsz=640)

        # Ejecutar inferencia con el modelo YOLOv8n en la imagen 'bus.jpg'
        results = model('path/to/bus.jpg')
        ```

    === "CLI"

        Los comandos CLI están disponibles para ejecutar directamente los modelos:

        ```bash
        # Cargar un modelo YOLOv8n preentrenado en COCO y entrenarlo en el conjunto de datos de ejemplo COCO8 durante 100 épocas
        yolo train model=yolov8n.pt data=coco8.yaml epochs=100 imgsz=640

        # Cargar un modelo YOLOv8n preentrenado en COCO y ejecutar inferencia en la imagen 'bus.jpg'
        yolo predict model=yolov8n.pt source=path/to/bus.jpg
        ```

## Contribuir con Nuevos Modelos

¿Interesado en contribuir con tu modelo a Ultralytics? ¡Genial! Siempre estamos abiertos a expandir nuestro portafolio de modelos.

1. **Haz un Fork del Repositorio**: Comienza haciendo un fork del [repositorio de GitHub de Ultralytics](https://github.com/ultralytics/ultralytics).

2. **Clona tu Fork**: Clona tu fork a tu máquina local y crea una nueva rama para trabajar.

3. **Implementa tu Modelo**: Añade tu modelo siguiendo los estándares de codificación y directrices proporcionadas en nuestra [Guía de Contribución](../../help/contributing.md).

4. **Prueba Rigurosamente**: Asegúrate de probar tu modelo rigurosamente, tanto de forma aislada como parte del proceso.

5. **Crea un Pull Request**: Una vez que estés satisfecho con tu modelo, crea un pull request al repositorio principal para revisión.

6. **Revisión de Código y Fusión**: Después de la revisión, si tu modelo cumple con nuestros criterios, será fusionado al repositorio principal.

Para pasos detallados, consulta nuestra [Guía de Contribución](../../help/contributing.md).
