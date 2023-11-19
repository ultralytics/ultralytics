---
comments: true
description: Explora la amplia gama de modelos de la familia YOLO, SAM, MobileSAM, FastSAM, YOLO-NAS y RT-DETR compatibles con Ultralytics. Comienza con ejemplos de uso tanto para CLI como para Python.
keywords: Ultralytics, documentación, YOLO, SAM, MobileSAM, FastSAM, YOLO-NAS, RT-DETR, modelos, arquitecturas, Python, CLI
---

# Modelos soportados por Ultralytics

¡Bienvenido a la documentación de modelos de Ultralytics! Ofrecemos soporte para una amplia gama de modelos, cada uno adaptado a tareas específicas como [detección de objetos](../tasks/detect.md), [segmentación de instancias](../tasks/segment.md), [clasificación de imágenes](../tasks/classify.md), [estimación de postura](../tasks/pose.md) y [seguimiento de múltiples objetos](../modes/track.md). Si estás interesado en contribuir con tu arquitectura de modelo a Ultralytics, consulta nuestra [Guía de Contribución](../../help/contributing.md).

!!! Note "Nota"

    🚧 Nuestra documentación en varios idiomas está actualmente en construcción y estamos trabajando arduamente para mejorarla. ¡Gracias por tu paciencia! 🙏

## Modelos Destacados

Aquí tienes algunos de los modelos clave soportados:

1. **[YOLOv3](../../models/yolov3.md)**: La tercera iteración de la familia de modelos YOLO, originalmente creada por Joseph Redmon, conocida por su capacidad de detección de objetos en tiempo real de manera eficiente.
2. **[YOLOv4](../../models/yolov4.md)**: Una actualización para la red oscura de YOLOv3, lanzada por Alexey Bochkovskiy en 2020.
3. **[YOLOv5](../../models/yolov5.md)**: Una versión mejorada de la arquitectura YOLO por Ultralytics, que ofrece mejores compensaciones de rendimiento y velocidad en comparación con versiones anteriores.
4. **[YOLOv6](../../models/yolov6.md)**: Lanzado por [Meituan](https://about.meituan.com/) en 2022, y utilizado en muchos de los robots autónomos de entrega de la compañía.
5. **[YOLOv7](../../models/yolov7.md)**: Modelos YOLO actualizados lanzados en 2022 por los autores de YOLOv4.
6. **[YOLOv8](../../models/yolov8.md)**: La última versión de la familia YOLO, que presenta capacidades mejoradas como segmentación de instancias, estimación de postura/puntos clave y clasificación.
7. **[Modelo de Segmentación de Cualquier Cosa (SAM)](../../models/sam.md)**: El Modelo de Segmentación de Cualquier Cosa (SAM) de Meta.
8. **[Modelo de Segmentación de Cualquier Cosa Móvil (MobileSAM)](../../models/mobile-sam.md)**: MobileSAM para aplicaciones móviles, por la Universidad Kyung Hee.
9. **[Modelo de Segmentación de Cualquier Cosa Rápida (FastSAM)](../../models/fast-sam.md)**: FastSAM del Grupo de Análisis de Imágenes y Video, Instituto de Automatización, Academia China de Ciencias.
10. **[YOLO-NAS](../../models/yolo-nas.md)**: Modelos de Búsqueda de Arquitectura Neural de YOLO (NAS).
11. **[Transformadores de Detección en Tiempo Real (RT-DETR)](../../models/rtdetr.md)**: Modelos de Transformadores de Detección en Tiempo Real (RT-DETR) de Baidu PaddlePaddle.

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

## Comenzando: Ejemplos de Uso

!!! Example "Ejemplo"

    === "Python"

        Los modelos preentrenados en PyTorch `*.pt` así como los archivos de configuración `*.yaml` pueden pasarse a las clases `YOLO()`, `SAM()`, `NAS()` y `RTDETR()` para crear una instancia de modelo en Python:

        ```python
        from ultralytics import YOLO

        # Cargar un modelo YOLOv8n preentrenado en COCO
        modelo = YOLO('yolov8n.pt')

        # Mostrar información del modelo (opcional)
        modelo.info()

        # Entrenar el modelo en el conjunto de datos de ejemplo COCO8 durante 100 épocas
        resultados = modelo.train(data='coco8.yaml', epochs=100, imgsz=640)

        # Ejecutar inferencia con el modelo YOLOv8n en la imagen 'bus.jpg'
        resultados = modelo('path/to/bus.jpg')
        ```

    === "CLI"

        Comandos CLI están disponibles para ejecutar directamente los modelos:

        ```bash
        # Cargar un modelo YOLOv8n preentrenado en COCO y entrenarlo en el conjunto de datos de ejemplo COCO8 durante 100 épocas
        yolo train model=yolov8n.pt data=coco8.yaml epochs=100 imgsz=640

        # Cargar un modelo YOLOv8n preentrenado en COCO y ejecutar inferencia en la imagen 'bus.jpg'
        yolo predict model=yolov8n.pt source=path/to/bus.jpg
        ```

## Contribuyendo con Nuevos Modelos

¿Interesado en contribuir con tu modelo a Ultralytics? ¡Genial! Siempre estamos abiertos a expandir nuestro portafolio de modelos.

1. **Haz un Fork del Repositorio**: Comienza haciendo un fork del [repositorio de GitHub de Ultralytics](https://github.com/ultralytics/ultralytics).

2. **Clona tu Fork**: Clona tu fork en tu máquina local y crea una nueva rama para trabajar.

3. **Implementa tu Modelo**: Añade tu modelo siguiendo los estándares y guías de codificación proporcionados en nuestra [Guía de Contribución](../../help/contributing.md).

4. **Prueba a Fondo**: Asegúrate de probar tu modelo rigurosamente, tanto de manera aislada como parte del pipeline.

5. **Crea un Pull Request**: Una vez que estés satisfecho con tu modelo, crea un pull request al repositorio principal para su revisión.

6. **Revisión de Código y Fusión**: Después de la revisión, si tu modelo cumple con nuestros criterios, se fusionará en el repositorio principal.

Consulta nuestra [Guía de Contribución](../../help/contributing.md) para los pasos detallados.
