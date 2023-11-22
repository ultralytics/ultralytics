---
comments: true
description: Explora Meituan YOLOv6, un modelo de detección de objetos de última generación que logra un equilibrio entre velocidad y precisión. Sumérgete en características, modelos pre-entrenados y el uso de Python.
keywords: Meituan YOLOv6, detección de objetos, Ultralytics, documentación de YOLOv6, Concatenación Bidireccional, Entrenamiento con Anclas, modelos pre-entrenados, aplicaciones en tiempo real
---

# Meituan YOLOv6

## Visión general

[Meituan](https://about.meituan.com/) YOLOv6 es un detector de objetos de última generación que ofrece un notable equilibrio entre velocidad y precisión, lo que lo convierte en una opción popular para aplicaciones en tiempo real. Este modelo presenta varias mejoras notables en su arquitectura y esquema de entrenamiento, que incluyen la implementación de un módulo de Concatenación Bidireccional (BiC), una estrategia de entrenamiento con anclas (AAT) y un diseño de columna vertebral y cuello mejorado para lograr una precisión de última generación en el conjunto de datos COCO.

![Meituan YOLOv6](https://user-images.githubusercontent.com/26833433/240750495-4da954ce-8b3b-41c4-8afd-ddb74361d3c2.png)
![Ejemplo de imagen del modelo](https://user-images.githubusercontent.com/26833433/240750557-3e9ec4f0-0598-49a8-83ea-f33c91eb6d68.png)
**Visión general de YOLOv6.** Diagrama de la arquitectura del modelo que muestra los componentes de la red redesdiseñados y las estrategias de entrenamiento que han llevado a mejoras significativas en el rendimiento. (a) El cuello de YOLOv6 (N y S se muestran). Señalar que, en M/L, RepBlocks es reemplazado por CSPStackRep. (b) La estructura de un módulo BiC. (c) Un bloque SimCSPSPPF. ([fuente](https://arxiv.org/pdf/2301.05586.pdf)).

### Características clave

- **Módulo de Concatenación Bidireccional (BiC):** YOLOv6 introduce un módulo de BiC en el cuello del detector, mejorando las señales de localización y ofreciendo mejoras en el rendimiento con una degradación de velocidad despreciable.
- **Estrategia de Entrenamiento con Anclas (AAT):** Este modelo propone AAT para disfrutar de los beneficios de los paradigmas basados en anclas y sin anclas sin comprometer la eficiencia de inferencia.
- **Diseño de Columna Vertebral y Cuello Mejorado:** Al profundizar en YOLOv6 para incluir otra etapa en la columna vertebral y el cuello, este modelo logra un rendimiento de última generación en el conjunto de datos COCO con una entrada de alta resolución.
- **Estrategia de Auto-Destilación:** Se implementa una nueva estrategia de auto-destilación para mejorar el rendimiento de los modelos más pequeños de YOLOv6, mejorando la rama de regresión auxiliar durante el entrenamiento y eliminándola durante la inferencia para evitar una marcada disminución de velocidad.

## Métricas de rendimiento

YOLOv6 proporciona varios modelos pre-entrenados con diferentes escalas:

- YOLOv6-N: 37.5% de precisión promedio (AP) en COCO val2017 a 1187 FPS con la GPU NVIDIA Tesla T4.
- YOLOv6-S: 45.0% de AP a 484 FPS.
- YOLOv6-M: 50.0% de AP a 226 FPS.
- YOLOv6-L: 52.8% de AP a 116 FPS.
- YOLOv6-L6: Precisión de última generación en tiempo real.

YOLOv6 también proporciona modelos cuantizados para diferentes precisiones y modelos optimizados para plataformas móviles.

## Ejemplos de uso

Este ejemplo proporciona ejemplos sencillos de entrenamiento e inferencia con YOLOv6. Para obtener documentación completa sobre estos y otros [modos](../modes/index.md), consulta las páginas de documentación de [Predict](../modes/predict.md), [Train](../modes/train.md), [Val](../modes/val.md) y [Export](../modes/export.md).

!!! Example "Ejemplo"

    === "Python"

        Los modelos pre-entrenados en `*.pt` de PyTorch, así como los archivos de configuración `*.yaml`, se pueden pasar a la clase `YOLO()` para crear una instancia del modelo en Python:

        ```python
        from ultralytics import YOLO

        # Construir un modelo YOLOv6n desde cero
        modelo = YOLO('yolov6n.yaml')

        # Mostrar información del modelo (opcional)
        modelo.info()

        # Entrenar el modelo en el conjunto de datos de ejemplo COCO8 durante 100 epochs
        resultados = modelo.train(data='coco8.yaml', epochs=100, imgsz=640)

        # Ejecutar inferencia con el modelo YOLOv6n en la imagen 'bus.jpg'
        resultados = modelo('path/to/bus.jpg')
        ```

    === "CLI"

        Se dispone de comandos de línea de comandos (CLI) para ejecutar directamente los modelos:

        ```bash
        # Construir un modelo YOLOv6n desde cero y entrenarlo en el conjunto de datos de ejemplo COCO8 durante 100 epochs
        yolo train model=yolov6n.yaml data=coco8.yaml epochs=100 imgsz=640

        # Construir un modelo YOLOv6n desde cero y ejecutar inferencia en la imagen 'bus.jpg'
        yolo predict model=yolov6n.yaml source=path/to/bus.jpg
        ```

## Tareas y Modos Soportados

La serie YOLOv6 ofrece una variedad de modelos, cada uno optimizado para [Detección de Objetos](../tasks/detect.md) de alto rendimiento. Estos modelos se adaptan a distintas necesidades computacionales y requisitos de precisión, lo que los hace versátiles para una amplia gama de aplicaciones.

| Tipo de Modelo | Pesos Pre-entrenados | Tareas Soportadas                          | Inferencia | Validación | Entrenamiento | Exportación |
|----------------|----------------------|--------------------------------------------|------------|------------|---------------|-------------|
| YOLOv6-N       | `yolov6-n.pt`        | [Detección de Objetos](../tasks/detect.md) | ✅          | ✅          | ✅             | ✅           |
| YOLOv6-S       | `yolov6-s.pt`        | [Detección de Objetos](../tasks/detect.md) | ✅          | ✅          | ✅             | ✅           |
| YOLOv6-M       | `yolov6-m.pt`        | [Detección de Objetos](../tasks/detect.md) | ✅          | ✅          | ✅             | ✅           |
| YOLOv6-L       | `yolov6-l.pt`        | [Detección de Objetos](../tasks/detect.md) | ✅          | ✅          | ✅             | ✅           |
| YOLOv6-L6      | `yolov6-l6.pt`       | [Detección de Objetos](../tasks/detect.md) | ✅          | ✅          | ✅             | ✅           |

Esta tabla proporciona una descripción detallada de las variantes del modelo YOLOv6, destacando sus capacidades en tareas de detección de objetos y su compatibilidad con varios modos operativos como [Inferencia](../modes/predict.md), [Validación](../modes/val.md), [Entrenamiento](../modes/train.md) y [Exportación](../modes/export.md). Este soporte integral garantiza que los usuarios puedan aprovechar al máximo las capacidades de los modelos YOLOv6 en una amplia gama de escenarios de detección de objetos.

## Citaciones y Agradecimientos

Nos gustaría agradecer a los autores por sus importantes contribuciones en el campo de la detección de objetos en tiempo real:

!!! Quote ""

    === "BibTeX"

        ```bibtex
        @misc{li2023yolov6,
              title={YOLOv6 v3.0: A Full-Scale Reloading},
              author={Chuyi Li and Lulu Li and Yifei Geng and Hongliang Jiang and Meng Cheng and Bo Zhang and Zaidan Ke and Xiaoming Xu and Xiangxiang Chu},
              year={2023},
              eprint={2301.05586},
              archivePrefix={arXiv},
              primaryClass={cs.CV}
        }
        ```

    Se puede encontrar el artículo original de YOLOv6 en [arXiv](https://arxiv.org/abs/2301.05586). Los autores han puesto su trabajo a disposición del público y el código fuente se puede acceder en [GitHub](https://github.com/meituan/YOLOv6). Agradecemos sus esfuerzos en avanzar en el campo y hacer que su trabajo sea accesible para la comunidad en general.
