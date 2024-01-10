---
comments: true
description: Descubra YOLOv5u, una versión mejorada del modelo YOLOv5 con un mejor equilibrio entre precisión y velocidad, y numerosos modelos pre-entrenados para diversas tareas de detección de objetos.
keywords: YOLOv5u, detección de objetos, modelos pre-entrenados, Ultralytics, Inferencia, Validación, YOLOv5, YOLOv8, sin anclas, sin atención al objeto, aplicaciones en tiempo real, aprendizaje automático
---

# YOLOv5

## Resumen

YOLOv5u representa un avance en las metodologías de detección de objetos. Originado a partir de la arquitectura fundamental del modelo [YOLOv5](https://github.com/ultralytics/yolov5) desarrollado por Ultralytics, YOLOv5u integra la división de la cabeza Ultralytics sin anclas y sin atención al objeto, una característica introducida previamente en los modelos [YOLOv8](yolov8.md). Esta adaptación perfecciona la arquitectura del modelo, resultando en un mejor equilibrio entre precisión y velocidad en tareas de detección de objetos. Con base en los resultados empíricos y sus características derivadas, YOLOv5u proporciona una alternativa eficiente para aquellos que buscan soluciones robustas tanto en investigación como en aplicaciones prácticas.

![Ultralytics YOLOv5](https://raw.githubusercontent.com/ultralytics/assets/main/yolov5/v70/splash.png)

## Características clave

- **Cabeza dividida Ultralytics sin anclas:** Los modelos tradicionales de detección de objetos dependen de cajas de anclaje predefinidas para predecir la ubicación de los objetos. Sin embargo, YOLOv5u moderniza este enfoque. Al adoptar una cabeza Ultralytics dividida sin anclas, se garantiza un mecanismo de detección más flexible y adaptable, lo que en consecuencia mejora el rendimiento en diversos escenarios.

- **Equilibrio óptimo entre precisión y velocidad:** La velocidad y la precisión suelen ser contrapuestas. Pero YOLOv5u desafía este equilibrio. Ofrece un balance calibrado, garantizando detecciones en tiempo real sin comprometer la precisión. Esta característica es especialmente valiosa para aplicaciones que requieren respuestas rápidas, como vehículos autónomos, robótica y análisis de video en tiempo real.

- **Variedad de modelos pre-entrenados:** Entendiendo que diferentes tareas requieren diferentes herramientas, YOLOv5u proporciona una gran cantidad de modelos pre-entrenados. Ya sea que te enfoques en Inferencia, Validación o Entrenamiento, hay un modelo a la medida esperándote. Esta variedad asegura que no estés utilizando una solución genérica, sino un modelo específicamente ajustado para tu desafío único.

## Tareas y Modos Soportados

Los modelos YOLOv5u, con diferentes pesos pre-entrenados, sobresalen en las tareas de [Detección de Objetos](../tasks/detect.md). Soportan una amplia gama de modos que los hacen adecuados para diversas aplicaciones, desde el desarrollo hasta la implementación.

| Tipo de Modelo | Pesos Pre-entrenados                                                                                                        | Tarea                                      | Inferencia | Validación | Entrenamiento | Exportación |
|----------------|-----------------------------------------------------------------------------------------------------------------------------|--------------------------------------------|------------|------------|---------------|-------------|
| YOLOv5u        | `yolov5nu`, `yolov5su`, `yolov5mu`, `yolov5lu`, `yolov5xu`, `yolov5n6u`, `yolov5s6u`, `yolov5m6u`, `yolov5l6u`, `yolov5x6u` | [Detección de Objetos](../tasks/detect.md) | ✅          | ✅          | ✅             | ✅           |

Esta tabla proporciona una descripción detallada de las variantes de modelos YOLOv5u, destacando su aplicabilidad en tareas de detección de objetos y el soporte para varios modos operativos como [Inferencia](../modes/predict.md), [Validación](../modes/val.md), [Entrenamiento](../modes/train.md) y [Exportación](../modes/export.md). Este soporte integral asegura que los usuarios puedan aprovechar al máximo las capacidades de los modelos YOLOv5u en una amplia gama de escenarios de detección de objetos.

## Métricas de Rendimiento

!!! Rendimiento

    === "Detección"

    Consulta la [Documentación de Detección](https://docs.ultralytics.com/tasks/detect/) para obtener ejemplos de uso con estos modelos entrenados en [COCO](https://docs.ultralytics.com/datasets/detect/coco/), los cuales incluyen 80 clases pre-entrenadas.

    | Modelo                                                                                       | YAML                                                                                                           | tamaño<br><sup>(píxeles) | mAP<sup>val<br>50-95 | Velocidad<br><sup>CPU ONNX<br>(ms) | Velocidad<br><sup>A100 TensorRT<br>(ms) | parámetros<br><sup>(M) | FLOPs<br><sup>(B) |
    |---------------------------------------------------------------------------------------------|----------------------------------------------------------------------------------------------------------------|-----------------------|----------------------|--------------------------------|-------------------------------------|--------------------|-------------------|
    | [yolov5nu.pt](https://github.com/ultralytics/assets/releases/download/v8.1.0/yolov5nu.pt)   | [yolov5n.yaml](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/models/v5/yolov5.yaml)     | 640                   | 34.3                 | 73.6                           | 1.06                                | 2.6                | 7.7               |
    | [yolov5su.pt](https://github.com/ultralytics/assets/releases/download/v8.1.0/yolov5su.pt)   | [yolov5s.yaml](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/models/v5/yolov5.yaml)     | 640                   | 43.0                 | 120.7                          | 1.27                                | 9.1                | 24.0              |
    | [yolov5mu.pt](https://github.com/ultralytics/assets/releases/download/v8.1.0/yolov5mu.pt)   | [yolov5m.yaml](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/models/v5/yolov5.yaml)     | 640                   | 49.0                 | 233.9                          | 1.86                                | 25.1               | 64.2              |
    | [yolov5lu.pt](https://github.com/ultralytics/assets/releases/download/v8.1.0/yolov5lu.pt)   | [yolov5l.yaml](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/models/v5/yolov5.yaml)     | 640                   | 52.2                 | 408.4                          | 2.50                                | 53.2               | 135.0             |
    | [yolov5xu.pt](https://github.com/ultralytics/assets/releases/download/v8.1.0/yolov5xu.pt)   | [yolov5x.yaml](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/models/v5/yolov5.yaml)     | 640                   | 53.2                 | 763.2                          | 3.81                                | 97.2               | 246.4             |
    |                                                                                             |                                                                                                                |                       |                      |                                |                                     |                    |                   |
    | [yolov5n6u.pt](https://github.com/ultralytics/assets/releases/download/v8.1.0/yolov5n6u.pt) | [yolov5n6.yaml](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/models/v5/yolov5-p6.yaml) | 1280                  | 42.1                 | 211.0                          | 1.83                                | 4.3                | 7.8               |
    | [yolov5s6u.pt](https://github.com/ultralytics/assets/releases/download/v8.1.0/yolov5s6u.pt) | [yolov5s6.yaml](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/models/v5/yolov5-p6.yaml) | 1280                  | 48.6                 | 422.6                          | 2.34                                | 15.3               | 24.6              |
    | [yolov5m6u.pt](https://github.com/ultralytics/assets/releases/download/v8.1.0/yolov5m6u.pt) | [yolov5m6.yaml](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/models/v5/yolov5-p6.yaml) | 1280                  | 53.6                 | 810.9                          | 4.36                                | 41.2               | 65.7              |
    | [yolov5l6u.pt](https://github.com/ultralytics/assets/releases/download/v8.1.0/yolov5l6u.pt) | [yolov5l6.yaml](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/models/v5/yolov5-p6.yaml) | 1280                  | 55.7                 | 1470.9                         | 5.47                                | 86.1               | 137.4             |
    | [yolov5x6u.pt](https://github.com/ultralytics/assets/releases/download/v8.1.0/yolov5x6u.pt) | [yolov5x6.yaml](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/models/v5/yolov5-p6.yaml) | 1280                  | 56.8                 | 2436.5                         | 8.98                                | 155.4              | 250.7             |

## Ejemplos de Uso

Este ejemplo proporciona ejemplos sencillos de entrenamiento e inferencia de YOLOv5. Para obtener documentación completa sobre estos y otros [modos](../modes/index.md), consulta las páginas de documentación de [Predict](../modes/predict.md), [Train](../modes/train.md), [Val](../modes/val.md) y [Export](../modes/export.md).

!!! Example "Ejemplo"

    === "Python"

        Los modelos pre-entrenados `*.pt` de PyTorch, así como los archivos de configuración `*.yaml`, se pueden pasar a la clase `YOLO()` para crear una instancia de modelo en Python:

        ```python
        from ultralytics import YOLO

        # Cargar un modelo YOLOv5n pre-entrenado en COCO
        modelo = YOLO('yolov5n.pt')

        # Mostrar información del modelo (opcional)
        modelo.info()

        # Entrenar el modelo con el conjunto de datos de ejemplo COCO8 durante 100 épocas
        resultados = modelo.train(data='coco8.yaml', epochs=100, imgsz=640)

        # Ejecutar inferencia con el modelo YOLOv5n en la imagen 'bus.jpg'
        resultados = modelo('path/to/bus.jpg')
        ```

    === "CLI"

        Hay comandos de CLI disponibles para ejecutar directamente los modelos:

        ```bash
        # Cargar un modelo YOLOv5n pre-entrenado en COCO y entrenarlo con el conjunto de datos de ejemplo COCO8 durante 100 épocas
        yolo train model=yolov5n.pt data=coco8.yaml epochs=100 imgsz=640

        # Cargar un modelo YOLOv5n pre-entrenado en COCO y ejecutar inferencia en la imagen 'bus.jpg'
        yolo predict model=yolov5n.pt source=path/to/bus.jpg
        ```

## Citaciones y Reconocimientos

Si utilizas YOLOv5 o YOLOv5u en tu investigación, por favor cita el repositorio de Ultralytics YOLOv5 de la siguiente manera:

!!! Quote ""

    === "BibTeX"
        ```bibtex
        @software{yolov5,
          title = {Ultralytics YOLOv5},
          author = {Glenn Jocher},
          year = {2020},
          version = {7.0},
          license = {AGPL-3.0},
          url = {https://github.com/ultralytics/yolov5},
          doi = {10.5281/zenodo.3908559},
          orcid = {0000-0001-5950-6979}
        }
        ```

Ten en cuenta que los modelos YOLOv5 se proporcionan bajo las licencias [AGPL-3.0](https://github.com/ultralytics/ultralytics/blob/main/LICENSE) y [Enterprise](https://ultralytics.com/license).
