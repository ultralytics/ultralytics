---
comments: true
description: ¡Explora las emocionantes características de YOLOv8, la última versión de nuestro detector de objetos en tiempo real! Aprende cómo las arquitecturas avanzadas, los modelos pre-entrenados y el equilibrio óptimo entre precisión y velocidad hacen de YOLOv8 la elección perfecta para tus tareas de detección de objetos.
keywords: YOLOv8, Ultralytics, detector de objetos en tiempo real, modelos pre-entrenados, documentación, detección de objetos, serie YOLO, arquitecturas avanzadas, precisión, velocidad
---

# YOLOv8

## Información general

YOLOv8 es la última versión de la serie YOLO de detectores de objetos en tiempo real, que ofrece un rendimiento de vanguardia en términos de precisión y velocidad. Basándose en los avances de versiones anteriores de YOLO, YOLOv8 presenta nuevas características y optimizaciones que lo convierten en una elección ideal para diversas tareas de detección de objetos en una amplia gama de aplicaciones.

![Ultralytics YOLOv8](https://raw.githubusercontent.com/ultralytics/assets/main/yolov8/yolo-comparison-plots.png)

## Funciones clave

- **Arquitecturas avanzadas de backbone y neck:** YOLOv8 utiliza arquitecturas de backbone y neck de última generación, lo que resulta en una mejor extracción de características y un mejor rendimiento en la detección de objetos.
- **Cabeza Ultralytics split sin anclas:** YOLOv8 adopta una cabeza Ultralytics dividida sin anclas, lo que contribuye a una mejor precisión y un proceso de detección más eficiente en comparación con los enfoques basados en anclas.
- **Equilibrio óptimo entre precisión y velocidad:** Con un enfoque en mantener un equilibrio óptimo entre precisión y velocidad, YOLOv8 es adecuado para tareas de detección de objetos en tiempo real en diversas áreas de aplicación.
- **Variedad de modelos pre-entrenados:** YOLOv8 ofrece una variedad de modelos pre-entrenados para adaptarse a diversas tareas y requisitos de rendimiento, lo que facilita encontrar el modelo adecuado para tu caso de uso específico.

## Tareas compatibles

| Tipo de modelo | Pesos pre-entrenados                                                                                                | Tarea                      |
|----------------|---------------------------------------------------------------------------------------------------------------------|----------------------------|
| YOLOv8         | `yolov8n.pt`, `yolov8s.pt`, `yolov8m.pt`, `yolov8l.pt`, `yolov8x.pt`                                                | Detección                  |
| YOLOv8-seg     | `yolov8n-seg.pt`, `yolov8s-seg.pt`, `yolov8m-seg.pt`, `yolov8l-seg.pt`, `yolov8x-seg.pt`                            | Segmentación de instancias |
| YOLOv8-pose    | `yolov8n-pose.pt`, `yolov8s-pose.pt`, `yolov8m-pose.pt`, `yolov8l-pose.pt`, `yolov8x-pose.pt`, `yolov8x-pose-p6.pt` | Pose/Puntos clave          |
| YOLOv8-cls     | `yolov8n-cls.pt`, `yolov8s-cls.pt`, `yolov8m-cls.pt`, `yolov8l-cls.pt`, `yolov8x-cls.pt`                            | Clasificación              |

## Modos compatibles

| Modo          | Compatible |
|---------------|------------|
| Inferencia    | ✅          |
| Validación    | ✅          |
| Entrenamiento | ✅          |

!!! Rendimiento

    === "Detección (COCO)"

        | Modelo                                                                                | tamaño<br><sup>(píxeles) | mAP<sup>val<br>50-95 | Velocidad<br><sup>CPU ONNX<br>(ms) | Velocidad<br><sup>A100 TensorRT<br>(ms) | parámetros<br><sup>(M) | FLOPs<br><sup>(B) |
        | ------------------------------------------------------------------------------------ | --------------------- | -------------------- | ------------------------------ | ----------------------------------- | ------------------ | ----------------- |
        | [YOLOv8n](https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n.pt) | 640                   | 37.3                 | 80.4                           | 0.99                                | 3.2                | 8.7               |
        | [YOLOv8s](https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8s.pt) | 640                   | 44.9                 | 128.4                          | 1.20                                | 11.2               | 28.6              |
        | [YOLOv8m](https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8m.pt) | 640                   | 50.2                 | 234.7                          | 1.83                                | 25.9               | 78.9              |
        | [YOLOv8l](https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8l.pt) | 640                   | 52.9                 | 375.2                          | 2.39                                | 43.7               | 165.2             |
        | [YOLOv8x](https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8x.pt) | 640                   | 53.9                 | 479.1                          | 3.53                                | 68.2               | 257.8             |

    === "Detección (Open Images V7)"

        Consulta la [documentación de detección](https://docs.ultralytics.com/tasks/detect/) para ver ejemplos de uso de estos modelos entrenados en [Open Image V7](https://docs.ultralytics.com/datasets/detect/open-images-v7/), que incluyen 600 clases pre-entrenadas.

        | Modelo                                                                                     | tamaño<br><sup>(píxeles) | mAP<sup>val<br>50-95 | Velocidad<br><sup>CPU ONNX<br>(ms) | Velocidad<br><sup>A100 TensorRT<br>(ms) | parámetros<br><sup>(M) | FLOPs<br><sup>(B) |
        | ----------------------------------------------------------------------------------------- | --------------------- | -------------------- | ------------------------------ | ----------------------------------- | ------------------ | ----------------- |
        | [YOLOv8n](https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n-oiv7.pt) | 640                   | 18.4                 | 142.4                          | 1.21                                | 3.5                | 10.5              |
        | [YOLOv8s](https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8s-oiv7.pt) | 640                   | 27.7                 | 183.1                          | 1.40                                | 11.4               | 29.7              |
        | [YOLOv8m](https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8m-oiv7.pt) | 640                   | 33.6                 | 408.5                          | 2.26                                | 26.2               | 80.6              |
        | [YOLOv8l](https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8l-oiv7.pt) | 640                   | 34.9                 | 596.9                          | 2.43                                | 44.1               | 167.4             |
        | [YOLOv8x](https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8x-oiv7.pt) | 640                   | 36.3                 | 860.6                          | 3.56                                | 68.7               | 260.6             |

    === "Segmentación (COCO)"

        | Modelo                                                                                        | tamaño<br><sup>(píxeles) | mAP<sup>caja<br>50-95 | mAP<sup>máscara<br>50-95 | Velocidad<br><sup>CPU ONNX<br>(ms) | Velocidad<br><sup>A100 TensorRT<br>(ms) | parámetros<br><sup>(M) | FLOPs<br><sup>(B) |
        | -------------------------------------------------------------------------------------------- | --------------------- | --------------------- | ----------------------- | ------------------------------ | ----------------------------------- | ------------------ | ----------------- |
        | [YOLOv8n-seg](https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n-seg.pt) | 640                   | 36.7                  | 30.5                    | 96.1                           | 1.21                                | 3.4                | 12.6              |
        | [YOLOv8s-seg](https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8s-seg.pt) | 640                   | 44.6                  | 36.8                    | 155.7                          | 1.47                                | 11.8               | 42.6              |
        | [YOLOv8m-seg](https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8m-seg.pt) | 640                   | 49.9                  | 40.8                    | 317.0                          | 2.18                                | 27.3               | 110.2             |
        | [YOLOv8l-seg](https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8l-seg.pt) | 640                   | 52.3                  | 42.6                    | 572.4                          | 2.79                                | 46.0               | 220.5             |
        | [YOLOv8x-seg](https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8x-seg.pt) | 640                   | 53.4                  | 43.4                    | 712.1                          | 4.02                                | 71.8               | 344.1             |

    === "Clasificación (ImageNet)"

        | Modelo                                                                                        | tamaño<br><sup>(píxeles) | acc<br><sup>top1 | acc<br><sup>top5 | Velocidad<br><sup>CPU ONNX<br>(ms) | Velocidad<br><sup>A100 TensorRT<br>(ms) | parámetros<br><sup>(M) | FLOPs<br><sup>(B) a 640 |
        | -------------------------------------------------------------------------------------------- | --------------------- | ---------------- | ---------------- | ------------------------------ | ----------------------------------- | ------------------ | ------------------------ |
        | [YOLOv8n-cls](https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n-cls.pt) | 224                   | 66.6             | 87.0             | 12.9                           | 0.31                                | 2.7                | 4.3                      |
        | [YOLOv8s-cls](https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8s-cls.pt) | 224                   | 72.3             | 91.1             | 23.4                           | 0.35                                | 6.4                | 13.5                     |
        | [YOLOv8m-cls](https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8m-cls.pt) | 224                   | 76.4             | 93.2             | 85.4                           | 0.62                                | 17.0               | 42.7                     |
        | [YOLOv8l-cls](https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8l-cls.pt) | 224                   | 78.0             | 94.1             | 163.0                          | 0.87                                | 37.5               | 99.7                     |
        | [YOLOv8x-cls](https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8x-cls.pt) | 224                   | 78.4             | 94.3             | 232.0                          | 1.01                                | 57.4               | 154.8                    |

    === "Pose (COCO)"

        | Modelo                                                                                                | tamaño<br><sup>(píxeles) | mAP<sup>pose<br>50-95 | mAP<sup>pose<br>50 | Velocidad<br><sup>CPU ONNX<br>(ms) | Velocidad<br><sup>A100 TensorRT<br>(ms) | parámetros<br><sup>(M) | FLOPs<br><sup>(B) |
        | ---------------------------------------------------------------------------------------------------- | --------------------- | --------------------- | ------------------ | ------------------------------ | ----------------------------------- | ------------------ | ----------------- |
        | [YOLOv8n-pose](https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n-pose.pt)       | 640                   | 50.4                  | 80.1               | 131.8                          | 1.18                                | 3.3                | 9.2               |
        | [YOLOv8s-pose](https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8s-pose.pt)       | 640                   | 60.0                  | 86.2               | 233.2                          | 1.42                                | 11.6               | 30.2              |
        | [YOLOv8m-pose](https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8m-pose.pt)       | 640                   | 65.0                  | 88.8               | 456.3                          | 2.00                                | 26.4               | 81.0              |
        | [YOLOv8l-pose](https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8l-pose.pt)       | 640                   | 67.6                  | 90.0               | 784.5                          | 2.59                                | 44.4               | 168.6             |
        | [YOLOv8x-pose](https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8x-pose.pt)       | 640                   | 69.2                  | 90.2               | 1607.1                         | 3.73                                | 69.4               | 263.2             |
        | [YOLOv8x-pose-p6](https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8x-pose-p6.pt) | 1280                  | 71.6                  | 91.2               | 4088.7                         | 10.04                               | 99.1               | 1066.4            |

## Uso

Puedes utilizar YOLOv8 para tareas de detección de objetos utilizando el paquete pip de Ultralytics. El siguiente es un fragmento de código de ejemplo que muestra cómo utilizar modelos YOLOv8 para inferencia:

!!! Ejemplo ""

    Este ejemplo proporciona un código simple de inferencia para YOLOv8. Para obtener más opciones, incluido el manejo de los resultados de la inferencia, consulta el modo de [Predicción](../modes/predict.md). Para utilizar YOLOv8 con modos adicionales, consulta los modos de [Entrenamiento](../modes/train.md), [Validación](../modes/val.md) y [Exportación](../modes/export.md).

    === "Python"

        Los modelos `*.pt` pre-entrenados en PyTorch, así como los archivos de configuración `*.yaml`, se pueden pasar a la clase `YOLO()` para crear una instancia del modelo en Python:

        ```python
        from ultralytics import YOLO

        # Carga un modelo YOLOv8n pre-entrenado en COCO
        model = YOLO('yolov8n.pt')

        # Muestra información del modelo (opcional)
        model.info()

        # Entrena el modelo en el conjunto de datos de ejemplo COCO8 durante 100 épocas
        results = model.train(data='coco8.yaml', epochs=100, imgsz=640)

        # Ejecuta la inferencia con el modelo YOLOv8n en la imagen 'bus.jpg'
        results = model('ruta/a/bus.jpg')
        ```

    === "CLI"

        Hay comandos de CLI disponibles para ejecutar los modelos directamente:

        ```bash
        # Carga un modelo YOLOv8n pre-entrenado en COCO y lo entrena en el conjunto de datos de ejemplo COCO8 durante 100 épocas
        yolo train model=yolov8n.pt data=coco8.yaml epochs=100 imgsz=640

        # Carga un modelo YOLOv8n pre-entrenado en COCO y ejecuta la inferencia en la imagen 'bus.jpg'
        yolo predict model=yolov8n.pt source=ruta/a/bus.jpg
        ```

## Citaciones y agradecimientos

Si utilizas el modelo YOLOv8 o cualquier otro software de este repositorio en tu trabajo, por favor cítalo utilizando el siguiente formato:

!!! Nota ""

    === "BibTeX"

        ```bibtex
        @software{yolov8_ultralytics,
          author = {Glenn Jocher and Ayush Chaurasia and Jing Qiu},
          title = {Ultralytics YOLOv8},
          version = {8.0.0},
          year = {2023},
          url = {https://github.com/ultralytics/ultralytics},
          orcid = {0000-0001-5950-6979, 0000-0002-7603-6750, 0000-0003-3783-7069},
          license = {AGPL-3.0}
        }
        ```

Ten en cuenta que el DOI está pendiente y se agregará a la cita una vez esté disponible. El uso del software está de acuerdo con la licencia AGPL-3.0.
