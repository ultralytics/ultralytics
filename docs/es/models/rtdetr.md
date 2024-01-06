---
comments: true
description: Descubre las características y beneficios de RT-DETR, un eficiente y adaptable detector de objetos en tiempo real desarrollado por Baidu y potenciado por Vision Transformers, que incluye modelos pre-entrenados.
keywords: RT-DETR, Baidu, Vision Transformers, detección de objetos, rendimiento en tiempo real, CUDA, TensorRT, selección de consultas IoU, Ultralytics, API de Python, PaddlePaddle
---

# RT-DETR de Baidu: Un Detector de Objetos en Tiempo Real Basado en Vision Transformers

## Resumen

Real-Time Detection Transformer (RT-DETR), desarrollado por Baidu, es un avanzado detector de objetos de extremo a extremo que proporciona un rendimiento en tiempo real manteniendo una alta precisión. Utiliza la potencia de Vision Transformers (ViT) para procesar de manera eficiente características de múltiples escalas mediante la descomposición de la interacción intra-escala y la fusión inter-escala. RT-DETR es altamente adaptable y permite ajustar de manera flexible la velocidad de inferencia utilizando diferentes capas de decodificador sin necesidad de volver a entrenar el modelo. El modelo se destaca en plataformas aceleradas como CUDA con TensorRT, superando a muchos otros detectores de objetos en tiempo real.

![Ejemplo de imagen del modelo](https://user-images.githubusercontent.com/26833433/238963168-90e8483f-90aa-4eb6-a5e1-0d408b23dd33.png)
**Resumen de RT-DETR de Baidu.** El diagrama de la arquitectura del modelo RT-DETR muestra las últimas tres etapas del canal (S3, S4, S5) como entrada al codificador. El eficiente codificador híbrido transforma características de múltiples escalas en una secuencia de características de imagen a través del módulo de interacción de características intra-escala (AIFI) y el módulo de fusión de características inter-escala (CCFM). Se utiliza la selección de consultas IoU-aware para seleccionar un número fijo de características de imagen que servirán como consultas iniciales de objetos para el decodificador. Finalmente, el decodificador con cabeceras de predicción auxiliares optimiza iterativamente las consultas de objetos para generar cajas y puntuaciones de confianza ([fuente](https://arxiv.org/pdf/2304.08069.pdf)).

### Características Clave

- **Codificador Híbrido Eficiente:** RT-DETR de Baidu utiliza un codificador híbrido eficiente que procesa características de múltiples escalas mediante la descomposición de la interacción intra-escala y la fusión inter-escala. Este diseño único basado en Vision Transformers reduce los costos computacionales y permite la detección de objetos en tiempo real.
- **Selección de Consultas IoU-aware:** RT-DETR de Baidu mejora la inicialización de las consultas de objetos utilizando la selección de consultas IoU-aware. Esto permite que el modelo se enfoque en los objetos más relevantes de la escena, mejorando la precisión en la detección.
- **Velocidad de Inferencia Adaptable:** RT-DETR de Baidu admite ajustes flexibles de la velocidad de inferencia utilizando diferentes capas de decodificador sin necesidad de volver a entrenar el modelo. Esta adaptabilidad facilita la aplicación práctica en diversos escenarios de detección de objetos en tiempo real.

## Modelos Pre-entrenados

La API de Python de Ultralytics proporciona modelos pre-entrenados de RT-DETR de PaddlePaddle en diferentes escalas:

- RT-DETR-L: 53.0% AP en COCO val2017, 114 FPS en GPU T4
- RT-DETR-X: 54.8% AP en COCO val2017, 74 FPS en GPU T4

## Ejemplos de Uso

Este ejemplo proporciona ejemplos sencillos de entrenamiento e inferencia de RT-DETRR. Para obtener una documentación completa sobre estos y otros [modos](../modes/index.md), consulta las páginas de documentación de [Predict](../modes/predict.md),  [Train](../modes/train.md), [Val](../modes/val.md) y [Export](../modes/export.md).

!!! Example "Ejemplo"

    === "Python"

        ```python
        from ultralytics import RTDETR

        # Cargar un modelo RT-DETR-l pre-entrenado en COCO
        model = RTDETR('rtdetr-l.pt')

        # Mostrar información del modelo (opcional)
        model.info()

        # Entrenar el modelo en el conjunto de datos de ejemplo COCO8 durante 100 épocas
        results = model.train(data='coco8.yaml', epochs=100, imgsz=640)

        # Realizar inferencia con el modelo RT-DETR-l en la imagen 'bus.jpg'
        results = model('path/to/bus.jpg')
        ```

    === "CLI"

        ```bash
        # Cargar un modelo RT-DETR-l pre-entrenado en COCO y entrenarlo en el conjunto de datos de ejemplo COCO8 durante 100 épocas
        yolo train model=rtdetr-l.pt data=coco8.yaml epochs=100 imgsz=640

        # Cargar un modelo RT-DETR-l pre-entrenado en COCO y realizar inferencia en la imagen 'bus.jpg'
        yolo predict model=rtdetr-l.pt source=path/to/bus.jpg
        ```

## Tareas y Modos Admitidos

Esta tabla presenta los tipos de modelos, los pesos pre-entrenados específicos, las tareas admitidas por cada modelo y los diversos modos ([Train](../modes/train.md) , [Val](../modes/val.md), [Predict](../modes/predict.md), [Export](../modes/export.md)) admitidos, indicados por los emojis ✅.

| Tipo de Modelo      | Pesos Pre-entrenados | Tareas Admitidas                           | Inferencia | Validación | Entrenamiento | Exportación |
|---------------------|----------------------|--------------------------------------------|------------|------------|---------------|-------------|
| RT-DETR Large       | `rtdetr-l.pt`        | [Detección de Objetos](../tasks/detect.md) | ✅          | ✅          | ✅             | ✅           |
| RT-DETR Extra-Large | `rtdetr-x.pt`        | [Detección de Objetos](../tasks/detect.md) | ✅          | ✅          | ✅             | ✅           |

## Citaciones y Agradecimientos

Si utilizas RT-DETR de Baidu en tu investigación o trabajo de desarrollo, por favor cita el [artículo original](https://arxiv.org/abs/2304.08069):

!!! Quote ""

    === "BibTeX"

        ```bibtex
        @misc{lv2023detrs,
              title={DETRs Beat YOLOs on Real-time Object Detection},
              author={Wenyu Lv and Shangliang Xu and Yian Zhao and Guanzhong Wang and Jinman Wei and Cheng Cui and Yuning Du and Qingqing Dang and Yi Liu},
              year={2023},
              eprint={2304.08069},
              archivePrefix={arXiv},
              primaryClass={cs.CV}
        }
        ```

Nos gustaría agradecer a Baidu y al equipo de [PaddlePaddle](https://github.com/PaddlePaddle/PaddleDetection) por crear y mantener este valioso recurso para la comunidad de visión por computadora. Apreciamos enormemente su contribución al campo con el desarrollo del detector de objetos en tiempo real basado en Vision Transformers, RT-DETR.

*keywords: RT-DETR, Transformer, ViT, Vision Transformers, Baidu RT-DETR, PaddlePaddle, Paddle Paddle RT-DETR, detección de objetos en tiempo real, detección de objetos basada en Vision Transformers, modelos pre-entrenados PaddlePaddle RT-DETR, uso de RT-DETR de Baidu, API de Python de Ultralytics*
