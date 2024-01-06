---
comments: true
description: Explora nuestra detallada guía sobre YOLOv4, un detector de objetos en tiempo real de vanguardia. Comprende sus aspectos arquitectónicos destacados, características innovadoras y ejemplos de aplicación.
keywords: ultralytics, YOLOv4, detección de objetos, red neuronal, detección en tiempo real, detector de objetos, aprendizaje automático
---

# YOLOv4: Detección de objetos rápida y precisa

Bienvenido a la página de documentación de Ultralytics para YOLOv4, un detector de objetos en tiempo real de vanguardia lanzado en 2020 por Alexey Bochkovskiy en [https://github.com/AlexeyAB/darknet](https://github.com/AlexeyAB/darknet). YOLOv4 está diseñado para ofrecer un equilibrio óptimo entre velocidad y precisión, lo que lo convierte en una excelente opción para muchas aplicaciones.

![Diagrama de arquitectura de YOLOv4](https://user-images.githubusercontent.com/26833433/246185689-530b7fe8-737b-4bb0-b5dd-de10ef5aface.png)
**Diagrama de arquitectura de YOLOv4**. Muestra el intrincado diseño de red de YOLOv4, incluyendo los componentes backbone, neck y head, y sus capas interconectadas para una detección de objetos en tiempo real óptima.

## Introducción

YOLOv4 significa You Only Look Once versión 4. Es un modelo de detección de objetos en tiempo real desarrollado para abordar las limitaciones de versiones anteriores de YOLO como [YOLOv3](yolov3.md) y otros modelos de detección de objetos. A diferencia de otros detectores de objetos basados en redes neuronales convolucionales (CNN), YOLOv4 no solo es aplicable para sistemas de recomendación, sino también para la gestión de procesos independientes y la reducción de la entrada humana. Su funcionamiento en unidades de procesamiento de gráficos (GPU) convencionales permite su uso masivo a un precio asequible, y está diseñado para funcionar en tiempo real en una GPU convencional, siendo necesario solo una GPU para el entrenamiento.

## Arquitectura

YOLOv4 utiliza varias características innovadoras que trabajan juntas para optimizar su rendimiento. Estas incluyen Conexiones Residuales Ponderadas (WRC), Conexiones Parciales Cruzadas en Etapas (CSP), Normalización Cruzada de Mini-Batch (CmBN), Entrenamiento Autoadversarial (SAT), Activación Mish, Aumento de Datos Mosaico, Regularización DropBlock y Pérdida CIoU. Estas características se combinan para lograr resultados de vanguardia.

Un detector de objetos típico está compuesto por varias partes, incluyendo la entrada, el backbone (espinazo), el neck (cuello) y el head (cabeza). El backbone de YOLOv4 está pre-entrenado en ImageNet y se utiliza para predecir las clases y las cajas delimitadoras de los objetos. El backbone puede ser de varios modelos, incluyendo VGG, ResNet, ResNeXt o DenseNet. La parte del neck del detector se utiliza para recolectar mapas de características de diferentes etapas y generalmente incluye varias rutas de abajo hacia arriba y varias rutas de arriba hacia abajo. La parte de la cabeza es la que se utiliza para realizar las detecciones y clasificaciones finales de objetos.

## Bolsa de regalos

YOLOv4 también utiliza métodos conocidos como "bolsa de regalos" (bag of freebies), que son técnicas que mejoran la precisión del modelo durante el entrenamiento sin aumentar el costo de la inferencia. La ampliación de datos es una técnica común de la bolsa de regalos utilizada en la detección de objetos, que aumenta la variabilidad de las imágenes de entrada para mejorar la robustez del modelo. Algunos ejemplos de ampliación de datos incluyen distorsiones fotométricas (ajuste del brillo, contraste, matiz, saturación y ruido de una imagen) y distorsiones geométricas (agregar escalado, recorte, volteo y rotación aleatorios). Estas técnicas ayudan al modelo a generalizar mejor para diferentes tipos de imágenes.

## Características y rendimiento

YOLOv4 está diseñado para obtener una velocidad y precisión óptimas en la detección de objetos. La arquitectura de YOLOv4 incluye CSPDarknet53 como backbone, PANet como neck y YOLOv3 como cabeza de detección. Este diseño permite que YOLOv4 realice la detección de objetos a una velocidad impresionante, lo que lo hace adecuado para aplicaciones en tiempo real. YOLOv4 también sobresale en precisión, logrando resultados de vanguardia en los benchmarks de detección de objetos.

## Ejemplos de uso

Hasta el momento de escribir este documento, Ultralytics actualmente no admite modelos YOLOv4. Por lo tanto, cualquier usuario interesado en usar YOLOv4 deberá consultar directamente el repositorio de YOLOv4 en GitHub para obtener instrucciones de instalación y uso.

Aquí hay un resumen breve de los pasos típicos que podrías seguir para usar YOLOv4:

1. Visita el repositorio de YOLOv4 en GitHub: [https://github.com/AlexeyAB/darknet](https://github.com/AlexeyAB/darknet).

2. Sigue las instrucciones proporcionadas en el archivo README para la instalación. Esto generalmente implica clonar el repositorio, instalar las dependencias necesarias y configurar las variables de entorno necesarias.

3. Una vez que la instalación esté completa, puedes entrenar y usar el modelo según las instrucciones de uso proporcionadas en el repositorio. Esto normalmente implica preparar tu conjunto de datos, configurar los parámetros del modelo, entrenar el modelo y luego usar el modelo entrenado para realizar la detección de objetos.

Ten en cuenta que los pasos específicos pueden variar dependiendo de tu caso de uso específico y del estado actual del repositorio de YOLOv4. Por lo tanto, se recomienda encarecidamente consultar directamente las instrucciones proporcionadas en el repositorio de YOLOv4 en GitHub.

Lamentamos cualquier inconveniente que esto pueda causar y nos esforzaremos por actualizar este documento con ejemplos de uso para Ultralytics una vez que se implemente el soporte para YOLOv4.

## Conclusión

YOLOv4 es un modelo de detección de objetos potente y eficiente que logra un equilibrio entre velocidad y precisión. Su uso de características únicas y técnicas de bolsa de regalos durante el entrenamiento le permite realizar un excelente desempeño en tareas de detección de objetos en tiempo real. YOLOv4 puede ser entrenado y utilizado por cualquier persona con una GPU convencional, lo que lo hace accesible y práctico para una amplia gama de aplicaciones.

## Citaciones y agradecimientos

Nos gustaría reconocer a los autores de YOLOv4 por sus importantes contribuciones en el campo de la detección de objetos en tiempo real:

!!! Quote ""

    === "BibTeX"

        ```bibtex
        @misc{bochkovskiy2020yolov4,
              title={YOLOv4: Optimal Speed and Accuracy of Object Detection},
              author={Alexey Bochkovskiy and Chien-Yao Wang and Hong-Yuan Mark Liao},
              year={2020},
              eprint={2004.10934},
              archivePrefix={arXiv},
              primaryClass={cs.CV}
        }
        ```

El artículo original de YOLOv4 se puede encontrar en [arXiv](https://arxiv.org/pdf/2004.10934.pdf). Los autores han puesto su trabajo a disposición del público, y el código se puede acceder en [GitHub](https://github.com/AlexeyAB/darknet). Apreciamos sus esfuerzos en el avance del campo y en hacer que su trabajo sea accesible para la comunidad en general.
