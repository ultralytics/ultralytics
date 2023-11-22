---
comments: true
description: Explora el YOLOv7, un detector de objetos en tiempo real. Comprende su velocidad superior, precisión impresionante y enfoque único en la optimización de entrenamiento de bolsas de características entrenables.
keywords: YOLOv7, detector de objetos en tiempo real, estado del arte, Ultralytics, conjunto de datos MS COCO, re-parametrización del modelo, asignación dinámica de etiquetas, escalado extendido, escalado compuesto
---

# YOLOv7: Bolsa de Características Entrenable

YOLOv7 es un detector de objetos en tiempo real de última generación que supera a todos los detectores de objetos conocidos tanto en velocidad como en precisión en el rango de 5 FPS a 160 FPS. Tiene la mayor precisión (56.8% AP) entre todos los detectores de objetos en tiempo real conocidos con una velocidad de 30 FPS o superior en la GPU V100. Además, YOLOv7 supera a otros detectores de objetos como YOLOR, YOLOX, Scaled-YOLOv4, YOLOv5 y muchos otros en cuanto a velocidad y precisión. El modelo se entrena desde cero utilizando el conjunto de datos MS COCO sin utilizar ningún otro conjunto de datos o pesos pre-entrenados. El código fuente de YOLOv7 está disponible en GitHub.

![Comparación de YOLOv7 con detectores de objetos SOTA](https://github.com/ultralytics/ultralytics/assets/26833433/5e1e0420-8122-4c79-b8d0-2860aa79af92)
**Comparación de los detectores de objetos de estado del arte.
** Según los resultados en la Tabla 2, sabemos que el método propuesto tiene el mejor equilibrio entre velocidad y precisión de manera integral. Si comparamos YOLOv7-tiny-SiLU con YOLOv5-N (r6.1), nuestro método es 127 fps más rápido y un 10.7% más preciso en AP. Además, YOLOv7 tiene un AP del 51.4% a una velocidad de cuadro de 161 fps, mientras que PPYOLOE-L con el mismo AP tiene solo una velocidad de cuadro de 78 fps. En términos de uso de parámetros, YOLOv7 utiliza un 41% menos que PPYOLOE-L. Si comparamos YOLOv7-X con una velocidad de inferencia de 114 fps con YOLOv5-L (r6.1) con una velocidad de inferencia de 99 fps, YOLOv7-X puede mejorar el AP en un 3.9%. Si se compara YOLOv7-X con YOLOv5-X (r6.1) de una escala similar, la velocidad de inferencia de YOLOv7-X es 31 fps más rápida. Además, en términos de cantidad de parámetros y cálculos, YOLOv7-X reduce un 22% de los parámetros y un 8% de los cálculos en comparación con YOLOv5-X (r6.1), pero mejora el AP en un 2.2% ([Fuente](https://arxiv.org/pdf/2207.02696.pdf)).

## Descripción general

La detección de objetos en tiempo real es un componente importante en muchos sistemas de visión por computadora, incluyendo el seguimiento de múltiples objetos, conducción autónoma, robótica y análisis de imágenes médicas. En los últimos años, el desarrollo de la detección de objetos en tiempo real se ha centrado en el diseño de arquitecturas eficientes y en la mejora de la velocidad de inferencia de diversas CPUs, GPUs y unidades de procesamiento neural (NPUs). YOLOv7 es compatible tanto con GPU para dispositivos móviles como con GPU para dispositivos de escritorio, desde el borde hasta la nube.

A diferencia de los detectores de objetos en tiempo real tradicionales que se centran en la optimización de la arquitectura, YOLOv7 introduce un enfoque en la optimización del proceso de entrenamiento. Esto incluye módulos y métodos de optimización diseñados para mejorar la precisión de la detección de objetos sin aumentar el costo de inferencia, un concepto conocido como "bolsas de características entrenables".

## Características clave

YOLOv7 introduce varias características clave:

1. **Re-parametrización del modelo**: YOLOv7 propone un modelo re-parametrizado planificado, que es una estrategia aplicable a capas en diferentes redes con el concepto de propagación del gradiente.

2. **Asignación dinámica de etiquetas**: El entrenamiento del modelo con múltiples capas de salida presenta un nuevo problema: "¿Cómo asignar objetivos dinámicos para las salidas de diferentes ramas?" Para resolver este problema, YOLOv7 introduce un nuevo método de asignación de etiquetas llamado asignación de etiquetas guiadas de manera gruesa a fina.

3. **Escalado extendido y compuesto**: YOLOv7 propone métodos de "escalado extendido" y "escalado compuesto" para el detector de objetos en tiempo real que pueden utilizar eficazmente los parámetros y cálculos.

4. **Eficiencia**: El método propuesto por YOLOv7 puede reducir eficazmente aproximadamente el 40% de los parámetros y el 50% de los cálculos del detector de objetos en tiempo real de última generación y tiene una velocidad de inferencia más rápida y una mayor precisión de detección.

## Ejemplos de uso

Hasta la fecha de redacción de este documento, Ultralytics no admite actualmente modelos YOLOv7. Por lo tanto, los usuarios interesados en utilizar YOLOv7 deberán consultar directamente el repositorio de GitHub de YOLOv7 para obtener instrucciones de instalación y uso.

Aquí hay un resumen breve de los pasos típicos que podrías seguir para usar YOLOv7:

1. Visita el repositorio de GitHub de YOLOv7: [https://github.com/WongKinYiu/yolov7](https://github.com/WongKinYiu/yolov7).

2. Sigue las instrucciones proporcionadas en el archivo README para la instalación. Esto generalmente implica clonar el repositorio, instalar las dependencias necesarias y configurar las variables de entorno necesarias.

3. Una vez que la instalación esté completa, puedes entrenar y utilizar el modelo según las instrucciones de uso proporcionadas en el repositorio. Esto generalmente implica preparar tu conjunto de datos, configurar los parámetros del modelo, entrenar el modelo y luego utilizar el modelo entrenado para realizar la detección de objetos.

Ten en cuenta que los pasos específicos pueden variar según tu caso de uso específico y el estado actual del repositorio YOLOv7. Por lo tanto, se recomienda encarecidamente consultar directamente las instrucciones proporcionadas en el repositorio de GitHub de YOLOv7.

Lamentamos cualquier inconveniente que esto pueda causar y nos esforzaremos por actualizar este documento con ejemplos de uso para Ultralytics una vez que se implemente el soporte para YOLOv7.

## Citaciones y Agradecimientos

Nos gustaría agradecer a los autores de YOLOv7 por sus importantes contribuciones en el campo de la detección de objetos en tiempo real:

!!! Quote ""

    === "BibTeX"

        ```bibtex
        @article{wang2022yolov7,
          title={{YOLOv7}: Trainable bag-of-freebies sets new state-of-the-art for real-time object detectors},
          author={Wang, Chien-Yao and Bochkovskiy, Alexey and Liao, Hong-Yuan Mark},
          journal={arXiv preprint arXiv:2207.02696},
          year={2022}
        }
        ```

El artículo original de YOLOv7 se puede encontrar en [arXiv](https://arxiv.org/pdf/2207.02696.pdf). Los autores han hecho su trabajo públicamente disponible y el código se puede acceder en [GitHub](https://github.com/WongKinYiu/yolov7). Agradecemos sus esfuerzos en el avance del campo y en hacer su trabajo accesible a la comunidad en general.
