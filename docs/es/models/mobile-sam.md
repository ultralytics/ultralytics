---
comments: true
description: Obtén más información sobre MobileSAM, su implementación, comparación con SAM original y cómo descargarlo y probarlo en el framework de Ultralytics. ¡Mejora tus aplicaciones móviles hoy mismo!
keywords: MobileSAM, Ultralytics, SAM, aplicaciones móviles, Arxiv, GPU, API, codificador de imágenes, decodificador de máscaras, descarga de modelos, método de prueba
---

![Logotipo de MobileSAM](https://github.com/ChaoningZhang/MobileSAM/blob/master/assets/logo2.png?raw=true)

# Segmentación Móvil de Cualquier Cosa (MobileSAM)

El artículo de MobileSAM ahora está disponible en [arXiv](https://arxiv.org/pdf/2306.14289.pdf).

Una demostración de MobileSAM funcionando en una CPU se puede acceder en este [enlace de demostración](https://huggingface.co/spaces/dhkim2810/MobileSAM). El rendimiento en una CPU Mac i5 tarda aproximadamente 3 segundos. En la demostración de Hugging Face, la interfaz y las CPUs de menor rendimiento contribuyen a una respuesta más lenta, pero sigue funcionando de manera efectiva.

MobileSAM se implementa en varios proyectos, incluyendo [Grounding-SAM](https://github.com/IDEA-Research/Grounded-Segment-Anything), [AnyLabeling](https://github.com/vietanhdev/anylabeling) y [Segment Anything in 3D](https://github.com/Jumpat/SegmentAnythingin3D).

MobileSAM se entrena en una sola GPU con un conjunto de datos de 100k (1% de las imágenes originales) en menos de un día. El código para este entrenamiento estará disponible en el futuro.

## Modelos Disponibles, Tareas Admitidas y Modos de Operación

Esta tabla presenta los modelos disponibles con sus pesos pre-entrenados específicos, las tareas que admiten y su compatibilidad con diferentes modos de operación como [Inference (Inferencia)](../modes/predict.md), [Validation (Validación)](../modes/val.md), [Training (Entrenamiento)](../modes/train.md) y [Export (Exportación)](../modes/export.md), indicados por emojis ✅ para los modos admitidos y emojis ❌ para los modos no admitidos.

| Tipo de Modelo | Pesos Pre-entrenados | Tareas Admitidas                                  | Inferencia | Validación | Entrenamiento | Exportación |
|----------------|----------------------|---------------------------------------------------|------------|------------|---------------|-------------|
| MobileSAM      | `mobile_sam.pt`      | [Segmentación de Instancias](../tasks/segment.md) | ✅          | ❌          | ❌             | ✅           |

## Adaptación de SAM a MobileSAM

Dado que MobileSAM mantiene el mismo pipeline que SAM original, hemos incorporado el pre-procesamiento, post-procesamiento y todas las demás interfaces del original. En consecuencia, aquellos que actualmente utilizan SAM original pueden hacer la transición a MobileSAM con un esfuerzo mínimo.

MobileSAM tiene un rendimiento comparable a SAM original y mantiene el mismo pipeline excepto por un cambio en el codificador de imágenes. Específicamente, reemplazamos el codificador de imágenes original ViT-H pesado (632M) por uno más pequeño, Tiny-ViT (5M). En una sola GPU, MobileSAM funciona a aproximadamente 12ms por imagen: 8ms en el codificador de imágenes y 4ms en el decodificador de máscaras.

La siguiente tabla proporciona una comparación de los codificadores de imágenes basados en ViT:

| Codificador de Imágenes | SAM Original | MobileSAM |
|-------------------------|--------------|-----------|
| Parámetros              | 611M         | 5M        |
| Velocidad               | 452ms        | 8ms       |

Tanto SAM original como MobileSAM utilizan el mismo decodificador de máscaras guiado por instrucciones:

| Decodificador de Máscaras | SAM Original | MobileSAM |
|---------------------------|--------------|-----------|
| Parámetros                | 3.876M       | 3.876M    |
| Velocidad                 | 4ms          | 4ms       |

Aquí está la comparación de todo el pipeline:

| Pipeline Completo (Enc+Dec) | SAM Original | MobileSAM |
|-----------------------------|--------------|-----------|
| Parámetros                  | 615M         | 9.66M     |
| Velocidad                   | 456ms        | 12ms      |

El rendimiento de MobileSAM y SAM original se demuestra utilizando tanto un punto como una caja como instrucciones.

![Imagen con Punto como Instrucción](https://raw.githubusercontent.com/ChaoningZhang/MobileSAM/master/assets/mask_box.jpg?raw=true)

![Imagen con Caja como Instrucción](https://raw.githubusercontent.com/ChaoningZhang/MobileSAM/master/assets/mask_box.jpg?raw=true)

Con su rendimiento superior, MobileSAM es aproximadamente 5 veces más pequeño y 7 veces más rápido que el actual FastSAM. Más detalles están disponibles en la [página del proyecto de MobileSAM](https://github.com/ChaoningZhang/MobileSAM).

## Probando MobileSAM en Ultralytics

Al igual que SAM original, ofrecemos un método sencillo de prueba en Ultralytics, que incluye modos tanto para instrucciones de Punto como para Caja.

### Descarga del Modelo

Puedes descargar el modelo [aquí](https://github.com/ChaoningZhang/MobileSAM/blob/master/weights/mobile_sam.pt).

### Instrucción de Punto

!!! Example "Ejemplo"

    === "Python"
        ```python
        from ultralytics import SAM

        # Carga el modelo
        model = SAM('mobile_sam.pt')

        # Predice un segmento basado en una instrucción de punto
        model.predict('ultralytics/assets/zidane.jpg', points=[900, 370], labels=[1])
        ```

### Instrucción de Caja

!!! Example "Ejemplo"

    === "Python"
        ```python
        from ultralytics import SAM

        # Carga el modelo
        model = SAM('mobile_sam.pt')

        # Predice un segmento basado en una instrucción de caja
        model.predict('ultralytics/assets/zidane.jpg', bboxes=[439, 437, 524, 709])
        ```

Hemos implementado `MobileSAM` y `SAM` utilizando la misma API. Para obtener más información sobre cómo usarlo, consulta la [página de SAM](sam.md).

## Citaciones y Reconocimientos

Si encuentras útil MobileSAM en tu investigación o trabajo de desarrollo, considera citar nuestro artículo:

!!! Quote ""

    === "BibTeX"

        ```bibtex
        @article{mobile_sam,
          title={Faster Segment Anything: Towards Lightweight SAM for Mobile Applications},
          author={Zhang, Chaoning and Han, Dongshen and Qiao, Yu and Kim, Jung Uk and Bae, Sung Ho and Lee, Seungkyu and Hong, Choong Seon},
          journal={arXiv preprint arXiv:2306.14289},
          year={2023}
        }
