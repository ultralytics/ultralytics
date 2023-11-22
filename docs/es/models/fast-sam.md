---
comments: true
description: Explora FastSAM, una solución basada en CNN para la segmentación en tiempo real de objetos en imágenes. Ofrece una interacción mejorada del usuario, eficiencia computacional y es adaptable a diversas tareas de visión.
keywords: FastSAM, aprendizaje automático, solución basada en CNN, segmentación de objetos, solución en tiempo real, Ultralytics, tareas de visión, procesamiento de imágenes, aplicaciones industriales, interacción del usuario
---

# Modelo para Segmentar Cualquier Cosa Rápidamente (FastSAM)

El Modelo para Segmentar Cualquier Cosa Rápidamente (FastSAM) es una solución novedosa basada en CNN que funciona en tiempo real para la tarea de Segmentar Cualquier Cosa. Esta tarea está diseñada para segmentar cualquier objeto dentro de una imagen basándose en diversas indicaciones posibles de interacción del usuario. FastSAM reduce significativamente las demandas computacionales a la vez que mantiene un rendimiento competitivo, lo que lo convierte en una opción práctica para una variedad de tareas de visión.

![Descripción general de la arquitectura del Modelo para Segmentar Cualquier Cosa Rápidamente (FastSAM)](https://user-images.githubusercontent.com/26833433/248551984-d98f0f6d-7535-45d0-b380-2e1440b52ad7.jpg)

## Descripción general

FastSAM está diseñado para abordar las limitaciones del [Modelo para Segmentar Cualquier Cosa (SAM)](sam.md), un modelo Transformer pesado con requerimientos sustanciales de recursos computacionales. FastSAM divide la tarea de segmentar cualquier cosa en dos etapas secuenciales: segmentación de todas las instancias y selección basada en indicaciones. La primera etapa utiliza [YOLOv8-seg](../tasks/segment.md) para producir las máscaras de segmentación de todas las instancias en la imagen. En la segunda etapa, produce la región de interés correspondiente a la indicación.

## Características principales

1. **Solución en tiempo real:** Al aprovechar la eficiencia computacional de las CNN, FastSAM proporciona una solución en tiempo real para la tarea de segmentar cualquier cosa, lo que lo hace valioso para aplicaciones industriales que requieren resultados rápidos.

2. **Eficiencia y rendimiento:** FastSAM ofrece una reducción significativa en las demandas computacionales y de recursos sin comprometer la calidad del rendimiento. Alcanza un rendimiento comparable al de SAM, pero con recursos computacionales drásticamente reducidos, lo que permite su aplicación en tiempo real.

3. **Segmentación guiada por indicaciones:** FastSAM puede segmentar cualquier objeto dentro de una imagen guiado por diversas indicaciones posibles de interacción del usuario, lo que proporciona flexibilidad y adaptabilidad en diferentes escenarios.

4. **Basado en YOLOv8-seg:** FastSAM se basa en [YOLOv8-seg](../tasks/segment.md), un detector de objetos equipado con una rama de segmentación de instancias. Esto le permite producir de manera efectiva las máscaras de segmentación de todas las instancias en una imagen.

5. **Resultados competitivos en pruebas de referencia:** En la tarea de propuesta de objetos de MS COCO, FastSAM alcanza puntuaciones altas a una velocidad significativamente más rápida que [SAM](sam.md) en una sola tarjeta NVIDIA RTX 3090, lo que demuestra su eficiencia y capacidad.

6. **Aplicaciones prácticas:** El enfoque propuesto proporciona una solución nueva y práctica para un gran número de tareas de visión a una velocidad muy alta, varias veces más rápida que los métodos actuales.

7. **Factibilidad de compresión del modelo:** FastSAM demuestra la factibilidad de un camino que puede reducir significativamente el esfuerzo computacional al introducir una prioridad artificial en la estructura, abriendo así nuevas posibilidades para la arquitectura de modelos grandes en tareas generales de visión.

## Modelos disponibles, tareas admitidas y modos de funcionamiento

Esta tabla presenta los modelos disponibles con sus pesos pre-entrenados específicos, las tareas que admiten y su compatibilidad con diferentes modos de funcionamiento, como [Inference](../modes/predict.md) (inferencia), [Validation](../modes/val.md) (validación), [Training](../modes/train.md) (entrenamiento) y [Export](../modes/export.md) (exportación), indicados mediante emojis ✅ para los modos admitidos y emojis ❌ para los modos no admitidos.

| Tipo de modelo | Pesos pre-entrenados | Tareas admitidas                                  | Inferencia | Validación | Entrenamiento | Exportación |
|----------------|----------------------|---------------------------------------------------|------------|------------|---------------|-------------|
| FastSAM-s      | `FastSAM-s.pt`       | [Segmentación de Instancias](../tasks/segment.md) | ✅          | ❌          | ❌             | ✅           |
| FastSAM-x      | `FastSAM-x.pt`       | [Segmentación de Instancias](../tasks/segment.md) | ✅          | ❌          | ❌             | ✅           |

## Ejemplos de uso

Los modelos FastSAM son fáciles de integrar en tus aplicaciones Python. Ultralytics proporciona una API y comandos de línea de comandos (CLI) fáciles de usar para agilizar el desarrollo.

### Uso de predicción

Para realizar la detección de objetos en una imagen, utiliza el método `predict` de la siguiente manera:

!!! Example "Ejemplo"

    === "Python"
        ```python
        from ultralytics import FastSAM
        from ultralytics.models.fastsam import FastSAMPrompt

        # Define una fuente de inferencia
        source = 'ruta/hacia/bus.jpg'

        # Crea un modelo FastSAM
        model = FastSAM('FastSAM-s.pt')  # o FastSAM-x.pt

        # Ejecuta la inferencia en una imagen
        everything_results = model(source, device='cpu', retina_masks=True, imgsz=1024, conf=0.4, iou=0.9)

        # Prepara un objeto de procesamiento de indicaciones
        prompt_process = FastSAMPrompt(source, everything_results, device='cpu')

        # Indicación Everything
        ann = prompt_process.everything_prompt()

        # Caja predeterminada [0,0,0,0] -> [x1,y1,x2,y2]
        ann = prompt_process.box_prompt(bbox=[200, 200, 300, 300])

        # Indicación de texto
        ann = prompt_process.text_prompt(text='una foto de un perro')

        # Indicación de punto
        # puntos predeterminados [[0,0]] [[x1,y1],[x2,y2]]
        # etiqueta_predeterminada [0] [1,0] 0:fondo, 1:primer plano
        ann = prompt_process.point_prompt(points=[[200, 200]], pointlabel=[1])
        prompt_process.plot(annotations=ann, output='./')
        ```

    === "CLI"
        ```bash
        # Carga un modelo FastSAM y segmenta todo con él
        yolo segment predict model=FastSAM-s.pt source=ruta/hacia/bus.jpg imgsz=640
        ```

Este fragmento de código demuestra la simplicidad de cargar un modelo pre-entrenado y realizar una predicción en una imagen.

### Uso de validación

La validación del modelo en un conjunto de datos se puede realizar de la siguiente manera:

!!! Example "Ejemplo"

    === "Python"
        ```python
        from ultralytics import FastSAM

        # Crea un modelo FastSAM
        model = FastSAM('FastSAM-s.pt')  # o FastSAM-x.pt

        # Valida el modelo
        results = model.val(data='coco8-seg.yaml')
        ```

    === "CLI"
        ```bash
        # Carga un modelo FastSAM y valida en el conjunto de datos de ejemplo COCO8 con un tamaño de imagen de 640
        yolo segment val model=FastSAM-s.pt data=coco8.yaml imgsz=640
        ```

Ten en cuenta que FastSAM solo admite la detección y segmentación de una sola clase de objeto. Esto significa que reconocerá y segmentará todos los objetos como si fueran de la misma clase. Por lo tanto, al preparar el conjunto de datos, debes convertir todos los IDs de categoría de objetos a 0.

## Uso oficial de FastSAM

FastSAM también está disponible directamente en el repositorio [https://github.com/CASIA-IVA-Lab/FastSAM](https://github.com/CASIA-IVA-Lab/FastSAM). Aquí hay una descripción general breve de los pasos típicos que podrías seguir para usar FastSAM:

### Instalación

1. Clona el repositorio de FastSAM:
   ```shell
   git clone https://github.com/CASIA-IVA-Lab/FastSAM.git
   ```

2. Crea y activa un entorno Conda con Python 3.9:
   ```shell
   conda create -n FastSAM python=3.9
   conda activate FastSAM
   ```

3. Navega hasta el repositorio clonado e instala los paquetes requeridos:
   ```shell
   cd FastSAM
   pip install -r requirements.txt
   ```

4. Instala el modelo CLIP:
   ```shell
   pip install git+https://github.com/openai/CLIP.git
   ```

### Ejemplo de uso

1. Descarga un [punto de control del modelo](https://drive.google.com/file/d/1m1sjY4ihXBU1fZXdQ-Xdj-mDltW-2Rqv/view?usp=sharing).

2. Utiliza FastSAM para inferencia. Ejemplos de comandos:

    - Segmentar todo en una imagen:
      ```shell
      python Inference.py --model_path ./weights/FastSAM.pt --img_path ./images/dogs.jpg
      ```

    - Segmentar objetos específicos utilizando una indicación de texto:
      ```shell
      python Inference.py --model_path ./weights/FastSAM.pt --img_path ./images/dogs.jpg --text_prompt "el perro amarillo"
      ```

    - Segmentar objetos dentro de una caja delimitadora (proporciona las coordenadas de la caja en formato xywh):
      ```shell
      python Inference.py --model_path ./weights/FastSAM.pt --img_path ./images/dogs.jpg --box_prompt "[570,200,230,400]"
      ```

    - Segmentar objetos cerca de puntos específicos:
      ```shell
      python Inference.py --model_path ./weights/FastSAM.pt --img_path ./images/dogs.jpg --point_prompt "[[520,360],[620,300]]" --point_label "[1,0]"
      ```

Además, puedes probar FastSAM a través de una [demostración en Colab](https://colab.research.google.com/drive/1oX14f6IneGGw612WgVlAiy91UHwFAvr9?usp=sharing) o en la [demostración web de HuggingFace](https://huggingface.co/spaces/An-619/FastSAM) para tener una experiencia visual.

## Citas y agradecimientos

Nos gustaría agradecer a los autores de FastSAM por sus importantes contribuciones en el campo de la segmentación de instancias en tiempo real:

!!! Quote ""

    === "BibTeX"

      ```bibtex
      @misc{zhao2023fast,
            title={Fast Segment Anything},
            author={Xu Zhao and Wenchao Ding and Yongqi An and Yinglong Du and Tao Yu and Min Li and Ming Tang and Jinqiao Wang},
            year={2023},
            eprint={2306.12156},
            archivePrefix={arXiv},
            primaryClass={cs.CV}
      }
      ```

El artículo original de FastSAM se puede encontrar en [arXiv](https://arxiv.org/abs/2306.12156). Los autores han puesto su trabajo a disposición del público, y el código base se puede acceder en [GitHub](https://github.com/CASIA-IVA-Lab/FastSAM). Agradecemos sus esfuerzos para avanzar en el campo y hacer que su trabajo sea accesible a la comunidad en general.
